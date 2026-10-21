"""Measure steady-state training throughput for one nanochat DISM model spec.

The benchmark uses fixed packed batches, real document-balanced backward and a
real optimizer step. It excludes dataloader I/O, logging and checkpoint writes.
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from nanochat.document_loss import DocumentLossAccumulator
from nanochat.models import ModelSpec, PretrainOptimizerConfig, build_model


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--tokens-per-microbatch", type=int, default=32768)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(
        ROOT / "dism_v3/build"
    ):
        parser.error("output must be a fresh path under dism_v3/build")
    spec = ModelSpec.from_dict(json.loads(args.model_spec.read_text()))
    sequence_len = spec.config["sequence_len"]
    if args.tokens_per_microbatch % sequence_len:
        parser.error("tokens-per-microbatch must contain whole sequences")
    device_batch = args.tokens_per_microbatch // sequence_len
    total_batch = args.tokens_per_microbatch * args.microbatches

    torch.manual_seed(spec.config.get("rng_seed", 1729))
    torch.cuda.manual_seed_all(spec.config.get("rng_seed", 1729))
    with torch.device("meta"):
        model = build_model(spec)
    model.to_empty(device="cuda")
    model.init_weights()
    model.train()
    model.set_training_step(max(0, spec.config["anneal_steps"] - 1))
    optimizer = model.setup_pretraining_optimizer(
        PretrainOptimizerConfig(0.0015, 0.0015, 0.0015, 0.0015, 0.01)
    )

    inputs = torch.randint(
        0,
        spec.config["vocab_size"],
        (device_batch, sequence_len),
        device="cuda",
    )
    targets = torch.randint_like(inputs, low=0, high=spec.config["vocab_size"])
    cu_seqlens = torch.arange(
        0,
        args.tokens_per_microbatch + 1,
        sequence_len,
        device="cuda",
        dtype=torch.int32,
    )
    compiled = torch.compile(model, dynamic=True)

    def step():
        optimizer.zero_grad(set_to_none=True)
        document_step = DocumentLossAccumulator(args.microbatches, "cuda")
        events = []

        def timed(name, function):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = function()
            end.record()
            events.append((name, start, end))
            return result

        for _ in range(args.microbatches):
            loss, count = timed(
                "forward",
                lambda: compiled(
                    inputs,
                    targets,
                    cu_seqlens=cu_seqlens,
                    loss_reduction="document_sum",
                ),
            )
            document_step.add(loss, count)
            timed(
                "backward",
                lambda: (loss / args.microbatches).backward(),
            )
        value = timed(
            "finalize",
            lambda: document_step.finalize(model.parameters()),
        )
        timed(
            "clip",
            lambda: torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf")
            ),
        )
        timed("optimizer", optimizer.step)
        return value, events

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    samples = []
    losses = []
    gpu_breakdown = []
    for _ in range(args.steps):
        start = time.perf_counter()
        loss, events = step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        samples.append(elapsed)
        losses.append(float(loss))
        gpu_breakdown.append({
            name: sum(
                start.elapsed_time(end)
                for event_name, start, end in events
                if event_name == name
            )
            for name in ("forward", "backward", "finalize", "clip", "optimizer")
        })

    report = {
        "model_spec": spec.to_dict(),
        "model_spec_sha256": hashlib.sha256(
            args.model_spec.read_bytes()
        ).hexdigest(),
        "git_head": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "gpu": torch.cuda.get_device_name(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "embedding_dtype": str(model.embedding.weight.dtype),
        "lm_head_dtype": str(model.lm_head.weight.dtype),
        "optimizer": type(optimizer).__name__,
        "tokens_per_microbatch": args.tokens_per_microbatch,
        "microbatches": args.microbatches,
        "total_batch_size": total_batch,
        "step_ms": [sample * 1000 for sample in samples],
        "tokens_per_second": [
            total_batch / sample for sample in samples
        ],
        "mean_step_ms": statistics.mean(samples) * 1000,
        "mean_tokens_per_second": statistics.mean(
            total_batch / sample for sample in samples
        ),
        "losses": losses,
        "gpu_breakdown_ms": gpu_breakdown,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
