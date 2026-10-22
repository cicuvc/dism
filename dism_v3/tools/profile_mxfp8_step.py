"""Profile one resident DISM training step with PyTorch CUDA activities."""
import argparse
import json
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
    parser.add_argument(
        "--cuda-only",
        action="store_true",
        help="omit CPU activities to reduce profiler-induced GPU gaps",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help="record Python stacks in the CPU/CUDA trace",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(
        ROOT / "dism_v3/build"
    ):
        parser.error("output must be a fresh path under dism_v3/build")

    spec = ModelSpec.from_dict(json.loads(args.model_spec.read_text()))
    sequence_len = spec.config["sequence_len"]
    device_batch = args.tokens_per_microbatch // sequence_len
    torch.manual_seed(spec.config.get("rng_seed", 1729))
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
        0, spec.config["vocab_size"],
        (device_batch, sequence_len), device="cuda",
    )
    targets = torch.randint_like(
        inputs, low=0, high=spec.config["vocab_size"]
    )
    cu_seqlens = torch.arange(
        0, args.tokens_per_microbatch + 1, sequence_len,
        device="cuda", dtype=torch.int32,
    )
    compiled = torch.compile(model, dynamic=True)

    def step():
        optimizer.zero_grad(set_to_none=True)
        document_step = DocumentLossAccumulator(args.microbatches, "cuda")
        for _ in range(args.microbatches):
            loss, count = compiled(
                inputs,
                targets,
                cu_seqlens=cu_seqlens,
                loss_reduction="document_sum",
            )
            document_step.add(loss, count)
            (loss / args.microbatches).backward()
        document_step.finalize(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        optimizer.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=(
            [torch.profiler.ProfilerActivity.CUDA]
            if args.cuda_only
            else [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=args.stack,
    ) as profile:
        step()
    torch.cuda.synchronize()
    table = profile.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=80,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".txt").write_text(table + "\n")
    profile.export_chrome_trace(str(args.output.with_suffix(".json")))
    print(table)


if __name__ == "__main__":
    main()
