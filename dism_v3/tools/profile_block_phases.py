"""Profile forward/backward CUDA time by model block and submodule."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from nanochat.models import ModelSpec, build_model
from nanochat.models.dism import model as dism_model


ROOT = Path(__file__).resolve().parents[2]


def make_event():
    return torch.cuda.Event(enable_timing=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and (
        args.output.exists()
        or not args.output.resolve().is_relative_to(ROOT / "dism_v3/build")
    ):
        parser.error("output must be a fresh path under dism_v3/build")

    spec = ModelSpec.from_dict(json.loads(args.model_spec.read_text()))
    sequence_len = spec.config["sequence_len"]
    if args.tokens % sequence_len:
        parser.error("tokens must contain whole sequences")
    device_batch = args.tokens // sequence_len

    torch.manual_seed(spec.config.get("rng_seed", 1729))
    with torch.device("meta"):
        model = build_model(spec)
    model.to_empty(device="cuda")
    model.init_weights()
    model.train()
    model.set_training_step(max(0, spec.config["anneal_steps"] - 1))

    inputs = torch.randint(
        0, spec.config["vocab_size"], (device_batch, sequence_len), device="cuda"
    )
    targets = torch.randint_like(
        inputs, low=0, high=spec.config["vocab_size"]
    )
    cu_seqlens = torch.arange(
        0,
        args.tokens + 1,
        sequence_len,
        device="cuda",
        dtype=torch.int32,
    )

    forward_records = defaultdict(list)
    backward_pending = defaultdict(list)
    backward_records = defaultdict(list)

    def instrument(name, module):
        original_forward = module.forward

        def profiled_forward(*forward_args, **forward_kwargs):
            start = make_event()
            stop = make_event()
            start.record()
            result = original_forward(*forward_args, **forward_kwargs)
            stop.record()
            forward_records[name].append((start, stop))
            return result

        module.forward = profiled_forward

        def backward_pre_hook(_module, _grad_output):
            start = make_event()
            start.record()
            backward_pending[name].append(start)

        def backward_hook(_module, _grad_input, _grad_output):
            stop = make_event()
            stop.record()
            backward_records[name].append(
                (backward_pending[name].pop(0), stop)
            )

        module.register_full_backward_pre_hook(backward_pre_hook)
        module.register_full_backward_hook(backward_hook)

    instrument("embedding", model.embedding)
    for index, layer in enumerate(model.layers):
        instrument(f"block{index:02d}", layer)
        instrument(f"block{index:02d}.attn", layer.attn)
        instrument(f"block{index:02d}.mlp", layer.mlp)
    instrument("norm", model.norm)

    original_value_first = dism_model._mixer_value_first

    def profiled_value_first(*forward_args, **forward_kwargs):
        start = make_event()
        stop = make_event()
        start.record()
        result = original_value_first(*forward_args, **forward_kwargs)
        stop.record()
        forward_records["value_first"].append((start, stop))
        return result

    dism_model._mixer_value_first = profiled_value_first
    original_document_ce = dism_model.packed_document_ce

    def profiled_document_ce(*forward_args, **forward_kwargs):
        start = make_event()
        stop = make_event()
        start.record()
        result = original_document_ce(*forward_args, **forward_kwargs)
        stop.record()
        forward_records["document_ce"].append((start, stop))
        return result

    dism_model.packed_document_ce = profiled_document_ce

    def one_step(record):
        model.zero_grad(set_to_none=True)
        if record:
            forward_records.clear()
            backward_pending.clear()
            backward_records.clear()
        forward_start = make_event()
        forward_stop = make_event()
        backward_start = make_event()
        backward_stop = make_event()
        forward_start.record()
        loss, _count = model(
            inputs,
            targets,
            cu_seqlens=cu_seqlens,
            loss_reduction="document_sum",
        )
        forward_stop.record()
        backward_start.record()
        loss.backward()
        backward_stop.record()
        torch.cuda.synchronize()
        return forward_start, forward_stop, backward_start, backward_stop

    for _ in range(args.warmup):
        one_step(record=False)
    fwd_start, fwd_stop, bwd_start, bwd_stop = one_step(record=True)

    def elapsed(records):
        return sum(start.elapsed_time(stop) for start, stop in records)

    forward_ms = fwd_start.elapsed_time(fwd_stop)
    backward_ms = bwd_start.elapsed_time(bwd_stop)

    block_forward = {
        index: elapsed(forward_records[f"block{index:02d}"])
        for index in range(len(model.layers))
    }
    block_backward = {
        index: elapsed(backward_records[f"block{index:02d}"])
        for index in range(len(model.layers))
    }
    attention_forward = {
        index: elapsed(forward_records[f"block{index:02d}.attn"])
        for index in range(len(model.layers))
    }
    attention_backward = {
        index: elapsed(backward_records[f"block{index:02d}.attn"])
        for index in range(len(model.layers))
    }
    mlp_forward = {
        index: elapsed(forward_records[f"block{index:02d}.mlp"])
        for index in range(len(model.layers))
    }
    mlp_backward = {
        index: elapsed(backward_records[f"block{index:02d}.mlp"])
        for index in range(len(model.layers))
    }

    def total(values):
        return sum(values.values())

    split = len(model.layers) // 2
    value_first_forward = elapsed(forward_records["value_first"])
    norm_forward = elapsed(forward_records["norm"])
    document_ce_forward = elapsed(forward_records["document_ce"])
    report = {
        "model_spec": str(args.model_spec),
        "tokens": args.tokens,
        "device_batch_size": device_batch,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "forward_share": {
            "embedding": elapsed(forward_records["embedding"]) / forward_ms,
            "pure_gdn_blocks": sum(block_forward[i] for i in range(split))
            / forward_ms,
            "dism_blocks": sum(block_forward[i] for i in range(split, len(model.layers)))
            / forward_ms,
            "value_first": value_first_forward / forward_ms,
            "final_norm": norm_forward / forward_ms,
            "document_ce": document_ce_forward / forward_ms,
            "remaining": (
                forward_ms
                - elapsed(forward_records["embedding"])
                - total(block_forward)
                - value_first_forward
                - norm_forward
                - document_ce_forward
            )
            / forward_ms,
        },
        "backward_share": {
            "pure_gdn_blocks": sum(block_backward[i] for i in range(split))
            / backward_ms,
            "dism_blocks": sum(block_backward[i] for i in range(split, len(model.layers)))
            / backward_ms,
            "head_loss_other": (
                backward_ms - total(block_backward)
            )
            / backward_ms,
        },
        "block_forward_ms": block_forward,
        "block_backward_ms": block_backward,
        "attention_forward_ms": attention_forward,
        "attention_backward_ms": attention_backward,
        "mlp_forward_ms": mlp_forward,
        "mlp_backward_ms": mlp_backward,
        "aggregate": {
            "embedding_forward_ms": elapsed(forward_records["embedding"]),
            "value_first_forward_ms": value_first_forward,
            "final_norm_forward_ms": norm_forward,
            "document_ce_forward_ms": document_ce_forward,
            "pure_gdn_block_forward_ms": sum(
                block_forward[i] for i in range(split)
            ),
            "dism_block_forward_ms": sum(
                block_forward[i] for i in range(split, len(model.layers))
            ),
            "pure_gdn_attention_forward_ms": sum(
                attention_forward[i] for i in range(split)
            ),
            "dism_attention_forward_ms": sum(
                attention_forward[i] for i in range(split, len(model.layers))
            ),
            "mlp_forward_ms": total(mlp_forward),
            "pure_gdn_block_backward_ms": sum(
                block_backward[i] for i in range(split)
            ),
            "dism_block_backward_ms": sum(
                block_backward[i] for i in range(split, len(model.layers))
            ),
            "pure_gdn_attention_backward_ms": sum(
                attention_backward[i] for i in range(split)
            ),
            "dism_attention_backward_ms": sum(
                attention_backward[i] for i in range(split, len(model.layers))
            ),
            "mlp_backward_ms": total(mlp_backward),
        },
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    dism_model._mixer_value_first = original_value_first
    dism_model.packed_document_ce = original_document_ce


if __name__ == "__main__":
    main()
