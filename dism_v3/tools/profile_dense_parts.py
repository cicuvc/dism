"""Compare steady-state dense sublayer timings for BF16 and MXFP8 models."""
import argparse
import json
import statistics
from pathlib import Path

import torch

from nanochat.models import ModelSpec, build_model


ROOT = Path(__file__).resolve().parents[2]


def timed(fn, *, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(values),
        "samples_ms": values,
    }


def timed_phases(forward, backward, *, warmup, repeats):
    for _ in range(warmup):
        contexts = forward()
        backward(contexts)
    torch.cuda.synchronize()
    forward_values, backward_values = [], []
    for _ in range(repeats):
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_stop = torch.cuda.Event(enable_timing=True)
        backward_stop = torch.cuda.Event(enable_timing=True)
        forward_start.record()
        contexts = forward()
        forward_stop.record()
        backward(contexts)
        backward_stop.record()
        backward_stop.synchronize()
        forward_values.append(forward_start.elapsed_time(forward_stop))
        backward_values.append(forward_stop.elapsed_time(backward_stop))
    return {
        "forward_ms": statistics.median(forward_values),
        "backward_ms": statistics.median(backward_values),
        "forward_samples_ms": forward_values,
        "backward_samples_ms": backward_values,
    }


def build(spec_path):
    spec = ModelSpec.from_dict(json.loads(spec_path.read_text()))
    with torch.device("meta"):
        model = build_model(spec)
    model.to_empty(device="cuda")
    model.init_weights()
    model.train()
    return spec, model


def measure_model(model, tokens):
    width = model.config.n_embd
    x = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
    out = {}

    def ffn_factory():
        for parameter in model.layers[0].mlp.parameters():
            parameter.grad = None
        inp = x.detach().clone().requires_grad_()
        context = torch.autocast("cuda", dtype=torch.bfloat16)
        context.__enter__()
        y = model.layers[0].mlp(inp)
        return context, y

    def finish(contexts):
        context, y = contexts
        y.float().square().mean().backward()
        context.__exit__(None, None, None)

    out["ffn"] = timed_phases(
        ffn_factory,
        finish,
        warmup=2,
        repeats=5,
    )

    for name, layer_idx in (("gdn", 0), ("hybrid", 6)):
        attn = model.layers[layer_idx].attn
        q_conv = getattr(attn, "q_conv", attn.gdn_q_conv)
        k_conv = getattr(attn, "k_conv", attn.gdn_k_conv)
        v_conv = getattr(attn, "v_conv", attn.gdn_v_conv)

        def qkv_factory():
            for parameter in list(q_conv.parameters()) + list(k_conv.parameters()) + list(v_conv.parameters()):
                parameter.grad = None
            inp = x.detach().clone().requires_grad_()
            context = torch.autocast("cuda", dtype=torch.bfloat16)
            context.__enter__()
            values = [q_conv.project_linear(inp),
                      k_conv.project_linear(inp),
                      v_conv.project_linear(inp)]
            if hasattr(attn, "gdn_v_conv") and attn.gdn_v_conv is not v_conv:
                values.append(attn.gdn_v_conv.project_linear(inp))
            return context, values

        def qkv_backward(contexts):
            context, values = contexts
            sum(value.float().square().mean() for value in values).backward()
            context.__exit__(None, None, None)

        out[f"{name}_qkv"] = timed_phases(
            qkv_factory,
            qkv_backward,
            warmup=2,
            repeats=5,
        )

        if hasattr(attn, "sq_proj"):
            def readout_factory():
                for parameter in list(attn.sq_proj.parameters()) + list(attn.sk_proj.parameters()):
                    parameter.grad = None
                inp = x.detach().clone().requires_grad_()
                context = torch.autocast("cuda", dtype=torch.bfloat16)
                context.__enter__()
                return context, (attn.sq_proj(inp), attn.sk_proj(inp))

            def readout_backward(contexts):
                context, values = contexts
                (values[0].float().square().mean()
                 + values[1].float().square().mean()).backward()
                context.__exit__(None, None, None)

            out[f"{name}_readout"] = timed_phases(
                readout_factory,
                readout_backward,
                warmup=2,
                repeats=5,
            )

        def o_proj_factory():
            for parameter in attn.o_proj.parameters():
                parameter.grad = None
            inp = x.detach().clone().requires_grad_()
            context = torch.autocast("cuda", dtype=torch.bfloat16)
            context.__enter__()
            return context, attn.o_proj(inp)

        def o_proj_backward(contexts):
            context, y = contexts
            y.float().square().mean().backward()
            context.__exit__(None, None, None)

        out[f"{name}_o_proj"] = timed_phases(
            o_proj_factory,
            o_proj_backward,
            warmup=2,
            repeats=5,
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-spec", type=Path, required=True)
    parser.add_argument("--mxfp8-spec", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(
        ROOT / "dism_v3/build"
    ):
        parser.error("output must be a fresh path under dism_v3/build")
    _, bf16 = build(args.bf16_spec)
    _, mxfp8 = build(args.mxfp8_spec)
    report = {
        "tokens": args.tokens,
        "bf16": measure_model(bf16, args.tokens),
        "mxfp8": measure_model(mxfp8, args.tokens),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
