"""Sweep MXFP8 GEMM/dgrad/wgrad configs for a dense LLM projection shape."""
import argparse
import json
import statistics
from pathlib import Path

import torch

from flash_mxfp8.core import quantize_weight
from flash_mxfp8 import tk_mxfp8_dgrad
from flash_mxfp8 import tk_mxfp8_gemm
from flash_mxfp8 import tk_mxfp8_wgrad
from flash_mxfp8.tk_silu_bwd import SiluBwdQuantizeLibrary


ROOT = Path(__file__).resolve().parents[2]


def measure(plan, repeats):
    for _ in range(2):
        plan()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        plan()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def sweep(library, args, name, a, b, as_, bs, out):
    results = []
    for config in range(len(library.CONFIGS)):
        try:
            plan = library.Library().plan(
                a,
                b,
                as_,
                bs,
                config=config,
                raster_group=13,
                out=out,
            )
        except (RuntimeError, ValueError):
            continue
        try:
            elapsed = measure(plan, args.repeats)
        finally:
            plan.release()
        results.append({"config": config, "ms": elapsed})
    results.sort(key=lambda item: item["ms"])
    return {"kind": name, "best": results[:10], "valid": len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--in-features", type=int, default=384)
    parser.add_argument("--out-features", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(
        ROOT / "dism_v3/build"
    ):
        parser.error("output must be a fresh path under dism_v3/build")

    torch.manual_seed(17)
    x = torch.randn(
        args.tokens, args.in_features, device="cuda", dtype=torch.bfloat16
    )
    dy = torch.randn(
        args.tokens, args.out_features, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(
        args.out_features,
        args.in_features,
        device="cuda",
        dtype=torch.float32,
    )
    xq = SiluBwdQuantizeLibrary().plan_direct(x)
    dyq = SiluBwdQuantizeLibrary().plan_direct(dy, transpose=True)
    xq()
    dyq()
    weight_fp8, weight_scale = quantize_weight(weight)
    report = {
        "tokens": args.tokens,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "gemm": sweep(
            tk_mxfp8_gemm,
            args,
            "gemm",
            xq.row_fp8,
            weight_fp8,
            xq.row_scale,
            weight_scale,
            torch.empty(
                args.tokens, args.out_features, device="cuda", dtype=torch.bfloat16
            ),
        ),
        "dgrad": sweep(
            tk_mxfp8_dgrad,
            args,
            "dgrad",
            dyq.row_fp8,
            weight_fp8,
            dyq.row_scale,
            weight_scale,
            torch.empty(
                args.tokens, args.in_features, device="cuda", dtype=torch.bfloat16
            ),
        ),
        "wgrad": sweep(
            tk_mxfp8_wgrad,
            args,
            "wgrad",
            dyq.column_fp8,
            xq.column_fp8,
            dyq.column_scale,
            xq.column_scale,
            torch.empty(
                args.out_features,
                args.in_features,
                device="cuda",
                dtype=torch.bfloat16,
            ),
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
