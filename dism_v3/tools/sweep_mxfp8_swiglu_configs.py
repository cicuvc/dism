"""Coordinate-config sweep for the complete 512/3968 MXFP8 SwiGLU layer."""
import argparse
import gc
import json
import statistics
from pathlib import Path

import torch

from flash_mxfp8 import MXFP8SwiGLU


ROOT = Path(__file__).resolve().parents[2]


def measure(module, x, repeats):
    def run():
        module.zero_grad(set_to_none=True)
        value = x.detach().clone().requires_grad_()
        y = module(value)
        y.float().square().mean().backward()

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    forward, backward, total = [], [], []
    for _ in range(repeats):
        module.zero_grad(set_to_none=True)
        value = x.detach().clone().requires_grad_()
        start = torch.cuda.Event(enable_timing=True)
        stop_forward = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        y = module(value)
        stop_forward.record()
        y.float().square().mean().backward()
        stop.record()
        stop.synchronize()
        forward.append(start.elapsed_time(stop_forward))
        backward.append(stop_forward.elapsed_time(stop))
        total.append(start.elapsed_time(stop))
    return {
        "forward_ms": statistics.median(forward),
        "backward_ms": statistics.median(backward),
        "total_ms": statistics.median(total),
        "forward_samples_ms": forward,
        "backward_samples_ms": backward,
    }


def evaluate(x, cfg, repeats):
    module = MXFP8SwiGLU(
        512,
        3968,
        512,
        device="cuda",
        forward_config=cfg[0],
        dx_config=cfg[1],
        wgrad_config=cfg[2],
        down_forward_config=cfg[3],
        down_dx_config=cfg[4],
        down_dw_config=cfg[5],
        raster_group=13,
    )
    try:
        return measure(module, x, repeats)
    finally:
        del module
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--config", type=int, nargs=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(
        ROOT / "dism_v3/build"
    ):
        parser.error("output must be a fresh path under dism_v3/build")

    torch.manual_seed(23)
    x = torch.randn(args.tokens, 512, device="cuda", dtype=torch.bfloat16)
    if args.config is not None:
        report = {
            "tokens": args.tokens,
            "config": args.config,
            **evaluate(x, args.config, args.repeats),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    current = [8, 12, 14, 8, 17, 12]
    labels = [
        "gate_forward",
        "gate_dx",
        "gate_wgrad",
        "down_forward",
        "down_dx",
        "down_wgrad",
    ]
    candidates = [
        [8, 109, 105, 106, 70, 68, 67, 110, 73, 72],
        [12, 14, 9, 13, 7, 11],
        [14, 13, 9, 12, 7, 11],
        [8, 70, 67, 68, 106, 105, 59],
        [17, 12, 13, 14, 7, 9],
        [12, 14, 13, 7, 9, 11],
    ]
    sweep = {}
    for index, (label, values) in enumerate(zip(labels, candidates)):
        trials = []
        for value in values:
            cfg = list(current)
            cfg[index] = value
            try:
                timing = evaluate(x, cfg, args.repeats)
            except ValueError:
                continue
            trials.append({"value": value, "config": cfg, **timing})
        trials.sort(key=lambda item: item["total_ms"])
        sweep[label] = trials
        current[index] = trials[0]["value"]

    report = {
        "tokens": args.tokens,
        "default": [8, 12, 14, 8, 17, 12],
        "coordinate_best": current,
        "coordinate_best_timing": evaluate(x, current, args.repeats),
        "sweep": sweep,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
