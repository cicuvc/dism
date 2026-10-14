"""Microbenchmark and numeric report for the fused value-residual operator.

Compares the eager common helper, the ``torch.compile``d helper and the fused
Triton operator on the production packed layout.  Timing needs an idle GPU;
pass ``--skip-timing`` to emit only the accuracy report.  Results (never
during training) go to an ignored JSON under ``dism_v3/build/``.

Usage:
    PYTHONPATH=dism_v3/python:dism_v3/python/flash_dism/kernels:nanochat \\
    python tools/bench_value_residual.py --out ../build/value_residual_bench.json
"""

import argparse
import json
import os
import statistics
import time

import torch

from flash_dism.kernels.value_residual import (
    torch_bwd_value_residual,
    torch_fwd_value_residual,
)
from flash_dism.module import value_residual_mix, value_residual_mix_reference


CASES = [
    # name, batch, seq, heads, dv, dtype, contiguous
    ("packed_72m_dv64_bf16", 1, 32768, 6, 64, torch.bfloat16, True),
    ("packed_72m_dv32_bf16", 1, 32768, 6, 32, torch.bfloat16, True),
    ("short_seq_dv64_bf16", 1, 2048, 6, 64, torch.bfloat16, True),
    ("packed_72m_dv64_fp32", 1, 16384, 6, 64, torch.float32, True),
    ("strided_heads_dv64_bf16", 2, 4096, 6, 64, torch.bfloat16, False),
]


def make_case(case):
    name, batch, seq, heads, dv, dtype, contiguous = case
    shape = (batch, seq, heads, dv)
    v = torch.randn(*shape, device="cuda", dtype=dtype) * 3
    v_first = torch.randn(*shape, device="cuda", dtype=dtype) * 3
    gate = torch.randn(batch, seq, heads, device="cuda", dtype=dtype) * 4
    dy = torch.randn(*shape, device="cuda", dtype=dtype)
    if not contiguous:
        def strided(tensor, gaps):
            strides = tuple(s * g for s, g in zip(tensor.stride(), gaps))
            out = torch.empty_strided(tensor.shape, strides, dtype=tensor.dtype,
                                      device=tensor.device)
            out.copy_(tensor)
            return out
        v = strided(v, (2, 2, 2, 1))
        v_first = strided(v_first, (1, 1, 1, 1))
        dy = strided(dy, (4, 4, 4, 1))
    return name, v, v_first, gate, dy


def metrics(actual, expected):
    a, e = actual.double().flatten(), expected.double().flatten()
    diff = a - e
    norm = e.norm()
    return {
        "max_abs": diff.abs().max().item(),
        "rel_l2": (diff.norm() / norm).item(),
        "cosine": torch.nn.functional.cosine_similarity(a, e, dim=0).item(),
        "norm_ratio": (a.norm() / norm).item(),
    }


def accuracy(case):
    name, v, v_first, gate, dy = make_case(case)
    fused_out = value_residual_mix(v, v_first, gate)
    eager_out = value_residual_mix_reference(v, v_first, gate)
    report = {"arguments": {"batch": v.shape[0], "seq": v.shape[1],
                            "heads": v.shape[2], "dv": v.shape[3],
                            "dtype": str(v.dtype),
                            "contiguous": v.is_contiguous()},
              "out": metrics(fused_out, eager_out)}
    leaves = [t.detach().clone().requires_grad_(True) for t in (v, v_first, gate)]
    eager_leaves = [t.detach().clone().requires_grad_(True) for t in (v, v_first, gate)]
    fused_grads = torch.autograd.grad(value_residual_mix(*leaves), leaves, dy)
    eager_grads = torch.autograd.grad(
        value_residual_mix_reference(*eager_leaves), eager_leaves, dy)
    r64 = [t.double() for t in (v, v_first, gate)]
    s64 = torch.sigmoid(r64[2]).unsqueeze(-1)
    delta64 = r64[0] - r64[1]
    exact = (r64[1] + delta64 * s64,
             dy.double() * s64,
             dy.double() * (1.0 - s64),
             (dy.double() * delta64).sum(-1) * s64.squeeze(-1) * (1.0 - s64.squeeze(-1)))
    for key, got, want, ref in (
            ("dv", fused_grads[0], eager_grads[0], exact[1]),
            ("dvfirst", fused_grads[1], eager_grads[1], exact[2]),
            ("dgate", fused_grads[2], eager_grads[2], exact[3])):
        report[key] = {
            "fused_vs_eager": metrics(got, want),
            "fused_vs_fp64": metrics(got, ref),
            "eager_vs_fp64": metrics(want, ref),
            "fused_bit_exact": bool(torch.equal(got, want)),
        }
    return name, report


def time_variant(run, warmup, repeats):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples),
            "p10_ms": sorted(samples)[max(0, len(samples) // 10)],
            "p90_ms": sorted(samples)[min(len(samples) - 1, 9 * len(samples) // 10)],
            "samples_ms": samples}


def launch_counts(run, iterations=3):
    """CUDA kernel launches per iteration, from the profiler (no tracing I/O)."""
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(iterations):
            run()
        torch.cuda.synchronize()
    counts = {}
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            counts[event.name] = counts.get(event.name, 0) + 1
    counts = {name: count / iterations for name, count in sorted(counts.items())}
    return {"per_iteration": counts, "total": sum(counts.values())}


def timing(case, warmup, repeats):
    name, v, v_first, gate, dy = make_case(case)
    # Leaves are pre-created so the timed region contains the operator, its
    # autograd bookkeeping and its allocations - not test scaffolding.
    leaves = [t.detach().requires_grad_(True) for t in (v, v_first, gate)]
    compiled_helper = torch.compile(value_residual_mix_reference, fullgraph=True, dynamic=True)

    def eager():
        torch.autograd.grad(value_residual_mix_reference(*leaves), leaves, dy)

    def compiled():
        torch.autograd.grad(compiled_helper(*leaves), leaves, dy)

    def fused():
        torch.autograd.grad(value_residual_mix(*leaves), leaves, dy)

    def no_grad(fn):
        def inner():
            with torch.no_grad():
                return fn(v, v_first, gate)
        return inner

    result = {"arguments": {"batch": v.shape[0], "seq": v.shape[1], "heads": v.shape[2],
                            "dv": v.shape[3], "dtype": str(v.dtype)}}
    for label, run in (("eager_reference", eager), ("compiled_helper", compiled),
                       ("fused", fused)):
        torch.cuda.reset_peak_memory_stats()
        result[label] = time_variant(run, warmup, repeats)
        result[label]["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        result[label]["kernels"] = launch_counts(run)
    forward_variants = (
        ("eager_forward", no_grad(value_residual_mix_reference)),
        ("compiled_forward", no_grad(compiled_helper)),
        ("fused_forward", no_grad(value_residual_mix)),
        ("fused_forward_raw", no_grad(torch_fwd_value_residual)),
        ("fused_backward_raw", lambda: torch_bwd_value_residual(v, v_first, gate, dy)),
    )
    for label, run in forward_variants:
        torch.cuda.reset_peak_memory_stats()
        result[label] = time_variant(run, warmup, repeats)
        result[label]["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        result[label]["kernels"] = launch_counts(run)
    for label, make_out in (("backward_only_eager",
                             lambda: value_residual_mix_reference(*leaves)),
                            ("backward_only_fused",
                             lambda: value_residual_mix(*leaves))):
        out = make_out()
        run = lambda: torch.autograd.grad(out, leaves, dy, retain_graph=True)
        result[label] = time_variant(run, warmup, repeats)
    result["host_wall_ms"] = {}
    for label, run in (("eager_reference", eager), ("fused", fused)):
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            run()
        torch.cuda.synchronize()
        result["host_wall_ms"][label] = (
            (time.perf_counter() - start) / repeats * 1e3)
    return name, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(0)
    report = {"torch": torch.__version__,
              "device": torch.cuda.get_device_name(0),
              "accuracy": {}, "timing": {}}
    for case in CASES:
        name, entry = accuracy(case)
        report["accuracy"][name] = entry
        if not args.skip_timing:
            _, entry = timing(case, args.warmup, args.repeats)
            report["timing"][name] = entry
    text = json.dumps(report, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
