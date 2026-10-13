"""GPU-only native-plan hard backward versus the production CUDA core.

Synthetic packed rows; planning/upload and dense forward saves are prepared
before timing. No model, checkpoint, RNG replay, or training changes.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np
import torch
import cu_flash_dism as cu
import flash_dism

from bench_training import gpu_status, measure, profile_kernels
from check_hard_prefill_backward_triton import inspect_kernel
from check_hard_prefill_training import metrics
from flash_dism.inference.training_planner import NativeTrainingPlanner
from flash_dism.varlen import VarlenLayout
from hard_prefill_backward_triton import HardPrefillBackward, _query_backward, _key_backward
from hard_prefill_backward_fused import FusedHardPrefillBackward


def repeated_tau_oracle(vectors, batch, n, heads, lengths, tau):
    """Independent FP64 dense derivative for identical labels, outside timing.

    A causal (i,j) pair has match length j+1 within its document. This uses
    no event decomposition, chunk prefixes, or CUDA forward normalizer.
    """
    xs = [x.reshape(batch, n, heads, -1).permute(0, 2, 1, 3).double() for x in vectors]
    result = torch.zeros((batch, heads), device='cuda', dtype=torch.float64)
    start = 0
    for length in lengths:
        logw, dl = np.full(length+1, -np.inf), np.zeros(length+1)
        for j in range(1, length+1):
            logw[j] = np.logaddexp(logw[j-1], j*tau)
            dl[j] = np.exp(logw[j-1]-logw[j])*dl[j-1]+np.exp(j*tau-logw[j])*j
        logden = np.logaddexp.accumulate(np.r_[0., logw[1:]])[1:]
        logw, logden, dl = [torch.tensor(a, device='cuda') for a in (logw[1:], logden, dl[1:])]
        causal = torch.arange(length, device='cuda')[:, None] >= torch.arange(length, device='cuda')[None, :]
        weights = torch.exp(torch.where(causal, logw[None, :]-logden[:, None], -torch.inf))
        derivative = weights*(dl[None, :]-(weights@dl)[:, None])
        for b in range(batch):
            sq, sk, v, do = [x[b, :, start:start+length] for x in xs]
            score = (sq@sk.transpose(-1, -2))*(do@v.transpose(-1, -2))
            result[b] += (score*derivative).sum((-1, -2))
        start += length
    return result.sum(0)


def graph_measure(fn, args):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Amortize Python replay overhead, also for tiny reduction stages.
        for _ in range(4):
            retained = fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.samples):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(args.repeats):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end)/(4*args.repeats))
    del retained
    return dict(median_ms=statistics.median(times), samples_ms=times)


def case(batch, pattern, layout_name, args):
    n, h, d, r, dv = args.tokens, 6, 64, 32, 64
    lengths = [n] if layout_name == 'uniform' else [256, 512, 256, 1024]*(n//2048)
    rng = np.random.default_rng(120)
    labels = rng.integers(0, 512, (2, batch, h, n), dtype=np.int32)
    if pattern == 'repeat':
        labels.fill(0)
    docs = []
    for b in range(batch):
        pos = 0
        for length in lengths:
            docs.append((b, pos, pos+length))
            pos += length
    host = NativeTrainingPlanner(16).prepare(labels, np.full(h, args.tau),
                                             documents=np.asarray(docs, np.int64), chunk_size=args.chunk_size)
    device = host.upload()
    device.wait()
    torch.cuda.synchronize()
    # Flatten B,N,H for both implementations; CUDA uses the packed varlen API.
    tokens = batch*n
    torch.manual_seed(318)
    vectors = [(torch.randn((1, tokens, h, c), device='cuda')*.2).bfloat16()
               for c in (r, r, dv, dv)]
    sq, sk, value, dout = vectors
    qk = torch.zeros((1, tokens, h, d), device='cuda', dtype=torch.bfloat16)
    lse = torch.zeros((1, tokens, h), device='cuda')
    iq, ik = [torch.tensor(x.transpose(1, 0, 2).reshape(1, h, tokens).copy(), device='cuda')
              for x in labels]
    direction = torch.ones((1, h), device='cuda', dtype=torch.bool)
    hard = torch.ones((1, h, tokens), device='cuda', dtype=torch.bool)
    tau = torch.full((h,), args.tau, device='cuda')
    cu_seqlens = torch.tensor([0, *np.cumsum(lengths*batch)], device='cuda', dtype=torch.int32)
    table = VarlenLayout.from_cu_seqlens(cu_seqlens, tokens).table
    raw = (qk, qk, sq, sk, value, lse, lse, iq, ik, direction, hard, tau)
    output, norm, state = cu.core_forward(raw, table, 0, True, False, True)
    operands, vertical = state['operands'], state['vertical']
    backend = cu.get_config(r, d, dv)
    delta = backend.varlen_delta(output, dout, table)
    common = (operands, vertical, dout.squeeze(0), norm.flatten(), delta)
    first = backend.varlen_backward_summary(*common, table, False)
    edge = backend.varlen_backward_chunk(first[3], first[4], table, h)
    cuda_full = lambda: cu.core_backward(operands, vertical, output, norm.flatten(),
                                         dout, table, 0, False, False)[0]
    baseline = cuda_full()
    for name in ('sq_vec', 'sk_vec', 'v', 'rtau'):
        if not torch.isfinite(baseline[name]).all():
            raise RuntimeError(f'nonfinite CUDA {name}')
    cuda_stages = {
        'b0_delta': lambda: backend.varlen_delta(output, dout, table),
        'b1_dsq_dsk_dv': lambda: backend.varlen_backward_summary(*common, table, False),
        'b2_chunk': lambda: backend.varlen_backward_chunk(first[3], first[4], table, h),
        'b3_qk_dtau': lambda: backend.varlen_backward_qk(*common, edge, table, False),
        'full': cuda_full,
    }
    result = dict(batch=batch, n=n, h=h, d=d, r=r, dv=dv, pattern=pattern,
                  layout=layout_name, lengths=lengths, tau=args.tau,
                  statistics=host.statistics, gpu_before=gpu_status(),
                  cuda={k: graph_measure(fn, args) for k, fn in cuda_stages.items()},
                  cuda_eager=measure(cuda_full, 3, args.repeats, args.samples),
                  cuda_profile=profile_kernels(cuda_full), triton={})
    flat = [x.reshape(-1, x.shape[-1]) for x in vectors]
    for precision in args.precisions:
        plan = HardPrefillBackward.from_native(device, mma_precision=precision)
        # Upload completed and synchronized above; retain 'device' throughout
        # this case but omit its external event wait from GPU-only capture.
        # This is not a claim that the async handoff API is graph-capturable.
        del plan.device_plan
        full = lambda: plan.backward(*flat)
        actual = full()
        agreement = {}
        for name in ('sq_vec', 'sk_vec', 'v', 'rtau'):
            if not torch.isfinite(actual[name]).all():
                raise RuntimeError(f'nonfinite Triton {name}')
            agreement[name] = metrics(actual[name].float().cpu().numpy().ravel(),
                                      baseline[name].float().cpu().numpy().ravel())
        # Explicit stage launches use fixed buffers. Each scan includes its
        # output zeroing; tau reduction uses the actual completed partials.
        dsq, dsk, grad_v = [torch.zeros_like(x, dtype=torch.float32) for x in flat[:3]]
        partials = torch.empty((2, plan.streams), device='cuda')
        meta = [plan.data[k] for k in ('offsets', 'rows', 'prefixes', 'weights', 'dlog', 'lengths', 'resets')]
        config = (r, dv, plan.c, max(32, r), dv, precision == 'bf16')
        def query():
            dsq.zero_()
            return _query_backward[(plan.streams, 1)](
                *flat, dsq, partials[0], *meta, *config, num_warps=8, num_stages=1)
        def key():
            dsk.zero_()
            grad_v.zero_()
            return _key_backward[(plan.streams, 1)](
                *flat, dsk, grad_v, partials[1], *meta, *config, num_warps=8, num_stages=1)
        query()
        key()
        def reduction():
            contributions = (partials[0]+partials[1]).reshape(plan.streams, 1).sum(-1)
            out = torch.zeros(h, device='cuda')
            return out.scatter_add_(0, plan.data['stream_heads'].long(), contributions)
        stages = dict(query_dsq_dtau=query, key_dsk_dv_dtau=key, tau_reduce=reduction, full=full)
        if not args.skip_register_baseline:
            result['triton'][precision] = dict(
                timings={k: graph_measure(fn, args) for k, fn in stages.items()},
                eager=measure(full, 3, args.repeats, args.samples),
                profile=profile_kernels(full),
                agreement_with_cuda_not_oracle=agreement,
                tau_actual=actual['rtau'].cpu().tolist(),
                tau_cuda=baseline['rtau'].cpu().tolist(),
                kernels=[inspect_kernel(k, args.output.with_suffix('')) for k in plan.last_kernels])
        for warps, long_mode in itertools.product(args.fused_warps, args.fused_long_modes):
            fused = FusedHardPrefillBackward.from_plan(plan, host.data['offsets'].numpy(),
                                                       host_lengths=host.data['lengths'].numpy(),
                                                       num_warps=None if warps == 'auto' else int(warps), long_mode=long_mode)
            fused_full = lambda: fused.backward(*flat)
            fused_actual = fused_full()
            agreement = {}
            for name in ('sq_vec', 'sk_vec', 'v', 'rtau'):
                if not torch.isfinite(fused_actual[name]).all():
                    raise RuntimeError(f'nonfinite fused {name}')
                agreement[name] = metrics(fused_actual[name].float().cpu().numpy().ravel(),
                                          baseline[name].float().cpu().numpy().ravel())
            result['triton'][f'{precision}_fused_w{warps}_{long_mode}'] = dict(
                timings={'full': graph_measure(fused_full, args)},
                eager=measure(fused_full, 3, args.repeats, args.samples),
                profile=profile_kernels(fused_full),
                single_streams=fused.single_count, multi_streams=fused.multi_count,
                multi_chunks=fused.multi_chunks, matrix_snapshot_bytes=fused.matrix_snapshot_bytes,
                single_buckets={str(width): ids.numel() for width, ids in fused.single_buckets},
                agreement_with_cuda_not_oracle=agreement,
                tau_actual=fused_actual['rtau'].cpu().tolist(),
                tau_cuda=baseline['rtau'].cpu().tolist(),
                kernels=[inspect_kernel(k, args.output.with_suffix('')) for k in fused.last_kernels])
    result['gpu_after'] = gpu_status()
    if pattern == 'repeat':
        reference = repeated_tau_oracle(flat, batch, n, h, lengths, args.tau).cpu().numpy()
        result['fp64_dense_tau'] = reference.tolist()
        result['cuda_tau_vs_fp64'] = metrics(baseline['rtau'].cpu().numpy(), reference)
        for item in result['triton'].values():
            item['tau_vs_fp64'] = metrics(item['tau_actual'], reference)
            item['tau_sign_flips_vs_fp64'] = int(np.sum(np.asarray(item['tau_actual'])*reference < 0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 16])
    parser.add_argument('--tokens', type=int, default=2048)
    parser.add_argument('--patterns', nargs='+', choices=['random', 'repeat'], default=['random', 'repeat'])
    parser.add_argument('--layouts', nargs='+', choices=['uniform', 'ragged'], default=['uniform'])
    parser.add_argument('--precisions', nargs='+', choices=['bf16', 'tf32x3'], default=['bf16', 'tf32x3'])
    parser.add_argument('--fused-warps', nargs='+', choices=['auto', '2', '4', '8'], default=[],
                        help='auto uses two warps for width16, four for width32/64')
    parser.add_argument('--fused-long-modes', nargs='+', choices=['scan', 'three_stage'], default=['three_stage'])
    parser.add_argument('--chunk-size', type=int, choices=[16, 32, 64], default=16)
    parser.add_argument('--skip-register-baseline', action='store_true',
                        help='omit repeated timing of the original unfused register scans')
    parser.add_argument('--tau', type=float, default=.5)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.tokens < 2048 or args.tokens % 2048:
        parser.error('fresh output path and tokens divisible by 2048 required')
    root = Path(__file__).resolve().parents[2]
    for module in (cu, flash_dism):
        if not Path(module.__file__).resolve().is_relative_to(root/'dism_v3/python'):
            raise RuntimeError(f'foreign import: {module.__file__}')
    torch.set_num_threads(1)
    report = dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  device=torch.cuda.get_device_name(), torch=torch.__version__,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  extension_sha256=hashlib.sha256(Path(cu.__file__).read_bytes()).hexdigest(),
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path(__file__).with_name('hard_prefill_backward_triton.py'),
                            Path(__file__).with_name('hard_prefill_backward_fused.py'))},
                  method='All-hard synthetic labels and BF16 vectors, prebuilt GPU-resident metadata '
                         'and dense forward saves; completed upload event wait omitted. '
                         'CUDA graphs include output clearing and GPU tau '
                         'reduction, exclude planning/H2D/forward/host allocations. Stage timings '
                         'are independent, not additive. Separate eager and profiler measurements.',
                  cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        for layout in args.layouts:
            for pattern in args.patterns:
                result = case(batch, pattern, layout, args)
                report['cases'].append(result)
                args.output.write_text(json.dumps(report, indent=2)+'\n')
                print(f'B{batch} {layout} {pattern} CUDA={result["cuda"]["full"]["median_ms"]:.4f} ms '
                      + ' '.join(f'{p}={v["timings"]["full"]["median_ms"]:.4f} ms'
                                 for p, v in result['triton'].items()), flush=True)


if __name__ == '__main__':
    main()
