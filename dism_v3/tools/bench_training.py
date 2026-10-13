"""Reproducible packed DISM training-operator benchmark (no model/data downloads).

Run with this checkout's python directory on PYTHONPATH and conda blkw.
Measures the same opaque vocabulary operators used by compiled training, plus
native core stages. Does not time projections, GDN, FFN, optimizer or DDP.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import triton
import cu_flash_dism as cu
import flash_dism
from flash_dism.compiler import voc_forward, voc_backward
from flash_dism.kernels.voc_glue import select_operands
from flash_dism.varlen import VarlenLayout


ROOT = Path(__file__).resolve().parents[2]


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def gpu_status():
    return command('nvidia-smi', '--query-gpu=name,uuid,driver_version,pstate,'
                   'temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,'
                   'memory.used,utilization.gpu', '--format=csv,noheader')


def measure(fn, warmup, repeats, samples):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(samples)]
    # Materialize event handles before timing.
    for start, end in pairs:
        start.record()
        end.record()
    torch.cuda.synchronize()
    stream_ms, wall_ms = [], []
    for start, end in pairs:
        t0 = time.perf_counter()
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000 / repeats)
        stream_ms.append(start.elapsed_time(end) / repeats)
    return {'stream_ms_median': statistics.median(stream_ms),
            'stream_ms_min': min(stream_ms), 'stream_ms_max': max(stream_ms),
            'stream_ms_samples': stream_ms, 'wall_ms_median': statistics.median(wall_ms)}


def memory(fn):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    stats = {'baseline_allocated_bytes': baseline,
             'peak_increment_bytes': torch.cuda.max_memory_allocated() - baseline,
             'retained_increment_bytes': torch.cuda.memory_allocated() - baseline}
    del result
    return stats


def profile_kernels(fn, iterations=3):
    """Separate instrumented device-duration attribution, never a latency claim."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
    kernels = {}
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        item = kernels.setdefault(event.name, {'count': 0, 'total_us': 0.})
        item['count'] += 1
        item['total_us'] += event.time_range.elapsed_us()
    return {'iterations': iterations,
            'device_us_per_call': sum(x['total_us'] for x in kernels.values()) / iterations,
            'kernels': sorted([{'name': name, 'calls_per_iteration': x['count'] / iterations,
                                'us_per_iteration': x['total_us'] / iterations}
                               for name, x in kernels.items()],
                              key=lambda x: -x['us_per_iteration'])}


def run_case(config, batch, pattern, probability, args):
    n, h = config['sequence_len'], config['n_head']
    d, r, dv, vocab = (config[k] for k in
                      ('head_dim', 'readout_dim', 'value_dim', 'qk_vocab_size'))
    if pattern == 'uniform':
        row_lengths = [n]
    else:
        # At N2048: [256,512,256,1024], repeated independently within each row.
        if n % 2048:
            raise ValueError('ragged fixture requires sequence_len divisible by 2048')
        row_lengths = [n // 8, n // 4, n // 8, n // 2]
    lengths = row_lengths * batch
    tokens = n * batch
    torch.manual_seed(args.seed)
    def vector(channels):
        return torch.nn.functional.silu(torch.randn(1, tokens, h, channels, device='cuda')).bfloat16()
    q, k, sq, sk, v = [vector(c) for c in (d, d, r, r, dv)]
    eq, ek = [torch.nn.functional.silu(torch.randn(h, vocab, d, device='cuda')) for _ in range(2)]
    tau = torch.full((h,), args.tau, device='cuda')
    direction = torch.ones(1, h, dtype=torch.bool, device='cuda')
    # Common random numbers across probability cases; one flag per row/head.
    row_hard = torch.rand(1, h, batch, device='cuda') < probability
    hard = row_hard.unsqueeze(-1).expand(1, h, batch, n).reshape(1, h, tokens).contiguous()
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                              dtype=torch.int32, device='cuda')
    layout = VarlenLayout.from_cu_seqlens(boundaries, tokens)
    table, flat = layout.table, layout.table.flatten()
    inputs = [q, k, sq, sk, v, eq, ek, tau, torch.tensor(1.), direction, hard]
    dout = (torch.randn_like(v) * .1).contiguous()
    forward = lambda: voc_forward(*inputs, boundaries, flat)
    saved = forward()
    backward = lambda: voc_backward(inputs, saved, boundaries, dout, flat)
    gradients = backward()
    if not all(torch.isfinite(t).all().item() for t in [saved[0], *gradients]):
        raise RuntimeError('nonfinite output/gradient; refusing to record timing')
    del gradients
    def pair():
        current = forward()
        return voc_backward(inputs, current, boundaries, dout, flat)

    settle_until = time.perf_counter() + args.settle_seconds
    while time.perf_counter() < settle_until:
        pair()
    torch.cuda.synchronize()
    result = {'batch_rows': batch, 'row_length': n, 'heads': h, 'D': d, 'R': r,
              'DV': dv, 'vocabulary': vocab, 'tokens': tokens, 'pattern': pattern,
              'document_lengths': lengths, 'sum_document_lengths_squared': sum(x*x for x in lengths),
              'hard_probability': probability, 'realized_hard_fraction': row_hard.float().mean().item(),
              'tau': args.tau, 'finite_output_and_gradients': True,
              'checkpoint_budget_bytes': layout.checkpoint_bytes(h), 'gpu_before': gpu_status()}
    result['memory'] = {'forward': memory(forward), 'backward': memory(backward), 'pair': memory(pair)}
    result['timings'] = timings = {}
    # Pair is measured directly; it is not inferred by adding isolated timings.
    for name, fn in [('voc_forward', forward), ('voc_backward', backward), ('voc_fwd_bwd', pair)]:
        timings[name] = measure(fn, args.warmup, args.repeats, args.samples)

    output, qfk, kfq, lk, lq, ik, iq, vertical, boundary, norm = saved
    qv, kv = select_operands(q, k, qfk, kfq, direction)
    raw = (qv, kv, sq, sk, v, lq.transpose(1, 2), lk.transpose(1, 2), iq, ik, direction, hard, tau)
    operands = list(cu.prepare_operands(raw, True, True)) + [boundary]
    core_fwd = lambda: cu.core_forward(raw, table, 0, True, False, True)
    core_bwd = lambda: cu.core_backward(operands, vertical, output, norm.flatten(), dout,
                                       table, 0, False, False)
    backend = cu.get_config(r, d, dv)
    a, b = backend.varlen_summary(operands[:12], table)
    delta = backend.varlen_delta(output, dout, table)
    common = (operands, vertical, dout.squeeze(0), norm.flatten(), delta)
    first = backend.varlen_backward_summary(*common, table, False)
    edge = backend.varlen_backward_chunk(first[3], first[4], table, h)
    stages = [('core_forward', core_fwd), ('core_backward', core_bwd),
              ('f1_summary', lambda: backend.varlen_summary(operands[:12], table)),
              ('f2_chunk', lambda: backend.varlen_chunk(a, b, table, h)),
              ('f3_forward', lambda: backend.varlen_forward(operands, table, True, False)),
              ('b0_delta', lambda: backend.varlen_delta(output, dout, table)),
              ('b1_summary', lambda: backend.varlen_backward_summary(*common, table, False)),
              ('b2_chunk', lambda: backend.varlen_backward_chunk(first[3], first[4], table, h)),
              ('b3_qk', lambda: backend.varlen_backward_qk(*common, edge, table, False))]
    for name, fn in stages:
        timings[name] = measure(fn, args.warmup, args.repeats, args.samples)
    result['operator_tokens_per_second'] = tokens * 1000 / timings['voc_fwd_bwd']['stream_ms_median']
    if args.profile:
        result['profile'] = {'forward': profile_kernels(forward), 'backward': profile_kernels(backward)}
    result['gpu_after'] = gpu_status()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'nanochat/configs/sequence_true_72m_valres.json')
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--patterns', nargs='+', choices=['uniform', 'ragged'], default=['uniform', 'ragged'])
    parser.add_argument('--hard-probs', type=float, nargs='+', default=[0., .5, .95, 1.])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--tau', type=float, default=.5)
    parser.add_argument('--settle-seconds', type=float, default=1., help='untimed workload before each case')
    parser.add_argument('--profile', action='store_true', help='separate instrumented per-kernel attribution')
    parser.add_argument('--interp-backward', choices=['baseline', 'atomic', 'atomic-fixed'], default='atomic')
    parser.add_argument('--interp-splits', type=int, default=16, help='token partitions for atomic-fixed only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.batches) < 1 or min(args.repeats, args.samples, args.warmup) < 1:
        parser.error('batch sizes and timing counts must be positive')
    if any(not 0 <= p <= 1 for p in args.hard_probs):
        parser.error('hard probabilities must be in [0,1]')
    if args.output.exists():
        parser.error('output exists; choose a fresh path')
    for module in (flash_dism, cu):
        if not Path(module.__file__).resolve().is_relative_to(ROOT / 'dism_v3/python'):
            raise RuntimeError(f'foreign import: {module.__file__}')
    torch.set_num_threads(1)
    if args.interp_backward != 'atomic':
        from functools import partial
        import flash_dism.compiler as compiler
        from flash_dism.kernels.interp_bwd_split import emb_bwd_split
        from flash_dism.emb_kernel import _emb_bwd_legacy
        # Process-local comparison; the production default is atomic + autotune.
        compiler.emb_bwd_wrapper = (_emb_bwd_legacy if args.interp_backward == 'baseline'
                                    else partial(emb_bwd_split, autotune=False, splits=args.interp_splits))
    config = json.loads(args.config.read_text())['config']
    if config['direction_policy'] != 'true' or config['hard_granularity'] != 'sequence':
        raise ValueError('benchmark requires the fixed-true, sequence-hard baseline')
    props = torch.cuda.get_device_properties(0)
    report = {'schema_version': 1, 'git_head': command('git', 'rev-parse', 'HEAD'),
              'git_status': command('git', 'status', '--short'), 'config': config,
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'torch': torch.__version__, 'triton': triton.__version__, 'torch_cuda': torch.version.cuda,
              'gpu': str(props), 'gpu_initial': gpu_status(),
              'extension': cu.__file__, 'extension_sha256': hashlib.sha256(Path(cu.__file__).read_bytes()).hexdigest(),
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'available_configs': sorted(cu.configs), 'fp32_diagnostics': cu.fp32_enabled(),
              'pythonpath': os.environ.get('PYTHONPATH'),
              'method': 'CUDA-event windows of repeated eager calls to compiled-training custom ops; '
                        'includes allocations, host launch starvation and hot metadata cache; no CUDA graphs. '
                        'Prebuilt layout and synthetic SiLU inputs, not checkpoint activations. '
                        'Backward reuses immutable forward saves, without parameter .grad accumulation. '
                        'Stage times are independent and not additive. Memory is incremental over live fixtures.',
              'cases': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        for pattern in args.patterns:
            for probability in args.hard_probs:
                result = run_case(config, batch, pattern, probability, args)
                report['cases'].append(result)
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                t = result['timings']
                print(f"B{batch} {pattern} p={probability:g} actual={result['realized_hard_fraction']:.3f}: "
                      f"fwd={t['voc_forward']['stream_ms_median']:.3f} "
                      f"bwd={t['voc_backward']['stream_ms_median']:.3f} "
                      f"pair={t['voc_fwd_bwd']['stream_ms_median']:.3f} ms", flush=True)
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
