"""Concurrent rebuild latency probe; optional GEMMs are a synthetic model proxy.

Per-token waits target a foreground event, never synchronize the whole device
while a background rebuild is pending. This measures eager operator latency,
including Python scheduling; it is not full-model decode throughput.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from flash_dism.inference.concurrent_probe import ConcurrentGpuPlannerCache


def run(args, mode, workers, proxy):
    h, r, d = args.heads, 32, 64
    n, steps = args.prefix, args.steps
    torch.manual_seed(42)
    if args.pattern == 'repeat':
        symbols = torch.zeros(n+steps, h, device='cuda', dtype=torch.int32)
    elif args.pattern == 'periodic':
        symbols = (torch.arange(n+steps, device='cuda', dtype=torch.int32)[:, None] % 7).expand(-1, h)
    else:
        symbols = torch.randint(0, 4, (n+steps, h), device='cuda', dtype=torch.int32)
    labels = torch.stack((symbols, symbols, torch.zeros_like(symbols)), 1).contiguous()
    k = torch.randn(h, n+steps, r, device='cuda', dtype=torch.bfloat16)*.2
    v = torch.randn(h, n+steps, d, device='cuda', dtype=torch.bfloat16)*.2
    # Resident contiguous step inputs avoid transposes/copies in timed region.
    ks, vs = k[:, n:].transpose(0, 1).contiguous(), v[:, n:].transpose(0, 1).contiguous()
    q = torch.randn_like(ks)*.2
    a, b = [torch.randn(args.gemm_size, args.gemm_size, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    out = torch.empty_like(a)
    for _ in range(10):
        torch.mm(a, b, out=out)
    stream = torch.cuda.current_stream()
    interval = steps+1 if mode == 'none' else args.interval
    rows = []
    with ConcurrentGpuPlannerCache(h, r, d, n+steps, [.5]*h,
            rebuild_interval=interval, workers=workers, slots=16, block=256, tile=128) as c:
        c.prime(labels[:n], k[:, :n].contiguous(), v[:, :n].contiguous())
        # Warm the probe's allocator/kernel path before measuring. This snapshot
        # is deliberately discarded; real jobs must still consume newer history.
        ready = torch.cuda.Event(); ready.record(stream)
        warm = c.pool.submit(c._build, n, c.snapshot, ready, time.perf_counter()).result()
        warm.ready.synchronize()
        del warm
        stream.synchronize()
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
        started = time.perf_counter()
        rejected_requests = 0
        for j in range(steps):
            t = time.perf_counter()
            events[j][0].record(stream)
            # Identical requested cutoffs for blocking and async runs; interval
            # is a safety horizon, trigger is the scheduled rebuild cadence.
            if j and j % args.trigger == 0 and mode != 'none':
                if not c.start_rebuild():
                    rejected_requests += 1
                if mode == 'blocking':
                    c.try_publish(wait=True)
            c.step(labels[n+j], ks[j], q[j], vs[j])
            for _ in range(proxy):
                torch.mm(a, b, out=out)
            events[j][1].record(stream)
            events[j][1].synchronize()
            rows.append((time.perf_counter()-t)*1000)
        foreground_ms = (time.perf_counter()-started)*1000
        c.check_status()
        stats = c.stats()
        summary_ranges = [[events[0][0].elapsed_time(e) for e in row['summary_events']]
                          for row in c.completed]
        token_ranges = [[events[0][0].elapsed_time(e) for e in pair] for pair in events]
        pending_at_end = c.pending is not None
        # Drain is explicitly outside measured foreground decoding.
    gpu = [a.elapsed_time(b) for a, b in events]
    def quantiles(x):
        return dict(zip(('p50', 'p95', 'p99', 'max', 'mean'),
                        [float(np.percentile(x, p)) for p in (50, 95, 99, 100)]+[float(np.mean(x))]))
    return dict(mode=mode, workers=workers, proxy_gemms=proxy, foreground_ms=foreground_ms,
        wall_ms=quantiles(rows), stream_ms=quantiles(gpu), pending_at_end=pending_at_end,
        rejected_rebuild_requests=rejected_requests,
        stats=stats, summary_ranges_ms=summary_ranges, token_ranges_ms=token_ranges,
        token_wall_ms=rows, token_stream_ms=gpu)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prefix', type=int, default=8192)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--steps', type=int, default=1536)
    p.add_argument('--heads', type=int, default=12)
    p.add_argument('--interval', type=int, default=1024)
    p.add_argument('--trigger', type=int, default=512)
    p.add_argument('--workers', type=int, nargs='+', default=[64, 128, 256])
    p.add_argument('--modes', nargs='+', choices=['none', 'blocking', 'async'], default=['blocking', 'async'])
    p.add_argument('--proxy-gemms', type=int, nargs='+', default=[0, 4])
    p.add_argument('--gemm-size', type=int, default=2048)
    p.add_argument('--pattern', choices=['repeat', 'periodic', 'random'], default='repeat')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 0 < args.trigger < args.interval:
        p.error('require 0 < trigger < interval')
    results = []
    stream = torch.cuda.Stream(priority=-1)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for proxy in args.proxy_gemms:
            for workers in args.workers:
                for repeat in range(args.repeats):
                    for mode in args.modes:
                        row = run(args, mode, workers, proxy)
                        row['repeat'] = repeat
                        results.append(row)
                        print(json.dumps({k: v for k, v in row.items() if not k.startswith('token_')}), flush=True)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(dict(config={**vars(args), 'output': str(args.output)}, results=results), indent=2))


if __name__ == '__main__':
    main()
