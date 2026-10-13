"""Diagnostic host/device handoff alongside the unchanged dense hard forward.

Not production scheduling. Labels are synthetic and already computed; an
optional preceding dense forward models queued GPU work before labels become
available. Native forward metadata + synthetic dlog transport, not a complete
native derivative planner. No checkpoint/training/model changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from bench_hard_plan_transfers import merge_plans, summary, transfer_time
from flash_dism.forward import forward_core
from flash_dism.inference import HardPrefillPlan
from flash_dism.inference.training_planner import NativeTrainingPlanner


def measure(batch, pattern, args):
    n, h = 2048, 6
    rng = np.random.default_rng(120)
    labels_cpu = rng.integers(0, 512, (2, batch, h, n), dtype=np.int32)
    if pattern == 'repeat':
        labels_cpu.fill(0)
    labels = torch.tensor(labels_cpu, device='cuda')
    pinned = torch.empty(labels.shape, dtype=labels.dtype, pin_memory=True)
    q = torch.zeros((batch, n, h, 64), device='cuda', dtype=torch.bfloat16)
    sq = torch.randn((batch, n, h, 32), device='cuda', dtype=torch.bfloat16)*.2
    sk, value = torch.randn_like(sq)*.2, torch.randn_like(q)*.2
    lse = torch.zeros((batch, n, h), device='cuda')
    direction = torch.ones((batch, h), device='cuda', dtype=torch.bool)
    hard = torch.ones((batch, h, n), device='cuda', dtype=torch.bool)
    tau = torch.full((h,), .5, device='cuda')
    def forward():
        return forward_core(q, q, sq, sk, value, lse, lse, labels[0], labels[1],
                            direction, hard, tau, save_state=True)
    baseline = transfer_time(forward, args.repeats)
    d2h_stream, h2d_stream = torch.cuda.Stream(), torch.cuda.Stream()
    label_ready, copied, uploaded = [torch.cuda.Event() for _ in range(3)]
    forward_start, forward_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    host = pinned.numpy()
    jobs = [(host[0, b, head], host[1, b, head], b*n, head)
            for b in range(batch) for head in range(h)]
    def build(job):
        return HardPrefillPlan(job[0], job[1], .5).native.chunks(16)
    cases = []
    native_planner = NativeTrainingPlanner(args.workers) if args.native_planner else None
    native_storage = None
    with ThreadPoolExecutor(max_workers=args.workers) as planners, ThreadPoolExecutor(max_workers=1) as coordinator:
        def worker(epoch):
            nonlocal native_storage, uploaded
            times = {'worker_start_ms': (time.perf_counter()-epoch)*1000}
            start = time.perf_counter()
            copied.synchronize()  # Worker waits, never the training/main thread.
            times['worker_d2h_wait_ms'] = (time.perf_counter()-start)*1000
            times['host_labels_ready_ms'] = (time.perf_counter()-epoch)*1000
            if native_planner is not None:
                prepared = native_planner.prepare(pinned, np.full(h, .5), storage=native_storage)
                native_storage = prepared.storage
                times.update({f'native_{k}': v for k, v in prepared.timings.items()})
                times['pinned_plan_ready_ms'] = (time.perf_counter()-epoch)*1000
                device_plan = prepared.upload(stream=h2d_stream)
                uploaded = device_plan.ready
                times['upload_submitted_ms'] = (time.perf_counter()-epoch)*1000
                return times, device_plan, prepared
            results = list(planners.map(build, jobs))
            times['plans_done_ms'] = (time.perf_counter()-epoch)*1000
            packed = merge_plans(results, jobs, h)
            times['merge_done_ms'] = (time.perf_counter()-epoch)*1000
            # Dynamic staging is included here, unlike isolated transfer tests.
            sources = {k: torch.from_numpy(v).pin_memory() for k, v in packed.items()}
            times['pin_stage_done_ms'] = (time.perf_counter()-epoch)*1000
            with torch.cuda.stream(h2d_stream):
                targets = {k: v.to('cuda', non_blocking=True) for k, v in sources.items()}
                uploaded.record()
            times['upload_submitted_ms'] = (time.perf_counter()-epoch)*1000
            return times, targets, sources  # Keep buffers alive until upload completes.
        for queued in (False, True):
            samples = []
            for iteration in range(args.repeats+1):
                torch.cuda.synchronize()
                epoch = time.perf_counter()
                if queued:
                    prior = forward()  # Explicit dependency-work scenario, not fence overhead.
                label_ready.record()
                with torch.cuda.stream(d2h_stream):
                    d2h_stream.wait_event(label_ready)
                    pinned.copy_(labels, non_blocking=True)
                    copied.record()
                future = coordinator.submit(worker, epoch)
                forward_start.record()
                host_forward_start = time.perf_counter()
                outputs = forward()
                host_forward_call_ms = (time.perf_counter()-host_forward_start)*1000
                forward_end.record()
                forward_end.synchronize()
                forward_observed_ms = (time.perf_counter()-epoch)*1000
                times, targets, sources = future.result()
                uploaded.synchronize()
                ready = (time.perf_counter()-epoch)*1000
                times.update(ready_observed_ms=ready, dense_forward_observed_ms=forward_observed_ms,
                             host_forward_call_ms=host_forward_call_ms,
                             dense_forward_stream_ms=forward_start.elapsed_time(forward_end),
                             exposed_at_forward_end_ms=max(0., ready-forward_observed_ms))
                if iteration:
                    samples.append(times)
            cases.append(dict(queued_dense_forward_before_labels=queued, samples=samples,
                              medians={k: float(np.median([s[k] for s in samples])) for k in samples[0]}))
    idle = []
    for _ in range(100):
        start = time.perf_counter()
        torch.cuda.synchronize()
        idle.append((time.perf_counter()-start)*1000)
    return dict(batch=batch, heads=h, tokens=n, pattern=pattern, workers=args.workers,
                native_planner=args.native_planner,
                dense_forward_baseline=baseline, idle_device_synchronize=summary(idle), cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 16])
    parser.add_argument('--patterns', nargs='+', default=['random', 'repeat'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--native-planner', action='store_true', help='full native derivatives and pinned batch packing')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a fresh output path')
    torch.set_num_threads(1)
    HardPrefillPlan([0], [0], .5)
    report = dict(device=torch.cuda.get_device_name(), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        for pattern in args.patterns:
            result = measure(batch, pattern, args)
            report['cases'].append(result)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(batch, pattern, [c['medians'] for c in result['cases']], flush=True)


if __name__ == '__main__':
    main()
