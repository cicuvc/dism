"""Isolate hard-label D2H, native planning/packing and event-metadata H2D.

Does not change model scheduling. The native planner lacks tau derivatives;
an explicitly synthetic dlog field measures their transport bytes, NOT their
generation cost. A one-document Python derivative-planner time is separate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from flash_dism.inference import HardPrefillPlan
from flash_dism.inference.prefill_triton import TritonPrefillPlan
from hard_prefill_backward_triton import pack_program
from hard_prefill_grad_reference import SymbolicPlanner


def summary(xs):
    return dict(median_ms=statistics.median(xs), min_ms=min(xs), max_ms=max(xs), samples_ms=xs)


def cpu_time(fn, repeats):
    result = fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter()-start)*1000)
    return summary(samples), result


def transfer_time(fn, repeats):
    # Event windows include any host submission gaps; they are not CUPTI DMA
    # durations. Buffers/events are allocated and warmed outside timed calls.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        keepalive = fn()
    torch.cuda.synchronize()
    enqueue, complete, device = [], [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        t = time.perf_counter()
        keepalive = fn()
        enqueue.append((time.perf_counter()-t)*1000)
        end.record()
        end.synchronize()
        complete.append((time.perf_counter()-t)*1000)
        device.append(start.elapsed_time(end))
    return dict(host_call=summary(enqueue), completed_wall=summary(complete), stream_window=summary(device))


def merge_plans(results, jobs, heads):
    fields = {name: [] for name in ('rows', 'prefixes', 'weights', 'lengths', 'resets')}
    offsets, base = [np.zeros(1, np.int64)], 0
    for packed, (_, _, token_base, head) in zip(results, jobs):
        row = packed['rows'].astype(np.int64)
        local = np.where(row >= 0, row, -row-1)
        absolute = (local+token_base)*heads+head
        fields['rows'].append(np.where(row >= 0, absolute, -absolute-1).astype(np.int32))
        for name in ('prefixes', 'weights', 'lengths', 'resets'):
            fields[name].append(packed[name])
        offsets.append(packed['offsets'][1:]+base)
        base += len(packed['lengths'])
    merged = {name: np.concatenate(parts) for name, parts in fields.items()}
    merged['offsets'] = np.concatenate(offsets)
    # Transport proxy only. Native derivative generation is NOT implemented.
    merged['dlog'] = np.zeros_like(merged['weights'])
    return merged


def measure_case(batch, pattern, layout, args, pool):
    n, h = args.tokens, args.heads
    rng = np.random.default_rng(120)
    host = rng.integers(0, 512, (2, batch, h, n), dtype=np.int32)
    if pattern == 'repeat':
        host.fill(0)
    labels = torch.tensor(host, device='cuda')
    pinned_labels = torch.empty(labels.shape, dtype=labels.dtype, pin_memory=True)
    d2h_pageable = transfer_time(lambda: labels.cpu(), args.transfer_repeats)
    d2h_pinned = transfer_time(lambda: pinned_labels.copy_(labels, non_blocking=True), args.transfer_repeats)
    np.testing.assert_array_equal(pinned_labels.numpy(), host)
    # Include the pack kernel as the current model would start from two arrays.
    d2h_pack = transfer_time(lambda: torch.stack((labels[0], labels[1])).cpu(), args.transfer_repeats)
    lengths = [n] if layout == 'uniform' else [256, 512, 256, 1024]*(n//2048)
    jobs = []
    for b in range(batch):
        offset = 0
        for length in lengths:
            for head in range(h):
                jobs.append((host[0, b, head, offset:offset+length],
                             host[1, b, head, offset:offset+length], b*n+offset, head))
            offset += length
    def build(job):
        return HardPrefillPlan(job[0], job[1], .5).native.chunks(16)
    planning, results = cpu_time(lambda: list(pool.map(build, jobs)), args.cpu_repeats)
    merging, merged = cpu_time(lambda: merge_plans(results, jobs, h), args.cpu_repeats)
    pageable = {k: torch.from_numpy(v) for k, v in merged.items()}
    pinned = {k: torch.empty_like(v, pin_memory=True) for k, v in pageable.items()}
    staging, _ = cpu_time(lambda: [pinned[k].copy_(v) for k, v in pageable.items()], args.cpu_repeats)
    targets = {k: torch.empty_like(v, device='cuda') for k, v in pageable.items()}
    h2d_pageable = transfer_time(lambda: [targets[k].copy_(v, non_blocking=True) for k, v in pageable.items()], args.transfer_repeats)
    h2d_pinned = transfer_time(lambda: [targets[k].copy_(v, non_blocking=True) for k, v in pinned.items()], args.transfer_repeats)
    blob = torch.empty(sum(v.numel()*v.element_size() for v in pageable.values()), dtype=torch.uint8, pin_memory=True)
    def stage_blob():
        offset = 0
        for v in pageable.values():
            raw = v.view(torch.uint8).flatten()
            blob[offset:offset+raw.numel()].copy_(raw)
            offset += raw.numel()
    blob_staging, _ = cpu_time(stage_blob, args.cpu_repeats)
    target_blob = torch.empty_like(blob, device='cuda')
    h2d_blob = transfer_time(lambda: target_blob.copy_(blob, non_blocking=True), args.transfer_repeats)
    for k, v in pageable.items():
        torch.testing.assert_close(targets[k].cpu(), v, rtol=0, atol=0)
    torch.testing.assert_close(target_blob.cpu(), blob, rtol=0, atol=0)
    # Current forward uploader: includes dispatch-ID creation and allocations,
    # but ignores synthetic dlog. Keep it separate from the backward proxy.
    uploader = transfer_time(lambda: TritonPrefillPlan.from_packed(merged, batch*h*n), args.cpu_repeats)
    derivative_one_doc, _ = cpu_time(
        lambda: pack_program(SymbolicPlanner(jobs[0][0], jobs[0][1]).compile(.5), 16), 1)
    return dict(batch=batch, heads=h, row_tokens=n, pattern=pattern, layout=layout,
                documents=len(jobs)//h, independent_plans=len(jobs), workers=args.workers,
                label_bytes=labels.numel()*labels.element_size(), metadata_bytes=blob.numel(),
                synthetic_derivative_bytes=pageable['dlog'].numel()*4,
                d2h_pageable=d2h_pageable, d2h_pinned=d2h_pinned, d2h_pack_pageable=d2h_pack,
                native_plan_and_pack=planning, cpu_merge=merging,
                cpu_pin_stage=staging, cpu_blob_stage=blob_staging,
                h2d_pageable=h2d_pageable, h2d_pinned=h2d_pinned, h2d_pinned_blob=h2d_blob,
                existing_forward_uploader=uploader, python_derivative_plan_one_document=derivative_one_doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--tokens', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--patterns', nargs='+', default=['random', 'repeat'])
    parser.add_argument('--layouts', nargs='+', default=['uniform', 'ragged'])
    parser.add_argument('--cpu-repeats', type=int, default=5)
    parser.add_argument('--transfer-repeats', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.tokens <= 0 or args.tokens % 2048:
        parser.error('fresh output path and positive tokens divisible by 2048 required')
    torch.set_num_threads(1)
    HardPrefillPlan([0], [0], .5)  # Native import/build excluded from measurements.
    report = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch in args.batches:
            for layout in args.layouts:
                for pattern in args.patterns:
                    result = measure_case(batch, pattern, layout, args, pool)
                    report['cases'].append(result)
                    args.output.write_text(json.dumps(report, indent=2)+'\n')
                    print(f'B{batch} {layout} {pattern}: D2H pinned '
                          f'{result["d2h_pinned"]["completed_wall"]["median_ms"]:.3f} ms, '
                          f'CPU plan {result["native_plan_and_pack"]["median_ms"]:.3f} ms, '
                          f'merge {result["cpu_merge"]["median_ms"]:.3f} ms, '
                          f'H2D blob {result["h2d_pinned_blob"]["completed_wall"]["median_ms"]:.3f} ms, '
                          f'metadata {result["metadata_bytes"]/2**20:.1f} MiB', flush=True)


if __name__ == '__main__':
    main()
