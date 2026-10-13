"""Full native derivative planning + parallel pinned packing + real H2D."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from bench_hard_plan_transfers import summary, transfer_time
from flash_dism.inference.training_planner import NativeTrainingPlanner


def case(planner, b, layout, pattern, args):
    n, h = args.tokens, args.heads
    rng = np.random.default_rng(120)
    labels = rng.integers(0, 512, (2, b, h, n), dtype=np.int32)
    if pattern == 'repeat':
        labels.fill(0)
    lengths = [n] if layout == 'uniform' else [256, 512, 256, 1024]*(n//2048)
    docs = []
    for row in range(b):
        start = 0
        for length in lengths:
            docs.append((row, start, start+length))
            start += length
    docs = np.asarray(docs, np.int64)
    hard = np.ones((b, h), np.uint8)
    tau = np.full(h, .5)
    first = planner.prepare(labels, tau, documents=docs, hard=hard)
    storage = first.storage
    for _ in range(2):
        host = planner.prepare(labels, tau, documents=docs, hard=hard, storage=storage)
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        host = planner.prepare(labels, tau, documents=docs, hard=hard, storage=storage)
        samples.append(dict(host.timings, observed_ms=(time.perf_counter()-start)*1000))
    gpu_labels = torch.tensor(labels, device='cuda')
    pinned_labels = torch.empty(gpu_labels.shape, dtype=torch.int32, pin_memory=True)
    d2h = transfer_time(lambda: pinned_labels.copy_(gpu_labels, non_blocking=True), args.repeats)
    gpu_blob = torch.empty_like(storage, device='cuda')
    h2d_copy = transfer_time(lambda: gpu_blob.copy_(storage, non_blocking=True), args.repeats)
    h2d_api = transfer_time(host.upload, args.repeats)
    torch.testing.assert_close(gpu_blob.cpu(), storage, rtol=0, atol=0)
    # Include dependency fences, native planning, direct pinned packing and
    # actual upload. No synthetic derivative fields and no CPU staging copy.
    totals = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        pinned_labels.copy_(gpu_labels, non_blocking=True)
        copied = torch.cuda.Event()
        copied.record()
        copied.synchronize()
        host = planner.prepare(pinned_labels, tau, documents=docs, hard=hard, storage=storage)
        uploaded = host.upload()
        uploaded.ready.synchronize()
        totals.append((time.perf_counter()-start)*1000)
    return dict(batch=b, heads=h, tokens=n, layout=layout, pattern=pattern,
                workers=planner.workers, statistics=host.statistics, cold_first_prepare=first.timings,
                cpu={k: summary([sample[k] for sample in samples]) for k in samples[0]},
                d2h=d2h, h2d_copy=h2d_copy, h2d_api=h2d_api, full_ready=summary(totals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 4, 8, 16])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 16])
    parser.add_argument('--patterns', nargs='+', default=['random', 'repeat'])
    parser.add_argument('--layouts', nargs='+', default=['uniform', 'ragged'])
    parser.add_argument('--tokens', type=int, default=2048)
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.tokens <= 0 or args.tokens % 2048:
        parser.error('fresh output path and tokens divisible by 2048 required')
    torch.set_num_threads(1)
    source = Path(__file__).resolve().parents[1]
    report = dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  device=torch.cuda.get_device_name(), cpu=subprocess.check_output(['lscpu'], text=True),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  sources={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), source/'python/flash_dism/inference/training_planner.py',
                                     *sorted((source/'python/flash_dism/inference/csrc').glob('prefill*'))]}, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for workers in args.workers:
        planner = NativeTrainingPlanner(workers)
        for b in args.batches:
            for layout in args.layouts:
                for pattern in args.patterns:
                    result = case(planner, b, layout, pattern, args)
                    report['cases'].append(result)
                    args.output.write_text(json.dumps(report, indent=2)+'\n')
                    print(f'W{workers} B{b} {layout} {pattern}: '
                          f'CPU {result["cpu"]["observed_ms"]["median_ms"]:.3f} ms '
                          f'(plan {result["cpu"]["plan_ms"]["median_ms"]:.3f}, '
                          f'pack {result["cpu"]["pack_ms"]["median_ms"]:.3f}), '
                          f'full ready {result["full_ready"]["median_ms"]:.3f} ms', flush=True)
        del planner


if __name__ == '__main__':
    main()
