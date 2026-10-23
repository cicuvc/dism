"""CPU-only compact planner latency and shared-pool contention probes.

Uses the production native plan/pack code with ordinary CPU output storage.
No CUDA context, pinned allocation, checkpoint updates, or production changes.
Synthetic labels are explicitly not trained-model label traces.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch
from flash_dism.inference.training_planner import (
    NativeTrainingPlanner, _compact_specification, _views,
)
from flash_dism.inference.build import load_prefill


def prepare(pool, labels, tau, docs, hard):
    start = time.perf_counter()
    native = pool.native.plan(labels, tau, docs, hard, 64, True)
    planned = time.perf_counter()
    stats = dict(native.statistics())
    spec, size = _compact_specification(stats)
    storage = torch.empty(size, dtype=torch.uint8)  # deliberately NOT pinned
    data = _views(storage, spec)
    buckets = []
    for width in (16, 32, 64):
        bucket = {k: data[f'b{width}_{k}'].numpy()
                  for k in ('rows', 'prefixes', 'weights', 'dlog', 'lengths')}
        bucket['stream_heads'] = data[f'b{width}_heads'].numpy()
        buckets.append(bucket)
    multi = {k: data[f'multi_{k}'].numpy() for k in (
        'offsets', 'rows', 'prefixes', 'weights', 'dlog', 'lengths', 'resets', 'stream_heads')}
    allocated = time.perf_counter()
    native.pack_into_compact(buckets, multi)
    packed = time.perf_counter()
    del native
    finished = time.perf_counter()
    return dict(start=start, finished=finished, plan_ms=(planned-start)*1000,
                native_plan_ms=stats.pop('native_plan_ms'),
                allocate_views_ms=(allocated-planned)*1000,
                pack_ms=(packed-allocated)*1000, release_ms=(finished-packed)*1000,
                total_ms=(finished-start)*1000, metadata_bytes=size, statistics=stats)


def summary(samples):
    return {key: dict(median=float(np.median([r[key] for r in samples])),
                     p90=float(np.percentile([r[key] for r in samples], 90)))
            for key in ('plan_ms', 'native_plan_ms', 'allocate_views_ms', 'pack_ms',
                        'release_ms', 'total_ms', 'metadata_bytes')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-state', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--symbols', type=int, nargs='+', default=[512, 32, 8, 1])
    ap.add_argument('--hard-probabilities', type=float, nargs='+', default=[0., .475, .95, 1.])
    ap.add_argument('--stagger-ms', type=float, nargs='+', default=[0., 3.])
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    if not args.output.resolve().is_relative_to(root/'dism_v3/build'):
        ap.error('output must be under dism_v3/build')
    if args.output.exists() or args.repeats < 2:
        ap.error('fresh output and at least two repeats required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        ap.error('set CUDA_VISIBLE_DEVICES="" to guarantee CPU-only execution')
    torch.set_num_threads(1)
    state = torch.load(args.train_state, map_location='cpu', weights_only=False)
    batch = state['prefetched_batch']
    b, n = batch['inputs'].shape
    h = 12
    cu = batch['cu_seqlens'].tolist()
    docs = []
    for lo, hi in zip(cu, cu[1:]):
        if lo == hi:
            continue
        assert lo//n == (hi-1)//n
        docs.append((lo//n, lo % n, (hi-1) % n+1))
    docs = np.asarray(docs, dtype=np.int64)
    del state, batch
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool = NativeTrainingPlanner(args.workers)
    native_path = load_prefill().__file__
    report = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  labels='synthetic iid symbols; independent q/k; tau=.5 per head',
                  geometry=[b,h,n], documents=docs.tolist(),
                  native_module=native_path,
                  native_sha256=hashlib.sha256(Path(native_path).read_bytes()).hexdigest(),
                  limitation='CPU ordinary allocation, no pin/H2D/D2H; shared host has downstream evaluation active',
                  cases=[])
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=args.layers) as executor:
        for symbols in args.symbols:
            labels = [np.random.default_rng(730+i).integers(0, symbols, (2,b,h,n), dtype=np.int32)
                      for i in range(args.layers)]
            uniforms = [np.random.default_rng(19231+i*104729).random((b,h)) for i in range(args.layers)]
            tau = np.full(h, .5, dtype=np.float64)
            for probability in args.hard_probabilities:
                flags = [(u < probability).astype(np.uint8) for u in uniforms]
                for _ in range(2):
                    prepare(pool, labels[0], tau, docs, flags[0])
                isolated = [prepare(pool, labels[0], tau, docs, flags[0]) for _ in range(args.repeats)]
                case = dict(symbols=symbols, hard_probability=probability,
                            active_heads=[int(f.sum()) for f in flags], isolated=summary(isolated),
                            statistics=isolated[-1]['statistics'], concurrent=[])
                for stagger in args.stagger_ms:
                    rounds = []
                    for _ in range(args.repeats):
                        barrier = threading.Barrier(args.layers+1)
                        origin = [None]
                        def job(i):
                            barrier.wait()
                            target = origin[0]+i*stagger/1000
                            delay = target-time.perf_counter()
                            if delay > 0:
                                time.sleep(delay)
                            row = prepare(pool, labels[i], tau, docs, flags[i])
                            row['layer'] = i
                            row['start_ms'] = (row.pop('start')-origin[0])*1000
                            row['finished_ms'] = (row.pop('finished')-origin[0])*1000
                            return row
                        futures = [executor.submit(job,i) for i in range(args.layers)]
                        origin[0] = time.perf_counter()
                        barrier.wait()
                        rounds.append([f.result() for f in futures])
                    case['concurrent'].append(dict(stagger_ms=stagger, rounds=rounds,
                        makespan_ms=float(np.median([max(r['finished_ms'] for r in run) for run in rounds])),
                        per_layer=[summary([run[i] for run in rounds]) for i in range(args.layers)]))
                report['cases'].append(case)
                save()
                print(f"symbols={symbols} h={probability}: isolated={case['isolated']['total_ms']['median']:.2f} ms; "
                      + '; '.join(f"stagger {c['stagger_ms']} ms => {c['makespan_ms']:.2f} ms total" for c in case['concurrent']), flush=True)
    assert not torch.cuda.is_initialized()


if __name__ == '__main__':
    main()
