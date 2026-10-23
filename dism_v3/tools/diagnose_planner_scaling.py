"""CPU-only worker scaling, process CPU time, and bounded layer-pool comparison.

Synthetic labels, real saved document boundaries; no CUDA or production changes.
CPU seconds / wall seconds measures average busy core equivalents, not work
efficiency or an equality between heterogeneous P/E cores.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import resource
import threading
import time

import numpy as np
import torch
from diagnose_planner_cpu import prepare, NativeTrainingPlanner


def measure(fn):
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    cpu0, start = time.process_time(), time.perf_counter()
    value = fn()
    wall, cpu = time.perf_counter()-start, time.process_time()-cpu0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    return dict(wall_ms=wall*1000, cpu_ms=cpu*1000, busy_cores=cpu/wall,
                voluntary_switches=ru1.ru_nvcsw-ru0.ru_nvcsw,
                involuntary_switches=ru1.ru_nivcsw-ru0.ru_nivcsw, layers=value)


def summarize(samples):
    return {k: float(np.median([s[k] for s in samples])) for k in (
        'wall_ms', 'cpu_ms', 'busy_cores', 'voluntary_switches', 'involuntary_switches')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--geometry-json', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=9)
    ap.add_argument('--symbols', type=int, nargs='+', default=[512,32])
    ap.add_argument('--focus', action='store_true', help='random-order equal-worker scheduling comparison')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    if (args.output.exists() or not args.output.resolve().is_relative_to(root/'dism_v3/build')
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.repeats < 2):
        ap.error('fresh output under build, repeats>=2, CUDA_VISIBLE_DEVICES="" required')
    torch.set_num_threads(1)
    original_affinity = os.sched_getaffinity(0)
    prior = json.loads(args.geometry_json.read_text())
    b,h,n = prior['geometry']
    docs = np.array(prior['documents'], dtype=np.int64)
    flags = [(np.random.default_rng(19231+i*104729).random((b,h)) < .95).astype(np.uint8) for i in range(6)]
    tau = np.full(h,.5,dtype=np.float64)
    result = dict(geometry=[b,h,n], probability=.95, ordinary_cpu_storage=True,
                  native_sha256=prior['native_sha256'], cpu_affinity=sorted(original_affinity),
                  symbols={}, limitation='synthetic iid labels, fixed tau; not trained-label trace; no transfer or pinned allocation')
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    for symbols in args.symbols:
        labels = [np.random.default_rng(730+i).integers(0,symbols,(2,b,h,n),dtype=np.int32) for i in range(6)]
        rows = result['symbols'][str(symbols)] = []
        # Linux cpu0..7 are the higher-frequency core group on this machine.
        # This affinity is only applied to this probe and its subsequently created workers.
        configurations = [('single',1,w,'all',0.) for w in (1,4,8,12,16,20,24)]
        configurations += [('single',1,8,'cpu0-7',0.)]
        configurations += [('concurrent',p,w,'all',stagger)
                           for stagger in (0.,3.) for p,w in ((1,16),(1,24),(3,8),(6,4))]
        configurations += [('serial',1,16,'all',0.),('serial',1,24,'all',0.)]
        if args.focus:
            configurations = [('concurrent',p,w,'all',3.)
                              for p,w in ((1,16),(2,8),(1,20),(1,24),(3,8),(6,4))]
            np.random.default_rng(1747+symbols).shuffle(configurations)
        for mode,pool_count,workers,affinity,stagger in configurations:
            allowed = original_affinity if affinity=='all' else original_affinity & set(range(8))
            if not allowed:
                continue
            os.sched_setaffinity(0,allowed)
            pools = [NativeTrainingPlanner(workers) for _ in range(pool_count)]
            def one(i):
                row=prepare(pools[i % pool_count],labels[i],tau,docs,flags[i])
                row['layer']=i
                return row
            with ThreadPoolExecutor(max_workers=6) as executor:
                def workload():
                    if mode=='single':
                        return [one(0)]
                    if mode=='serial':
                        return [one(i) for i in range(6)]
                    gate=threading.Barrier(7)
                    origin=[None]
                    def job(i):
                        gate.wait()
                        delay=origin[0]+i*stagger/1000-time.perf_counter()
                        if delay>0:
                            time.sleep(delay)
                        row=one(i)
                        row['start_ms']=(row.pop('start')-origin[0])*1000
                        row['finish_ms']=(row.pop('finished')-origin[0])*1000
                        return row
                    futures=[executor.submit(job,i) for i in range(6)]
                    origin[0]=time.perf_counter()
                    gate.wait()
                    return [f.result() for f in futures]
                for _ in range(2):
                    workload()
                samples=[measure(workload) for _ in range(args.repeats)]
            row=dict(mode=mode,pools=pool_count,workers_per_pool=workers,
                     native_workers=pool_count*workers,affinity=affinity,stagger_ms=stagger,
                     medians=summarize(samples),samples=samples)
            if mode=='concurrent':
                row['median_layer_finish_ms']=[float(np.median([s['layers'][i]['finish_ms'] for s in samples])) for i in range(6)]
            rows.append(row)
            save()
            print(symbols,mode,f'{pool_count}x{workers}',affinity,stagger,
                  {k:round(v,2) for k,v in row['medians'].items()},flush=True)
            del pools
            gc.collect()
            os.sched_setaffinity(0,original_affinity)
    assert not torch.cuda.is_initialized()


if __name__=='__main__':
    main()
