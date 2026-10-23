"""Sequential CPU sweeps on frozen real inputs; no CUDA work or production edits."""
import argparse
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess


def summarize(run):
    totals=[]
    for sample in run['samples']:
        total={k:sum(r[k] for r in sample) for k in ('plan_ms','plan_cpu_ms','allocate_ms','pack_ms','release_ms','total_ms','cpu_ms','metadata_bytes','events')}
        total['plan_busy_cores']=total['plan_cpu_ms']/total['plan_ms']
        total['plan_worker_wall_occupancy']=sum(sum(r['job_ms']) for r in sample)/total['plan_ms']
        totals.append(total)
    return {k:median(t[k] for t in totals) for k in totals[0]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--quick',action='store_true');args=p.parse_args()
    config=[]
    for workers in (1,4,8,12,16,24):
        for variant in ('baseline','lut'):
            config.append((8192,workers,'prefill16',variant,'all'))
    for workers in (8,16,24):
        for mode in ('training16','training64compact','prefill64','inference64compact','bare16'):
            config.append((8192,workers,mode,'baseline','all'))
    config.extend((8192,8,'prefill16',v,'0-7') for v in ('baseline','lut'))
    config.extend((2048,w,m,v,'all') for w in (8,16,24) for m,v in
                  (('prefill16','baseline'),('prefill16','lut'),('training64compact','baseline')))
    if args.quick:
        config=[(8192,8,m,v,'all') for m,v in (('prefill16','baseline'),('prefill16','lut'),('training16','baseline'),('training64compact','baseline'))]
    random.Random(123).shuffle(config)
    result=dict(cpu_affinity=sorted(os.sched_getaffinity(0)),allocator='ordinary new[]; no pinned allocation or CUDA',
                capture_manifest=json.loads((args.root/'inputs/manifest.json').read_text()),runs=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for n,w,mode,variant,affinity in config:
        cmd=[str(args.root/f'bin/compare_{variant}'),str(w),mode,str(args.repeats),'2']
        cmd += [str(args.root/f'inputs/n{n}_layer{i}.bin') for i in range(6)]
        if affinity!='all':cmd=['taskset','-c',affinity]+cmd
        output=subprocess.check_output(cmd,text=True)
        row=json.loads(output);row.update(length=n,variant=variant,affinity=affinity)
        row['medians']=summarize(row);result['runs'].append(row)
        # Same forward program bytes for every control, independent of chunk layout.
        groups=[r['forward_program_hashes'] for r in result['runs'] if r['length']==n]
        if any(h!=groups[0] for h in groups):raise AssertionError('forward program hash mismatch')
        args.output.write_text(json.dumps(result,indent=2))
        print(n,w,mode,variant,affinity,json.dumps(row['medians']),flush=True)


if __name__=='__main__':main()
