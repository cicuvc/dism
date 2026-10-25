"""Interleaved CPU-only baseline/weighted/binary planner timing on saved inputs.

Run without concurrent builds, tests or profiles. W1 binds CPU 0; W16 uses OS
scheduling. Timings include planning, allocation, packing and destruction.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,nargs='+',required=True)
    ap.add_argument('--workers',type=int,nargs='+',default=[1,16])
    ap.add_argument('--seconds',type=float,default=1.2)
    ap.add_argument('--repeats',type=int,default=5)
    args=ap.parse_args();root=Path(__file__).resolve().parents[2];out=args.experiment.resolve()
    if not out.is_relative_to(root/'dism_v3/build') or (out/'benchmark.json').exists():
        ap.error('experiment must be under dism_v3/build with no existing benchmark.json')
    if args.seconds<=0 or args.repeats<1 or any(w<1 for w in args.workers):ap.error('positive timing/repeat/worker values required')
    variants=('baseline','weighted','binary');records=[]
    for inp in args.inputs:
        for w in args.workers:
            prefix=['taskset','-c','0'] if w==1 else []
            for v in variants:
                subprocess.run(prefix+[str(out/v/'profile/native'),str(inp),'.2',str(w)],check=True,stdout=subprocess.DEVNULL)
            for rep in range(args.repeats):
                order=list(variants);random.Random(9042+rep).shuffle(order)
                for v in order:
                    cmd=prefix+[str(out/v/'profile/native'),str(inp),str(args.seconds),str(w)]
                    row=json.loads(subprocess.check_output(cmd,text=True));row.update(input=inp.stem,workers=w,variant=v,rep=rep)
                    records.append(row)
                (out/'benchmark.json').write_text(json.dumps(records,indent=2)+'\n')
            med={v:statistics.median(a['ms_per_iteration'] for a in records if (a['input'],a['workers'],a['variant'])==(inp.stem,w,v)) for v in variants}
            print(inp.stem,w,med,flush=True)

if __name__=='__main__':main()
