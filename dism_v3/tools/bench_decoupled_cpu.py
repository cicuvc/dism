"""Randomized paired CPU timings, including format allocation and destruction."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from bench_decoupled_readout import load_input
from check_decoupled_readout import load_native
from check_decoupled_direct import identical


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=11)
    ap.add_argument('--direct',action='store_true',help='also benchmark direct legacy and direct split packing')
    ap.add_argument('--native',type=Path)
    ap.add_argument('--contexts',nargs='+',type=int,default=[2048,8192])
    ap.add_argument('--layers',nargs='+',type=int,default=list(range(6)))
    args=ap.parse_args();torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    planner=load_native(args.native or args.directory/'native/_dism_decoupled.so').Planner(8)
    order=random.Random(749);results=[]
    for n in args.contexts:
        for layer in args.layers:
            labels,tau,candidates=load_input(args.inputs/f'n{n}_layer{layer}.bin')
            for topk in (1,4):
                for backward in (False,True):
                    thresholds=[-1,1 if backward else 64];methods=['prepare']*2
                    if args.direct:
                        for threshold in thresholds:
                            params=(labels,tau,candidates,topk,backward,64,64,threshold)
                            identical(planner.prepare(*params),planner.prepare_direct(*params))
                        thresholds*=2;methods+=['prepare_direct']*2
                    samples=[[] for _ in thresholds]
                    for rep in range(args.repeats+3):
                        indices=list(range(len(thresholds)));order.shuffle(indices)
                        for i in indices:
                            start=time.perf_counter()
                            raw=getattr(planner,methods[i])(labels,tau,candidates,topk,backward,64,64,thresholds[i])
                            stats=raw['statistics'];del raw
                            elapsed=(time.perf_counter()-start)*1000
                            if rep>=3:samples[i].append(dict(wall_ms=elapsed,build_ms=stats['build_ms'],pack_ms=stats['pack_ms']))
                    row=dict(context=n,layer=layer,topk=topk,stage='backward' if backward else 'prefill',variants=[])
                    for method,threshold,measurements in zip(methods,thresholds,samples):
                        row['variants'].append(dict(method=method,threshold=threshold,samples=measurements,
                            **{key:float(np.median([x[key] for x in measurements])) for key in ('wall_ms','build_ms','pack_ms')}))
                    results.append(row);args.output.write_text(json.dumps(results,indent=2)+'\n')
                    print(n,layer,topk,row['stage'],[(x['threshold'],round(x['wall_ms'],2),round(x['pack_ms'],2)) for x in row['variants']],flush=True)


if __name__=='__main__':main()
