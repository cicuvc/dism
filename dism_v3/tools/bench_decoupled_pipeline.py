"""Serial prepare -> pinned upload -> eager GPU execution, excluding model work.

This deliberately measures a non-overlapped operator pipeline, not training
throughput. All four variants use identical vectors and unchanged GPU kernels.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from bench_decoupled_readout import load_input
from check_decoupled_readout import load_native
from decoupled_readout import Executor,baseline_module


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--native',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--contexts',nargs='+',type=int,default=[2048,8192]);ap.add_argument('--layers',nargs='+',type=int,default=list(range(6)))
    ap.add_argument('--repeats',type=int,default=9)
    ap.add_argument('--topk',nargs='+',type=int,default=[1,4])
    ap.add_argument('--direct-only',action='store_true')
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    kernel_dir=args.directory/'direct_kernels';kernel_dir.mkdir(exist_ok=True)
    base=baseline_module(args.directory);planner=load_native(args.native).Planner(8)
    order=random.Random(932);results=[]
    def pinned(size):return torch.empty(size,dtype=torch.uint8,pin_memory=True).numpy()
    for n in args.contexts:
        for layer in args.layers:
            labels,tau,candidates=load_input(args.inputs/f'n{n}_layer{layer}.bin');count=int(np.prod(labels.shape[1:]))
            for k in args.topk:
                torch.manual_seed(99)
                sq=(torch.randn(count,32,device='cuda')*.2).bfloat16().repeat_interleave(k,0)
                sk=(torch.randn(count,32,device='cuda')*.2).bfloat16();v=(torch.randn(count,64,device='cuda')*.2).bfloat16()
                do=(torch.randn(count*k,64,device='cuda')*.2).bfloat16();torch.cuda.synchronize()
                for backward in (False,True):
                    variants=[('prepare',-1),('prepare',1 if backward else 64),('prepare_direct',-1),('prepare_direct',1 if backward else 64)]
                    if args.direct_only:variants=variants[2:]
                    samples=[[] for _ in variants];resources=[[] for _ in variants]
                    for rep in range(args.repeats+3):
                        indices=list(range(len(variants)));order.shuffle(indices)
                        for i in indices:
                            method,threshold=variants[i];start=time.perf_counter()
                            kwargs=dict(allocator=pinned) if method=='prepare_direct' else {}
                            raw=getattr(planner,method)(labels,tau,candidates,k,backward,64,64,threshold,**kwargs)
                            prepared=time.perf_counter();executor=Executor(raw,base,backward);uploaded=time.perf_counter()
                            output=executor.backward(sq,sk,v,do) if backward else executor.forward(sq,sk,v)
                            torch.cuda.synchronize();executed=time.perf_counter()
                            metadata=executor.metadata_bytes;workspace=executor.workspace_bytes
                            if method=='prepare_direct':assert executor.host_storage.is_pinned() and executor.host_storage.data_ptr()==raw['_storage'].ctypes.data
                            if rep==2:
                                for compiled in executor.last_kernels:
                                    ptx=compiled.asm['ptx'];key=hashlib.sha256(ptx.encode()).hexdigest()[:12]
                                    resources[i].append(dict(name=compiled.name,key=key,registers=compiled.n_regs,spills=compiled.n_spills))
                                    (kernel_dir/f'{key}.ptx').write_text(ptx);(kernel_dir/f'{key}.cubin').write_bytes(compiled.asm['cubin'])
                            del output,executor,raw;done=time.perf_counter()
                            if rep>=3:samples[i].append(dict(total_ms=(done-start)*1000,prepare_ms=(prepared-start)*1000,
                                upload_ms=(uploaded-prepared)*1000,execute_ms=(executed-uploaded)*1000,release_ms=(done-executed)*1000,
                                metadata_bytes=metadata,workspace_bytes=workspace))
                    row=dict(context=n,layer=layer,topk=k,stage='backward' if backward else 'prefill',variants=[])
                    for (method,threshold),values,kernels in zip(variants,samples,resources):
                        row['variants'].append(dict(method=method,threshold=threshold,samples=values,resources=kernels,
                            **{key:float(np.median([x[key] for x in values])) for key in values[0]}))
                    results.append(row);args.output.write_text(json.dumps(results,indent=2)+'\n')
                    print(n,layer,k,row['stage'],[round(x['total_ms'],2) for x in row['variants']],flush=True)
                del sq,sk,v,do;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
