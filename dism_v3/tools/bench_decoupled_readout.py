"""Real-symbol paired timings for old events vs shared K-state / Q-read tiles."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from check_decoupled_readout import load_native
from decoupled_readout import Executor,baseline_module
from bench_sparse_planner import measure
from check_hard_prefill_training import assert_gradient


def load_input(path):
    with path.open('rb') as f:
        b,h,n,nd=np.fromfile(f,np.int32,4)
        labels=np.fromfile(f,np.int32,2*b*h*n).reshape(2,b,h,n)
        hard=np.fromfile(f,np.uint8,b*h);tau=np.fromfile(f,np.float64,h)
        docs=np.fromfile(f,np.int64,nd*3).reshape(nd,3)
        candidates=np.fromfile(f,np.int32,b*h*n*4).reshape(b,h,n,4)
    assert hard.all() and nd==b and (docs[:,1]==0).all() and (docs[:,2]==n).all()
    return labels,tau,candidates


def error_metrics(a,b):
    a,b=a.flatten().float(),b.flatten().float();an=a.norm();bn=b.norm();diff=a-b
    return dict(relative_l2=(diff.norm()/bn.clamp_min(1e-30)).item(),cosine=(torch.dot(a,b)/(an*bn).clamp_min(1e-30)).item(),
                norm_ratio=(an/bn.clamp_min(1e-30)).item(),signed_bias_over_rms=(diff.mean()/b.square().mean().sqrt().clamp_min(1e-30)).item())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--contexts',nargs='+',type=int,default=[2048,8192])
    ap.add_argument('--layers',nargs='+',type=int,default=list(range(6)));ap.add_argument('--topk',nargs='+',type=int,default=[1,4])
    ap.add_argument('--tiles',nargs='+',default=['32x32','64x32','64x64']);ap.add_argument('--cpu-repeats',type=int,default=5)
    args=ap.parse_args();torch.set_num_threads(1);base=baseline_module(args.directory)
    (args.directory/'kernels').mkdir(parents=True,exist_ok=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    planner=load_native(args.directory/'native/_dism_decoupled.so').Planner(8)
    results=[]
    for n in args.contexts:
        for layer in args.layers:
            path=args.inputs/f'n{n}_layer{layer}.bin';labels,tau,candidates=load_input(path)
            for topk in args.topk:
                count=int(np.prod(labels.shape[1:]));torch.manual_seed(99)
                sq=(torch.randn(count,32,device='cuda')*.2).bfloat16().repeat_interleave(topk,0)
                sk=(torch.randn(count,32,device='cuda')*.2).bfloat16();v=(torch.randn(count,64,device='cuda')*.2).bfloat16()
                do=(torch.randn(count*topk,64,device='cuda')*.2).bfloat16();xs=[sq,sk,v,do]
                for backward in [False,True]:
                    stage='backward' if backward else 'prefill';executors=[];functions=[];records=[]
                    configs=[(64,32,-1)]+[(*map(int,t.split('x')),1 if backward else 64) for t in args.tiles]
                    for bk,bq,threshold in configs:
                        times=[]
                        for rep in range(args.cpu_repeats+1):
                            begin=time.perf_counter();raw=planner.prepare(labels,tau,candidates,topk,backward,bk,bq,threshold)
                            if rep:times.append((time.perf_counter()-begin)*1000)
                        executor=Executor(raw,base,backward);executors.append(executor)
                        fn=(lambda e=executor:e.backward(*xs)) if backward else (lambda e=executor:e.forward(*xs[:3]))
                        functions.append(fn)
                        record=dict(bk=bk,bq=bq,threshold=threshold,cpu_ms=float(np.median(times)),statistics=raw['statistics'],metadata_bytes=executor.metadata_bytes)
                        records.append(record)
                    ref=functions[0]()
                    for i,fn in enumerate(functions):
                        actual=fn();records[i]['errors']={}
                        for name in (['sq_vec','sk_vec','v','rtau'] if backward else ['out']):
                            a,b=(actual[name],ref[name]) if backward else (actual,ref)
                            assert_gradient(a,b,name);records[i]['errors'][name]=error_metrics(a,b)
                        records[i]['matrix_workspace_bytes']=executors[i].workspace_bytes
                        resources=[]
                        for compiled in executors[i].last_kernels:
                            ptx=compiled.asm['ptx'];key=hashlib.sha256(ptx.encode()).hexdigest()[:12]
                            resources.append(dict(name=compiled.name,key=key,registers=compiled.n_regs,spills=compiled.n_spills))
                            (args.directory/'kernels'/f'{key}.ptx').write_text(ptx)
                            (args.directory/'kernels'/f'{key}.cubin').write_bytes(compiled.asm['cubin'])
                        records[i]['resources']=resources
                    del ref,actual,a,b
                    timings=measure(functions)
                    for record,timing in zip(records,timings):record['gpu']=timing
                    row=dict(context=n,layer=layer,topk=topk,stage=stage,variants=records,input=str(path))
                    results.append(row);args.output.write_text(json.dumps(results,indent=2)+'\n')
                    print(n,layer,topk,stage,[(x['threshold'],x['bk'],x['bq'],round(x['cpu_ms'],2),round(x['gpu']['median_ms'],3),round(x['matrix_workspace_bytes']/2**20,1)) for x in records],flush=True)
                    del executors,functions,executor,fn,raw,xs,do,sq,sk,v
                    gc.collect();torch.cuda.empty_cache()
                    # Keep identical vector values between stages without retaining
                    # graph pools or stale lambda references from the previous one.
                    torch.manual_seed(99)
                    sq=(torch.randn(count,32,device='cuda')*.2).bfloat16().repeat_interleave(topk,0)
                    sk=(torch.randn(count,32,device='cuda')*.2).bfloat16();v=(torch.randn(count,64,device='cuda')*.2).bfloat16()
                    do=(torch.randn(count*topk,64,device='cuda')*.2).bfloat16();xs=[sq,sk,v,do]


if __name__=='__main__':main()
