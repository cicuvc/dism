"""Measure full CPU/GPU step latency, including transfers and periodic rebuilds."""
import argparse
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from runtime import NativeDecodeCache,load_cuda


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=2048)
    ap.add_argument('--steps',type=int,default=129)
    ap.add_argument('--heads',type=int,default=6)
    ap.add_argument('--r',type=int,default=32)
    ap.add_argument('--dv',type=int,default=64)
    ap.add_argument('--interval',type=int,default=128)
    ap.add_argument('--sample',type=int,default=128)
    ap.add_argument('--threshold',type=int,default=160)
    ap.add_argument('--chunk',type=int,default=128)
    ap.add_argument('--pattern',choices=['random','repeat','zipf'],default='random')
    ap.add_argument('--output')
    args=ap.parse_args()
    load_cuda();torch.manual_seed(431)
    total=args.n+args.steps;h=args.heads
    labels=torch.randint(0,512,(total,3,h),device='cuda',dtype=torch.int32)
    labels[:,2]=0
    if args.pattern=='repeat':labels[:,:2]=0
    if args.pattern=='zipf':
        probs=torch.arange(1,513,device='cuda',dtype=torch.float32).pow(-1.2)
        labels[:,:2]=torch.multinomial(probs,total*2*h,replacement=True).reshape(total,2,h).int()
    k=torch.randn(total,h,args.r,device='cuda',dtype=torch.bfloat16)
    q=torch.randn_like(k);v=torch.randn(total,h,args.dv,device='cuda',dtype=torch.bfloat16)
    cache=NativeDecodeCache(h,args.r,args.dv,total,[.7]*h,rebuild_interval=args.interval,
        sample_interval=args.sample,materialize_threshold=args.threshold,rebuild_chunk=args.chunk)
    kp=k[:args.n].transpose(0,1).contiguous();vp=v[:args.n].transpose(0,1).contiguous()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();cache.prime(labels[:args.n],kp,vp);torch.cuda.synchronize()
    prime=time.perf_counter()-start
    ordinary=[];rebuilding=[]
    for i in range(args.n,total):
        old=cache.memory_stats()['rebuilds'];torch.cuda.synchronize();start=time.perf_counter()
        cache.step(labels[i],k[i],q[i],v[i]);torch.cuda.synchronize()
        dt=(time.perf_counter()-start)*1e6
        (rebuilding if cache.memory_stats()['rebuilds']!=old else ordinary).append(dt)
    all_steps=ordinary+rebuilding
    result=dict(config=vars(args),prime_ms=prime*1000,ordinary_median_us=float(np.median(ordinary)),
        ordinary_p95_us=float(np.percentile(ordinary,95)),rebuild_us=rebuilding,
        amortized_us=float(np.mean(all_steps)),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        live=cache.memory_stats(),torch=torch.__version__,gpu=torch.cuda.get_device_name())
    linear=load_cuda().LinearCache(h,args.r,args.dv,total,[.7]*h,k[0])
    torch.cuda.synchronize();start=time.perf_counter()
    linear.prime(labels[:args.n],kp,vp);torch.cuda.synchronize()
    result['linear_prime_ms']=(time.perf_counter()-start)*1000
    times=[]
    for i in range(args.n,total):
        torch.cuda.synchronize();start=time.perf_counter()
        linear.step(labels[i],k[i],q[i],v[i]);torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1e6)
    result['linear_median_us']=float(np.median(times))
    result['linear_mean_us']=float(np.mean(times))
    result['linear_over_snapshot_amortized']=result['linear_mean_us']/result['amortized_us']
    root=Path(__file__).parent
    result['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for pattern in ('*.cu','*.cuh','*.cpp','*.hpp','runtime.py') for p in root.glob(pattern)}
    print(json.dumps(result,indent=2),flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
