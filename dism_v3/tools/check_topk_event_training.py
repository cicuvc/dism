"""Production top-k event/fallback timing and SASS resource audit; no training.

Optional --input consumes the local top4_event_probe binary format. Vectors and
router weights are synthetic, so timings are not end-to-end LM throughput.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import torch
import triton
from flash_dism.kernels import topk_hard
from flash_dism.kernels.topk_event import TopkEventBackward
from flash_dism.topk_planner import NativeTopkPlanner,upload_topk


def measure(fn):
    fn();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    for _ in range(3):graph.replay()
    times=[]
    for _ in range(9):
        a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
        a.record()
        for _ in range(10):graph.replay()
        b.record();b.synchronize();times.append(a.elapsed_time(b)/10)
    return float(np.median(times))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path)
    p.add_argument('--tokens',type=int,default=512)
    p.add_argument('--topk',nargs='+',type=int,choices=(1,2,4),default=[1,2,4])
    p.add_argument('--output',type=Path,default=Path('dism_v3/build/topk_event_training'))
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(551)
    if args.input:
        with args.input.open('rb') as f:
            b,h,n,nd=map(int,np.fromfile(f,np.int32,4))
            labels=np.fromfile(f,np.int32,2*b*h*n).reshape(2,b,h,n)
            hard=np.fromfile(f,np.uint8,b*h).reshape(b,h).astype(bool)
            tau=np.fromfile(f,np.float64,h)
            docs=np.fromfile(f,np.int64,nd*3).reshape(nd,3)
            candidates=np.fromfile(f,np.int32,b*h*n*4).reshape(b,h,n,4)
    else:
        b,h,n=2,2,args.tokens
        rng=np.random.default_rng(19)
        labels=rng.integers(0,8,(2,b,h,n),dtype=np.int32)
        candidates=np.argsort(rng.normal(size=(b,h,n,8)),axis=-1)[...,:4].astype(np.int32)
        labels[0]=candidates[...,0]
        hard=np.array([[True,False],[True,True]])
        tau=np.array([0.,.1]);docs=np.array([(i,0,n) for i in range(b)],np.int64)
    labels=torch.from_numpy(labels.copy());hard=torch.from_numpy(hard.copy());tau=torch.from_numpy(tau.copy()).float()
    iq,ik=labels.cuda().unbind();dhard=hard.cuda();dtau=tau.cuda()
    sq=(torch.randn(b,n,h,32,device='cuda')*.2).bfloat16();sk=torch.randn_like(sq)*.2
    v=torch.randn(b,n,h,64,device='cuda').bfloat16();do=torch.randn_like(v)
    seg=torch.zeros(b,n,dtype=torch.int32)
    for index,(row,start,end) in enumerate(docs):seg[row,start:end]=index
    seg=seg.cuda();native=NativeTopkPlanner(4);records=[]
    for topk in args.topk:
        ids=torch.from_numpy(candidates[...,:topk].copy()).cuda()
        pi=torch.randn(b,h,n,topk,device='cuda').softmax(-1)
        _,state=topk_hard.forward(sq,sk,v,iq,ik,ids,pi,dtau,dhard,seg)
        fallback=lambda choice=None:topk_hard.backward(sq,sk,v,do,iq,ik,ids,pi,dhard,seg,state,choice=choice)
        cpu=[]
        for _ in range(3):
            begin=time.perf_counter();raw=native.prepare(labels,tau,ids.cpu(),hard,docs,2**29)
            cpu.append((time.perf_counter()-begin)*1000)
        payload=upload_topk(raw,device=sq.device,stream=torch.cuda.current_stream(),budget=2**30)
        desc=payload.descriptor.to(sq.device);choice=torch.ones(1,dtype=torch.int32,device=sq.device)
        engine=TopkEventBackward(desc,choice)
        event=lambda:engine.backward(sq,sk,v,do,pi)
        def routed():return tuple(a+b for a,b in zip(event(),fallback(choice)))
        expected=fallback();actual=event();errors={}
        for name,a,z in zip(('sq','sk','v','tau','router'),actual,expected):
            a=a.double().flatten();z=z.double().flatten();norm=z.norm()
            error=(a-z).norm()/norm.clamp_min(1e-12)
            cosine=torch.nn.functional.cosine_similarity(a,z,dim=0)
            errors[name]=dict(relative_l2=error.item(),cosine=cosine.item(),norm_ratio=(a.norm()/norm.clamp_min(1e-12)).item(),
                              signed_bias_over_rms=((a-z).mean()/z.square().mean().sqrt().clamp_min(1e-12)).item())
            assert torch.isfinite(a).all() and (norm<1e-8 and a.norm()<1e-8 or error<.025 and cosine>.999),(name,errors[name])
        row=dict(topk=topk,batch=b,heads=h,tokens=n,cpu_ms=float(np.median(cpu[1:])),statistics=raw['statistics'],errors=errors)
        if not args.check_only:
            row['quadratic_ms']=measure(fallback)
            row['ready_event_ms']=measure(routed)
            choice.zero_();row['late_fallback_ms']=measure(routed);choice.fill_(1)
        # Compile and audit the quadratic gate as well as all persistent stages.
        bound,log,dlog,den,lam=state
        dq=torch.zeros_like(sq,dtype=torch.float32);dk=torch.zeros_like(sk,dtype=torch.float32)
        dv=torch.zeros_like(v,dtype=torch.float32);dt=torch.zeros(h,device='cuda');u=torch.zeros_like(pi)
        compiled=topk_hard._backward[(n//32,b*h)](sq,sk,v,do,iq,ik,seg,dhard,bound,ids,pi,log,dlog,den,lam,
            dq,dk,dv,dt,u,n,h,32,64,TOPK=topk,Choice=choice,BK=32,num_warps=4,num_stages=1)
        # Include selection/router and every changed quadratic specialization.
        q=torch.randn(b,n,h,32,device='cuda',dtype=torch.bfloat16)*.2
        c=max(4,int(ids.max())+1,int(iq.max())+1)
        eq=torch.randn(h,c,32,device='cuda')*.2
        selected,_=topk_hard.select(q,eq,iq,dhard,1.,topk)
        topk_hard.router_backward(q,eq,ids,actual[-1],dhard,1.)
        variants=[]
        for name in ('_select','_router_dq','_router_deq','_normalizers','_forward','_backward'):
            variants.extend(getattr(topk_hard,name).device_caches[torch.cuda.current_device()][0].values())
        resources=[];seen=set()
        for kernel in [*engine.last_kernels,compiled,*variants]:
            binary=kernel.asm['cubin'];key=hashlib.sha256(binary).hexdigest()[:16]
            if key in seen:continue
            seen.add(key)
            path=args.output/(key+'.cubin');path.write_bytes(binary)
            sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(path)],text=True)
            resources.append(dict(name=kernel.name,sha=key,registers=kernel.n_regs,spills=kernel.n_spills,
                                  calls=len(re.findall(r'\bCALL\b',sass))))
        row['resources']=resources
        # Null descriptors after rejection exercise all persistent early exits.
        choice.zero_();desc.zero_()
        for value in event():assert torch.count_nonzero(value)==0
        for a,z in zip(routed(),expected):torch.testing.assert_close(a,z,atol=1e-4,rtol=1e-4)
        records.append(row);(args.output/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('resources','statistics')}),flush=True)


if __name__=='__main__':main()
