"""Paired frozen-weight RMSNorm probe, including vocabulary projections.

No optimizer or training job. Does not install the probe in production modules.
The temporary adapter is restored on exit. See rmsnorm_probe.py for semantics.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import flash_dism_v4.module as module
from flash_dism_v4.blocked import dism_v4_blocked_from_embeddings
from rmsnorm_probe import rms_readout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['operator','model'],default='operator')
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--length',type=int,default=2048)
    p.add_argument('--vocab',type=int,default=512)
    p.add_argument('--steps',type=int,default=5)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(123)
    original=module.dism_v4_blocked_from_embeddings
    if args.kind=='operator':
        b,n,h,d=args.batch,args.length,12,64
        q=torch.randn(b,n,h,d,device='cuda')*.2
        k=torch.randn_like(q)*.2
        qw=torch.randn(h,args.vocab,d,device='cuda')*.125
        kw=torch.randn_like(qw)*.125
        v=torch.randn_like(q)
        tau=torch.linspace(0,2,h,device='cuda')
        parameters=[t.requires_grad_() for t in [q,k,qw,kw,v,tau]]
        cot=torch.randn_like(v)
    else:
        from nanochat.models import ModelSpec,build_model
        repo=Path(__file__).resolve().parents[2]
        cfg=json.loads((repo/'nanochat/configs/dism_v4_123m_ggggggdddddd_slimpajama_3b.json').read_text())
        cfg['config']['sequence_len']=args.length
        model=build_model(ModelSpec.from_dict(cfg)).cuda();model.init_weights();model.train()
        ids=torch.randint(model.config.vocab_size,(args.batch,args.length+1),device='cuda')
        x,y=ids[:,:-1].contiguous(),ids[:,1:].contiguous()
        cu=torch.arange(args.batch+1,device='cuda',dtype=torch.int32)*args.length
        parameters=list(model.parameters())
    results=[]
    try:
        for probability in (0.,.5,1.):
            if args.kind=='operator':
                flags=torch.zeros(args.batch,12,device='cuda',dtype=torch.bool)
                if probability==.5:flags[:,::2]=True
                if probability==1.:flags.fill_(True)
            else:model.hard_probability.fill_(probability)
            for variant in ('arithmetic','jensen','rmsnorm_jensen'):
                def dispatch(*pos,**kwargs):
                    if variant=='rmsnorm_jensen':return rms_readout(*pos,**kwargs)
                    kwargs['soft_readout']=variant
                    return dism_v4_blocked_from_embeddings(*pos,**kwargs)
                module.dism_v4_blocked_from_embeddings=dispatch
                def step():
                    for t in parameters:t.grad=None
                    if args.kind=='operator':
                        out=dispatch(q,k,qw,kw,v,tau,hard=flags)
                        loss=(out*cot).sum()
                    else:
                        model.rng_counter.zero_()
                        loss=model(x,y,cu_seqlens=cu)
                    loss.backward()
                    return loss.detach()
                for _ in range(2):step()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                times=[]
                for _ in range(args.repeats):
                    start=time.perf_counter()
                    for _ in range(args.steps):loss=step()
                    torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)/args.steps)
                assert bool(torch.isfinite(loss))
                assert all(t.grad is None or bool(torch.isfinite(t.grad).all()) for t in parameters)
                row=dict(kind=args.kind,variant=variant,hard_probability=probability,
                         seconds=statistics.median(times),samples=times,loss=float(loss),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),finite=True)
                results.append(row);print(json.dumps(row),flush=True)
    finally:module.dism_v4_blocked_from_embeddings=original
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(config=vars(args),parameter_count=sum(t.numel() for t in parameters),
                                   results=results),indent=2)+'\n')


if __name__=='__main__':main()
