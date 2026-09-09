"""CUPTI summary/output timing on actual embedding inputs; no per-launch sync."""
import argparse
import importlib.util
import json
import statistics
import torch
from . import core
from .embedding import forward as embedding
from .kernel_config import TILE_LSE


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-binary')
    parser.add_argument('--direction',default='q_from_k',choices=['q_from_k','k_from_q'])
    parser.add_argument('--hard-prob',type=float,default=.5)
    parser.add_argument('--d',type=int,default=64,choices=[32,64,128])
    parser.add_argument('--dv',type=int,choices=[32,64,128])
    parser.add_argument('--n',type=int,default=1024)
    parser.add_argument('--kernel',choices=['summary','output'],default='summary')
    parser.add_argument('--label-dtype',choices=['int32','int64'],default='int32')
    args=parser.parse_args()
    if args.dv is None: args.dv=args.d
    if args.baseline_binary:
        name='dism_v2_core_sm120a'+('' if TILE_LSE=='full' else '_'+TILE_LSE)
        spec=importlib.util.spec_from_file_location(name,args.baseline_binary)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        core._extension=lambda:module
    torch.manual_seed(0)
    q,k=[torch.randn(64,4,args.n,args.d,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    v=torch.randn(64,4,args.n,args.dv,device='cuda',dtype=torch.bfloat16)
    qv,kv=[torch.randn(4,512,args.d,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    emb=embedding(q,k,qv,kv,1.)
    tau=torch.full((4,),3.,device='cuda')
    a,b,lse=(q,emb[0],emb[3]) if args.direction=='q_from_k' else (emb[1],k,emb[2])
    dtype=torch.int32 if args.label_dtype=='int32' else torch.int64
    qlabel,klabel=emb[7].to(dtype),emb[6].to(dtype)
    state=None
    def run():
        nonlocal state
        result=core.forward(a,b,v,lse,tau,qlabel,klabel,sm_scale=1.,direction=args.direction,
            hard_prob=args.hard_prob,rng_state=state,return_rng_state=True,save_boundaries=True)
        state=result[-1]
        return result
    torch.cuda.manual_seed(777)
    for _ in range(20): run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(30): result=run()
        torch.cuda.synchronize()
    times=[e.device_time_total for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA
           and (('summary_persistent<' in e.name) if args.kernel=='summary'
                else ('void dism_v2::core<' in e.name or 'output_persistent<' in e.name))]
    assert len(times)==30,[(e.name,e.device_time_total) for e in prof.events()
                         if e.device_type==torch.autograd.DeviceType.CUDA][:20]
    print(json.dumps(dict(**vars(args),tile_lse=TILE_LSE,b=64,h=4,vocab=512,
        gpu=torch.cuda.get_device_name(),samples_us=times,median_us=statistics.median(times),
        finite=bool(torch.isfinite(result[0]).all()))))


if __name__=='__main__': main()
