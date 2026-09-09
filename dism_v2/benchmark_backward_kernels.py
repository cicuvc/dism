"""CUPTI WS B1/B3 on fixed actual-embedding inputs and saved forward states."""
import argparse
import collections
import json
import importlib.util
import os
from pathlib import Path
import statistics
import torch
from . import core, backward
from .embedding import forward as embedding
from .kernel_config import TILE_LSE

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--direction',choices=('q_from_k','k_from_q'),default='q_from_k')
    p.add_argument('--d',type=int,default=64)
    p.add_argument('--dv',type=int,default=64)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--n',type=int,default=1024)
    p.add_argument('--hard-prob',type=float,default=.5)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--output')
    p.add_argument('--ncu',action='store_true',help='Run without CUPTI profiler for external NCU')
    p.add_argument('--baseline-binary',help='Saved unmodified backward extension')
    p.add_argument('--baseline-module',help='Original PyInit name of a saved experiment')
    args=p.parse_args()
    if args.baseline_binary:
        name=args.baseline_module or 'dism_v2_backward_sm120a'+('' if TILE_LSE=='full' else '_'+TILE_LSE)
        spec=importlib.util.spec_from_file_location(name,args.baseline_binary)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backward._extension=lambda:module
    torch.manual_seed(0)
    def rand(shape):return torch.randn(shape,device='cuda',dtype=torch.bfloat16)
    q,k=[rand((args.batch,args.heads,args.n,args.d)) for _ in range(2)]
    v,do=[rand((args.batch,args.heads,args.n,args.dv)) for _ in range(2)]
    qv,kv=[rand((args.heads,512,args.d)) for _ in range(2)]
    emb=embedding(q,k,qv,kv,1.)
    a,b,lse=(q,emb[0],emb[3]) if args.direction=='q_from_k' else (emb[1],k,emb[2])
    ql,kl=emb[7],emb[6]
    tau=torch.full((args.heads,),3.,device='cuda')
    out,norm,bounds,state=core.forward(a,b,v,lse,tau,ql,kl,sm_scale=1.,direction=args.direction,
        hard_prob=args.hard_prob,return_rng_state=True,save_boundaries=True,
        generator=torch.Generator(device='cuda').manual_seed(777))
    delta=backward.delta(do,out)
    def run():
        dv,summary,boundary=backward.value_gradient(a,b,do,lse,tau,ql,kl,norm,bounds,
            sm_scale=1.,rng_state=state,v=v,delta=delta,warp_specialized=True)
        gradients=backward.operand_gradient(a,b,v,do,lse,tau,ql,kl,norm,delta,bounds,boundary,
            sm_scale=1.,rng_state=state,warp_specialized=True)
        return dv,gradients
    for _ in range(args.warmup):run()
    torch.cuda.synchronize()
    if args.ncu:
        for _ in range(args.repeats):run()
        torch.cuda.synchronize()
        return
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(args.repeats):result=run()
        torch.cuda.synchronize()
    groups=collections.defaultdict(list)
    for e in prof.events():
        if e.device_type==torch.autograd.DeviceType.CUDA:
            groups[e.name].append(e.device_time_total)
    report=dict(**vars(args),gpu=torch.cuda.get_device_name(),tile_lse=TILE_LSE,
        optimization=os.environ.get('DISM_BWD_OPT','0'),vocab=512,scale=1.,tau=3.,
        requested_stages=os.environ.get('DISM_BWD_STAGES','2'),
        scope='GPU kernel durations, B1+passing+B3+scalar reduction and wrapper auxiliaries; fixed forward states; no embedding/forward/delta time',
        finite=all(bool(torch.isfinite(x).all()) for x in (result[0],*result[1])),
        kernels={k:dict(samples_us=v,median_us=statistics.median(v),mean_per_iteration_us=sum(v)/args.repeats)
                 for k,v in groups.items()},
        total_gpu_mean_us=sum(sum(v) for v in groups.values())/args.repeats)
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
