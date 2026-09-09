"""CUPTI A/B: complete CUDA embedding + core forward/backward + embedding backward."""
import argparse
import collections
import json
import os
import statistics
import torch
from .autograd import voc_dism
from .kernel_config import TILE_LSE


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--direction',default='q_from_k',choices=('q_from_k','k_from_q'))
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--forward-only',action='store_true')
    p.add_argument('--output',help='Optional JSON artifact path')
    args=p.parse_args()
    torch.manual_seed(0)
    b,h,n,d=64,4,1024,64
    def rand(shape): return torch.randn(shape,device='cuda',dtype=torch.bfloat16).requires_grad_()
    inputs=[rand((b,h,n,d)) for _ in range(3)]+[
        torch.full((h,),3.,device='cuda',requires_grad=True),rand((h,512,d)),rand((h,512,d))]
    dout=torch.randn_like(inputs[2])
    state=None
    def run():
        nonlocal state
        out,state=voc_dism(*inputs,sm_scale=1.,direction=args.direction,hard_prob=.5,
            rng_state=state,return_rng_state=True,embedding_backend='cuda',embedding_backward_backend='cuda')
        return out if args.forward_only else torch.autograd.grad(out,inputs,dout)
    results=[]
    for trial,modes in enumerate((('0','1'),('1','0'))):
        for mode in modes:
            os.environ['DISM_ROW_BITSET']=mode
            for _ in range(10): run()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(args.repeats): run()
                torch.cuda.synchronize()
            groups=collections.defaultdict(list)
            for e in prof.events():
                if e.device_type==torch.autograd.DeviceType.CUDA:
                    groups[e.name].append(e.device_time_total)
            # Each named group can have several launches per iteration (e.g. casts).
            kernels={name:sum(times)/args.repeats for name,times in groups.items()}
            medians={name:statistics.median(times) for name,times in groups.items()}
            result=dict(tile_lse=TILE_LSE,bitset=mode,trial=trial,direction=args.direction,
                forward_only=args.forward_only,output_q_alias=os.environ.get('DISM_OUTPUT_Q_ALIAS','kv'),
                gpu=torch.cuda.get_device_name(),b=b,h=h,n=n,d=d,dv=d,vocab=512,
                repeats=args.repeats,total_gpu_mean_us=sum(kernels.values()),
                kernel_mean_per_iteration_us=kernels,kernel_launch_median_us=medians)
            results.append(result)
            print(json.dumps(result),flush=True)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__': main()
