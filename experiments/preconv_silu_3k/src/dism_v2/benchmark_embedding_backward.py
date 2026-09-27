"""Embedding backward config sweep: warm-cache CUPTI GPU time, not API latency.

python -m dism_v2.benchmark_embedding_backward --repeats 20 --rounds 3
Every CUDA call includes preprocessing and all four gradients. Token WS tiling
stays fixed. No candidate is excluded for spilling. Paired D128/T64 is excluded
because shared memory exceeds device capacity. Triton computes the same requested
gradients through its general wrapper (including its extra inactive-branch work).
"""
import argparse
import collections
import json
import random
import statistics
import torch
from .embedding import forward, backward
from .emb_kernel import emb_bwd_wrapper


def configurations(d):
    yield "single", dict(warp_specialized=False)
    for symmetric in (False,True):
        for shared in ((False,True) if symmetric and d==64 else (False,)):
            for step in (16,32,64):
                if not symmetric and d==128 and step==64: continue
                name=f"{'sym' if symmetric else 'paired'}-{'smem' if shared else 'reg'}-t{step}"
                yield name,dict(warp_specialized=True,vocab_symmetric=symmetric,
                               vocab_token_step=step,vocab_shared=shared)


def measure(fn,repeats):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats): fn()
        torch.cuda.synchronize()
    events=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
    groups=collections.defaultdict(list)
    for event in events: groups[event.name].append(event.time_range.elapsed_us())
    assert events and all(len(x)==repeats for x in groups.values()),list(groups)
    kernels={name:statistics.median(times) for name,times in groups.items()}
    return dict(total_us=sum(kernels.values()),vocab_us=sum(t for n,t in kernels.items()
                if 'vocabulary' in n),kernels_us=kernels)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--dims',type=int,nargs='+',default=[32,64,128])
    parser.add_argument('--shapes',nargs='+',default=['1,4,65,129','1,4,1024,1024',
        '1,4,4096,1024','1,4,4096,64','2,4,1025,257'],help='B,H,N,V per shape')
    opts=parser.parse_args()
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        cuda=torch.version.cuda,scope='CUPTI sum of per-kernel medians; hot inputs; excludes host gaps',
        repeats=opts.repeats,rounds=opts.rounds)),flush=True)
    torch.manual_seed(941)
    rng=random.Random(941)
    for b,h,n,v in [tuple(map(int,s.split(','))) for s in opts.shapes]:
        for d in opts.dims:
            q,k=[torch.randn(b,h,n,d,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
            eq,ek=[torch.randn(h,v,d,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
            raw=forward(q,k,eq,ek,d**-.5)
            u=torch.randn(q.shape,device='cuda',dtype=torch.float32)
            lam=torch.randn(q.shape[:3],device='cuda',dtype=torch.float32)
            zero=torch.zeros_like(u)
            functions={name:(lambda cfg=cfg:backward(q,k,eq,ek,raw[0],raw[2],raw[3],u,lam,
                direction='q_from_k',sm_scale=d**-.5,**cfg)) for name,cfg in configurations(d)}
            functions['triton']=lambda:emb_bwd_wrapper(q,k,eq,ek,*raw[:4],u,zero,None,lam,d**-.5)
            for trial in range(opts.rounds):
                names=list(functions);rng.shuffle(names)
                for name in names:
                    result=measure(functions[name],opts.repeats)
                    print(json.dumps(dict(b=b,h=h,n=n,v=v,d=d,config=name,trial=trial,
                                          **result)),flush=True)


if __name__=='__main__': main()
