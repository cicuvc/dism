"""CUPTI timings of tt_dism.py's six original soft-path kernels, unmodified.

Three independent attention input sets, forward 0/1/2 then backward 2/1/0.
This is an attention launch benchmark, not an old-model training benchmark.
"""
import argparse
import collections
import json
import statistics
import torch
import tt_dism as old


NAMES=('perprocess_kernel_hh','chunk_passing_kernel','attn_fwd_kernel_hh',
       'attn_bwd_kernel_hh','chunk_passing_kernel_bwd','attn_bwd_kernel_post_hh')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--warmup',type=int,default=3)
    args=parser.parse_args()
    if min(args.steps,args.warmup)<1: parser.error('positive steps/warmup required')
    torch.set_default_device('cuda')
    torch.manual_seed(0)
    sets=[];upstream=[]
    for _ in range(3):
        q,k,v=[torch.randn(64,4,1024,64,dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
        beta=torch.randn(64,4,1024,dtype=torch.float32,requires_grad=True)
        tau=torch.full((4,),3.,dtype=torch.float32,requires_grad=True)
        sets.append((q,k,v,beta,tau));upstream.append(torch.randn_like(v))
    def run(check=False):
        outputs=[old.ParallelSoftDiscreteAttention.apply(*s) for s in sets]
        for i in (2,1,0):
            grads=torch.autograd.grad(outputs[i],sets[i],upstream[i])
            if check:
                assert torch.isfinite(outputs[i]).all()
                assert all(torch.isfinite(g).all() for g in grads)
    print(json.dumps(dict(kind='config',batch=64,heads=4,n=1024,n_headdim=64,n_vocab=64,
        query_chunk=32,diagonal_chunk=32,beta='randn FP32',tau=3.,eps=1e-4,
        inputs='randn BF16 logits and values; original wrapper softmax',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=old.triton.__version__,
        replicas=3,steps=args.steps,warmup=args.warmup,
        scope='CUPTI GPU launch time; original soft path, no model/optimizer')),flush=True)
    for i in range(args.warmup):
        run(check=i==0);torch.cuda.synchronize()
        print(json.dumps(dict(kind='warmup',step=i)),flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):run()
        torch.cuda.synchronize()
    groups=collections.defaultdict(list)
    for e in sorted(prof.events(),key=lambda x:x.time_range.start):
        if e.device_type==torch.autograd.DeviceType.CUDA and e.name in NAMES:
            groups[e.name].append(e.time_range.elapsed_us())
    assert set(groups)==set(NAMES),list(groups)
    for name in NAMES:
        times=groups[name]
        assert len(times)==args.steps*3,(name,len(times))
        print(json.dumps(dict(kind='kernel',name=name,count=len(times),median_us=statistics.median(times),
            mean_us=statistics.mean(times),min_us=min(times),max_us=max(times),launch_times_us=times)),flush=True)


if __name__=='__main__':main()
