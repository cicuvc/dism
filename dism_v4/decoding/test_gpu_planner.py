"""Experimental device planner vs independent FP64 recurrence and graph replay."""
import argparse
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dism_v4.decoding.gpu_runtime import GpuPlannerCache
from dism_v4.decoding.runtime import NativeDecodeCache


@torch.inference_mode()
def guard_test():
    for invalid_reset in (False,True):
        cache=GpuPlannerCache(1,16,32,32,[.7],rebuild_interval=2)
        labels=torch.zeros(3,1,device='cuda',dtype=torch.int32)
        sk=torch.ones(1,16,device='cuda',dtype=torch.bfloat16)
        v=torch.ones(1,32,device='cuda',dtype=torch.bfloat16)
        if invalid_reset:labels[2]=2
        else:
            for _ in range(2):cache.step(labels,sk,sk,v)
        assert cache.step(labels,sk,sk,v).isnan().all()
        try:cache.check_status()
        except RuntimeError:pass
        else:raise AssertionError('missing horizon/reset guard')
    print('GPU_GUARDS_PASS',flush=True)


@torch.inference_mode()
def run(dtype,r,d,pattern,n=97,b=7,prime=0,graph=False):
    torch.manual_seed(733)
    h=3
    tau=torch.tensor([0.,1e-8,4.2],device='cuda')
    labels=torch.randint(0,5,(n,3,h),device='cuda',dtype=torch.int32)
    labels[:,2]=0;labels[::29,2]=1
    if pattern=='repeat':labels[:,:2]=0;labels[:,2]=0
    if pattern=='no_match':labels[:,0]=0;labels[:,1]=1
    sk=torch.randn(n,h,r,device='cuda').to(dtype);sq=torch.randn_like(sk)
    v=torch.randn(n,h,d,device='cuda').to(dtype)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    worst=0.
    with torch.cuda.stream(stream):
        cache=GpuPlannerCache(h,r,d,n,tau.cpu().tolist(),rebuild_interval=b,
            sample_interval=3,materialize_threshold=r+3,rebuild_chunk=64,cache_dtype=dtype)
        if prime:cache.prime(labels[:prime],sk[:prime].transpose(0,1).contiguous(),v[:prime].transpose(0,1).contiguous())
        previous=torch.empty(h,0,device='cuda',dtype=torch.float64)
        since=0;executable=None
        static_labels=torch.empty_like(labels[0]);static_sk=torch.empty_like(sk[0]);static_sq=torch.empty_like(sq[0]);static_v=torch.empty_like(v[0])
        for i in range(n):
            pred=torch.nn.functional.pad(previous,(1,0),value=-torch.inf)
            pred[labels[i,2].bool()]=-torch.inf
            previous=torch.where(labels[:i+1,0].T==labels[i,1,:,None],tau.double()[:,None]+torch.logaddexp(pred,torch.zeros_like(pred)),-torch.inf)
            if i<prime:continue
            if since==b:
                cache.rebuild();since=0;executable=None
            if graph:
                static_labels.copy_(labels[i]);static_sk.copy_(sk[i]);static_sq.copy_(sq[i]);static_v.copy_(v[i])
                if executable is None:
                    executable=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(executable,stream=stream):
                        actual=cache.step(static_labels,static_sk,static_sq,static_v)
                executable.replay()
            else:actual=cache.step(labels[i],sk[i],sq[i],v[i])
            since+=1
            maximum=previous.amax(-1).clamp_min(0)
            w=(previous-maximum[:,None]).exp()
            readout=torch.einsum('hr,thr->ht',sq[i].double(),sk[:i+1].double())
            expected=torch.einsum('ht,thd->hd',w*readout,v[:i+1].double())/(w.sum(-1)+(-maximum).exp())[:,None]
            torch.testing.assert_close(actual.double(),expected,atol=3e-4,rtol=3e-4)
            worst=max(worst,(actual.double()-expected).abs().max().item())
        assert cache.check_status()==n
        # Capacity failure is device-guarded even under graph replay, not OOB.
        failed=cache.step(labels[-1],sk[-1],sq[-1],v[-1])
        assert failed.isnan().all()
        try:cache.check_status()
        except RuntimeError:pass
        else:raise AssertionError('missing capacity error')
    stream.synchronize()
    print('PASS',dtype,r,d,pattern,'prime',prime,'graph',graph,'max_abs',worst,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--long',action='store_true');a=p.parse_args()
    guard_test()
    if a.smoke:
        run(torch.bfloat16,16,32,'repeat',n=41,b=7,prime=11,graph=True)
        run(torch.float32,32,64,'random',n=41,b=7,prime=11)
    elif a.long:
        for pattern in ('repeat','random','no_match'):
            run(torch.bfloat16,32,64,pattern,n=2081,b=16,prime=2048,graph=True)
    else:
        for dtype in (torch.float32,torch.bfloat16):
            for r,d in ((16,32),(16,64),(32,32),(32,64)):
                for pattern in ('random','repeat','no_match'):
                    run(dtype,r,d,pattern,prime=0)
                    run(dtype,r,d,pattern,prime=37,graph=True)
    print('GPU_PLANNER_PASS',flush=True)
