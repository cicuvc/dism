"""Long-chain, alternate tuning, no-match and prefill/rebuild regression."""
import numpy as np
import torch
from runtime import NativeDecodeCache,load_cuda


def main():
    torch.manual_seed(819)
    for pattern in ('repeat','random','no_match'):
      for sample,threshold,chunk in ((16,48,32),(128,160,512)):
        n,steps,h,r,d=2048,33,3,32,64
        total=n+steps;tau=[0.,1e-8,4.2]
        rng=np.random.default_rng(38)
        labels=rng.integers(0,8,(total,3,h),dtype=np.int32);labels[:,2]=0
        if pattern=='repeat':labels[:,:2]=0
        if pattern=='no_match':labels[:,0]=0;labels[:,1]=1
        gpu=torch.tensor(labels,device='cuda')
        sk=torch.randn(total,h,r,device='cuda',dtype=torch.bfloat16)
        sq=torch.randn_like(sk);v=torch.randn(total,h,d,device='cuda',dtype=torch.bfloat16)
        cache=NativeDecodeCache(h,r,d,total,tau,rebuild_interval=32,sample_interval=sample,
            materialize_threshold=threshold,rebuild_chunk=chunk)
        cache.prime(gpu[:n],sk[:n].transpose(0,1).contiguous(),v[:n].transpose(0,1).contiguous())
        linear=load_cuda().LinearCache(h,r,d,total,tau,sk[0])
        linear.prime(gpu[:n],sk[:n].transpose(0,1).contiguous(),v[:n].transpose(0,1).contiguous())
        previous=np.empty((h,0),dtype=np.int32)
        worst=0.
        for i in range(total):
            previous=np.where(labels[:i+1,0].T==labels[i,1,:,None],np.pad(previous,((0,0),(1,0)))+1,0)
            if i<n:continue
            lengths=torch.tensor(previous,device='cuda',dtype=torch.float64)
            t=torch.tensor(tau,device='cuda',dtype=torch.float64)[:,None]
            maximum=lengths.amax(-1,keepdim=True)
            geometric=torch.where(t==0,lengths,-torch.expm1(-t*lengths)/-torch.expm1(-t))
            weights=(t*(lengths-maximum)).exp()*geometric
            score=torch.einsum('hr,thr->ht',sq[i].double(),sk[:i+1].double())
            expected=torch.einsum('ht,thd->hd',weights*score,v[:i+1].double())/(weights.sum(-1,keepdim=True)+(-t*maximum).exp())
            actual=cache.step(gpu[i],sk[i],sq[i],v[i])
            torch.testing.assert_close(actual.double(),expected,atol=3e-4,rtol=3e-4)
            torch.testing.assert_close(linear.step(gpu[i],sk[i],sq[i],v[i]).double(),expected,atol=3e-4,rtol=3e-4)
            worst=max(worst,(actual.double()-expected).abs().max().item())
        print('LONG_PASS',pattern,sample,threshold,chunk,'max_abs',worst,flush=True)


if __name__=='__main__':main()
