"""FP64 dense comparison for the production C++ packing and BF16/FP32 payloads."""
import torch
from runtime import NativeDecodeCache,load_cuda


def main():
    torch.manual_seed(172)
    for dtype in (torch.float32,torch.bfloat16):
        for r,d in ((16,32),(16,64),(32,32),(32,64)):
            for repeated in (False,True):
                h,n=3,97
                tau=torch.tensor([0.,1e-8,4.2],device='cuda')
                labels=torch.randint(0,5,(n,3,h),device='cuda',dtype=torch.int32)
                labels[:,2]=0
                labels[::29,2]=1
                if repeated: labels[:,:2]=0;labels[:,2]=0
                sk=torch.randn(n,h,r,device='cuda').to(dtype)
                sq=torch.randn(n,h,r,device='cuda').to(dtype)
                v=torch.randn(n,h,d,device='cuda').to(dtype)
                cache=NativeDecodeCache(h,r,d,n,tau.cpu().tolist(),rebuild_interval=7,
                    sample_interval=3,materialize_threshold=r+3,rebuild_chunk=64,cache_dtype=dtype)
                previous=torch.empty(h,0,device='cuda',dtype=torch.float64)
                linear=load_cuda().LinearCache(h,r,d,n,tau.cpu().tolist(),torch.empty(0,device='cuda',dtype=dtype))
                worst=0.
                primed=None
                for i in range(n):
                    if i==47:
                        primed=NativeDecodeCache(h,r,d,n,tau.cpu().tolist(),rebuild_interval=7,
                            sample_interval=3,materialize_threshold=r+3,rebuild_chunk=64,cache_dtype=dtype)
                        primed.prime(labels[:i],sk[:i].transpose(0,1).contiguous(),v[:i].transpose(0,1).contiguous())
                    actual=cache.step(labels[i],sk[i],sq[i],v[i])
                    pred=torch.nn.functional.pad(previous,(1,0),value=-torch.inf)
                    pred[labels[i,2].bool()]=-torch.inf
                    previous=torch.where(labels[:i+1,0].T==labels[i,1,:,None],tau.double()[:,None]+torch.logaddexp(pred,torch.zeros_like(pred)),-torch.inf)
                    maximum=previous.amax(-1).clamp_min(0)
                    w=(previous-maximum[:,None]).exp()
                    readout=torch.einsum('hr,thr->ht',sq[i].double(),sk[:i+1].double())
                    expected=torch.einsum('ht,thd->hd',w*readout,v[:i+1].double())/(w.sum(-1)+(-maximum).exp())[:,None]
                    torch.testing.assert_close(actual.double(),expected,atol=3e-4,rtol=3e-4)
                    torch.testing.assert_close(linear.step(labels[i],sk[i],sq[i],v[i]).double(),expected,atol=3e-4,rtol=3e-4)
                    if primed is not None:
                        torch.testing.assert_close(primed.step(labels[i],sk[i],sq[i],v[i]).double(),expected,atol=3e-4,rtol=3e-4)
                    worst=max(worst,(actual.double()-expected).abs().max().item())
                print(f'PASS {dtype} R{r}/DV{d} repeated={repeated} max_abs={worst:.3g} {cache.memory_stats()}',flush=True)
    print('NATIVE_PASS',flush=True)


if __name__=='__main__':main()
