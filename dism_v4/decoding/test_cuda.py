import torch
from runtime import DecodeCache


def main():
    torch.manual_seed(71)
    for r,d in ((16,32),(32,64)):
        for pattern in ('random','repeated','long_chain'):
            h,n=3,81
            tau=torch.tensor([0.,1e-8,4.2],device='cuda')
            k=torch.randint(0,4,(n,h),device='cuda');q=torch.randint(0,4,(n,h),device='cuda')
            if pattern!='random': k.zero_();q.zero_()
            sk=torch.randn(n,h,r,device='cuda');sq=torch.randn_like(sk);v=torch.randn(n,h,d,device='cuda')
            cache=DecodeCache(h,r,d,n,tau.cpu().tolist(),rebuild_interval=7,sample_interval=3,materialize_threshold=r+3)
            previous=torch.empty(h,0,device='cuda',dtype=torch.float64)
            worst=0.
            for i in range(n):
                reset=torch.full((h,),pattern!='long_chain' and i%23==0,device='cuda',dtype=torch.bool)
                actual=cache.step(k[i],q[i],sk[i],sq[i],v[i],reset)
                pred=torch.nn.functional.pad(previous,(1,0),value=-torch.inf)
                pred[reset]=-torch.inf
                previous=torch.where(k[:i+1].T==q[i,:,None],tau.double()[:,None]+torch.logaddexp(pred,torch.zeros_like(pred)),-torch.inf)
                maximum=previous.amax(-1).clamp_min(0)
                w=(previous-maximum[:,None]).exp()
                score=torch.einsum('hr,thr->ht',sq[i].double(),sk[:i+1].double())
                expected=torch.einsum('ht,thd->hd',w*score,v[:i+1].double())/(w.sum(-1)+(-maximum).exp())[:,None]
                torch.testing.assert_close(actual.double(),expected,atol=3e-4,rtol=3e-4)
                worst=max(worst,(actual.double()-expected).abs().max().item())
            print(f'PASS R={r} DV={d} {pattern}: max_abs={worst:.3g}',flush=True)


if __name__=='__main__': main()
