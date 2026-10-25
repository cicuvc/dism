"""Diagnostic eager top-k Q readout with one committed hard history.

No new CUDA kernel, cache, planner, or production dispatch. SQ/SK affect only
the numerator. Each candidate retains its own 1+Z denominator. Uses bounded
pair tiles, with PyTorch diagonal scans to recover mainline match lengths.
"""
import torch


def history_tiles(iq,ik,tau,segments,block=64):
    """G,N labels -> natural-log continuation weight A before current match."""
    groups,n=iq.shape
    prev=torch.zeros_like(iq)
    diagonal=torch.arange(n,device=iq.device)
    keypos=torch.arange(n,device=iq.device)
    for start in range(0,n,block):
        count=min(block,n-start)
        r=torch.arange(count,device=iq.device)
        qi=start+r
        kj=qi[:,None]-diagonal[None,:]
        previous_j=start-1-diagonal
        incoming=prev[:,previous_j.clamp_min(0)]*(previous_j>=0)
        qprev=iq[:,(qi-1).clamp_min(0)]
        kprev=ik[:,(kj-1).clamp_min(0)]
        qs=segments[:,qi]
        qps=segments[:,(qi-1).clamp_min(0)]
        ks=segments[:,kj.clamp_min(0)]
        kps=segments[:,(kj-1).clamp_min(0)]
        broken=((r[None,:,None]>0)&((qprev[:,:,None]!=kprev)|(kj[None]<=0)))
        broken=broken | (qs[:,:,None]!=qps[:,:,None]) | (ks!=kps)
        last=torch.where(broken,r[None,:,None],-1).cummax(1).values
        length=torch.where(last>=0,r[None,:,None]-last,incoming[:,None,:]+r[None,:,None])
        end=start+count
        new=length[:,-1,:end]+1
        js=end-1-diagonal[:end]
        new=new*((iq[:,end-1,None]==ik[:,js])&(segments[:,end-1,None]==segments[:,js]))
        prev=torch.zeros_like(prev)
        prev[:,:end]=new.flip(-1).to(prev.dtype)
        # Diagonal coordinates -> ordinary pair tile [G,BQ,N].
        idx=(qi[:,None]-keypos[None,:]).clamp_min(0)
        terms=length.gather(2,idx[None].expand(groups,-1,-1)).to(tau.dtype)+1
        t=tau[:,None,None]
        a=t.abs();la=terms*a
        # Small-tau series also avoids 0/0 at tau=0.
        loggeom=torch.log(-torch.expm1(-la))-torch.log(-torch.expm1(-a))
        loga=torch.where(t>0,terms*t,t)+loggeom
        series=terms.log()+t*(terms+1)/2+t.square()*(terms.square()-1)/24
        series-=t.pow(4)*(terms.pow(4)-1)/2880
        loga=torch.where(la<.01,series,loga)
        valid=(keypos[None,:]<=qi[:,None])[None] & (qs[:,:,None]==segments[:,None,:])
        yield start,loga.masked_fill(~valid,-torch.inf)


def read_candidates(logits,iq,ik,sq,sk,v,tau,segments,k,block=64):
    """All inputs G,N,...; returns mixed and mainline-only FP32/FP64 outputs.

The native argmax is always first, followed by highest other logits. This
preserves native tie-breaking despite tiny vocabulary GEMM rounding differences.
"""
    if not 1<=k<=logits.shape[-1]:raise ValueError('invalid candidate count')
    groups,n,c=logits.shape
    main=iq.long()[...,None]
    if k==1:ids=main
    else:
        others=logits.scatter(-1,main,-torch.inf).topk(k-1,-1).indices
        ids=torch.cat([main,others],-1)
    selected=logits.gather(-1,ids)
    coefficients=torch.zeros_like(logits).scatter(-1,ids,selected.softmax(-1))
    mass=(selected.logsumexp(-1)-logits.logsumexp(-1)).exp()
    out=torch.empty_like(v);base=torch.empty_like(v)
    for start,loga in history_tiles(iq,ik,tau,segments,block):
        end=start+loga.shape[1]
        index=ik.long()[:,None,:].expand_as(loga)
        maximum=torch.zeros_like(logits[:,start:end]).scatter_reduce(2,index,loga,reduce='amax',include_self=True)
        a=(loga-maximum.gather(2,index)).exp()
        den=(-maximum).exp().scatter_add(2,index,a)
        w=a/den.gather(2,index)
        similarity=torch.bmm(sq[:,start:end],sk.transpose(1,2))
        m=w*coefficients[:,start:end].gather(2,index)
        out[:,start:end]=torch.bmm(m*similarity,v)
        own=w*(iq[:,start:end,None]==ik[:,None,:])
        base[:,start:end]=torch.bmm(own*similarity,v)
    return out,base,mass


def oracle(logits,iq,ik,sq,sk,v,tau,segments,k):
    """Independent small dense reference, sequential mainline recurrence."""
    groups,n,c=logits.shape
    ids=torch.cat([iq.long()[...,None],logits.scatter(-1,iq.long()[...,None],-torch.inf).topk(k-1,-1).indices],-1)
    prob=logits.gather(-1,ids).softmax(-1)
    previous=torch.zeros((groups,n),device=v.device,dtype=v.dtype)
    rows=[]
    for i in range(n):
        shifted=torch.nn.functional.pad(previous[:,:-1],(1,0))
        if i:shifted=shifted*(segments[:,i,None]==segments[:,i-1,None])
        a=tau.exp()[:,None]*(1+shifted)
        valid=(torch.arange(n,device=v.device)[None]<=i)&(segments[:,i,None]==segments)
        a=a*valid
        values=(sq[:,i,None,:]*sk).sum(-1)[...,None]*v
        row=torch.zeros_like(v[:,0])
        for slot in range(k):
            w=a*(ik==ids[:,i,slot,None])
            candidate=(w[...,None]*values).sum(1)/(1+w.sum(-1,keepdim=True))
            row=row+prob[:,i,slot,None]*candidate
        rows.append(row)
        previous=a*(iq[:,i,None]==ik)
    return torch.stack(rows,1)


def validate():
    torch.manual_seed(81)
    for same in (False,True):
        g,n,c=2,73,9
        z=torch.randn(g,n,c,device='cuda',dtype=torch.float64)
        iq=z.argmax(-1).int();ik=torch.randint(c,(g,n),device='cuda',dtype=torch.int32)
        if same:iq.zero_();ik.zero_()
        sq=torch.randn(g,n,3,device='cuda',dtype=torch.float64)
        sk=torch.randn_like(sq);v=torch.randn(g,n,5,device='cuda',dtype=torch.float64)
        tau=torch.tensor([0.,.3],device='cuda',dtype=torch.float64)
        seg=torch.zeros(g,n,device='cuda',dtype=torch.int32);seg[:,37:]=1
        for k in (1,2,4,8):
            actual,base,mass=read_candidates(z,iq,ik,sq,sk,v,tau,seg,k,block=32)
            expected=oracle(z,iq,ik,sq,sk,v,tau,seg,k)
            torch.testing.assert_close(actual,expected,atol=2e-10,rtol=2e-9)
            torch.testing.assert_close(base,oracle(z,iq,ik,sq,sk,v,tau,seg,1),atol=2e-10,rtol=2e-9)
            assert bool(((mass>=0)&(mass<=1+1e-12)).all())
        # Current alternative must not enter later mainline states; only its
        # own row changes when weights vary with externally fixed labels.
        z2=z.clone();z2[:,11]=torch.randn_like(z2[:,11])
        a=read_candidates(z,iq,ik,sq,sk,v,tau,seg,4)[0]
        b=read_candidates(z2,iq,ik,sq,sk,v,tau,seg,4)[0]
        mask=torch.arange(n,device=v.device)!=11
        # cuBLAS FP64 may differ by one rounding unit across allocations.
        torch.testing.assert_close(a[:,mask],b[:,mask],atol=2e-12,rtol=2e-12)
    print('8 FP64 candidate cases and local-history checks passed',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    validate()
