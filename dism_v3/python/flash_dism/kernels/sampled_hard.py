"""One sampled suffix read plus an exact top-four zero-history control.

Q/K mainline labels never change. The first implementation shares one quadratic
pair traversal for control and residual; no prefix-state optimization yet.
"""
import torch
import triton as tr
import triton.language as tl
from .topk_hard import _boundaries, _lut, _lengths, _document_extent


@tr.jit
def _sample(PROB, COUNTER, DRAW, N:tl.constexpr, SEED:tl.constexpr, BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    seed=(tl.load(COUNTER).to(tl.uint64)+SEED)^0xD1B54A32D192ED03
    u=tl.rand(seed,x)
    p0=tl.load(PROB+x*4,x<N,0)
    p1=tl.load(PROB+x*4+1,x<N,0)
    p2=tl.load(PROB+x*4+2,x<N,0)
    draw=tl.where(u<p0,0,tl.where(u<p0+p1,1,tl.where(u<p0+p1+p2,2,3)))
    tl.store(DRAW+x,draw,x<N)


def sample(prob,counter,seed):
    draw=torch.empty(prob.shape[:-1],device=prob.device,dtype=torch.int32)
    _sample[(tr.cdiv(draw.numel(),256),)](prob,counter,draw,draw.numel(),seed,256)
    return draw


@tr.jit
def _histogram(IK,HARD,PREFIX,N:tl.constexpr,C:tl.constexpr):
    block=tl.program_id(0);bh=tl.program_id(1)
    if tl.load(HARD+bh):
        keys=tl.load(IK+bh*N+block*32+tl.arange(0,32))
        counts=tl.histogram(keys,C)
        tl.store(PREFIX+(bh*(N//32)+block)*C+tl.arange(0,C),counts)


@tr.jit
def _prefix_counts(PREFIX,HARD,N:tl.constexpr,C:tl.constexpr,NB:tl.constexpr,BC:tl.constexpr=32):
    bh=tl.program_id(1)
    if tl.load(HARD+bh):
        blocks=tl.arange(0,NB);symbols=tl.program_id(0)*BC+tl.arange(0,BC)
        ptr=PREFIX+(bh*(N//32)+blocks[:,None])*C+symbols[None,:]
        mask=(blocks[:,None]<N//32)&(symbols[None,:]<C)
        counts=tl.load(ptr,mask,0)
        prefix=tl.cumsum(counts,0)
        tl.store(ptr,prefix,mask)


def prefix_counts(ik,hard,vocab_size):
    b,h,n=ik.shape;c=tr.next_power_of_2(vocab_size)
    prefix=torch.empty((b*h,n//32,c),device=ik.device,dtype=torch.int32)
    _histogram[(n//32,b*h)](ik,hard,prefix,n,c,num_warps=4)
    _prefix_counts[(tr.cdiv(c,32),b*h)](prefix,hard,n,c,tr.next_power_of_2(n//32),num_warps=4)
    return prefix


@tr.jit
def _normalizers(IQ,IK,SEG,HARD,BOUND,IDS,DRAW,LOG,DLOG,DEN,LAMBDA,PREFIX,
                 N:tl.constexpr,H:tl.constexpr,C:tl.constexpr,BQ:tl.constexpr=32,BK:tl.constexpr=64):
    q0=tl.program_id(0)*BQ;bh=tl.program_id(1)
    if tl.load(HARD+bh):
        rows=q0+tl.arange(0,BQ);slots=tl.arange(0,4)
        symbol=tl.load(IDS+(bh*N+rows[:,None])*4+slots[None,:])
        draw=tl.load(DRAW+bh*N+rows)
        chosen=tl.sum(tl.where(slots[None,:]==draw[:,None],symbol,0),1)
        mx=tl.full((BQ,),0,tl.float32);den=tl.full((BQ,),1,tl.float32)
        moment=tl.full((BQ,),0,tl.float32)
        first=_document_extent(SEG,bh,q0,N,H,False)
        count=tl.load(PREFIX+(bh*(N//32)+q0//32-1)*C+symbol,q0>0,0)
        count-=tl.load(PREFIX+(bh*(N//32)+first//32-1)*C+symbol,first>0,0)
        inside=q0+tl.arange(0,BQ)
        local_keys=tl.load(IK+bh*N+inside)
        count+=tl.sum(((inside[None,None,:]<=rows[:,None,None])&
                       (local_keys[None,None,:]==symbol[:,:,None])).to(tl.int32),2)
        for k0 in range(first,q0+BQ,BK):
            length,valid=_lengths(IQ,IK,SEG,BOUND,bh,q0,k0,N,H,BQ,BK)
            keys=tl.load(IK+bh*N+k0+tl.arange(0,BK),k0+tl.arange(0,BK)<N,-1)
            score=tl.load(LOG+(bh%H)*(N+1)+length)
            deriv=tl.load(DLOG+(bh%H)*(N+1)+length)
            score=tl.where(valid&(keys[None,:]==chosen[:,None]),score,-float('inf'))
            new=tl.maximum(mx,tl.max(score,1));rescale=tl.exp2(mx-new)
            w=tl.exp2(score-new[:,None])
            den=den*rescale+tl.sum(w,1);moment=moment*rescale+tl.sum(w*deriv,1)
            mx=new
        # Columns 0..3: base edge weight and its log derivative. Column 4:
        # selected hard log denominator and its tau derivative.
        tau2=tl.load(LOG+(bh%H)*(N+1)+1)
        pivot=tl.maximum(tau2,0.)
        e=tl.exp2(tau2-pivot);fallback=tl.exp2(-pivot)
        base_den=fallback+count.to(tl.float32)*e
        unit=tl.where(count>0,e/base_den,0.)
        deriv0=tl.where(count>0,fallback/base_den,0.)
        tl.store(DEN+(bh*N+rows[:,None])*5+slots[None,:],unit)
        tl.store(LAMBDA+(bh*N+rows[:,None])*5+slots[None,:],deriv0)
        tl.store(DEN+(bh*N+rows)*5+4,mx+tl.log2(den))
        tl.store(LAMBDA+(bh*N+rows)*5+4,moment/den)


@tr.jit
def _weights(IQ,IK,SEG,BOUND,IDS,PROB,DRAW,LOG,DLOG,DEN,LAMBDA,bh,q0,k0,
             N:tl.constexpr,H:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr,BACKWARD:tl.constexpr):
    rows=q0+tl.arange(0,BQ);cols=k0+tl.arange(0,BK)
    length,valid=_lengths(IQ,IK,SEG,BOUND,bh,q0,k0,N,H,BQ,BK)
    key=tl.load(IK+bh*N+cols,cols<N,-1)
    draw=tl.load(DRAW+bh*N+rows)
    base=tl.full((BQ,BK),0,tl.float32);dbase=tl.full((BQ,BK),0,tl.float32)
    slotid=tl.full((BQ,BK),-1,tl.int32)
    for s in tl.static_range(4):
        symbol=tl.load(IDS+(bh*N+rows)*4+s)
        p=tl.load(PROB+(bh*N+rows)*4+s)
        unit=tl.load(DEN+(bh*N+rows)*5+s)
        coef=(p-(draw==s).to(tl.float32))*unit
        match=valid&(key[None,:]==symbol[:,None])
        base=tl.where(match,coef[:,None],base)
        slotid=tl.where(match,s,slotid)
        if BACKWARD:
            ld=tl.load(LAMBDA+(bh*N+rows)*5+s)
            dbase=tl.where(match,(coef*ld)[:,None],dbase)
    score=tl.load(LOG+(bh%H)*(N+1)+length)
    hardden=tl.load(DEN+(bh*N+rows)*5+4)
    hw=tl.exp2(tl.where((slotid>=0)&(slotid==draw[:,None]),score-hardden[:,None],-float('inf')))
    if BACKWARD:
        deriv=tl.load(DLOG+(bh%H)*(N+1)+length)
        ld=tl.load(LAMBDA+(bh*N+rows)*5+4)
        dbase+=hw*(deriv-ld[:,None])
    return base+hw,dbase,slotid


@tr.jit
def _forward(SQ,SK,V,IQ,IK,SEG,HARD,BOUND,IDS,PROB,DRAW,LOG,DLOG,DEN,LAMBDA,OUT,
             N:tl.constexpr,H:tl.constexpr,R:tl.constexpr,DV:tl.constexpr,BQ:tl.constexpr=32,BK:tl.constexpr=64):
    q0=tl.program_id(0)*BQ;bh=tl.program_id(1)
    if tl.load(HARD+bh):
        rows=q0+tl.arange(0,BQ);r=tl.arange(0,R);d=tl.arange(0,DV)
        qoff=((bh//H*N+rows)*H+bh%H)
        sq=tl.load(SQ+qoff[:,None]*R+r[None,:]);acc=tl.full((BQ,DV),0,tl.float32)
        first=_document_extent(SEG,bh,q0,N,H,False)
        for k0 in range(first,q0+BQ,BK):
            cols=k0+tl.arange(0,BK);koff=((bh//H*N+cols)*H+bh%H)
            sk=tl.load(SK+koff[:,None]*R+r[None,:],cols[:,None]<N,0)
            v=tl.load(V+koff[:,None]*DV+d[None,:],cols[:,None]<N,0)
            m,_,_=_weights(IQ,IK,SEG,BOUND,IDS,PROB,DRAW,LOG,DLOG,DEN,LAMBDA,bh,q0,k0,N,H,BQ,BK,False)
            score=tl.dot(sq,tl.trans(sk))
            acc+=tl.dot((m*score).to(v.dtype),v)
        tl.store(OUT+qoff[:,None]*DV+d[None,:],acc)


@tr.jit
def _backward(SQ,SK,V,DO,IQ,IK,SEG,HARD,BOUND,IDS,PROB,DRAW,LOG,DLOG,DEN,LAMBDA,
              DSQ,DSK,DVOUT,DT,U,N:tl.constexpr,H:tl.constexpr,R:tl.constexpr,DV:tl.constexpr,
              BQ:tl.constexpr=32,BK:tl.constexpr=32):
    k0=tl.program_id(0)*BK;bh=tl.program_id(1)
    if tl.load(HARD+bh):
        cols=k0+tl.arange(0,BK);r=tl.arange(0,R);d=tl.arange(0,DV)
        koff=((bh//H*N+cols)*H+bh%H)
        sk=tl.load(SK+koff[:,None]*R+r[None,:]);v=tl.load(V+koff[:,None]*DV+d[None,:])
        dsk=tl.full((BK,R),0,tl.float32);dv=tl.full((BK,DV),0,tl.float32)
        dt=tl.full((),0,tl.float32)
        end=_document_extent(SEG,bh,k0,N,H,True)
        for q0 in range(k0//BQ*BQ,end,BQ):
            rows=q0+tl.arange(0,BQ);qoff=((bh//H*N+rows)*H+bh%H)
            sq=tl.load(SQ+qoff[:,None]*R+r[None,:]);do=tl.load(DO+qoff[:,None]*DV+d[None,:])
            m,dm,slot=_weights(IQ,IK,SEG,BOUND,IDS,PROB,DRAW,LOG,DLOG,DEN,LAMBDA,bh,q0,k0,N,H,BQ,BK,True)
            score=tl.dot(sq,tl.trans(sk));e=tl.dot(do,tl.trans(v))
            f=m*score*e
            dt+=tl.sum(tl.sum(dm*score*e,1),0)
            # U_s=pi_s*<g,B_s> + 1[s=C]*<g,H_C-B_C>.
            # U-pi*sum(U) is the corrected, frozen-cotangent router estimator.
            for s in tl.static_range(4):
                u=tl.sum(tl.where(slot==s,f,0.),1)
                tl.atomic_add(U+(bh*N+rows)*4+s,u,sem='relaxed')
            ds=(m*e).to(sq.dtype)
            tl.atomic_add(DSQ+qoff[:,None]*R+r[None,:],tl.dot(ds,sk),sem='relaxed')
            dsk+=tl.dot(tl.trans(ds),sq)
            dv+=tl.dot(tl.trans((m*score).to(do.dtype)),do)
        tl.store(DSK+koff[:,None]*R+r[None,:],dsk)
        tl.store(DVOUT+koff[:,None]*DV+d[None,:],dv)
        tl.atomic_add(DT+bh%H,dt,sem='relaxed')


def forward(sq,sk,v,iq,ik,ids,prob,tau,hard,segments,draw,vocab_size=512):
    b,n,h,r=sq.shape
    bound=torch.empty((b,h,(n-1)//32,n),device=sq.device,dtype=torch.int32)
    log=torch.empty((h,n+1),device=sq.device);dlog=torch.empty_like(log)
    den=torch.empty((b,h,n,5),device=sq.device);lam=torch.empty_like(den)
    out=torch.zeros_like(v)
    prefix=prefix_counts(ik,hard,vocab_size)
    _boundaries[(b*h,tr.cdiv(n,128))](iq,ik,segments,hard,bound,n,h,128,num_warps=8)
    _lut[(h,tr.cdiv(n+1,256))](tau,log,dlog,n,256)
    _normalizers[(n//32,b*h)](iq,ik,segments,hard,bound,ids,draw,log,dlog,den,lam,prefix,n,h,prefix.shape[-1],num_warps=4,num_stages=1)
    _forward[(n//32,b*h)](sq,sk,v,iq,ik,segments,hard,bound,ids,prob,draw,log,dlog,den,lam,out,n,h,r,v.shape[-1],num_warps=4,num_stages=1)
    return out,(bound,log,dlog,den,lam)


def backward(sq,sk,v,do,iq,ik,ids,prob,hard,segments,state,draw):
    b,n,h,r=sq.shape;bound,log,dlog,den,lam=state
    dsq=torch.zeros_like(sq,dtype=torch.float32);dsk=torch.zeros_like(sk,dtype=torch.float32)
    dv=torch.zeros_like(v,dtype=torch.float32);dt=torch.zeros(h,device=v.device);u=torch.zeros_like(prob)
    _backward[(n//32,b*h)](sq,sk,v,do,iq,ik,segments,hard,bound,ids,prob,draw,log,dlog,den,lam,
                          dsq,dsk,dv,dt,u,n,h,r,v.shape[-1],num_warps=4,num_stages=1)
    return dsq,dsk,dv,dt,u-prob*u.sum(-1,keepdim=True)
