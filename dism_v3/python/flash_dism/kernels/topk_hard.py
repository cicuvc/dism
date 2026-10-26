"""Top-k hard training kernels with a shared, committed argmax history.

Vectors BNHD, labels BHN; packed documents are represented by BN segment IDs.
The public integration validates 256 alignment. This is the quadratic backend;
there are no shared FP32 atomics. Scalar denominators and router moments are per query/candidate.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .planner_graph import acquire


@tr.jit
def _max(a,b):return tl.maximum(a,b)


@tr.jit
def _select(Q,EQ,IQ,HARD,IDS,PROB,beta,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,C:tl.constexpr,
            BC:tl.constexpr,TOPK:tl.constexpr=4,M:tl.constexpr=16):
    bh=tl.program_id(1);row=tl.program_id(0)*M+tl.arange(0,M)
    if tl.load(HARD+bh):
        d=tl.arange(0,D);c=tl.arange(0,BC)
        q=tl.load(Q+((bh//H*N+row[:,None])*H+bh%H)*D+d[None,:],row[:,None]<N,0)
        eq=tl.load(EQ+((bh%H)*C+c[:,None])*D+d[None,:],c[:,None]<C,0).to(q.dtype)
        z=tl.dot(q,tl.trans(eq))*beta
        z=tl.where(c[None,:]<C,z,-float('inf'))
        i0=tl.load(IQ+bh*N+row,row<N,0)
        z0=tl.sum(tl.where(c[None,:]==i0[:,None],z,0.),1)
        if TOPK == 1:
            tl.store(IDS+(bh*N+row),i0,row<N)
            tl.store(PROB+(bh*N+row),1.,row<N)
        elif TOPK == 2:
            z=tl.where(c[None,:]==i0[:,None],-float('inf'),z)
            z1=tl.max(z,1);i1=tl.min(tl.where(z==z1[:,None],c[None,:],2147483647),1)
            maximum=tl.maximum(z0,z1)
            p0=tl.exp(z0-maximum);p1=tl.exp(z1-maximum)
            tl.store(IDS+(bh*N+row)*2,i0,row<N)
            tl.store(IDS+(bh*N+row)*2+1,i1,row<N)
            tl.store(PROB+(bh*N+row)*2,p0/(p0+p1),row<N)
            tl.store(PROB+(bh*N+row)*2+1,p1/(p0+p1),row<N)
        else:
            z=tl.where(c[None,:]==i0[:,None],-float('inf'),z)
            z1=tl.max(z,1);i1=tl.min(tl.where(z==z1[:,None],c[None,:],2147483647),1)
            z=tl.where(c[None,:]==i1[:,None],-float('inf'),z)
            z2=tl.max(z,1);i2=tl.min(tl.where(z==z2[:,None],c[None,:],2147483647),1)
            z=tl.where(c[None,:]==i2[:,None],-float('inf'),z)
            z3=tl.max(z,1);i3=tl.min(tl.where(z==z3[:,None],c[None,:],2147483647),1)
            maximum=tl.maximum(tl.maximum(z0,z1),tl.maximum(z2,z3))
            p0=tl.exp(z0-maximum);p1=tl.exp(z1-maximum);p2=tl.exp(z2-maximum);p3=tl.exp(z3-maximum)
            denominator=p0+p1+p2+p3
            for slot in tl.static_range(4):
                label=i0 if slot==0 else (i1 if slot==1 else (i2 if slot==2 else i3))
                p=p0 if slot==0 else (p1 if slot==1 else (p2 if slot==2 else p3))
                tl.store(IDS+(bh*N+row)*4+slot,label,row<N)
                tl.store(PROB+(bh*N+row)*4+slot,p/denominator,row<N)



@tr.jit
def _router_dq(EQ,IDS,DZ,HARD,DQ,beta,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,C:tl.constexpr,TOPK:tl.constexpr=4,M:tl.constexpr=32):
    bh=tl.program_id(1);row=tl.program_id(0)*M+tl.arange(0,M)
    if tl.load(HARD+bh):
        d=tl.arange(0,D)
        total=tl.full((M,D),0,tl.float32)
        for slot in tl.static_range(TOPK):
            index=tl.load(IDS+(bh*N+row)*TOPK+slot,row<N,0)
            dz=tl.load(DZ+(bh*N+row)*TOPK+slot,row<N,0)*beta
            eq=tl.load(EQ+((bh%H)*C+index[:,None])*D+d[None,:]).to(tl.bfloat16).to(tl.float32)
            total+=dz[:,None]*eq
        tl.store(DQ+((bh//H*N+row[:,None])*H+bh%H)*D+d[None,:],total,row[:,None]<N)


@tr.jit
def _router_deq(Q,IDS,DZ,DEQ,beta,B:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,C:tl.constexpr,
                K:tl.constexpr=32,M:tl.constexpr=64,TOPK:tl.constexpr=4):
    h=tl.program_id(1);c=tl.program_id(0)*K+tl.arange(0,K);d=tl.arange(0,D)
    acc=tl.full((K,D),0,tl.float32)
    for start in range(0,B*N,M):
        t=start+tl.arange(0,M)
        location=((t//N)*H+h)*N+t%N
        w=tl.full((M,K),0,tl.float32)
        for slot in tl.static_range(TOPK):
            label=tl.load(IDS+location*TOPK+slot,t<B*N,0)
            dz=tl.load(DZ+location*TOPK+slot,t<B*N,0)*beta
            w+=tl.where(label[:,None]==c[None,:],dz[:,None],0.)
        q=tl.load(Q+(t[:,None]*H+h)*D+d[None,:],t[:,None]<B*N,0).to(tl.float32)
        acc+=tl.dot(tl.trans(w),q,input_precision='tf32x3')
    tl.store(DEQ+(h*C+c[:,None])*D+d[None,:],acc,c[:,None]<C)


def select(q,eq,iq,hard,beta,topk=4):
    if topk not in (1,2,4):raise ValueError("topk must be 1, 2 or 4")
    b,n,h,d=q.shape;c=eq.shape[1]
    ids=torch.zeros((b,h,n,topk),device=q.device,dtype=torch.int32)
    prob=torch.zeros((b,h,n,topk),device=q.device,dtype=torch.float32)
    _select[(tr.cdiv(n,16),b*h)](q,eq,iq,hard,ids,prob,beta,n,h,d,c,max(16,tr.next_power_of_2(c)),TOPK=topk,num_warps=4)
    return ids,prob


def router_backward(q,eq,ids,dz,hard,beta):
    b,n,h,d=q.shape;c=eq.shape[1]
    dq=torch.zeros_like(q,dtype=torch.float32);deq=torch.empty_like(eq,dtype=torch.float32)
    _router_dq[(tr.cdiv(n,32),b*h)](eq,ids,dz,hard,dq,beta,n,h,d,c,TOPK=ids.shape[-1])
    _router_deq[(tr.cdiv(c,32),h)](q,ids,dz,deq,beta,b,n,h,d,c,TOPK=ids.shape[-1],num_warps=4,num_stages=1)
    return dq,deq


@tr.jit
def _boundaries(IQ,IK,SEG,HARD,BOUND,N:tl.constexpr,H:tl.constexpr,BD:tl.constexpr):
    bh,di=tl.program_id(0),tl.program_id(1)
    if tl.load(HARD+bh):
        d=di*BD+tl.arange(0,BD)
        r=tl.arange(0,32)
        carried=tl.full((BD,),0,tl.int32)
        for start in range((di*BD//32)*32,N,32):
            qi=start+r
            kj=qi[:,None]-d[None,:]
            q=tl.load(IQ+bh*N+qi,qi<N,-1)
            k=tl.load(IK+bh*N+kj,(kj>=0)&(kj<N),-2)
            qs=tl.load(SEG+(bh//H)*N+qi,qi<N,-3)
            ks=tl.load(SEG+(bh//H)*N+kj,(kj>=0)&(kj<N),-4)
            qps=tl.load(SEG+(bh//H)*N+qi-1,(qi>0)&(qi<N),-5)
            kps=tl.load(SEG+(bh//H)*N+kj-1,(kj>0)&(kj<N),-6)
            match=(q[:,None]==k)&(qs[:,None]==ks)&(kj>=0)
            # Segment boundary breaks the incoming chain, but the current
            # matching pair can start a fresh length-one chain.
            reset=(qs[:,None]!=qps[:,None])|(ks!=kps)
            reset_first=tl.sum(tl.where(r[:,None]==0,reset.to(tl.int32),0),0)>0
            carried=tl.where(reset_first,0,carried)
            bad=tl.where(~match,r[:,None],tl.where(reset,r[:,None]-1,-1))
            last=tl.associative_scan(bad,0,_max)
            length=tl.where(last>=0,r[:,None]-last,carried[None,:]+r[:,None]+1)
            length=tl.where(match,length,0)
            carried=tl.sum(tl.where(r[:,None]==31,length,0),0)
            j=start+31-d
            if start+32<N:
                tl.store(BOUND+(bh*((N-1)//32)+start//32)*N+j,carried,(j>=0)&(j<N))


@tr.jit
def _lengths(IQ,IK,SEG,BOUND,bh,q0,k0,N:tl.constexpr,H:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr):
    # Diagonal coordinates cover every predecessor needed by a BQ x BK tile.
    r=tl.arange(0,BQ)
    d=tl.arange(0,2*BK)-BQ
    qi=q0+r
    kj=k0+d[None,:]+r[:,None]
    qprev=tl.load(IQ+bh*N+qi-1,(qi>0)&(qi<N),-1)
    kprev=tl.load(IK+bh*N+kj-1,(kj>0)&(kj<=N),-2)
    # Public training contract: document starts are 256-aligned. A 32x64
    # tile cannot contain an interior document boundary. Only its first
    # query/key can break the incoming history; valid diagonals cross k0 at
    # most once. This avoids loading a full 32x128 segment-ID tile.
    qs=tl.load(SEG+(bh//H)*N+q0)
    qps=tl.load(SEG+(bh//H)*N+q0-1,q0>0,-4)
    ks=tl.load(SEG+(bh//H)*N+k0)
    kps=tl.load(SEG+(bh//H)*N+k0-1,k0>0,-6)
    broken=((r[:,None]>0)&(qprev[:,None]!=kprev))|((r[:,None]==0)&(qs!=qps))|((kj==k0)&(ks!=kps))
    # One bit per row is sufficient: find the latest mismatch with CLZ,
    # instead of retaining a full integer scan tile through vector MMAs.
    bits=tl.sum(tl.where(broken,tl.full((),1,tl.uint32)<<r[:,None],0),0)
    col=tl.arange(0,BK)
    indices=col[None,:]-r[:,None]+BQ
    pair_bits=tl.reshape(tl.gather(bits,tl.reshape(indices,(BQ*BK,)),0),(BQ,BK))
    mask=tl.full((),0xffffffff,tl.uint32)>>(31-r)
    last=31-libdevice.clz((pair_bits&mask[:,None]).to(tl.int32))
    incoming_j=k0+col[None,:]-r[:,None]-1
    incoming=tl.load(BOUND+(bh*((N-1)//32)+q0//32-1)*N+incoming_j,
                     (q0>0)&(incoming_j>=0)&(incoming_j<q0),0)
    length=tl.where(last>=0,r[:,None]-last,incoming+r[:,None])+1
    key=k0+col
    valid=(key[None,:]<=qi[:,None])&(key[None,:]<N)&(qi[:,None]<N)&(qs==ks)
    return length,valid


@tr.jit
def _lut(TAU,LOG,DLOG,N:tl.constexpr,BLOCK:tl.constexpr):
    h=tl.program_id(0)
    l=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    terms=tl.maximum(l,1).to(tl.float32)
    t=tl.load(TAU+h);a=tl.abs(t);la=terms*a
    geom=tl.log(-libdevice.expm1(-la))-tl.log(-libdevice.expm1(-a))
    logw=tl.where(t>0,terms*t,t)+geom
    ra=tl.exp(-a)/(-libdevice.expm1(-a))
    rl=tl.exp(-la)/(-libdevice.expm1(-la))
    dw=tl.where(t>0,terms+terms*rl-ra,1.-terms*rl+ra)
    series=tl.log(terms)+t*(terms+1)/2+t*t*(terms*terms-1)/24-t*t*t*t*(terms*terms*terms*terms-1)/2880
    ds=(terms+1)/2+t*(terms*terms-1)/12-t*t*t*(terms*terms*terms*terms-1)/720
    logw=tl.where(la<.01,series,logw)
    dw=tl.where(la<.01,ds,dw)
    tl.store(LOG+h*(N+1)+l,logw*1.4426950408889634,l<=N)
    tl.store(DLOG+h*(N+1)+l,dw,l<=N)


@tr.jit
def _document_extent(SEG,bh,position,N:tl.constexpr,H:tl.constexpr,END:tl.constexpr):
    # Validated IDs identify contiguous documents; every start is a multiple
    # of 256. Find the containing interval entirely on device during replay.
    chunks=tl.arange(0,tr.next_power_of_2(N//256))
    wanted=tl.load(SEG+(bh//H)*N+position)
    ids=tl.load(SEG+(bh//H)*N+chunks*256,chunks<N//256,-1)
    match=(chunks<N//256)&(ids==wanted)
    if END:
        return tl.max(tl.where(match,(chunks+1)*256,0),0)
    return tl.min(tl.where(match,chunks*256,N),0)


@tr.jit
def _normalizers(IQ,IK,SEG,HARD,BOUND,IDS,LOG,DLOG,DEN,LAMBDA,
                 N:tl.constexpr,H:tl.constexpr,BQ:tl.constexpr=32,BK:tl.constexpr=64,TOPK:tl.constexpr=4):
    q0=tl.program_id(0)*BQ;bh=tl.program_id(1)
    if tl.load(HARD+bh):
        rows=q0+tl.arange(0,BQ)
        slots=tl.arange(0,TOPK)
        symbol=tl.load(IDS+(bh*N+rows[:,None])*TOPK+slots[None,:])
        mx=tl.full((BQ,TOPK),0,tl.float32)
        den=tl.full((BQ,TOPK),1,tl.float32)
        moment=tl.full((BQ,TOPK),0,tl.float32)
        first=_document_extent(SEG,bh,q0,N,H,False)
        for k0 in range(first,q0+BQ,BK):
            length,valid=_lengths(IQ,IK,SEG,BOUND,bh,q0,k0,N,H,BQ,BK)
            keys=tl.load(IK+bh*N+k0+tl.arange(0,BK),k0+tl.arange(0,BK)<N,-1)
            match=valid[:,None,:]&(keys[None,None,:]==symbol[:,:,None])
            # Load directly in the candidate layout. Broadcasting a separately
            # loaded [BQ,BK] derivative produced redundant shared STS writes
            # in Triton's K4 layout conversion (racecheck WAW at STS.128).
            # Candidate-dependent masks avoid that conversion without changing
            # the normalizer or its derivative.
            index=tl.broadcast_to(length[:,None,:],(BQ,TOPK,BK))
            score=tl.load(LOG+(bh%H)*(N+1)+index,match,-float('inf'))
            deriv=tl.load(DLOG+(bh%H)*(N+1)+index,match,0.)
            new=tl.maximum(mx,tl.max(score,2))
            rescale=tl.exp2(mx-new)
            w=tl.exp2(score-new[:,:,None])
            den=den*rescale+tl.sum(w,2)
            moment=moment*rescale+tl.sum(w*deriv,2)
            mx=new
        tl.store(DEN+(bh*N+rows[:,None])*TOPK+slots[None,:],mx+tl.log2(den))
        tl.store(LAMBDA+(bh*N+rows[:,None])*TOPK+slots[None,:],moment/den)


@tr.jit
def _weights(IQ,IK,SEG,BOUND,IDS,PROB,LOG,DEN,LAMBDA,bh,q0,k0,
             N:tl.constexpr,H:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr,TOPK:tl.constexpr=4):
    rows=q0+tl.arange(0,BQ);cols=k0+tl.arange(0,BK)
    length,valid=_lengths(IQ,IK,SEG,BOUND,bh,q0,k0,N,H,BQ,BK)
    key=tl.load(IK+bh*N+cols,cols<N,-1)
    score=tl.load(LOG+(bh%H)*(N+1)+length)
    selected_den=tl.full((BQ,BK),0,tl.float32)
    selected_prob=tl.full((BQ,BK),0,tl.float32)
    lam=tl.full((BQ,BK),0,tl.float32)
    slotid=tl.full((BQ,BK),-1,tl.int32)
    for s in tl.static_range(TOPK):
        symbol=tl.load(IDS+(bh*N+rows)*TOPK+s)
        p=tl.load(PROB+(bh*N+rows)*TOPK+s)
        den=tl.load(DEN+(bh*N+rows)*TOPK+s)
        ld=tl.load(LAMBDA+(bh*N+rows)*TOPK+s)
        match=valid&(key[None,:]==symbol[:,None])
        selected_den=tl.where(match,den[:,None],selected_den)
        selected_prob=tl.where(match,p[:,None],selected_prob)
        lam+=tl.where(match,ld[:,None],0.)
        slotid=tl.where(match,s,slotid)
    weight=tl.exp2(tl.where(slotid>=0,score-selected_den,-float('inf')))*selected_prob
    return weight,length,lam,slotid


@tr.jit
def _forward(SQ,SK,V,IQ,IK,SEG,HARD,BOUND,IDS,PROB,LOG,DEN,LAMBDA,OUT,
             N:tl.constexpr,H:tl.constexpr,R:tl.constexpr,DV:tl.constexpr,BQ:tl.constexpr=32,BK:tl.constexpr=64,TOPK:tl.constexpr=4):
    q0=tl.program_id(0)*BQ;bh=tl.program_id(1)
    if tl.load(HARD+bh):
        rows=q0+tl.arange(0,BQ);r=tl.arange(0,R);d=tl.arange(0,DV)
        qoff=((bh//H*N+rows)*H+bh%H)
        sq=tl.load(SQ+qoff[:,None]*R+r[None,:])
        acc=tl.full((BQ,DV),0,tl.float32)
        first=_document_extent(SEG,bh,q0,N,H,False)
        for k0 in range(first,q0+BQ,BK):
            cols=k0+tl.arange(0,BK)
            koff=((bh//H*N+cols)*H+bh%H)
            sk=tl.load(SK+koff[:,None]*R+r[None,:],cols[:,None]<N,0)
            v=tl.load(V+koff[:,None]*DV+d[None,:],cols[:,None]<N,0)
            m,_,_,_=_weights(IQ,IK,SEG,BOUND,IDS,PROB,LOG,DEN,LAMBDA,bh,q0,k0,N,H,BQ,BK,TOPK)
            score=tl.dot(sq,tl.trans(sk))
            acc+=tl.dot((m*score).to(v.dtype),v)
        tl.store(OUT+qoff[:,None]*DV+d[None,:],acc)


@tr.jit
def _backward(SQ,SK,V,DO,IQ,IK,SEG,HARD,BOUND,IDS,PROB,LOG,DLOG,DEN,LAMBDA,
              DSQ,DSK,DVOUT,DT,U,N:tl.constexpr,H:tl.constexpr,R:tl.constexpr,DV:tl.constexpr,
              BQ:tl.constexpr=32,BK:tl.constexpr=64,TOPK:tl.constexpr=4,Choice=None):
    k0=tl.program_id(0)*BK;bh=tl.program_id(1)
    enabled=True
    if Choice is not None:enabled=acquire(Choice)!=1
    if enabled and tl.load(HARD+bh):
        cols=k0+tl.arange(0,BK);r=tl.arange(0,R);d=tl.arange(0,DV)
        koff=((bh//H*N+cols)*H+bh%H)
        sk=tl.load(SK+koff[:,None]*R+r[None,:])
        v=tl.load(V+koff[:,None]*DV+d[None,:])
        dsk=tl.full((BK,R),0,tl.float32);dv=tl.full((BK,DV),0,tl.float32)
        dt=tl.full((),0,tl.float32)
        end=_document_extent(SEG,bh,k0,N,H,True)
        for q0 in range(k0//BQ*BQ,end,BQ):
            rows=q0+tl.arange(0,BQ);qoff=((bh//H*N+rows)*H+bh%H)
            sq=tl.load(SQ+qoff[:,None]*R+r[None,:])
            do=tl.load(DO+qoff[:,None]*DV+d[None,:])
            m,length,lam,slot=_weights(IQ,IK,SEG,BOUND,IDS,PROB,LOG,DEN,LAMBDA,bh,q0,k0,N,H,BQ,BK,TOPK)
            score=tl.dot(sq,tl.trans(sk));e=tl.dot(do,tl.trans(v))
            f=m*score*e
            dl=tl.load(DLOG+(bh%H)*(N+1)+length)
            dt+=tl.sum(tl.sum(f*(dl-lam),1),0)
            for s in tl.static_range(TOPK):
                u=tl.sum(tl.where(slot==s,f,0.),1)
                tl.atomic_add(U+(bh*N+rows)*TOPK+s,u,sem='relaxed')
            ds=(m*e).to(sq.dtype)
            dsq=tl.dot(ds,sk)
            tl.atomic_add(DSQ+qoff[:,None]*R+r[None,:],dsq,sem='relaxed')
            dsk+=tl.dot(tl.trans(ds),sq)
            dv+=tl.dot(tl.trans((m*score).to(do.dtype)),do)
        tl.store(DSK+koff[:,None]*R+r[None,:],dsk)
        tl.store(DVOUT+koff[:,None]*DV+d[None,:],dv)
        tl.atomic_add(DT+bh%H,dt,sem='relaxed')


def forward(sq,sk,v,iq,ik,ids,prob,tau,hard,segments):
    b,n,h,r=sq.shape;topk=ids.shape[-1]
    bound=torch.empty((b,h,(n-1)//32,n),device=sq.device,dtype=torch.int32)
    log=torch.empty((h,n+1),device=sq.device);dlog=torch.empty_like(log)
    den=torch.empty((b,h,n,topk),device=sq.device);lam=torch.empty_like(den)
    out=torch.zeros_like(v)
    _boundaries[(b*h,tr.cdiv(n,128))](iq,ik,segments,hard,bound,n,h,128,num_warps=8)
    _lut[(h,tr.cdiv(n+1,256))](tau,log,dlog,n,256)
    _normalizers[(n//32,b*h)](iq,ik,segments,hard,bound,ids,log,dlog,den,lam,n,h,TOPK=topk,BK=32,num_warps=4,num_stages=1)
    _forward[(n//32,b*h)](sq,sk,v,iq,ik,segments,hard,bound,ids,prob,log,den,lam,out,n,h,r,v.shape[-1],TOPK=topk,num_warps=4,num_stages=1)
    return out,(bound,log,dlog,den,lam)


def backward(sq,sk,v,do,iq,ik,ids,prob,hard,segments,state,*,choice=None):
    b,n,h,r=sq.shape
    bound,log,dlog,den,lam=state
    dsq=torch.zeros_like(sq,dtype=torch.float32);dsk=torch.zeros_like(sk,dtype=torch.float32)
    dv=torch.zeros_like(v,dtype=torch.float32);dt=torch.zeros(h,device=v.device)
    u=torch.zeros_like(prob)
    _backward[(n//32,b*h)](sq,sk,v,do,iq,ik,segments,hard,bound,ids,prob,log,dlog,den,lam,
                           dsq,dsk,dv,dt,u,n,h,r,v.shape[-1],TOPK=ids.shape[-1],Choice=choice,BK=32,num_warps=4,num_stages=1)
    dz=u-prob*u.sum(-1,keepdim=True)
    return dsq,dsk,dv,dt,dz
