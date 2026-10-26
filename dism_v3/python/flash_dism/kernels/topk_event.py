"""Decoupled K-state/Q-read backward, fixed-grid graph consumer.

Descriptor: legacy compact records [0:64], immutable legacy weights [64:68],
counts (blocks, tiles, streams) [68:71], split pointers [71:88],
H/G/alpha [88:91], immutable split query weights [91].
All descriptor reads require an acquired, frozen route. No host shape reads.
"""
import torch
import triton as tr
import triton.language as tl

from flash_dism.kernels.hard_backward_fused import _bf16_fp32_dot, _summary_scan


@tr.jit
def _key_summary(SK,V,H,A,KR,KP,KW,KL,RESET,R:tl.constexpr,D:tl.constexpr,
                 BK:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr,PRECISE:tl.constexpr,BF16:tl.constexpr,TASK=None):
    block=tl.program_id(0) if TASK is None else TASK;k=tl.arange(0,BK);r=tl.arange(0,BR);d=tl.arange(0,BD)
    length=tl.load(KL+block);row=tl.load(KR+block*BK+k);p=tl.load(KP+block*BK+k);w=tl.load(KW+block*BK+k)
    sk=tl.load(SK+row[:,None]*R+r[None,:],(k[:,None]<length)&(r[None,:]<R),0).to(tl.float32)
    v=tl.load(V+row[:,None]*D+d[None,:],(k[:,None]<length)&(d[None,:]<D),0).to(tl.float32)
    last=tl.load(KP+block*BK+length-1)
    scale=tl.where(k<length,w*tl.exp(tl.minimum(last-p,0.)),0.)
    if PRECISE and BF16:
        h=tl.trans(_bf16_fp32_dot(tl.trans(v),sk*scale[:,None]))
    elif BF16:
        h=tl.dot(tl.trans(sk).to(tl.bfloat16),(v*scale[:,None]).to(tl.bfloat16))
    else:
        h=tl.dot(tl.trans(sk),v*scale[:,None],input_precision="tf32x3")
    tl.store(H+block.to(tl.int64)*R*D+r[:,None]*D+d[None,:],h,(r[:,None]<R)&(d[None,:]<D))
    tl.store(A+block,tl.where(tl.load(RESET+block)!=0,0.,tl.exp(last)))


@tr.jit
def _query_summary(SQ,DO,G,OFFSETS,RESET,QR,QP,QW,QL,R:tl.constexpr,D:tl.constexpr,
                   BQ:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr,BF16:tl.constexpr,TASK=None):
    block=tl.program_id(0) if TASK is None else TASK;q=tl.arange(0,BQ);r=tl.arange(0,BR);d=tl.arange(0,BD)
    begin=tl.load(OFFSETS+block);end=tl.load(OFFSETS+block+1)
    total=tl.full((BR,BD),0.,tl.float32)
    # The first block has no dependence on a preceding state.
    if tl.load(RESET+block)==0:
        for tile in range(begin,end):
            length=tl.load(QL+tile);qi=tl.load(QR+tile*BQ+q)
            scale=tl.load(QW+tile*BQ+q)*tl.exp(tl.load(QP+tile*BQ+q))
            sq=tl.load(SQ+qi[:,None]*R+r[None,:],(q[:,None]<length)&(r[None,:]<R),0).to(tl.float32)
            do=tl.load(DO+qi[:,None]*D+d[None,:],(q[:,None]<length)&(d[None,:]<D),0).to(tl.float32)
            if BF16:piece=_bf16_fp32_dot(tl.trans(sq),do*scale[:,None])
            else:piece=tl.dot(tl.trans(sq),do*scale[:,None],input_precision="tf32x3")
            total+=piece
    tl.store(G+block.to(tl.int64)*R*D+r[:,None]*D+d[None,:],total,(r[:,None]<R)&(d[None,:]<D))


@tr.jit
def _query_backward(SQ,SK,V,DO,DSQ,DSK,DV,TAU,H,KR,KP,KW,KD,KL,RESET,KTAU,
                    QR,QP,QW,QD,QL,QCOUNT,QBLOCK,R:tl.constexpr,D:tl.constexpr,
                    BK:tl.constexpr,BQ:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr,BF16:tl.constexpr,TASK=None):
    tile=tl.program_id(0) if TASK is None else TASK;block=tl.load(QBLOCK+tile)
    q=tl.arange(0,BQ);k=tl.arange(0,BK);r=tl.arange(0,BR);d=tl.arange(0,BD)
    qlen=tl.load(QL+tile);klen=tl.load(KL+block)
    qi=tl.load(QR+tile*BQ+q);ki=tl.load(KR+block*BK+k)
    qp=tl.load(QP+tile*BQ+q);kp=tl.load(KP+block*BK+k)
    qw=tl.load(QW+tile*BQ+q);kw=tl.load(KW+block*BK+k)
    qdw=tl.load(QD+tile*BQ+q);kdw=tl.load(KD+block*BK+k);count=tl.load(QCOUNT+tile*BQ+q)
    sq=tl.load(SQ+qi[:,None]*R+r[None,:],(q[:,None]<qlen)&(r[None,:]<R),0).to(tl.float32)
    sk=tl.load(SK+ki[:,None]*R+r[None,:],(k[:,None]<klen)&(r[None,:]<R),0).to(tl.float32)
    v=tl.load(V+ki[:,None]*D+d[None,:],(k[:,None]<klen)&(d[None,:]<D),0).to(tl.float32)
    do=tl.load(DO+qi[:,None]*D+d[None,:],(q[:,None]<qlen)&(d[None,:]<D),0).to(tl.float32)
    pair=tl.where((q[:,None]<qlen)&(k[None,:]<count[:,None]),qw[:,None]*kw[None,:]*tl.exp(tl.minimum(qp[:,None]-kp[None,:],0.)),0.)
    if BF16:
        ds=tl.dot(do.to(tl.bfloat16),tl.trans(v).to(tl.bfloat16))*pair
        score=tl.dot(sq.to(tl.bfloat16),tl.trans(sk).to(tl.bfloat16))
    else:
        ds=tl.dot(do,tl.trans(v),input_precision="tf32x3")*pair
        score=tl.dot(sq,tl.trans(sk),input_precision="tf32x3")
    tau=tl.sum(tl.sum(ds*score*(qdw[:,None]+kdw[None,:]),1),0)
    if BF16:
        dv=tl.dot(tl.trans((score*pair).to(tl.bfloat16)),do.to(tl.bfloat16))
        dq=tl.dot(ds.to(tl.bfloat16),sk.to(tl.bfloat16))
        dk=tl.dot(tl.trans(ds.to(tl.bfloat16)),sq.to(tl.bfloat16))
    else:
        dv=tl.dot(tl.trans(score*pair),do,input_precision="tf32x3")
        dq=tl.dot(ds,sk,input_precision="tf32x3")
        dk=tl.dot(tl.trans(ds),sq,input_precision="tf32x3")
    h=tl.load(H+block.to(tl.int64)*R*D+r[:,None]*D+d[None,:],(r[:,None]<R)&(d[None,:]<D),0)
    if BF16:history=_bf16_fp32_dot(do,tl.trans(h))
    else:history=tl.dot(do,tl.trans(h),input_precision="tf32x3")
    qs=tl.where(tl.load(RESET+block)==0,qw*tl.exp(qp),0.)
    tau+=tl.sum(qdw*tl.sum(sq*history,1)*qs,0)
    tl.atomic_add(DSQ+qi[:,None]*R+r[None,:],dq+history*qs[:,None],(q[:,None]<qlen)&(r[None,:]<R),sem="relaxed")
    tl.atomic_add(DSK+ki[:,None]*R+r[None,:],dk,(k[:,None]<klen)&(r[None,:]<R),sem="relaxed")
    tl.atomic_add(DV+ki[:,None]*D+d[None,:],dv,(k[:,None]<klen)&(d[None,:]<D),sem="relaxed")
    tl.atomic_add(TAU+tl.load(KTAU+block),tau,sem="relaxed")


@tr.jit
def _key_backward(SK,V,DSK,DV,TAU,G,KR,KP,KW,KD,KL,KTAU,R:tl.constexpr,D:tl.constexpr,
                  BK:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr,BF16:tl.constexpr,TASK=None):
    block=tl.program_id(0) if TASK is None else TASK;k=tl.arange(0,BK);r=tl.arange(0,BR);d=tl.arange(0,BD)
    length=tl.load(KL+block);row=tl.load(KR+block*BK+k)
    p=tl.load(KP+block*BK+k);w=tl.load(KW+block*BK+k);dw=tl.load(KD+block*BK+k)
    last=tl.load(KP+block*BK+length-1);scale=tl.where(k<length,w*tl.exp(tl.minimum(last-p,0.)),0.)
    sk=tl.load(SK+row[:,None]*R+r[None,:],(k[:,None]<length)&(r[None,:]<R),0).to(tl.float32)
    v=tl.load(V+row[:,None]*D+d[None,:],(k[:,None]<length)&(d[None,:]<D),0).to(tl.float32)
    g=tl.load(G+block.to(tl.int64)*R*D+r[:,None]*D+d[None,:],(r[:,None]<R)&(d[None,:]<D),0)
    if BF16:
        dk=_bf16_fp32_dot(v,tl.trans(g))
        dv=tl.dot(sk.to(tl.bfloat16),g.to(tl.bfloat16))
    else:
        dk=tl.dot(v,tl.trans(g),input_precision="tf32x3")
        dv=tl.dot(sk,g,input_precision="tf32x3")
    tau=tl.sum(dw*tl.sum(sk*dk,1)*scale,0)
    tl.atomic_add(DSK+row[:,None]*R+r[None,:],dk*scale[:,None],(k[:,None]<length)&(r[None,:]<R),sem="relaxed")
    tl.atomic_add(DV+row[:,None]*D+d[None,:],dv*scale[:,None],(k[:,None]<length)&(d[None,:]<D),sem="relaxed")
    tl.atomic_add(TAU+tl.load(KTAU+block),tau,sem="relaxed")


from .planner_graph import acquire, pointer, GraphCompactBackward


@tr.jit
def _split_weights(D,Choice,PI,BLOCK:tl.constexpr=256):
    if acquire(Choice)==1:
        count=tl.load(D+69)*64;rows=pointer(D,79,tl.int32)
        src=pointer(D,91,tl.float32);dst=pointer(D,81,tl.float32)
        for start in range(tl.program_id(0)*BLOCK,count,tl.num_programs(0)*BLOCK):
            i=start+tl.arange(0,BLOCK)
            row=tl.load(rows+i,i<count,0);w=tl.load(src+i,i<count,0.)
            p=tl.load(PI+row,i<count,1.)
            tl.store(dst+i,w*p,i<count)


@tr.jit
def _split_stage(D,Choice,SQ,SK,V,DO,DQ,DK,DV,DT,STAGE:tl.constexpr):
    if acquire(Choice)==1:
        count=tl.load(D+(69 if STAGE==3 else 70 if STAGE==2 else 68)).to(tl.int32)
        kr=pointer(D,71,tl.int32);kp=pointer(D,72,tl.float32);kw=pointer(D,73,tl.float32);kd=pointer(D,74,tl.float32)
        kl=pointer(D,75,tl.int32);reset=pointer(D,76,tl.int32);ktau=pointer(D,77,tl.int32)
        # Slot 78 is reserved for stream offsets; Q fields start at 79.
        offsets=pointer(D,78,tl.int64)
        qr=pointer(D,79,tl.int32);qp=pointer(D,80,tl.float32);qw=pointer(D,81,tl.float32);qd=pointer(D,82,tl.float32)
        ql=pointer(D,83,tl.int32);qc=pointer(D,84,tl.int32);qb=pointer(D,85,tl.int32);qo=pointer(D,86,tl.int64)
        h=pointer(D,88,tl.float32);g=pointer(D,89,tl.float32);a=pointer(D,90,tl.float32)
        for task in range(tl.program_id(0),count,tl.num_programs(0)):
            if STAGE==0:
                _key_summary(SK,V,h,a,kr,kp,kw,kl,reset,32,64,64,32,64,True,True,TASK=task)
            elif STAGE==1:
                _query_summary(SQ,DO,g,qo,reset,qr,qp,qw,ql,32,64,64,32,64,True,TASK=task)
            elif STAGE==2:
                _summary_scan(h,g,a,offsets,2048,2048,TASK=task)
            elif STAGE==3:
                _query_backward(SQ,SK,V,DO,DQ,DK,DV,DT,h,kr,kp,kw,kd,kl,reset,ktau,
                                qr,qp,qw,qd,ql,qc,qb,32,64,64,64,32,64,True,TASK=task)
            else:
                _key_backward(SK,V,DK,DV,DT,g,kr,kp,kw,kd,kl,ktau,32,64,64,32,64,True,TASK=task)


@tr.jit
def _expand_queries(Choice,SQ,DO,PROB,QS,DOS,PI,N:tl.constexpr,H:tl.constexpr,K:tl.constexpr,TOTAL:tl.constexpr,M:tl.constexpr=16):
    if acquire(Choice)==1:
        row=tl.program_id(0)*M+tl.arange(0,M);d=tl.arange(0,64)
        sq=tl.load(SQ+row[:,None]*32+d[None,:],(row[:,None]<TOTAL)&(d[None,:]<32),0)
        do=tl.load(DO+row[:,None]*64+d[None,:],row[:,None]<TOTAL,0)
        source=((row//(N*H))*H+row%H)*N+row//H%N
        for slot in tl.static_range(K):
            tl.store(QS+(row[:,None]*K+slot)*32+d[None,:],sq,(row[:,None]<TOTAL)&(d[None,:]<32))
            tl.store(DOS+(row[:,None]*K+slot)*64+d[None,:],do,row[:,None]<TOTAL)
            prob=tl.load(PROB+source*K+slot,row<TOTAL,0.)
            tl.store(PI+row*K+slot,prob,row<TOTAL)


@tr.jit
def _collapse_queries(Choice,DQ,SQ,PI,DSQ,DZ,N:tl.constexpr,H:tl.constexpr,K:tl.constexpr,TOTAL:tl.constexpr,M:tl.constexpr=16):
    row=tl.program_id(0)*M+tl.arange(0,M);slot=tl.arange(0,K);r=tl.arange(0,32)
    dsq=tl.full((M,32),0.,tl.float32);dz=tl.full((M,K),0.,tl.float32)
    if acquire(Choice)==1:
        dq=tl.load(DQ+(row[:,None,None]*K+slot[None,:,None])*32+r[None,None,:],row[:,None,None]<TOTAL,0.)
        sq=tl.load(SQ+row[:,None]*32+r[None,:],row[:,None]<TOTAL,0.).to(tl.float32)
        dsq=tl.sum(dq,1)
        u=tl.sum(dq*sq[:,None,:],2)
        prob=tl.load(PI+row[:,None]*K+slot[None,:],row[:,None]<TOTAL,0.)
        dz=u-prob*tl.sum(u,1)[:,None]
    dst=((row//(N*H))*H+row%H)*N+row//H%N
    tl.store(DSQ+row[:,None]*32+r[None,:],dsq,row[:,None]<TOTAL)
    tl.store(DZ+dst[:,None]*K+slot[None,:],dz,row[:,None]<TOTAL)


class TopkEventBackward:
    def __init__(self,descriptor,choice):
        if descriptor.shape!=(96,):raise ValueError('topk descriptor must contain 96 words')
        self.descriptor,self.choice=descriptor,choice
        self.base=GraphCompactBackward(descriptor[:64],choice,virtual_queries=True)
        self.last_kernels=[]

    @torch.no_grad()
    def backward(self,sq,sk,v,do,prob):
        if (sq.ndim!=4 or min(sq.shape[:3])<1 or sq.shape[-1]!=32 or sk.shape!=sq.shape
                or v.shape!=(*sq.shape[:3],64) or do.shape!=v.shape
                or prob.ndim!=4 or prob.shape[:3]!=(sq.shape[0],sq.shape[2],sq.shape[1])
                or prob.shape[-1] not in (1,2,4) or prob.dtype!=torch.float32
                or any(x.device!=self.descriptor.device or not x.is_contiguous() for x in (sq,sk,v,do,prob))
                or any(x.dtype!=torch.bfloat16 for x in (sq,sk,v,do))):
            raise ValueError('topk event requires contiguous BF16 BNHR32/BNHD64 and FP32 BHNK probabilities')
        b,n,h,r=sq.shape;topk=prob.shape[-1]
        if b*n*h*topk*64>2**31-1:raise ValueError("topk vectors exceed int32 addressing")
        # Exact replication is gated; late plans do not copy virtual Q/dO.
        qs=torch.empty((b*n*h*topk,32),dtype=sq.dtype,device=sq.device)
        ds=torch.empty((b*n*h*topk,64),dtype=do.dtype,device=do.device)
        pi=torch.empty(b*n*h*topk,dtype=torch.float32,device=sq.device)
        D=self.descriptor
        self.last_kernels=[_expand_queries[(tr.cdiv(b*n*h,16),)](
            self.choice,sq,do,prob,qs,ds,pi,n,h,topk,b*n*h,num_warps=4)]
        for i,c in enumerate((16,32,64,64)):
            # The immutable weight pointer belongs to the full descriptor.
            self.last_kernels.append(_legacy_weights[(128,)](D,self.choice,pi,i,c))
        self.last_kernels.append(_split_weights[(128,)](D,self.choice,pi))
        out=self.base.backward(qs,sk.reshape(-1,r),v.reshape(-1,64),ds,heads=h)
        self.last_kernels+=self.base.last_kernels
        args=[out[name] for name in ('sq_vec','sk_vec','v','rtau')]
        for stage in range(5):
            self.last_kernels.append(_split_stage[(512,1)](D,self.choice,qs,sk,v,ds,*args,stage,num_warps=4,num_stages=1))
        dq=torch.empty_like(sq,dtype=torch.float32);dz=torch.empty_like(prob)
        # Output is linear in SQ: dot(SQ,dSQ_c) = pi_c * dot(dO,O_c).
        self.last_kernels.append(_collapse_queries[(tr.cdiv(b*n*h,16),)](
            self.choice,out['sq_vec'],sq,pi,dq,dz,n,h,topk,b*n*h,num_warps=4))
        return dq,out['sk_vec'].view_as(sk),out['v'].view_as(v),out['rtau'],dz



@tr.jit
def _legacy_weights(D,Choice,PI,I:tl.constexpr,C:tl.constexpr,BLOCK:tl.constexpr=256):
    if acquire(Choice)==1:
        count=tl.load(D+16*I)*C
        rows=pointer(D+16*I,2,tl.int32);src=pointer(D,64+I,tl.float32);dst=pointer(D+16*I,4,tl.float32)
        for start in range(tl.program_id(0)*BLOCK,count,tl.num_programs(0)*BLOCK):
            i=start+tl.arange(0,BLOCK)
            row=tl.load(rows+i,i<count,0);w=tl.load(src+i,i<count,0.)
            p=tl.load(PI-row-1,(i<count)&(row<0),1.)
            tl.store(dst+i,w*p,i<count)
