"""Opt-in top-four hard autograd; native v3 soft and sampling stay unchanged."""
import math
import os
import torch
from .kernels import topk_hard as kernels


class _NoPlanner:
    def submit(self,*args,**kwargs):return None


class _HardForward:
    def __init__(self,q,eq,beta,segments,rows):
        self.q,self.eq,self.beta,self.segments,self.rows=q,eq,beta,segments,rows

    def __call__(self,sq,sk,v,iq,ik,hard,tau,**kwargs):
        rows=self.rows;n=self.q.shape[0]*self.q.shape[1]//rows;h=self.q.shape[2]
        labels=lambda x:x.transpose(1,2).reshape(rows,n,h).transpose(1,2).contiguous()
        iq,ik=labels(iq),labels(ik)
        q=self.q.reshape(rows,n,h,-1)
        ids,pi=kernels.select(q,self.eq,iq,hard,self.beta)
        output,state=kernels.forward(sq.reshape(rows,n,h,-1),sk.reshape(rows,n,h,-1),v.reshape(rows,n,h,-1),
            iq,ik,ids,pi,tau,hard,self.segments.reshape(rows,n))
        self.saved=(iq,ik,ids,pi,*state)
        return output.reshape_as(v),output.new_empty((0,))


class _Vocabulary(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,sq,sk,v,eq,ek,tau,direction,hard,segments,boundaries,layout,beta):
        from .compiler import _voc_forward_impl
        rows=hard.shape[0]
        engine=_HardForward(q,eq,beta,segments,rows)
        saved=_voc_forward_impl(q,k,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout,True,
                               direction_true=True,planner=_NoPlanner(),hard_forward_override=engine)
        ctx.beta=beta;ctx.rows=rows
        ctx.has_boundaries=boundaries is not None;ctx.has_layout=layout is not None
        ctx.save_for_backward(q,k,sq,sk,v,eq,ek,tau,direction,hard,segments,*saved,*engine.saved,
            *([] if boundaries is None else [boundaries]),*([] if layout is None else [layout]))
        return saved[0]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,dout):
        from .compiler import _voc_backward_impl
        s=ctx.saved_tensors
        raw=list(s[:8]);direction,hard,segments=s[8:11]
        native=list(s[11:23]);iq,ik,ids,pi=s[23:27];state=s[27:32]
        boundaries=s[32] if ctx.has_boundaries else None
        layout=s[32+int(ctx.has_boundaries)] if ctx.has_layout else None
        grads=_voc_backward_impl([*raw,ctx.beta,direction,hard],native,boundaries,dout,layout,direction_true=True)
        rows=ctx.rows;n=raw[0].shape[0]*raw[0].shape[1]//rows;h=raw[0].shape[2]
        q,sq,sk,v=[raw[i].reshape(rows,n,h,-1) for i in (0,2,3,4)]
        dsq,dsk,dv,dt,dz=kernels.backward(sq,sk,v,dout.contiguous().reshape(rows,n,h,-1),
                                         iq,ik,ids,pi,hard,segments.reshape(rows,n),state)
        dq,deq=kernels.router_backward(q,raw[5],ids,dz,hard,ctx.beta)
        for index,extra in ((0,dq),(2,dsq),(3,dsk),(4,dv),(5,deq),(7,dt)):
            grads[index]=(grads[index].float()+extra.reshape_as(grads[index])).to(raw[index].dtype)
        return (*grads,None,None,None,None,None,None)


def vocabulary(q,k,sq,sk,v,eq,ek,tau,*,direction,hard,segments,boundaries=None,layout=None,beta=1.):
    # Segment IDs come from the same validated, 256-aligned document layout
    # as boundaries/layout. They are graph-staged metadata, not arbitrary masks.
    if os.environ.get('DISM_HARD_PLANNER', '0') != '0':
        raise ValueError('top-four currently requires DISM_HARD_PLANNER=0')
    if not q.is_cuda or q.ndim!=4 or q.shape!=k.shape or not q.shape[0] or not q.shape[1] or q.shape[1]%256:
        raise ValueError('top-four requires aligned equal Q/K BNHD geometry')
    b,n,h,d=q.shape
    if d not in (32,64) or sq.shape!=sk.shape or sq.shape[:3]!=(b,n,h) or sq.shape[-1] not in (16,32):
        raise ValueError('top-four supports D32/64 and R16/32')
    if v.shape[:3]!=(b,n,h) or v.shape[-1] not in (32,64):
        raise ValueError('top-four supports DV32/64')
    if eq.ndim!=3 or eq.shape!=ek.shape or eq.shape[0]!=h or eq.shape[2]!=d or eq.shape[1]<4:
        raise ValueError('top-four requires per-head HCD codebooks with at least four symbols')
    if hard.ndim!=2 or not hard.shape[0] or hard.shape[1]!=h or (b*n)%hard.shape[0] or (b*n//hard.shape[0])%256:
        raise ValueError('top-four requires aligned equal-length training rows')
    if segments.shape!=(b,n) or segments.device!=q.device or segments.dtype not in (torch.int32,torch.int64):
        raise ValueError('segments must describe the packed token/document mapping')
    if any(x.dtype!=torch.bfloat16 or not x.is_contiguous() for x in (q,k,sq,sk,v)):
        raise ValueError('top-four vectors must be contiguous BF16')
    if (any(x.device!=q.device or not x.is_contiguous() for x in (q,k,sq,sk,v,eq,ek,tau,hard,segments,direction))
            or eq.dtype not in (torch.float32,torch.bfloat16) or ek.dtype!=eq.dtype
            or tau.shape!=(h,) or tau.dtype!=torch.float32 or hard.dtype!=torch.bool
            or direction.dtype!=torch.bool or direction.shape!=(b,h)):
        raise ValueError('top-four requires contiguous operands on one CUDA device, FP32 H tau and bool flags')
    if not math.isfinite(float(beta)) or beta<=0:
        raise ValueError('top-four beta must be finite and positive')
    return _Vocabulary.apply(q,k,sq,sk,v,eq,ek,tau,direction,hard,segments,boundaries,layout,float(beta))
