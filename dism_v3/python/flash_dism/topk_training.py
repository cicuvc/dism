"""Opt-in top-k hard autograd; native v3 soft and sampling stay unchanged."""
import math
import os
from functools import wraps
import torch
from .kernels import topk_hard as kernels


class _NoPlanner:
    def submit(self,*args,**kwargs):return None


class _HardForward:
    def __init__(self,q,eq,beta,segments,rows,sampled,counter,seed,topk,graph_state,plan_eager):
        self.q,self.eq,self.beta,self.segments,self.rows=q,eq,beta,segments,rows
        self.sampled,self.counter,self.seed=sampled,counter,seed
        self.topk,self.graph_state,self.ticket=topk,graph_state,None
        self.plan_eager=plan_eager

    def __call__(self,sq,sk,v,iq,ik,hard,tau,**kwargs):
        rows=self.rows;n=self.q.shape[0]*self.q.shape[1]//rows;h=self.q.shape[2]
        labels=lambda x:x.transpose(1,2).reshape(rows,n,h).transpose(1,2).contiguous()
        iq,ik=labels(iq),labels(ik)
        q=self.q.reshape(rows,n,h,-1)
        ids,pi=kernels.select(q,self.eq,iq,hard,self.beta,self.topk)
        if self.graph_state is not None:
            self.graph_state.slot.snapshot_topk(iq,ik,hard,tau,ids)
        elif self.plan_eager and not self.sampled and os.environ.get("DISM_HARD_PLANNER","0")!="0":
            from .topk_planner import default_topk_planner
            self.ticket=default_topk_planner(q.device).submit(iq,ik,hard,tau,ids,kwargs.get("layout"))
        backend=kernels;extra=()
        if self.sampled:
            from .kernels import sampled_hard as backend
            extra=(backend.sample(pi,self.counter,self.seed),)
        output,state=backend.forward(sq.reshape(rows,n,h,-1),sk.reshape(rows,n,h,-1),v.reshape(rows,n,h,-1),
            iq,ik,ids,pi,tau,hard,self.segments.reshape(rows,n),*extra,
            **({'vocab_size':self.eq.shape[1]} if self.sampled else {}))
        self.saved=(iq,ik,ids,pi,*state)
        if self.sampled:self.saved+=extra
        return output.reshape_as(v),output.new_empty((0,))


def _guard_binding(method):
    @wraps(method)
    def guarded(ctx,*args):
        try:return method(ctx,*args)
        except Exception as exc:
            state=getattr(ctx,'graph_state',None)
            if state is not None:state.failed=exc
            raise
    return guarded


class _Vocabulary(torch.autograd.Function):
    @staticmethod
    @_guard_binding
    def forward(ctx,q,k,sq,sk,v,eq,ek,tau,direction,hard,segments,boundaries,layout,beta,sampled,counter,seed,topk,graph_binding,plan_eager):
        from .compiler import _voc_forward_impl
        rows=hard.shape[0]
        state=None
        if graph_binding is not None:
            from .graph_binding import _state
            state=_state(graph_binding.handle,'vocabulary')
            if getattr(state.slot,'topk',None)!=topk:raise ValueError('topk graph slot mismatch')
            ctx.graph_state=state
            state.begin('vocabulary',q,sq,v,hard)
            direction=state.direction
        ctx.graph_state=state
        engine=_HardForward(q,eq,beta,segments,rows,sampled,counter,seed,topk,state,plan_eager)
        saved=_voc_forward_impl(q,k,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout,True,
                               direction_true=True,planner=_NoPlanner(),hard_forward_override=engine)
        ctx.beta=beta;ctx.rows=rows;ctx.sampled=sampled;ctx.ticket=engine.ticket
        ctx.has_boundaries=boundaries is not None;ctx.has_layout=layout is not None
        ctx.save_for_backward(q,k,sq,sk,v,eq,ek,tau,direction,hard,segments,*saved,*engine.saved,
            *([] if boundaries is None else [boundaries]),*([] if layout is None else [layout]))
        return saved[0]

    @staticmethod
    @torch.autograd.function.once_differentiable
    @_guard_binding
    def backward(ctx,dout):
        from .compiler import _voc_backward_impl
        if ctx.graph_state is not None:
            binding=ctx.graph_state
            binding.check()
            stream,capture_id=binding.capture()
            if not binding.pending or stream!=binding.stream or capture_id!=binding.capture_id:
                raise RuntimeError('topk bound backward must follow forward in the same stream/capture')
        s=ctx.saved_tensors
        raw=list(s[:8]);direction,hard,segments=s[8:11]
        native=list(s[11:23]);iq,ik,ids,pi=s[23:27];state=s[27:32]
        offset=32+int(ctx.sampled)
        boundaries=s[offset] if ctx.has_boundaries else None
        layout=s[offset+int(ctx.has_boundaries)] if ctx.has_layout else None
        grads=_voc_backward_impl([*raw,ctx.beta,direction,hard],native,boundaries,dout,layout,direction_true=True)
        rows=ctx.rows;n=raw[0].shape[0]*raw[0].shape[1]//rows;h=raw[0].shape[2]
        q,sq,sk,v=[raw[i].reshape(rows,n,h,-1) for i in (0,2,3,4)]
        backend=kernels;extra=()
        if ctx.sampled:
            from .kernels import sampled_hard as backend
            extra=(s[32],)
        do=dout.contiguous().reshape(rows,n,h,-1)
        fallback=lambda choice: backend.backward(sq,sk,v,do,iq,ik,ids,pi,hard,segments.reshape(rows,n),state,*extra,
                                                **({} if ctx.sampled else {'choice':choice}))
        if ctx.graph_state is not None:
            dsq,dsk,dv,dt,dz=binding.slot.backward_topk(sq,sk,v,do,pi,fallback)
        elif ctx.ticket is not None:
            dsq,dsk,dv,dt,dz=ctx.ticket.backward(sq,sk,v,do,pi,fallback)
        else:dsq,dsk,dv,dt,dz=fallback(None)
        dq,deq=kernels.router_backward(q,raw[5],ids,dz,hard,ctx.beta)
        for index,extra in ((0,dq),(2,dsq),(3,dsk),(4,dv),(5,deq),(7,dt)):
            grads[index]=(grads[index].float()+extra.reshape_as(grads[index])).to(raw[index].dtype)
        if ctx.graph_state is not None:
            ctx.graph_state.slot.finish()
            ctx.graph_state.pending=False
        return (*grads,None,None,None,None,None,None,None,None,None,None,None,None)


def vocabulary(q,k,sq,sk,v,eq,ek,tau,*,direction,hard,segments,boundaries=None,layout=None,beta=1.,
               sampled=False,counter=None,seed=0,topk=4,graph_binding=None):
    # Segment IDs come from the same validated, 256-aligned document layout
    # as boundaries/layout. They are graph-staged metadata, not arbitrary masks.
    if topk not in (1,2,4) or type(topk) is not int:raise ValueError('topk must be 1, 2 or 4')
    if sampled and (topk!=4 or graph_binding is not None or os.environ.get('DISM_HARD_PLANNER','0')!='0'):
        raise ValueError('sampled-CV requires K4 and planner disabled')
    if (graph_binding is not None or os.environ.get('DISM_HARD_PLANNER','0')!='0') and (sq.shape[-1]!=32 or v.shape[-1]!=64):
        raise ValueError('topk planner requires R32/DV64')
    if not q.is_cuda or q.ndim!=4 or q.shape!=k.shape or not q.shape[0] or not q.shape[1] or q.shape[1]%256:
        raise ValueError('top-k requires aligned equal Q/K BNHD geometry')
    b,n,h,d=q.shape
    if d not in (32,64) or sq.shape!=sk.shape or sq.shape[:3]!=(b,n,h) or sq.shape[-1] not in (16,32):
        raise ValueError('top-k supports D32/64 and R16/32')
    if v.shape[:3]!=(b,n,h) or v.shape[-1] not in (32,64):
        raise ValueError('top-k supports DV32/64')
    if eq.ndim!=3 or eq.shape!=ek.shape or eq.shape[0]!=h or eq.shape[2]!=d or eq.shape[1]<topk:
        raise ValueError('top-k requires per-head HCD codebooks with at least K symbols')
    if hard.ndim!=2 or not hard.shape[0] or hard.shape[1]!=h or (b*n)%hard.shape[0] or (b*n//hard.shape[0])%256:
        raise ValueError('top-k requires aligned equal-length training rows')
    if segments.shape!=(b,n) or segments.device!=q.device or segments.dtype not in (torch.int32,torch.int64):
        raise ValueError('segments must describe the packed token/document mapping')
    if any(x.dtype!=torch.bfloat16 or not x.is_contiguous() for x in (q,k,sq,sk,v)):
        raise ValueError('top-k vectors must be contiguous BF16')
    if (any(x.device!=q.device or not x.is_contiguous() for x in (q,k,sq,sk,v,eq,ek,tau,hard,segments,direction))
            or eq.dtype not in (torch.float32,torch.bfloat16) or ek.dtype!=eq.dtype
            or tau.shape!=(h,) or tau.dtype!=torch.float32 or hard.dtype!=torch.bool
            or direction.dtype!=torch.bool or direction.shape!=(b,h)):
        raise ValueError('top-k requires contiguous operands on one CUDA device, FP32 H tau and bool flags')
    if not math.isfinite(float(beta)) or beta<=0:
        raise ValueError('top-k beta must be finite and positive')
    if type(sampled) is not bool:
        raise ValueError('sampled must be bool')
    if sampled and (type(seed) is not int or not 0<=seed<2**63):
        raise ValueError('sampled seed must be a nonnegative int64')
    if sampled and (not isinstance(counter,torch.Tensor) or counter.shape!=() or
                    counter.dtype!=torch.int64 or counter.device!=q.device):
        raise ValueError('sampled reads require an explicit CUDA scalar int64 counter')
    if graph_binding is not None:
        graph_binding._training((q,k,sq,sk,v,eq,ek,tau))
        from .graph_binding import _state
        bound=_state(graph_binding.handle,'vocabulary')
        if boundaries is not None or ((bound.layout is None)!=(layout is None)) or (layout is not None and layout.data_ptr()!=bound.layout.table.data_ptr()):
            raise ValueError('topk binding must use its owned graph layout')
    return _Vocabulary.apply(q,k,sq,sk,v,eq,ek,tau,direction,hard,segments,boundaries,layout,float(beta),sampled,counter,seed,topk,graph_binding,
                             torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,sq,sk,v,eq,ek,tau)))
