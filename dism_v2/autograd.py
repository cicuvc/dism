"""Fixed-length sm120 Dism, using the existing Triton embedding kernels."""
import math
import torch
from .core import forward,RowRNGState,ScanBoundaries
from .backward import delta,value_gradient,operand_gradient
from .emb_kernel import emb_fwd_wrapper,emb_bwd_wrapper


class _VocDism(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,tau,qvoc,kvoc,scale,direction,probability,generator,replay,embedding_backend):
        # Actual order: q_from_k, k_from_q, k_lse, q_lse, k_top, q_top, k_idx, q_idx.
        if embedding_backend=="cuda":
            from .embedding import forward as embedding_forward
            raw=embedding_forward(q,k,qvoc,kvoc,scale,warp_specialized=True)
        else:
            raw=emb_fwd_wrapper(q,k,qvoc,kvoc,scale)
        oq,ok,lk,lq,_,_,ki,qi=raw
        qi=qi.long();ki=ki.long()
        if direction=="q_from_k": a,b,lse=q,oq,lq
        elif direction=="k_from_q": a,b,lse=ok,k,lk
        else: a,b,lse=(q,ok),(oq,k),(lq,lk)
        out,norm,edges,state=forward(a,b,v,lse,tau,qi,ki,sm_scale=scale,direction=direction,
            hard_prob=probability,generator=generator,rng_state=replay,
            save_boundaries=True,return_rng_state=True)
        ctx.save_for_backward(q,k,v,tau,qvoc,kvoc,oq,ok,lk,lq,qi,ki,out,norm,
            edges.vertical,edges.horizontal)
        ctx.state=state
        ctx.scale=scale
        ctx.set_materialize_grads(False)
        return out,state

    @staticmethod
    def backward(ctx,dout,_state_grad):
        if torch.is_grad_enabled():
            raise NotImplementedError("higher-order backward is not implemented")
        if dout is None: return (None,)*12
        q,k,v,tau,qvoc,kvoc,oq,ok,lk,lq,qi,ki,out,norm,vertical,horizontal=ctx.saved_tensors
        state=ctx.state
        if state.direction=="q_from_k": a,b,lse=q,oq,lq
        else: a,b,lse=ok,k,lk
        dout=dout.to(v.dtype).contiguous()
        dd=delta(dout,out)
        edges=ScanBoundaries(vertical,horizontal)
        kw=dict(sm_scale=ctx.scale,rng_state=state)
        dv,_,g32=value_gradient(a,b,dout,lse,tau,qi,ki,norm,edges,**kw,
            v=v,delta=dd,warp_specialized=True)
        da,db,dlse,dtau=operand_gradient(a,b,v,dout,lse,tau,qi,ki,norm,dd,edges,g32,**kw)
        zero=torch.zeros_like(a,dtype=torch.float32)
        # emb wrapper dlq belongs to out_q/k_lse; dlk to out_k/q_lse.
        if state.direction=="q_from_k":
            dq,dk,dqvoc,dkvoc=emb_bwd_wrapper(q,k,qvoc,kvoc,oq,ok,lk,lq,
                db,zero,None,dlse,ctx.scale)
            dq=dq+da
        else:
            dq,dk,dqvoc,dkvoc=emb_bwd_wrapper(q,k,qvoc,kvoc,oq,ok,lk,lq,
                zero,da,dlse,None,ctx.scale)
            dk=dk+db
        # Merge direct/embedding FP32 terms BEFORE the final input-dtype cast.
        grads=(dq,dk,dv,dtau,dqvoc,dkvoc)
        inputs=(q,k,v,tau,qvoc,kvoc)
        return tuple(g.to(x.dtype) for g,x in zip(grads,inputs))+(None,)*6


def voc_dism(q,k,v,rtau,q_voc,k_voc,*,sm_scale=1.0,direction="random",hard_prob=0.0,
             generator=None,rng_state=None,return_rng_state=False,embedding_backend="triton"):
    """BF16 output and six-input first-order autograd; optional replay state.

    q/k [B,H,N,D], v [B,H,N,DV], vocab [H,V,D]: contiguous BF16.
    rtau [H]: FP32, natural logarithm. Uses Triton embedding + CUDA WS core.
    Fixed-length sm120 only; no varlen, CUDA Graphs, deterministic backward,
    probability broadcasting or higher-order gradients. Known precision limits
    of BF16 outputs/delta are retained; no global attention or random masks.
    embedding_backend="cuda" opts into the fused CUDA embedding forward;
    embedding backward remains Triton for both backends.
    """
    if embedding_backend not in ("triton","cuda"):
        raise ValueError("embedding_backend must be triton or cuda")
    if q.ndim!=4 or not q.is_cuda:
        raise ValueError("q must be CUDA [B,H,N,D]")
    batch,heads,n,d=q.shape
    tensors=(q,k,v,rtau,q_voc,k_voc)
    if any(x.device!=q.device or not x.is_contiguous() for x in tensors):
        raise ValueError("inputs must be contiguous on the same CUDA device")
    if min(batch,heads,n)<=0 or d not in (32,64,128):
        raise ValueError("unsupported q dimensions")
    if k.shape!=q.shape or v.ndim!=4 or v.shape[:3]!=q.shape[:3] or v.shape[-1] not in (32,64,128):
        raise ValueError("q/k/v shape mismatch or unsupported DV")
    if q_voc.ndim!=3 or q_voc.shape!=k_voc.shape or q_voc.shape[0]!=heads or q_voc.shape[2]!=d or q_voc.shape[1]<=0:
        raise ValueError("vocab must be matching [H,V,D] with V>0")
    if any(x.dtype!=torch.bfloat16 for x in (q,k,v,q_voc,k_voc)):
        raise TypeError("q/k/v/vocab must be BF16")
    if rtau.shape!=(heads,) or rtau.dtype!=torch.float32:
        raise TypeError("rtau must be FP32 [H]")
    if direction not in ("q_from_k","k_from_q","random"):
        raise ValueError("unknown direction")
    if not isinstance(sm_scale,(int,float)) or not math.isfinite(sm_scale):
        raise ValueError("sm_scale must be finite scalar")
    if not isinstance(hard_prob,(int,float)) or not math.isfinite(hard_prob) or not 0<=hard_prob<=1:
        raise ValueError("hard_prob must be scalar in [0,1]")
    if generator is not None and rng_state is not None:
        raise ValueError("generator and rng_state are mutually exclusive")
    if rng_state is not None and not isinstance(rng_state,RowRNGState):
        raise TypeError("rng_state must be RowRNGState")
    with torch.cuda.device(q.device):
        if torch.cuda.get_device_capability(q.device)!=(12,0):
            raise RuntimeError("initial autograd supports sm120 only")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CUDA Graph capture is not supported")
        out,state=_VocDism.apply(q,k,v,rtau,q_voc,k_voc,float(sm_scale),direction,
            float(hard_prob),generator,rng_state,embedding_backend)
    return (out,state) if return_rng_state else out
