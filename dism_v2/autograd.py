"""Fixed-length sm120 Dism with selectable Triton/CUDA embedding kernels."""
import math
import os
import torch
from .core import forward,RowRNGState,ScanBoundaries
from .backward import delta,value_gradient,operand_gradient
from .emb_kernel import emb_fwd_wrapper,emb_bwd_wrapper


class _VocDism(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,tau,qvoc,kvoc,scale,direction,probability,generator,replay,embedding_backend,embedding_backward_backend):
        # Actual order: q_from_k, k_from_q, k_lse, q_lse, k_top, q_top, k_idx, q_idx.
        bits=None
        packed=(embedding_backend=="cuda" and 0<probability<1 and os.environ.get("DISM_ROW_BITSET","0")=="1")
        if packed and replay is None:
            from .core import _extension
            seed,offset,column=_extension().reserve_rows(q,direction=="random",direction=="k_from_q",probability,generator)
            replay=RowRNGState(seed,offset,tuple(q.shape[:3]),"k_from_q" if column else "q_from_k",probability)
            generator=None
        if embedding_backend=="cuda":
            from .embedding import forward as embedding_forward
            raw=embedding_forward(q,k,qvoc,kvoc,scale,warp_specialized=True,
                row_rng=(replay.seed,replay.offset,probability) if packed else None)
            if packed: raw,bits=raw[:8],raw[8]
        else:
            raw=emb_fwd_wrapper(q,k,qvoc,kvoc,scale)
        oq,ok,lk,lq,_,_,ki,qi=raw
        # CUDA/Triton argmax labels fit int32; do not widen before the core.
        qi=qi.int();ki=ki.int()
        if direction=="q_from_k": a,b,lse=q,oq,lq
        elif direction=="k_from_q": a,b,lse=ok,k,lk
        else: a,b,lse=(q,ok),(oq,k),(lq,lk)
        out,norm,edges,state=forward(a,b,v,lse,tau,qi,ki,sm_scale=scale,direction=direction,
            hard_prob=probability,generator=generator,rng_state=replay,
            save_boundaries=True,return_rng_state=True,hard_bits=bits)
        ctx.save_for_backward(q,k,v,tau,qvoc,kvoc,oq,ok,lk,lq,qi,ki,out,norm,
            edges.vertical,edges.horizontal,bits)
        ctx.state=state
        ctx.scale=scale
        ctx.embedding_backward_backend=embedding_backward_backend
        ctx.set_materialize_grads(False)
        return out,state

    @staticmethod
    def backward(ctx,dout,_state_grad):
        if torch.is_grad_enabled():
            raise NotImplementedError("higher-order backward is not implemented")
        if dout is None: return (None,)*13
        q,k,v,tau,qvoc,kvoc,oq,ok,lk,lq,qi,ki,out,norm,vertical,horizontal,bits=ctx.saved_tensors
        state=ctx.state
        if state.direction=="q_from_k": a,b,lse=q,oq,lq
        else: a,b,lse=ok,k,lk
        dout=dout.to(v.dtype).contiguous()
        dd=delta(dout,out)
        edges=ScanBoundaries(vertical,horizontal)
        kw=dict(sm_scale=ctx.scale,rng_state=state,hard_bits=bits)
        dv,_,g32=value_gradient(a,b,dout,lse,tau,qi,ki,norm,edges,**kw,
            v=v,delta=dd,warp_specialized=True)
        da,db,dlse,dtau=operand_gradient(a,b,v,dout,lse,tau,qi,ki,norm,dd,edges,g32,**kw)
        if ctx.embedding_backward_backend!="triton":
            from .embedding import backward as embedding_backward
            symmetric=ctx.embedding_backward_backend=="cuda_symmetric"
            full=state.direction=="q_from_k"
            dq,dk,dqvoc,dkvoc=embedding_backward(q,k,qvoc,kvoc,oq if full else ok,lk,lq,
                db if full else da,dlse,direction=state.direction,sm_scale=ctx.scale,
                warp_specialized=True,vocab_symmetric=symmetric,
                vocab_token_step=None if symmetric else (64 if q.shape[-1]==32 else 32))
        else:
            zero=torch.zeros_like(a,dtype=torch.float32)
            dq,dk,dqvoc,dkvoc=emb_bwd_wrapper(q,k,qvoc,kvoc,oq,ok,lk,lq,
                db if state.direction=="q_from_k" else zero,
                zero if state.direction=="q_from_k" else da,
                None if state.direction=="q_from_k" else dlse,
                dlse if state.direction=="q_from_k" else None,ctx.scale)
        # emb wrapper dlq belongs to out_q/k_lse; dlk to out_k/q_lse.
        if state.direction=="q_from_k":
            dq=dq+da
        else:
            dk=dk+db
        # Merge direct/embedding FP32 terms BEFORE the final input-dtype cast.
        grads=(dq,dk,dv,dtau,dqvoc,dkvoc)
        inputs=(q,k,v,tau,qvoc,kvoc)
        return tuple(g.to(x.dtype) for g,x in zip(grads,inputs))+(None,)*7


def voc_dism(q,k,v,rtau,q_voc,k_voc,*,sm_scale=1.0,direction="random",hard_prob=0.0,
             generator=None,rng_state=None,return_rng_state=False,embedding_backend="triton",
             embedding_backward_backend="triton"):
    """BF16 output and six-input first-order autograd; optional replay state.

    q/k [B,H,N,D], v [B,H,N,DV], vocab [H,V,D]: contiguous BF16.
    rtau [H]: FP32, natural logarithm. Uses Triton embedding + CUDA WS core.
    Fixed-length sm120 only; no varlen, CUDA Graphs, deterministic backward,
    probability broadcasting or higher-order gradients. Known precision limits
    of BF16 outputs/delta are retained; no global attention matrix.
    DISM_ROW_BITSET=1 caches mixed row decisions in a packed bitset generated by
    CUDA embedding; default/Triton paths replay Philox without storing a mask.
    embedding_backend="cuda" opts into the fused CUDA embedding forward;
    embedding_backward_backend="cuda" selects paired CUDA WS backward;
    "cuda_symmetric" selects symmetric WS. Default backward remains Triton.
    """
    if embedding_backend not in ("triton","cuda"):
        raise ValueError("embedding_backend must be triton or cuda")
    if embedding_backward_backend not in ("triton","cuda","cuda_symmetric"):
        raise ValueError("embedding_backward_backend must be triton, cuda or cuda_symmetric")
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
            float(hard_prob),generator,rng_state,embedding_backend,embedding_backward_backend)
    return (out,state) if return_rng_state else out
