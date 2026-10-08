"""Inference-only Torch reference with linear-space KV/diagonal-state cache.

No CUDA extension calls, finite sentinel, approximate softplus or BF16 MMA.
One step costs O(history * (D+R+DV)); this is not the hard-label SAM algorithm.
"""
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .dism_v3_ref import _float


@dataclass(frozen=True)
class DismDecodeCache:
    k_vec: Tensor             # [B,T,H,D], selected/interpolated score operand
    sk_vec: Tensor            # [B,T,H,R], final caller-parameterized readout feature
    v: Tensor                # [B,T,H,DV]
    k_lse: Tensor            # [B,T,H], raw natural-log LSE (tau NOT absorbed)
    idx_k: Tensor            # [B,H,T]
    last_w: Tensor           # [B,H,T], last causal row in natural-log units
    direction: Tensor        # [B,H], fixed over cache lifetime
    rtau: Tensor             # [H], fixed over cache lifetime

    @property
    def length(self):
        return self.k_vec.shape[1]


@torch.no_grad()
def dism_decode_ref(q_vec, k_vec, sq_vec, sk_vec, q_lse, k_lse,
                    idx_q, idx_k, direction, hard, v, rtau, *, cache=None):
    """Same operands/order as dism_ref, but only NEW tokens are supplied.

    Returns (output [B,N,H,DV], new_cache); accepts N>=1 for prefill/chunks.
    Includes each current token's own key. Old cache is not mutated and inputs
    are not aliased by the returned cache. Half inputs accumulate/output FP32;
    FP64 stays FP64. No backward graph is retained. Reset with cache=None for a
    new document; batch entries must have equal lengths (no packed-varlen API).
    Direction/tau and the upstream model/embedding parameters must stay fixed.
    The caller supplies row hard decisions; decoding does not resample history.
    """
    if q_vec.ndim != 4 or q_vec.shape != k_vec.shape:
        raise ValueError('q_vec/k_vec must be equal [B,N,H,D]')
    b,n,h,d = q_vec.shape
    if min(b,n,h,d) <= 0:
        raise ValueError('empty dimensions are not supported')
    if (sq_vec.ndim != 4 or sq_vec.shape != sk_vec.shape
            or sq_vec.shape[:3] != (b,n,h) or sq_vec.shape[-1] <= 0):
        raise ValueError('sq_vec/sk_vec must be equal [B,N,H,R]')
    if v.ndim != 4 or v.shape[:3] != (b,n,h) or v.shape[-1] <= 0:
        raise ValueError('v must be [B,N,H,DV]')
    if q_lse.shape != (b,n,h) or k_lse.shape != (b,n,h) or rtau.shape != (h,):
        raise ValueError('raw LSE must be [B,N,H], rtau [H]')
    if direction.shape != (b,h) or direction.dtype != torch.bool:
        raise ValueError('direction must be bool [B,H]')
    if hard.shape != (b,h,n) or hard.dtype != torch.bool:
        raise ValueError('hard must be bool [B,H,N]')
    for label in (idx_q,idx_k):
        if label.shape != (b,h,n) or label.dtype not in (torch.int32,torch.int64):
            raise ValueError('labels must be int32/int64 [B,H,N]')
    values=(q_vec,k_vec,sq_vec,sk_vec,q_lse,k_lse,v,rtau)
    if any(x.device != q_vec.device for x in (*values,idx_q,idx_k,direction,hard)):
        raise ValueError('all operands must be on the same device')
    if any(not x.is_floating_point() for x in values):
        raise ValueError('vectors, LSE and tau must be floating point')
    dtype=torch.float64 if any(x.dtype==torch.float64 for x in values) else torch.float32
    q,k,sq,sk,lq,lk,value,tau=(x.to(dtype) for x in values)

    if cache is None:
        start=0
        previous=q.new_empty((b,h,0))
        keys,soft_keys,values,lses,labels=k.clone(),sk.clone(),value.clone(),lk.clone(),idx_k.clone()
    else:
        if not isinstance(cache,DismDecodeCache):
            raise TypeError('cache must be DismDecodeCache or None')
        if ((cache.k_vec.shape[0],cache.k_vec.shape[2]) != (b,h) or cache.k_vec.shape[-1] != d
                or cache.sk_vec.shape[-1] != sq.shape[-1] or cache.v.shape[-1] != v.shape[-1]
                or cache.k_vec.device != q.device or cache.k_vec.dtype != dtype):
            raise ValueError('cache batch/head/channels/device/accumulation dtype changed')
        if not torch.equal(direction,cache.direction) or not torch.equal(tau,cache.rtau):
            raise ValueError('direction and rtau must remain fixed while using a cache')
        start=cache.length
        previous=cache.last_w
        keys=torch.cat((cache.k_vec,k),dim=1)
        soft_keys=torch.cat((cache.sk_vec,sk),dim=1)
        values=torch.cat((cache.v,value),dim=1)
        lses=torch.cat((cache.k_lse,lk),dim=1)
        labels=torch.cat((cache.idx_k,idx_k),dim=2)

    outputs=[]
    for i in range(n):
        end=start+i+1
        dot=torch.einsum('bhd,bthd->bht',q[:,i],keys[:,:end])
        bias=torch.where(direction[...,None],lq[:,i,:,None],lses[:,:end].transpose(1,2))
        hard_score=torch.zeros_like(dot).masked_fill(idx_q[:,:,i,None]!=labels[:,:,:end],-torch.inf)
        logm=torch.where(hard[:,:,i,None],hard_score,dot-bias)+tau[None,:,None]
        # Old W has length end-1. Shift it right by one, including its last
        # diagonal element; the new key's predecessor is the old self-key.
        previous=logm+F.softplus(F.pad(previous,(1,0),value=-torch.inf))
        maximum=previous.amax(-1,keepdim=True).clamp_min(0.)
        weight=(previous-maximum).exp()
        denominator=weight.sum(-1,keepdim=True)+(-maximum).exp()
        readout=torch.einsum('bhr,bthr->bht',sq[:,i],soft_keys[:,:end])
        numerator=torch.einsum('bht,bthd->bhd',weight*readout,values[:,:end])
        outputs.append(numerator/denominator)

    updated=DismDecodeCache(keys,soft_keys,values,lses,labels,previous,
                            direction.clone(),tau.clone())
    return torch.stack(outputs,dim=1),updated


@torch.no_grad()
def dism_wrapper_decode(q,k,sq_vec,sk_vec,q_weight,k_weight,v,rtau,*,
                        cache=None,direction=None,hard=None,hard_prob=.5,generator=None):
    """Cached counterpart of dism_wrapper, including Torch vocabulary interpolation.

    Tables are shared [V,D] or per-head [H,V,D]. Activations belong to
    the caller. A missing direction is sampled once, then reused from cache.
    RNG is the Torch oracle's RNG, not the production CUDA Philox mapping;
    supply explicit direction/hard for chunk-invariant replay. Keep tables and
    model weights unchanged across calls (cache does not fingerprint weights).
    """
    if q.ndim!=4 or q.shape!=k.shape:
        raise ValueError('q/k must be equal [B,N,H,D]')
    b,n,h,d=q.shape
    if (q_weight.ndim not in (2,3) or q_weight.shape!=k_weight.shape
            or q_weight.shape[-1]!=d or (q_weight.ndim==3 and q_weight.shape[0]!=h)):
        raise ValueError('tables must be equal shared [V,D] or per-head [H,V,D]')
    if not 0<=hard_prob<=1:
        raise ValueError('hard_prob must be in [0,1]')
    q,k,eq,ek=map(_float,(q,k,q_weight,k_weight))
    if eq.ndim==2:
        eq,ek=eq.expand(h,-1,-1),ek.expand(h,-1,-1)
    qs=torch.einsum('bnhd,hvd->bnhv',q,eq)
    ks=torch.einsum('bnhd,hvd->bnhv',k,ek)
    lq,lk=qs.logsumexp(-1),ks.logsumexp(-1)
    iq,ik=qs.argmax(-1).transpose(1,2),ks.argmax(-1).transpose(1,2)
    qfk=torch.einsum('bnhv,hvd->bnhd',ks.softmax(-1),eq)
    kfq=torch.einsum('bnhv,hvd->bnhd',qs.softmax(-1),ek)
    if direction is None:
        direction=(cache.direction if cache is not None else
                   torch.randint(0,2,(),device=q.device,generator=generator).bool().expand(b,h))
    if hard is None:
        hard=(torch.full((b,h,n),bool(hard_prob),device=q.device,dtype=torch.bool)
              if hard_prob in (0.,1.) else
              torch.rand((b,h,n),device=q.device,generator=generator)<hard_prob)
    choose=direction[:,None,:,None]
    return dism_decode_ref(torch.where(choose,q,kfq),torch.where(choose,qfk,k),
                           sq_vec,sk_vec,lq,lk,iq,ik,direction,hard,v,rtau,cache=cache)
