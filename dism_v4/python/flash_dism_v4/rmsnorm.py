"""Opt-in denominator-free Jensen readout with per-head RMSNorm.

Recomputes pair tiles in backward; reuses the unchanged hard-history kernel.
No SQ/SK or slow numerical fallback.
"""
import math
import torch
from flash_dism_v4.history import history_blocks


def _read(q, e, lse, v, loga, epsilon):
    score = loga + torch.bmm(q, e.transpose(1, 2)) - lse
    scale = score.amax(-1, keepdim=True).clamp_min(0)
    m = (score-scale).exp()
    u = torch.bmm(m, v)
    inv = (u.square().mean(-1, keepdim=True) + epsilon*(-2*scale).exp()).rsqrt()
    return m, u, inv


class _Read(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, e, lse, v, tau, iq, ik, seg, reset, block, epsilon):
        out = torch.empty_like(v)
        for start, loga, _ in history_blocks(iq, ik, tau, seg, reset, block):
            end = start+loga.shape[1]
            _, u, inv = _read(q[:, start:end], e, lse[:, start:end], v, loga, epsilon)
            out[:, start:end] = u*inv
        ctx.save_for_backward(q,e,lse,v,tau,iq,ik,seg,reset)
        ctx.block, ctx.epsilon = block, epsilon
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        q,e,lse,v,tau,iq,ik,seg,reset = ctx.saved_tensors
        dq,de,dl,dv,dt = [torch.zeros_like(x) for x in (q,e,lse,v,tau)]
        for start, loga, dloga in history_blocks(iq,ik,tau,seg,reset,ctx.block):
            end = start+loga.shape[1]
            qb = q[:, start:end]
            m,u,inv = _read(qb,e,lse[:,start:end],v,loga,ctx.epsilon)
            g = grad[:,start:end].contiguous()
            du = inv*(g-u*(g*u).mean(-1,keepdim=True)*inv.square())
            ds = torch.bmm(du,v.transpose(1,2))*m
            dq[:,start:end] = torch.bmm(ds,e)
            de += torch.bmm(ds.transpose(1,2),qb)
            dl[:,start:end] = -ds.sum(-1,keepdim=True)
            dv += torch.bmm(m.transpose(1,2),du)
            dt += (ds*dloga).sum((1,2))
        return dq,de,dl,dv,dt,None,None,None,None,None,None


@torch.compiler.disable
def rms_readout(q,k,q_weight,k_weight,v,tau,*,hard,segment_ids=None,
                block_size=64,precision='fp32',soft_readout=None,epsilon=1e-6):
    """No-denominator hard/Jensen readout, then per-head RMSNorm, gain=1.

    Hard flags select K compatibility; history always uses argmax Q/K.
    Epsilon is defined in the unscaled numerator coordinates.
    """
    if precision != 'fp32':
        raise ValueError("RMSNorm readout currently requires fp32")
    if block_size not in (32,64,128):
        raise ValueError("block_size must be 32, 64 or 128")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    if q.ndim != 4 or q.shape != k.shape or min(q.shape) <= 0:
        raise ValueError("q/k must have equal nonempty BNHD shapes")
    b,n,h,d = q.shape
    if q_weight.ndim != 3 or q_weight.shape != k_weight.shape or q_weight.shape[0] != h or q_weight.shape[2] != d or q_weight.shape[1] <= 0:
        raise ValueError("codebooks must have equal HCD shapes")
    if v.ndim != 4 or v.shape[:3] != (b,n,h) or v.shape[-1] <= 0 or tau.shape != (h,):
        raise ValueError("v must be BNHDv and tau H")
    tensors=(q,k,q_weight,k_weight,v,tau)
    if not q.is_cuda or any(t.device != q.device for t in tensors):
        raise ValueError("inputs must share one CUDA device")
    if any(t.dtype not in (torch.float16,torch.bfloat16,torch.float32) for t in tensors):
        raise ValueError("readout supports fp16/bf16/fp32 inputs")
    if hard.shape != (b,h) or hard.dtype != torch.bool or hard.device != q.device:
        raise ValueError("hard must be bool BH on the input device")
    if segment_ids is not None:
        if segment_ids.shape != (b,n) or segment_ids.device != q.device or segment_ids.dtype not in (torch.int32,torch.int64):
            raise ValueError("segment_ids must be integer BN on the input device")
        if bool((segment_ids[:,1:] < segment_ids[:,:-1]).any()):
            raise ValueError("segment_ids must be nondecreasing")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("v4 prototype does not support CUDA graph capture")
    with torch.autocast('cuda',enabled=False):
        b,n,h,d = q.shape
        def flatten(x):
            return x.float().permute(0,2,1,3).reshape(b*h,n,x.shape[-1]).contiguous()
        ql = torch.einsum('bnhd,hcd->bnhc',q.float(),q_weight.float())
        kl = torch.einsum('bnhd,hcd->bnhc',k.float(),k_weight.float())
        qf,kf = flatten(ql),flatten(kl)
        iq,ik = qf.argmax(-1).int(),kf.argmax(-1).int()
        codebook = q_weight.float()[None].expand(b,-1,-1,-1).reshape(b*h,-1,d)
        e = torch.zeros((b*h,n,d),device=q.device,dtype=torch.float32)
        # Split the K embedding work according to the same row/head flags as
        # production. Hard heads do not evaluate a differentiable K softmax.
        for flag in (True,False):
            indices = (hard.flatten()==flag).nonzero().flatten()
            if indices.numel()==0:
                continue
            book = codebook.index_select(0,indices)
            if flag:
                part = book.gather(1,ik.index_select(0,indices).long()[...,None].expand(-1,-1,d))
            else:
                part = torch.bmm(kf.index_select(0,indices).softmax(-1),book)
            e = e.index_copy(0,indices,part)
        if segment_ids is None:
            segment_ids = torch.zeros((b,n),device=q.device,dtype=torch.int32)
        seg = segment_ids[:,None].expand(b,h,n).reshape(b*h,n).int().contiguous()
        reset = torch.zeros_like(seg,dtype=torch.bool)
        out = _Read.apply(flatten(q),e,qf.logsumexp(-1,keepdim=True),flatten(v),
                          tau.float().repeat(b),iq,ik,seg,reset,block_size,epsilon)
        return out.reshape(b,h,n,v.shape[-1]).permute(0,2,1,3).contiguous()
