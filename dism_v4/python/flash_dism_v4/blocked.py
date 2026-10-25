"""CUDA blockwise v4 prototype with hard history and candidate denominators.

Soft reads support arithmetic mixing or a normalized Jensen surrogate.
No SQ/SK, planner, cache, or CUDA graph. Pair scratch is O(BH * block_size * N).
Backward regenerates history in forward block order and differentiates its
geometric weight w.r.t. tau, without saving quadratic boundary checkpoints.
"""
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .history import history_blocks


def _mm(a, b, precision):
    if precision == "bf16":
        return torch.bmm(a.to(torch.bfloat16), b.to(torch.bfloat16), out_dtype=torch.float32)
    return torch.bmm(a, b)


def _hard_weights(loga, p, labels):
    index = labels[:, None, :].expand_as(loga)
    maximum = torch.zeros_like(p).scatter_reduce_(2, index, loga, reduce="amax", include_self=True)
    w = (loga - maximum.gather(2, index)).exp()
    denominator = (-maximum).exp().scatter_add_(2, index, w)
    w = w / denominator.gather(2, index)
    return index, w, w * p.gather(2, index)


def _soft_weights(loga, p, k, precision):
    maximum = loga.amax(-1, keepdim=True).clamp_min(0)
    a = (loga - maximum).exp()
    denominator = (-maximum).exp() + _mm(a, k, precision)
    c = p / denominator
    b = _mm(c, k.transpose(1, 2), precision)
    return a, denominator, c, b, a*b


def _jensen_weights(loga, logp, k, precision):
    """Jensen on the complete normalized coefficient, retaining candidate Z.

    The common row scale cancels because each K distribution sums to one.
    No probability threshold, clipping, or slow log-domain fallback.
    """
    maximum = loga.amax(-1, keepdim=True).clamp_min(0)
    loga_scaled = loga - maximum
    a = loga_scaled.exp()
    denominator = (-maximum).exp() + _mm(a, k, precision)
    r = logp - denominator.log()
    m = (loga_scaled + _mm(r, k.transpose(1, 2), precision)).exp()
    return a, denominator, r, m


class _Blocked(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, logp, logk, v, tau, iq, ik, segments, reset, mode, block, precision):
        p, k = logp.exp(), logk.exp()
        out = torch.empty_like(v)
        for start, loga, _ in history_blocks(iq, ik, tau, segments, reset, block):
            end = start + loga.shape[1]
            if mode == "hard":
                _, _, m = _hard_weights(loga, p[:, start:end], ik)
                out[:, start:end] = _mm(m, v, precision)
            elif mode == "jensen":
                *_, m = _jensen_weights(loga, logp[:, start:end], k, precision)
                out[:, start:end] = _mm(m, v, precision)
            else:
                *_, m = _soft_weights(loga, p[:, start:end], k, precision)
                out[:, start:end] = _mm(m, v, precision)
        ctx.save_for_backward(logp, logk, v, tau, iq, ik, segments, reset)
        ctx.mode, ctx.block, ctx.precision = mode, block, precision
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad):
        logp, logk, v, tau, iq, ik, segments, reset = ctx.saved_tensors
        p, k = logp.exp(), logk.exp()
        dp, dk, dv, dt = torch.zeros_like(p), torch.zeros_like(k), torch.zeros_like(v), torch.zeros_like(tau)
        precision = ctx.precision
        for start, loga, dloga in history_blocks(iq, ik, tau, segments, reset, ctx.block):
            end = start + loga.shape[1]
            pb, g = p[:, start:end], grad[:, start:end].contiguous()
            e = _mm(g, v.transpose(1, 2), precision)
            if ctx.mode == "hard":
                index, w, m = _hard_weights(loga, pb, ik)
                dprob = torch.zeros_like(pb).scatter_add_(2, index, w*e)
                dp[:, start:end] = pb*dprob
                dlog_a = m*(e-dprob.gather(2, index))
                dv += _mm(m.transpose(1, 2), g, precision)
            elif ctx.mode == "jensen":
                a, denominator, r, m = _jensen_weights(loga, logp[:, start:end], k, precision)
                t = e*m
                dlogp = _mm(t, k, precision)
                dp[:, start:end] = dlogp
                dz = -dlogp/denominator
                # Dropping the common log-scale from r changes dK only by a
                # candidate-independent offset, canceled by K's log-softmax.
                dk += (_mm(t.transpose(1, 2), r, precision)
                       + _mm(a.transpose(1, 2), dz, precision))*k
                dlog_a = t + a*_mm(dz, k.transpose(1, 2), precision)
                dv += _mm(m.transpose(1, 2), g, precision)
            else:
                a, denominator, c, b, m = _soft_weights(loga, pb, k, precision)
                dc = _mm(e*a, k, precision)
                dp[:, start:end] = dc*c
                # This ordering avoids squaring tiny candidate denominators.
                dz = -(dc*c)/denominator
                dk += (_mm((e*a).transpose(1, 2), c, precision)
                       + _mm(a.transpose(1, 2), dz, precision))*k
                dlog_a = a*(e*b + _mm(dz, k.transpose(1, 2), precision))
                dv += _mm(m.transpose(1, 2), g, precision)
            dt += (dlog_a*dloga).sum((1, 2))
        return dp, (dk if ctx.mode != "hard" else None), dv, dt, None, None, None, None, None, None, None


@torch.compiler.disable
def dism_v4_blocked(
    q_logits: Tensor, k_logits: Tensor, v: Tensor, tau: Tensor, *,
    key_mode="hard", hard=None, block_size=64, precision="fp32", soft_readout="arithmetic",
    cu_seqlens=None, segment_ids=None, query_reset=None,
):
    """BNHC logits -> BNHDv output; float32 CUDA, optional BF16 GEMMs.

    `hard`: optional bool [B,H] mask overriding key_mode per packed training row.
    All modes retain hard Q/K history. `precision='bf16'` is an explicit inexact
    speed option; the default fp32 uses native PyTorch float32 GEMM settings.
    Segment IDs must be nondecreasing [B,N]; packed cu_seqlens requires B=1.
    soft_readout='jensen' selects the normalized Jensen surrogate for soft
    heads. key_mode='jensen' is shorthand for all-soft Jensen evaluation.
    Both soft paths always use GEMM; extreme ranges may underflow or overflow.
    """
    if q_logits.ndim != 4 or q_logits.shape != k_logits.shape or min(q_logits.shape) <= 0:
        raise ValueError("q/k logits must have equal nonempty [B,N,H,C] shapes")
    b, n, h, c = q_logits.shape
    if v.ndim != 4 or v.shape[:3] != (b, n, h) or v.shape[-1] <= 0 or tau.shape != (h,):
        raise ValueError("v must be [B,N,H,Dv] and tau [H]")
    if not all(x.dtype in (torch.float16, torch.bfloat16, torch.float32) for x in (q_logits,k_logits,v,tau)):
        raise ValueError("blocked prototype accepts fp16/bf16/fp32 inputs; use reference for fp64")
    if key_mode not in ("hard", "soft", "jensen") or block_size not in (32, 64, 128) or precision not in ("fp32", "bf16"):
        raise ValueError("require hard/soft/jensen mode, block_size 32/64/128, precision fp32/bf16")
    if soft_readout not in ("arithmetic", "jensen"):
        raise ValueError("soft_readout must be arithmetic or jensen")
    if not q_logits.is_cuda or any(x.device != q_logits.device for x in (k_logits, v, tau)):
        raise ValueError("blocked prototype requires inputs on one CUDA device")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("v4 prototype does not support CUDA graph capture")
    if cu_seqlens is not None:
        if segment_ids is not None or b != 1 or cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens requires B=1 and no segment_ids")
        bounds = cu_seqlens.tolist()
        if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != n or any(x >= y for x, y in zip(bounds, bounds[1:])):
            raise ValueError("cu_seqlens must strictly increase from 0 to N")
        segment_ids = torch.bucketize(torch.arange(n, device=v.device), cu_seqlens[1:-1].to(v.device), right=True)[None]
    if segment_ids is None:
        segment_ids = torch.zeros((b, n), device=v.device, dtype=torch.int32)
    if segment_ids.shape != (b, n) or segment_ids.device != v.device or segment_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("segment_ids must be integer [B,N] on the input device")
    if bool((segment_ids[:, 1:] < segment_ids[:, :-1]).any()):
        raise ValueError("segment_ids must be nondecreasing")
    if query_reset is None:
        query_reset = torch.zeros((b, n), device=v.device, dtype=torch.bool)
    if query_reset.shape != (b, n) or query_reset.dtype != torch.bool or query_reset.device != v.device:
        raise ValueError("query_reset must be bool [B,N] on the input device")
    if hard is None:
        hard = torch.full((b, h), key_mode == "hard", device=v.device, dtype=torch.bool)
    if hard.shape != (b, h) or hard.dtype != torch.bool or hard.device != v.device:
        raise ValueError("hard must be bool [B,H] on the input device")
    logp = q_logits.float().permute(0, 2, 1, 3).reshape(b*h, n, c).log_softmax(-1)
    logk = k_logits.float().permute(0, 2, 1, 3).reshape(b*h, n, c).log_softmax(-1)
    values = v.float().permute(0, 2, 1, 3).reshape(b*h, n, v.shape[-1]).contiguous()
    iq, ik = logp.argmax(-1).int(), logk.argmax(-1).int()
    seg = segment_ids[:, None].expand(b, h, n).reshape(b*h, n).int().contiguous()
    reset = query_reset[:, None].expand(b, h, n).reshape(b*h, n).contiguous()
    taus = tau.float().repeat(b)
    out = torch.zeros_like(values)
    soft_mode = "jensen" if key_mode == "jensen" or soft_readout == "jensen" else "soft"
    for decision, mode in [(True, "hard"), (False, soft_mode)]:
        indices = (hard.flatten() == decision).nonzero().flatten()
        if indices.numel() == 0:
            continue
        selected = [x.index_select(0, indices).contiguous() for x in (logp, logk, values, taus, iq, ik, seg, reset)]
        part = _Blocked.apply(*selected, mode, block_size, precision)
        out = out.index_copy(0, indices, part)
    return out.reshape(b, h, n, v.shape[-1]).permute(0, 2, 1, 3).contiguous()


def dism_v4_blocked_from_embeddings(q, k, q_weight, k_weight, v, tau, **kwargs):
    """Per-head codebooks [H,C,D]; no implicit scaling or codebook activation."""
    # Keep vocabulary projection in FP32 even inside model BF16 autocast.
    with torch.autocast("cuda", enabled=False):
        q_logits = torch.einsum("bnhd,hcd->bnhc", q.float(), q_weight.float())
        k_logits = torch.einsum("bnhd,hcd->bnhc", k.float(), k_weight.float())
        return dism_v4_blocked(q_logits, k_logits, v, tau, **kwargs)
