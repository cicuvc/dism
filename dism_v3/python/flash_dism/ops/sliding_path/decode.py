"""Forward-only single-token sliding-window PaTH with a GPU KV ring.

Each live key is H_i ... H_{j+1} k_j; values are raw v_j. One fused kernel
replaces the oldest slot, transforms retained keys, inserts the current key
without H_i, and computes stable window-only softmax. The initial empty slots
are MASKED from the denominator; a saturating per-head length counter tracks
warmup. Keys/softmax reductions use FP32, stored values and outputs use BF16.

The input/layout/stride contract matches sliding_delta_net_decode, but scale
multiplies logits INSIDE softmax. No implicit normalization or beta*v writes.
States are updated in place on the current stream. With preallocated out,
no allocation or host cursor/count synchronization occurs during a step.
CUDA Graph replay is supported; forward-only even when inputs require grad.
"""

from dataclasses import dataclass
import torch
import triton
import triton.language as tl

from ..sliding_delta_net.decode import (
    SlidingDeltaDecodeState, create_decode_state as _create_ring, _validate,
)


@triton.jit
def _decode(Q, K, V, Beta, Keys, Values, Cursor, Lengths, O,
            H: tl.constexpr, W: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            QB: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
            KB: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
            VB: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
            BB: tl.constexpr, BH: tl.constexpr,
            OB: tl.constexpr, OH: tl.constexpr, OD: tl.constexpr,
            DK: tl.constexpr, DV: tl.constexpr, BW: tl.constexpr,
            SCALE: tl.constexpr):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    d, e = tl.arange(0, DK), tl.arange(0, DV)
    q = tl.load(Q + b * QB + h * QH + d * QD, d < D, 0).to(tl.float32)
    k = tl.load(K + b * KB + h * KH + d * KD, d < D, 0).to(tl.float32)
    v = tl.load(V + b * VB + h * VH + e * VD, e < E, 0).to(tl.float32)
    beta = tl.load(Beta + b * BB + h * BH)
    cursor = tl.load(Cursor + bh)
    length = tl.load(Lengths + bh)
    output = tl.full((DV,), 0., tl.float32)
    maximum, normalizer = 0., 0.
    for block in range(tl.cdiv(W, BW)):
        r = block * BW + tl.arange(0, BW)
        current = r == cursor
        live = (r < length) | current
        kp = Keys + (bh * W + r[:, None]) * D + d[None, :]
        # Ignore the evicted key/value entirely, even if that slot contains NaN.
        old_k = tl.load(kp, (r[:, None] < length) & ~current[:, None] & (d[None, :] < D), 0)
        projection = tl.sum(old_k * k[None, :], 1)
        updated = old_k - (beta * projection)[:, None] * k[None, :]
        updated = tl.where(current[:, None], k[None, :], updated)
        tl.store(kp, updated, (r[:, None] < W) & (d[None, :] < D))
        scores = tl.sum(updated * q[None, :], 1) * SCALE
        scores = tl.where((r < W) & live, scores, -float("inf"))
        has_live = tl.sum(((r < W) & live).to(tl.int32), 0) > 0
        block_maximum = tl.max(scores, 0)
        new_maximum = tl.where(has_live,
                               tl.where(normalizer == 0., block_maximum,
                                        tl.maximum(maximum, block_maximum)), maximum)
        alpha = tl.where(normalizer == 0., 0., tl.exp(maximum - new_maximum))
        p = tl.exp(scores - new_maximum)
        normalizer = normalizer * alpha + tl.sum(p, 0)
        maximum = new_maximum

        vp = Values + (bh * W + r[:, None]) * E + e[None, :]
        old_v = tl.load(vp, (r[:, None] < length) & ~current[:, None] & (e[None, :] < E), 0).to(tl.float32)
        values = tl.where(current[:, None], v[None, :], old_v)
        # Old values do not change. Only the replaced slot is written.
        tl.store(vp, values, (r[:, None] < W) & current[:, None] & (e[None, :] < E))
        output = output * alpha + tl.sum(p[:, None] * values, 0)

    tl.store(O + b * OB + h * OH + e * OD, output / normalizer, e < E)
    tl.store(Lengths + bh, tl.minimum(length + 1, W))
    tl.store(Cursor + bh, tl.where(cursor + 1 == W, 0, cursor + 1))


@dataclass(frozen=True, kw_only=True)
class SlidingPathDecodeState(SlidingDeltaDecodeState):
    """Ring with GPU int32 [B,H] lengths, saturated at window_size.

    During warmup the first lengths[b,h] slots are live. Once full, every
    slot is live and cursor identifies the oldest one. Counter values are
    kernel-owned; reset_ clears all sequences without changing tensor objects.
    """

    lengths: torch.Tensor

    @torch.no_grad()
    def reset_(self):
        super().reset_()
        self.lengths.zero_()
        return self


def create_path_decode_state(batch_size, num_heads, key_dim, value_dim, window_size, *, device="cuda"):
    ring = _create_ring(batch_size, num_heads, key_dim, value_dim, window_size, device=device)
    return SlidingPathDecodeState(ring.keys, ring.values, ring.cursor,
                                   lengths=torch.zeros_like(ring.cursor))


@torch.no_grad()
def sliding_window_path_decode(q, k, v, beta, state, *, scale=1.0, out=None):
    """Consume [B,1,H,D] Q/K/V and [B,H,1] beta; update the state in place.

    Returns BF16 [B,1,H,Dv]. Noncontiguous inputs and nonoverlapping strided
    output buffers are supported. The output may not alias inputs or state.
    Each batch/head may have its own valid history length and ring cursor.
    """
    if not isinstance(state, SlidingPathDecodeState):
        raise TypeError("state must be a SlidingPathDecodeState")
    if state.write_beta:
        raise ValueError("PaTH stores raw values; write_beta must be False")
    _validate(q, k, v, beta, state, out, scale)
    count = state.lengths
    if (count.shape != state.cursor.shape or count.dtype != torch.int32
            or count.device != q.device or not count.is_contiguous() or count.requires_grad):
        raise ValueError("lengths must be contiguous int32 [B,H] on the input device")
    ptr = count.untyped_storage().data_ptr()
    if any(ptr == z.untyped_storage().data_ptr()
           for z in (state.keys, state.values, state.cursor, q, k, v, beta)):
        raise ValueError("lengths must not alias other buffers")
    if out is not None and ptr == out.untyped_storage().data_ptr():
        raise ValueError("out must not alias lengths")
    b, _, h, d = q.shape
    e, w = v.shape[-1], state.window_size
    with torch.cuda.device(q.device):
        if out is None:
            out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        _decode[(b*h,)](
            q, k, v, beta, state.keys, state.values, state.cursor, state.lengths, out,
            H=h, W=w, D=d, E=e,
            QB=q.stride(0), QH=q.stride(2), QD=q.stride(3),
            KB=k.stride(0), KH=k.stride(2), KD=k.stride(3),
            VB=v.stride(0), VH=v.stride(2), VD=v.stride(3),
            BB=beta.stride(0), BH=beta.stride(1),
            OB=out.stride(0), OH=out.stride(2), OD=out.stride(3),
            DK=triton.next_power_of_2(d), DV=triton.next_power_of_2(e),
            BW=min(64, triton.next_power_of_2(w)), SCALE=float(scale),
            num_warps=4, num_stages=1,
        )
    return out

