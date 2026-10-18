"""Fused forward-only sliding-window DeltaNet decoding with a KV ring buffer.

Input Q/K: BF16 [B,1,H,Dk], V: BF16 [B,1,H,Dv], beta: FP32 [B,H,1].
Output: BF16 [B,1,H,Dv]. The state is updated in place on the current stream.

After token i, a live slot j contains
    keys[j] = H_i ... H_{j+1} k_j,  values[j] = v_j,
where H_i = I - beta_i k_i k_i^T. With write_beta=True, values[j] instead
stores beta_j*v_j. Slots hold exactly the last window_size writes; before
the first wraparound, unwritten slots contain zero. The cursor identifies
the next free / oldest slot, so no growing token counter or CPU sync is used.

Each step discards the oldest slot, applies H_i to the remaining old keys,
inserts k_i/v_i at the discarded slot, and sums q_i^T keys[j] * values[j].
The new key MUST bypass H_i: the diagonal in the original formula is q_i^T k_i.
No matrix inverse, beta division, softmax, or decay gate is needed.

The recurrent key cache is FP32 to avoid repeated BF16 rounding over the
window. Values are BF16 (FP32 when write_beta=True to retain beta*v exactly).
All reductions and readout accumulation use FP32. One CTA owns one batch/head
and streams over window tiles; the update, query, and cursor advance are fused.
Cost per token: O(B H w (Dk+Dv)); persistent space: O(B H w (Dk+Dv)).

This API is inference-only and does not build an autograd graph. Use a separate
state for each independent decoding stream. Calls sharing one state must be
ordered on the same stream (or externally synchronized). Device cursors and an
optional preallocated output allow repeated CUDA Graph replay without host
counter updates. State allocation/reset is separate from stepping.

Example:
    state = create_decode_state(B, H, Dk, Dv, window_size=128, device=q.device)
    o = sliding_window_delta_net_decode(q, k, v, beta, state)
"""

from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _decode(Q, K, V, Beta, Keys, Values, Cursor, O,
            H: tl.constexpr, W: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            QB: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
            KB: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
            VB: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
            BB: tl.constexpr, BH: tl.constexpr,
            OB: tl.constexpr, OH: tl.constexpr, OD: tl.constexpr,
            DK: tl.constexpr, DV: tl.constexpr, BW: tl.constexpr,
            SCALE: tl.constexpr, WRITE_BETA: tl.constexpr):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    d, e = tl.arange(0, DK), tl.arange(0, DV)
    q = tl.load(Q + b * QB + h * QH + d * QD, d < D, 0).to(tl.float32)
    k = tl.load(K + b * KB + h * KH + d * KD, d < D, 0).to(tl.float32)
    v = tl.load(V + b * VB + h * VH + e * VD, e < E, 0).to(tl.float32)
    beta = tl.load(Beta + b * BB + h * BH)
    if WRITE_BETA:
        v = beta * v
    cursor = tl.load(Cursor + bh)
    output = tl.full((DV,), 0., tl.float32)
    for block in range(tl.cdiv(W, BW)):
        r = block * BW + tl.arange(0, BW)
        current = r == cursor
        kp = Keys + (bh * W + r[:, None]) * D + d[None, :]
        # Ignore the evicted key/value entirely, even if that slot contains NaN.
        old_k = tl.load(kp, (r[:, None] < W) & ~current[:, None] & (d[None, :] < D), 0)
        projection = tl.sum(old_k * k[None, :], 1)
        updated = old_k - (beta * projection)[:, None] * k[None, :]
        updated = tl.where(current[:, None], k[None, :], updated)
        tl.store(kp, updated, (r[:, None] < W) & (d[None, :] < D))
        scores = tl.sum(updated * q[None, :], 1)

        vp = Values + (bh * W + r[:, None]) * E + e[None, :]
        old_v = tl.load(vp, (r[:, None] < W) & ~current[:, None] & (e[None, :] < E), 0).to(tl.float32)
        values = tl.where(current[:, None], v[None, :], old_v)
        # Old values do not change. Only the replaced slot is written.
        tl.store(vp, values, (r[:, None] < W) & current[:, None] & (e[None, :] < E))
        output += tl.sum(scores[:, None] * values, 0)

    tl.store(O + b * OB + h * OH + e * OD, output * SCALE, e < E)
    tl.store(Cursor + bh, tl.where(cursor + 1 == W, 0, cursor + 1))


@dataclass(frozen=True)
class SlidingDeltaDecodeState:
    """Mutable GPU ring contents; tensor objects and write convention are fixed.

    keys: FP32 [B,H,W,Dk], already transformed keys, not a raw-key cache.
    values: [B,H,W,Dv], BF16 or FP32 for beta-weighted writes.
    cursor: int32 [B,H], next write position in [0,W). Treat as kernel-owned.
    reset_() clears all sequences. Clone all three tensors to fork a stream.
    """

    keys: torch.Tensor
    values: torch.Tensor
    cursor: torch.Tensor
    write_beta: bool = False

    @property
    def window_size(self):
        return self.keys.shape[2]

    @torch.no_grad()
    def reset_(self):
        self.keys.zero_()
        self.values.zero_()
        self.cursor.zero_()
        return self


def create_decode_state(batch_size, num_heads, key_dim, value_dim, window_size, *,
                        device="cuda", write_beta=False):
    """Allocate an empty ring; no prompt history or initial state is assumed."""
    dims = (batch_size, num_heads, key_dim, value_dim, window_size)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in dims):
        raise ValueError("Batch, heads, dimensions, and window_size must be positive integers")
    if not isinstance(write_beta, bool):
        raise TypeError("write_beta must be bool")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Decoding state must be on a CUDA device")
    keys = torch.zeros(batch_size, num_heads, window_size, key_dim, device=device, dtype=torch.float32)
    values = torch.zeros(batch_size, num_heads, window_size, value_dim, device=device,
                         dtype=torch.float32 if write_beta else torch.bfloat16)
    cursor = torch.zeros(batch_size, num_heads, device=device, dtype=torch.int32)
    return SlidingDeltaDecodeState(keys, values, cursor, write_beta)


def _validate(q, k, v, beta, state, out, scale):
    if q.ndim != 4 or q.shape[1] != 1 or min(q.shape) < 1 or k.shape != q.shape:
        raise ValueError("Q/K must have identical nonempty [B,1,H,Dk] shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3] or v.shape[-1] < 1:
        raise ValueError("V must have shape [B,1,H,Dv]")
    b, _, h, d = q.shape
    if beta.shape != (b, h, 1):
        raise ValueError("beta must have shape [B,H,1]")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)) or beta.dtype != torch.float32:
        raise TypeError("Q/K/V must be BF16 and beta must be FP32")
    if q.device.type != "cuda" or any(x.device != q.device for x in (k, v, beta)):
        raise ValueError("All inputs must be on the same CUDA device")
    if not isinstance(state, SlidingDeltaDecodeState):
        raise TypeError("state must be a SlidingDeltaDecodeState")
    if not isinstance(state.write_beta, bool):
        raise TypeError("state.write_beta must be bool")
    if state.keys.ndim != 4 or state.keys.shape[:2] != (b, h) or state.keys.shape[-1] != d:
        raise ValueError("state.keys must have shape [B,H,W,Dk]")
    w = state.keys.shape[2]
    if w < 1 or state.values.shape != (b, h, w, v.shape[-1]) or state.cursor.shape != (b, h):
        raise ValueError("State value/cursor shapes do not match Q/K/V or window_size")
    value_dtype = torch.float32 if state.write_beta else torch.bfloat16
    if state.keys.dtype != torch.float32 or state.values.dtype != value_dtype or state.cursor.dtype != torch.int32:
        raise TypeError("Invalid cache dtype: keys FP32, cursor int32, values BF16 (FP32 for write_beta)")
    if any(x.device != q.device or not x.is_contiguous() or x.requires_grad
           for x in (state.keys, state.values, state.cursor)):
        raise ValueError("State tensors must be contiguous, on the input device, and not require gradients")
    if not isinstance(scale, (float, int)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite Python number")
    if out is not None:
        if out.shape != v.shape or out.dtype != v.dtype or out.device != v.device or out.requires_grad:
            raise ValueError("out must match V's shape/dtype/device and not require gradients")
        # An output sharing any input/state storage could corrupt later steps or
        # race with another head. Requiring separate storage keeps ownership clear.
        output_storage = out.untyped_storage().data_ptr()
        if any(x.untyped_storage().data_ptr() == output_storage
               for x in (q, k, v, beta, state.keys, state.values, state.cursor)):
            raise ValueError("out must not share storage with inputs or state")
        # Non-overlapping dense slices (including output[:, i:i+1]) are fine;
        # overlapping/expanded outputs would introduce multiple writers.
        dims = sorted((out.stride(i), out.shape[i]) for i in (0, 2, 3) if out.shape[i] > 1)
        extent = 1
        for stride, size in dims:
            if stride < extent:
                raise ValueError("out must have non-overlapping positive strides")
            extent += (size - 1) * stride


@torch.no_grad()
def sliding_window_delta_net_decode(q, k, v, beta, state, *, scale=1.0, out=None):
    """Consume exactly one token, return its output, and advance state in place.

    No backward is provided, even when inputs require gradients. Inputs may
    have arbitrary positive/zero strides; no contiguous copies are made.
    out can be preallocated (including a non-overlapping sequence slice).
    scale is applied only to output and does not change the stored state.
    """
    _validate(q, k, v, beta, state, out, scale)
    b, _, h, d = q.shape
    e, w = v.shape[-1], state.window_size
    with torch.cuda.device(q.device):
        if out is None:
            out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        _decode[(b*h,)](
            q, k, v, beta, state.keys, state.values, state.cursor, out,
            H=h, W=w, D=d, E=e,
            QB=q.stride(0), QH=q.stride(2), QD=q.stride(3),
            KB=k.stride(0), KH=k.stride(2), KD=k.stride(3),
            VB=v.stride(0), VH=v.stride(2), VD=v.stride(3),
            BB=beta.stride(0), BH=beta.stride(1),
            OB=out.stride(0), OH=out.stride(2), OD=out.stride(3),
            DK=triton.next_power_of_2(d), DV=triton.next_power_of_2(e), BW=min(64, triton.next_power_of_2(w)),
            SCALE=float(scale), WRITE_BETA=state.write_beta, num_warps=4, num_stages=1,
        )
    return out
