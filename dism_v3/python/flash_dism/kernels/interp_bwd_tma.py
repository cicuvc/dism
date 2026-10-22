"""Optional joint interpolation backward: reuse dKV scores and TMA-reduce dX.

Each vocabulary tile contributes an FP32 [tokens, channels] partial dX. TMA
reduces it into a zeroed BNHD tensor; dKey/dValue retain the existing global
FP32 atomic reduction. The separate production dQ kernel remains the default.
"""
import triton
import triton.language as tl
from .interp_bwd_split import _TUNE_KEY, _bench, _prune_splits


def _descriptor_block(args):
    args['DXDesc'].block_shape = [args['M'], args['D']]


@triton.autotune(
    configs=[triton.Config({'M': m, 'K': k, 'SPLITS': splits},
                           num_warps=warps, num_stages=stages, pre_hook=_descriptor_block)
             for m, k, splits, stages, warps in [(*c,4) for c in [
                 (32,64,1,2), (32,64,4,2), (64,64,1,2), (64,64,4,2),
                 (128,64,1,2), (128,64,4,2),
                 (32,64,16,2), (64,32,16,2),
                 (64,64,16,2), (64,64,32,2), (64,64,64,2),
                 (64,128,16,2), (64,128,32,2),
                 (128,64,16,2), (128,64,32,2)]] + [
                 (64,128,32,1,4), (128,64,16,1,4), (128,64,32,1,4),
                 (128,128,16,1,4), (128,128,32,1,4),
                 (64,128,32,1,8), (128,64,32,1,8), (128,128,32,1,8),
                 (64,32,16,1,4), (64,32,32,1,4),
                 (64,32,16,1,8), (64,32,32,1,8), (64,32,64,1,8),
                 (64,32,16,2,8), (64,32,32,2,8),
                 (64,64,32,2,8), (32,64,32,2,8)]],
    key=_TUNE_KEY, restore_value=['DK', 'DV', 'DX'],
    prune_configs_by={'early_config_prune': _prune_splits}, do_bench=_bench)
@triton.jit
def _dkv_dq_tma(X, Key, Value, DO, LSE, DLSE, Delta, DK, DV, DX, DXDesc, Tau,
         B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, V: tl.constexpr,
         D: tl.constexpr, beta, HAS_TAU: tl.constexpr, DEVICE: tl.constexpr,
         xb: tl.constexpr, xn: tl.constexpr, xh: tl.constexpr,
         kh: tl.constexpr, kv: tl.constexpr, vh: tl.constexpr, vv: tl.constexpr,
         ob: tl.constexpr, on: tl.constexpr, oh: tl.constexpr,
         M: tl.constexpr, K: tl.constexpr, SPLITS: tl.constexpr, LSE_ONLY: tl.constexpr = False):
    head = tl.program_id(1)
    split = tl.program_id(2)
    col = tl.program_id(0) * K + tl.arange(0, K)
    d = tl.arange(0, D)
    key = tl.load(Key + head * kh + col[:, None] * kv + d[None, :], col[:, None] < V, 0)
    if not LSE_ONLY:
        value = tl.load(Value + head * vh + col[:, None] * vv + d[None, :], col[:, None] < V, 0)
    key_grad = tl.zeros((K, D), tl.float32)
    if not LSE_ONLY:
        value_grad = tl.zeros((K, D), tl.float32)
    # Split in tiles; masks handle incomplete token/vocabulary tiles and B>1.
    TILES: tl.constexpr = tl.cdiv(B * N, M * SPLITS)
    for tile in range(TILES):
        row = (split * TILES + tile) * M + tl.arange(0, M)
        batch, token = row // N, row % N
        valid = row < B * N
        x = tl.load(X + batch[:, None] * xb + token[:, None] * xn + head * xh + d[None, :],
                    valid[:, None], 0)
        if not LSE_ONLY:
            do = tl.load(DO + batch[:, None] * ob + token[:, None] * on + head * oh + d[None, :],
                         valid[:, None], 0).to(Key.dtype.element_ty)
        offset = (batch * H + head) * N + token
        lse = tl.load(LSE + offset, valid, 0)
        if HAS_TAU:
            lse += tl.load(Tau + head)
        lse *= 1.4426950408889634
        dlse = tl.load(DLSE + offset, valid, 0)
        if not LSE_ONLY:
            delta = tl.load(Delta + offset, valid, 0)
        score = tl.dot(x, tl.trans(key))
        score = tl.where(valid[:, None] & (col[None, :] < V), score, float('-inf'))
        p = tl.exp2(score * (beta * 1.4426950408889634) - lse[:, None])
        if LSE_ONLY:
            ds = p * dlse[:, None]
        else:
            dp = tl.dot(do, tl.trans(value))
            ds = p * (dp - delta[:, None]) + p * dlse[:, None]
        # TMA output layout stages this tile in shared memory, then reduces
        # into global FP32 DX. No shared-memory atomic operation is used.
        dx = tl.dot(ds.to(Key.dtype.element_ty), key) * beta
        DXDesc.atomic_add([(split * TILES + tile) * M, head * D], dx)
        key_grad += tl.dot(tl.trans(ds).to(Key.dtype.element_ty), x) * beta
        if not LSE_ONLY:
            value_grad += tl.dot(tl.trans(p).to(Key.dtype.element_ty), do)
    offset = (head * V + col[:, None]) * D + d[None, :]
    # Global FP32 atomics only; zeroed outputs, no shared atomics.
    tl.atomic_add(DK + offset, key_grad, col[:, None] < V, sem='relaxed')
    if not LSE_ONLY:
        tl.atomic_add(DV + offset, value_grad, col[:, None] < V, sem='relaxed')
