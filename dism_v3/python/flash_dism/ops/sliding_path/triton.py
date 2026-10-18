"""Native sliding-window PaTH with block-parallel CUDA forward/backward.

For max(document_start, i-window_size+1) <= j <= i:
  s_ij = scale * q_i^T H_i ... H_{j+1} k_j,
  o_i = sum_j softmax_j(s_ij) v_j, H_t = I - beta_t k_t k_t^T.
The j=i product is the identity. Scale defaults to 1 and is applied INSIDE
softmax. Beta only controls the transformations; values are not beta-weighted.
Q/K normalization is external. This implements the user's tied key/reflector
variant of PaTH, not the paper's independent reflector parameterization.

Inputs: BF16 Q/K [B,N,H,Dk], V [B,N,H,Dv]; FP32 beta [B,H,N].
Output: BF16 [B,N,H,Dv]. cu_seqlens enables packed varlen (physical B=1).
Chunk/window metadata, WY preparation, and inverse helpers are shared with
sliding_delta_net_triton. Both training paths are native Triton; no Python
loop over numerical documents or tokens and no dense N-by-N score matrix.
The inverse, local WY preparation/adjoint, solved diagonal score coefficients,
and cancellation-sensitive query adjoints use TF32x3. Window scan and value
products default to BF16 with FP32 accumulation; matmul_precision='tf32x3'
also raises these to TF32x3 for diagnosis. Keeping the local WY operations
precise prevents softmax from amplifying cancellation errors for repeated
keys and beta near two.

Each query tile scans its window's key tiles with stable online softmax.
Backward recomputes probabilities from saved LSE and transformed queries.
A separate parallel reduction computes sum_j p_ij * dP_ij instead of using
rounded BF16 outputs, keeping the softmax row adjoint accurate (in particular
for singleton windows). WY gradients include the key's dual role as key and
Householder vector. FP32 atomic reductions are not bitwise deterministic.
Only first derivatives are supported. Saved storage is
O(B H N ceil(w/C) Dk + B H N (C+Dk)); inference omits query history and LSE.
Packed metadata follows sliding_delta_net_varlen's validation/cache semantics
(first-use host synchronization; varlen CUDA Graph capture is unsupported).

Reference: https://arxiv.org/html/2505.16381v2#S3 (UT/WY + tiled softmax).
"""

import math
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from ..sliding_delta_net.triton import (
    _allocate,
    _chunk_bounds,
    _config,
    _dot,
    _prepare,
)
from ..sliding_delta_net.varlen import prepare_varlen_metadata


@triton.jit(do_not_specialize=["NT"])
def _softmax_delta(Q, K, V, R, Kr, History, LSE, DO, Delta,
                   N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                   NT, C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                   W: tl.constexpr, LAGS: tl.constexpr, SCALE: tl.constexpr,
                   HIGH: tl.constexpr, Cu=None, ChunkIndices=None):
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    r, d, e = tl.arange(0, C), tl.arange(0, DK), tl.arange(0, DV)
    start, end, local = _chunk_bounds(block, Cu, ChunkIndices, N, C)
    t = start + r
    tile = (bh * NT + block) * C
    qo = ((b * N + t[:, None]) * H + h) * D + d[None, :]
    vo = ((b * N + t[:, None]) * H + h) * E + e[None, :]
    valid = (t[:, None] < end) & (d[None, :] < D)
    q, k = tl.load(Q + qo, valid, 0), tl.load(K + qo, valid, 0)
    v = tl.load(V + vo, (t[:, None] < end) & (e[None, :] < E), 0)
    do = tl.load(DO + vo, (t[:, None] < end) & (e[None, :] < E), 0)
    inv = tl.load(R + tile * C + r[:, None] * C + r[None, :])
    lse = tl.load(LSE + tile + r)
    lower = r[:, None] >= r[None, :]
    base = tl.where(lower, _dot(q, tl.trans(k), HIGH), 0.)
    score = _dot(base, inv, True)
    mask = lower & (r[:, None] - r[None, :] < W) & (t[:, None] < end)
    p = tl.where(mask, tl.exp(score * SCALE - lse[:, None]), 0.)
    delta = tl.sum(p * _dot(do, tl.trans(v), HIGH), 1)
    for lag in range(1, tl.minimum(local, LAGS) + 1):
        jt = (bh * NT + block - lag) * C
        tj = start - lag * C + r
        ho = ((bh * NT + block) * LAGS + lag - 1) * C * D
        x = tl.load(History + ho + r[:, None] * D + d[None, :], d[None, :] < D, 0)
        kr = tl.load(Kr + jt * D + r[:, None] * D + d[None, :], d[None, :] < D, 0)
        score = _dot(x, tl.trans(kr), HIGH)
        mask = (lag * C + r[:, None] - r[None, :] < W) & (t[:, None] < end)
        p = tl.where(mask, tl.exp(score * SCALE - lse[:, None]), 0.)
        v = tl.load(V + ((b * N + tj[:, None]) * H + h) * E + e[None, :],
                    (tj[:, None] < end) & (e[None, :] < E), 0)
        delta += tl.sum(p * _dot(do, tl.trans(v), HIGH), 1)
    tl.store(Delta + tile + r, delta)


@triton.jit(do_not_specialize=["NT"])
def _forward(Q, K, V, Beta, R, U, Kr, Ql, History, O, LSE,
             N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
             NT, C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
             W: tl.constexpr, LAGS: tl.constexpr, SCALE: tl.constexpr,
             SAVE: tl.constexpr, HIGH: tl.constexpr,
             Cu=None, ChunkIndices=None):
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    r, d, e = tl.arange(0, C), tl.arange(0, DK), tl.arange(0, DV)
    start, end, local = _chunk_bounds(block, Cu, ChunkIndices, N, C)
    ti = start + r
    offq = ((b * N + ti[:, None]) * H + h) * D + d[None, :]
    q = tl.load(Q + offq, (ti[:, None] < end) & (d[None, :] < D), 0)
    k = tl.load(K + offq, (ti[:, None] < end) & (d[None, :] < D), 0)
    tile = (bh * NT + block) * C
    inv = tl.load(R + tile * C + r[:, None] * C + r[None, :])
    lower = r[:, None] >= r[None, :]
    base = tl.where(lower, _dot(q, tl.trans(k), HIGH), 0.)
    a = _dot(base, inv, True)
    mask = lower & (r[:, None] - r[None, :] < W) & (ti[:, None] < end)
    logits = tl.where(mask, a * SCALE, -float("inf"))
    maximum = tl.where(ti < end, tl.max(logits, 1), 0.)
    p = tl.exp(logits - maximum[:, None])
    normalizer = tl.sum(p, 1)
    voff = ((b * N + ti[:, None]) * H + h) * E + e[None, :]
    v = tl.load(V + voff, (ti[:, None] < end) & (e[None, :] < E), 0).to(tl.float32)
    out = _dot(p, v, HIGH)
    x = tl.load(Ql + tile * D + r[:, None] * D + d[None, :], d[None, :] < D, 0)
    last = tl.minimum(local, LAGS)
    for lag in range(1, last + 1):
        j = block - lag
        tj = start - lag * C + r
        jt = (bh * NT + j) * C
        if SAVE:
            ho = ((bh * NT + block) * LAGS + lag - 1) * C * D
            tl.store(History + ho + r[:, None] * D + d[None, :], x, d[None, :] < D)
        kr = tl.load(Kr + jt * D + r[:, None] * D + d[None, :], d[None, :] < D, 0)
        score = _dot(x, tl.trans(kr), HIGH)
        mask = (lag * C + r[:, None] - r[None, :] < W) & (ti[:, None] < end)
        logits = tl.where(mask, score * SCALE, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(logits, 1))
        alpha = tl.exp(maximum - new_maximum)
        p = tl.exp(logits - new_maximum[:, None])
        normalizer = normalizer * alpha + tl.sum(p, 1)
        maximum = new_maximum
        v = tl.load(V + ((b * N + tj[:, None]) * H + h) * E + e[None, :],
                    (tj[:, None] < end) & (e[None, :] < E), 0).to(tl.float32)
        out = out * alpha[:, None] + _dot(p, v, HIGH)
        if lag < last:
            k = tl.load(K + ((b * N + tj[:, None]) * H + h) * D + d[None, :],
                        (tj[:, None] < end) & (d[None, :] < D), 0)
            u = tl.load(U + jt * D + r[:, None] * D + d[None, :], d[None, :] < D, 0)
            x -= _dot(_dot(x, tl.trans(k), HIGH), u, HIGH)
    denominator = tl.where(normalizer > 0., normalizer, 1.)
    tl.store(O + voff, out / denominator[:, None], (ti[:, None] < end) & (e[None, :] < E))
    if SAVE:
        tl.store(LSE + tile + r, maximum + tl.log(denominator))


@triton.jit(do_not_specialize=["NT"])
def _backward_scan(K, V, Beta, U, Kr, History, LSE, Delta, DO, GQl, GU, GKr, GK, GV, GB,
                   N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                   NT, C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                   W: tl.constexpr, LAGS: tl.constexpr, SCALE: tl.constexpr,
                   HIGH: tl.constexpr,
                   Cu=None, ChunkIndices=None):
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    r, d, e = tl.arange(0, C), tl.arange(0, DK), tl.arange(0, DV)
    start, end, local = _chunk_bounds(block, Cu, ChunkIndices, N, C)
    ti = start + r
    do = tl.load(DO + ((b * N + ti[:, None]) * H + h) * E + e[None, :],
                 (ti[:, None] < end) & (e[None, :] < E), 0).to(tl.float32)
    lse = tl.load(LSE + (bh * NT + block) * C + r)
    delta = tl.load(Delta + (bh * NT + block) * C + r)
    gx = tl.full((C, DK), 0., tl.float32)
    last = tl.minimum(local, LAGS)
    # Reverse the forward right-to-left scan, starting with the oldest block.
    for step in range(last):
        lag = last - step
        j = block - lag
        tj = start - lag * C + r
        jt = (bh * NT + j) * C
        ho = ((bh * NT + block) * LAGS + lag - 1) * C * D
        x = tl.load(History + ho + r[:, None] * D + d[None, :], d[None, :] < D, 0)
        ko = jt * D + r[:, None] * D + d[None, :]
        kr = tl.load(Kr + ko, d[None, :] < D, 0)
        mask = (lag * C + r[:, None] - r[None, :] < W) & (ti[:, None] < end)
        logits = _dot(x, tl.trans(kr), HIGH) * SCALE
        p = tl.where(mask, tl.exp(logits - lse[:, None]), 0.)
        v = tl.load(V + ((b * N + tj[:, None]) * H + h) * E + e[None, :],
                    (tj[:, None] < end) & (e[None, :] < E), 0).to(tl.float32)
        dp = _dot(do, tl.trans(v), HIGH)
        ds = tl.where(mask, p * (dp - delta[:, None]) * SCALE, 0.)
        gv = _dot(tl.trans(p), do, HIGH)
        tl.atomic_add(GV + jt * E + r[:, None] * E + e[None, :], gv,
                      e[None, :] < E, sem="relaxed")
        gkr = _dot(tl.trans(ds), x, HIGH)
        tl.atomic_add(GKr + ko, gkr, d[None, :] < D, sem="relaxed")
        if step > 0:
            k = tl.load(K + ((b * N + tj[:, None]) * H + h) * D + d[None, :],
                        (tj[:, None] < end) & (d[None, :] < D), 0)
            u = tl.load(U + ko, d[None, :] < D, 0)
            xk = _dot(x, tl.trans(k), HIGH)
            gu = -_dot(tl.trans(xk), gx, HIGH)
            g_xk = -_dot(gx, tl.trans(u), HIGH)
            gk = _dot(tl.trans(g_xk), x, HIGH)
            tl.atomic_add(GU + ko, gu, d[None, :] < D, sem="relaxed")
            tl.atomic_add(GK + ko, gk, d[None, :] < D, sem="relaxed")
            gx += _dot(g_xk, k, True)
        gx += _dot(ds, kr, True)
    tile = (bh * NT + block) * C
    tl.store(GQl + tile * D + r[:, None] * D + d[None, :], gx, d[None, :] < D)


@triton.jit(do_not_specialize=["NT"])
def _backward_prepare(Q, K, V, Beta, R, LSE, Delta, DO, GQl, GU, GKr, GK, GV, GB, DQ, DKOut, DVOut, DB,
                      N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                      NT, C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                      W: tl.constexpr, SCALE: tl.constexpr, HIGH: tl.constexpr,
                      Cu=None, ChunkIndices=None):
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    r, d, e = tl.arange(0, C), tl.arange(0, DK), tl.arange(0, DV)
    start, end, local = _chunk_bounds(block, Cu, ChunkIndices, N, C)
    t = start + r
    qoff = ((b * N + t[:, None]) * H + h) * D + d[None, :]
    voff = ((b * N + t[:, None]) * H + h) * E + e[None, :]
    qvalid = (t[:, None] < end) & (d[None, :] < D)
    vvalid = (t[:, None] < end) & (e[None, :] < E)
    q = tl.load(Q + qoff, qvalid, 0).to(tl.float32)
    k = tl.load(K + qoff, qvalid, 0).to(tl.float32)
    v = tl.load(V + voff, vvalid, 0).to(tl.float32)
    beta = tl.load(Beta + bh * N + t, t < end, 0)
    do = tl.load(DO + voff, vvalid, 0).to(tl.float32)
    tile = (bh * NT + block) * C
    off = tile * D + r[:, None] * D + d[None, :]
    inv = tl.load(R + tile * C + r[:, None] * C + r[None, :])
    gql = tl.load(GQl + off, d[None, :] < D, 0)
    gu = tl.load(GU + off, d[None, :] < D, 0)
    gkr = tl.load(GKr + off, d[None, :] < D, 0)
    gk = tl.load(GK + off, d[None, :] < D, 0)
    gb = tl.load(GB + tile + r)
    lower = r[:, None] >= r[None, :]
    strict = r[:, None] > r[None, :]
    mask = lower & (r[:, None] - r[None, :] < W)
    base = tl.where(lower, _dot(q, tl.trans(k), HIGH), 0.)
    a = tl.where(lower, _dot(base, inv, True), 0.)
    bk = beta[:, None] * k
    lse = tl.load(LSE + tile + r)
    delta = tl.load(Delta + tile + r)
    mask = mask & (t[:, None] < end)
    p = tl.where(mask, tl.exp(a * SCALE - lse[:, None]), 0.)
    dp = _dot(do, tl.trans(v), HIGH)
    ga = tl.where(mask, p * (dp - delta[:, None]) * SCALE, 0.)
    gv = _dot(tl.trans(p), do, HIGH)
    gv += tl.load(GV + tile * E + r[:, None] * E + e[None, :], e[None, :] < E, 0)
    tl.store(DVOut + voff, gv, vvalid)

    # Ql=Q-A beta K; U=R beta K; Kr=R^T K; A=lower(QK^T)R.
    # In the query adjoint, large terms can cancel (e.g. repeated keys with
    # beta=1/2). Keep this short solve/readout chain in TF32x3, rather than
    # letting independent BF16 roundings leave a spurious query gradient.
    ga = tl.where(lower, ga - _dot(gql, tl.trans(bk), True), 0.)
    gbk = -_dot(tl.trans(a), gql, True) + _dot(tl.trans(inv), gu, True)
    gr = (_dot(tl.trans(base), ga, True) + _dot(gu, tl.trans(bk), True)
          + _dot(k, tl.trans(gkr), True))
    gbase = tl.where(lower, _dot(ga, tl.trans(inv), True), 0.)
    gq = gql + _dot(gbase, k, True)
    gk += _dot(tl.trans(gbase), q, True) + _dot(inv, gkr, True) + beta[:, None] * gbk
    gb += tl.sum(gbk * k, 1)

    # The inverse and its adjoint are the precision-sensitive part.
    gl = -tl.dot(tl.dot(tl.trans(inv), gr, input_precision="tf32x3"),
                 tl.trans(inv), input_precision="tf32x3")
    gl = tl.where(strict, gl, 0.)
    s = tl.where(strict, _dot(k, tl.trans(k), True), 0.)
    gb += tl.sum(gl * s, 1)
    gs = beta[:, None] * gl
    gk += _dot(gs + tl.trans(gs), k, True)
    tl.store(DQ + qoff, gq, qvalid)
    tl.store(DKOut + qoff, gk, qvalid)
    tl.store(DB + bh * N + t, gb, t < end)


class _SlidingPath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, window_size, chunk_size, scale, precision, save, metadata):
        cfg = _config(q, v, window_size, chunk_size, scale, False, precision, metadata)
        cfg.pop('WRITE_BETA')
        meta = {} if metadata is None else dict(Cu=metadata.cu_seqlens,
                                                ChunkIndices=metadata.chunk_indices)
        grid = (cfg['NT'], q.shape[0] * cfg['H'])
        with torch.cuda.device(q.device):
            inv, u, kr, ql, history = _allocate(q, cfg, save)
            out = torch.empty_like(v)
            lse = torch.empty(u.shape[:-1] if save else (0,), device=q.device, dtype=torch.float32)
            prep = {key: cfg[key] for key in ('N', 'H', 'D', 'NT', 'C', 'DK', 'HIGH')}
            prep['HIGH'] = True
            _prepare[grid](q, k, beta, inv, u, kr, ql, **prep, **meta, num_warps=4, num_stages=1)
            _forward[grid](q, k, v, beta, inv, u, kr, ql, history, out, lse,
                           **cfg, **meta, SAVE=save, num_warps=4, num_stages=1)
        if save:
            ctx.save_for_backward(q, k, v, beta, inv, u, kr, history, lse, *meta.values())
            ctx.cfg = cfg
            ctx.has_varlen = metadata is not None
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        q, k, v, beta, inv, u, kr, history, lse = ctx.saved_tensors[:9]
        meta = {} if not ctx.has_varlen else dict(Cu=ctx.saved_tensors[9],
                                                 ChunkIndices=ctx.saved_tensors[10])
        cfg = ctx.cfg
        grid = (cfg['NT'], q.shape[0] * cfg['H'])
        with torch.cuda.device(q.device):
            do = do.contiguous()
            kw = dict(device=q.device, dtype=torch.float32)
            delta = torch.empty_like(lse)
            gql = torch.empty(u.shape, **kw)
            gu, gkr, gk = [torch.zeros(u.shape, **kw) for _ in range(3)]
            gv = torch.zeros((*u.shape[:-1], cfg['E']), **kw)
            gb = torch.zeros(u.shape[:-1], **kw)
            dq, dk, dv, db = (torch.empty_like(x) for x in (q, k, v, beta))
            _softmax_delta[grid](q, k, v, inv, kr, history, lse, do, delta,
                                 **cfg, **meta, num_warps=4, num_stages=1)
            _backward_scan[grid](k, v, beta, u, kr, history, lse, delta, do,
                                 gql, gu, gkr, gk, gv, gb,
                                 **cfg, **meta, num_warps=4, num_stages=1)
            prep_cfg = {key: value for key, value in cfg.items() if key != 'LAGS'}
            _backward_prepare[grid](q, k, v, beta, inv, lse, delta, do,
                                    gql, gu, gkr, gk, gv, gb, dq, dk, dv, db,
                                    **prep_cfg, **meta, num_warps=4, num_stages=1)
        return dq, dk, dv, db, None, None, None, None, None, None


def sliding_window_path(q, k, v, beta, window_size, *, chunk_size=32,
                        scale=1.0, matmul_precision="bf16", cu_seqlens=None):
    """Return sliding-window softmax PaTH with native analytical backward.

    Q/K: BF16 [B,N,H,Dk], V: BF16 [B,N,H,Dv], beta: FP32 [B,H,N].
    The softmax is over EXACTLY max(doc_start,i-w+1) <= j <= i, including
    the diagonal with no reflector applied. No beta-weighted value writes.
    scale multiplies logits, not the output. No implicit Q/K normalization.

    cu_seqlens: CPU or same-device CUDA int32/int64 document offsets, start
    0, end N, nondecreasing; requires B=1. Empty documents are supported but
    total N must be positive. Noncontiguous inputs are copied with autograd.
    """
    if q.ndim != 4 or min(q.shape) < 1 or k.shape != q.shape:
        raise ValueError("Q/K must have identical nonempty [B,N,H,Dk] shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3] or v.shape[-1] < 1:
        raise ValueError("V must have shape [B,N,H,Dv]")
    b, n, h, _ = q.shape
    if beta.shape != (b, h, n):
        raise ValueError("beta must have shape [B,H,N]")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)) or beta.dtype != torch.float32:
        raise TypeError("Q/K/V must be BF16 and beta must be FP32")
    if q.device.type != "cuda" or any(x.device != q.device for x in (k, v, beta)):
        raise ValueError("All inputs must be on the same CUDA device")
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        raise ValueError("BF16/TF32x3 kernels require CUDA capability >= 8.0")
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size not in (16, 32, 64):
        raise ValueError("chunk_size must be 16, 32, or 64")
    if matmul_precision not in ("bf16", "tf32x3"):
        raise ValueError("matmul_precision must be 'bf16' or 'tf32x3'")
    if not isinstance(scale, (float, int)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite Python number")
    metadata = prepare_varlen_metadata(cu_seqlens, batch=b, total_tokens=n,
                                       chunk_size=chunk_size, device=q.device)
    save = torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, beta))
    return _SlidingPath.apply(*(x.contiguous() for x in (q, k, v, beta)), window_size,
                              chunk_size, float(scale), matmul_precision, save, metadata)
