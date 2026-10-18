"""Sliding-window DeltaNet: BF16 CUDA forward and analytical Triton backward.

Q/K: [B, N, H, Dk], V/O: [B, N, H, Dv], beta: FP32 [B, H, N].
Optional cu_seqlens [documents+1] enables native packed varlen training (B=1).
Each document has independent chunks and windows; no transform, attention,
or gradient crosses its boundaries. See sliding_delta_net_varlen.py for
metadata validation/cache semantics (first-use host sync; no graph capture).
The default is exactly the formula in sliding_delta_net.py, including the
current token and excluding j <= i-window_size. No decay, softmax, implicit
normalization, or implicit scaling. write_beta=True uses beta_j*v_j writes.

Algorithm references (independent implementation, no runtime FLA dependency):
  fla/ops/gated_delta_rule/{chunk.py,wy_fast.py}
  fla/ops/utils/solve_tril.py
  https://github.com/fla-org/flash-linear-attention
  https://arxiv.org/html/2505.16381v2#S3

For each block, S=strictLower(K K^T), R=(I+diag(beta) S)^-1,
U=R diag(beta) K, Kr=R^T K, A=lower(Q K^T) R, Ql=Q-A diag(beta) K.
The inter-block query update is X <- X - (X K^T) U. Masking scores, rather
than resetting states, gives exact sliding windows independent of block size.

The inverse uses FP32 substitution in 16x16 leaves and TF32x3 block merges;
its backward uses dL=-R^T dR R^T at the same precision. The cancellation-prone
query adjoint through the WY transform also uses TF32x3. Other products use
BF16 operands with FP32 accumulation by default, as in GDN.
FP32 transformed queries and gradient accumulators avoid rounding every sum.
``matmul_precision='tf32x3'`` is available for numerical diagnosis; the input
contract remains BF16/FP32. CUDA capability >= 8.0 is required.

Forward saves one BF16 query tile per interacting block pair for backward
(FP32 in diagnostic mode):
O(B H N ceil(w/C) Dk) storage, in addition to O(B H N (C+Dk)) WY buffers.
Inference does not allocate that query history. Backward uses FP32 atomics
to reduce contributions to shared K/V/beta and WY buffers; it is not bitwise
deterministic. Only first derivatives are supported. No Torch math/autograd
is used inside the kernels' forward/backward paths (only allocations/copies).
"""

import math

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .varlen import prepare_varlen_metadata


@triton.jit
def _dot(a, b, HIGH: tl.constexpr):
    if HIGH:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")
    return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))


@triton.jit
def _inverse16(m):
    r = tl.arange(0, 16)
    inv = (r[:, None] == r[None, :]).to(tl.float32)
    for row in tl.static_range(1, 16):
        values = tl.sum(tl.where(r[:, None] == row, m, 0.), 0)
        update = -tl.sum(values[:, None] * inv, 0)
        update = tl.where(r < row, update, tl.where(r == row, 1., 0.))
        inv = tl.where(r[:, None] == row, update[None, :], inv)
    return inv


@triton.jit
def _quadrants(m, HALF: tl.constexpr):
    left, right = tl.split(m.reshape(2 * HALF, 2, HALF).permute(0, 2, 1))
    a11, a21 = tl.split(left.reshape(2, HALF, HALF).permute(1, 2, 0))
    a12, a22 = tl.split(right.reshape(2, HALF, HALF).permute(1, 2, 0))
    return a11, a21, a22


@triton.jit
def _join_inverse(a11, a21, a22, HALF: tl.constexpr):
    zero = tl.full((HALF, HALF), 0., tl.float32)
    top = tl.join(a11, zero).permute(0, 2, 1).reshape(HALF, 2 * HALF)
    bottom = tl.join(a21, a22).permute(0, 2, 1).reshape(HALF, 2 * HALF)
    return tl.join(top, bottom).permute(2, 0, 1).reshape(2 * HALF, 2 * HALF)


@triton.jit
def _inverse32(m):
    a11, a21, a22 = _quadrants(m, 16)
    r11, r22 = _inverse16(a11), _inverse16(a22)
    r21 = -tl.dot(tl.dot(r22, a21, input_precision="tf32x3"), r11, input_precision="tf32x3")
    return _join_inverse(r11, r21, r22, 16)


@triton.jit
def _inverse_unit_lower(m, C: tl.constexpr):
    # FP32 leaf substitution avoids the catastrophic cancellation of a
    # nilpotent power-series inverse for repeated keys and beta near two.
    if C == 16:
        return _inverse16(m)
    elif C == 32:
        return _inverse32(m)
    else:
        a11, a21, a22 = _quadrants(m, 32)
        r11, r22 = _inverse32(a11), _inverse32(a22)
        r21 = -tl.dot(tl.dot(r22, a21, input_precision="tf32x3"), r11, input_precision="tf32x3")
        return _join_inverse(r11, r21, r22, 32)


@triton.jit
def _chunk_bounds(block, Cu, ChunkIndices, N: tl.constexpr, C: tl.constexpr):
    if Cu is None:
        return block * C, N, block
    else:
        doc = tl.load(ChunkIndices + block * 2)
        local = tl.load(ChunkIndices + block * 2 + 1)
        start = tl.load(Cu + doc)
        end = tl.load(Cu + doc + 1)
        return start + local * C, end, local


@triton.jit(do_not_specialize=["NT"])
def _prepare(Q, K, Beta, R, U, Kr, Ql,
             N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
             NT, C: tl.constexpr, DK: tl.constexpr, HIGH: tl.constexpr,
             Cu=None, ChunkIndices=None):
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    r, d = tl.arange(0, C), tl.arange(0, DK)
    start, end, local = _chunk_bounds(block, Cu, ChunkIndices, N, C)
    t = start + r
    offsets = ((b * N + t[:, None]) * H + h) * D + d[None, :]
    valid = (t[:, None] < end) & (d[None, :] < D)
    q = tl.load(Q + offsets, valid, 0).to(tl.float32)
    k = tl.load(K + offsets, valid, 0).to(tl.float32)
    beta = tl.load(Beta + bh * N + t, t < end, 0)
    strict = r[:, None] > r[None, :]
    lower = r[:, None] >= r[None, :]
    s = tl.where(strict, _dot(k, tl.trans(k), HIGH), 0.)
    inv = _inverse_unit_lower(beta[:, None] * s, C)
    bk = beta[:, None] * k
    u = _dot(inv, bk, HIGH)
    kr = _dot(tl.trans(inv), k, HIGH)
    base = tl.where(lower, _dot(q, tl.trans(k), HIGH), 0.)
    a = tl.where(lower, _dot(base, inv, HIGH), 0.)
    ql = q - _dot(a, bk, HIGH)
    tile = (bh * NT + block) * C
    tl.store(R + tile * C + r[:, None] * C + r[None, :], inv)
    off = tile * D + r[:, None] * D + d[None, :]
    tl.store(U + off, u, d[None, :] < D)
    tl.store(Kr + off, kr, d[None, :] < D)
    tl.store(Ql + off, ql, d[None, :] < D)


@triton.jit(do_not_specialize=["NT"])
def _forward(Q, K, V, Beta, R, U, Kr, Ql, History, O,
             N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
             NT, C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
             W: tl.constexpr, LAGS: tl.constexpr, SCALE: tl.constexpr,
             WRITE_BETA: tl.constexpr, SAVE: tl.constexpr, HIGH: tl.constexpr,
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
    a = _dot(base, inv, HIGH)
    mask = lower & (r[:, None] - r[None, :] < W)
    a = tl.where(mask, a, 0.)
    voff = ((b * N + ti[:, None]) * H + h) * E + e[None, :]
    v = tl.load(V + voff, (ti[:, None] < end) & (e[None, :] < E), 0).to(tl.float32)
    if WRITE_BETA:
        beta = tl.load(Beta + bh * N + ti, ti < end, 0)
        v *= beta[:, None]
    out = _dot(a, v, HIGH)
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
        score = tl.where(lag * C + r[:, None] - r[None, :] < W, score, 0.)
        v = tl.load(V + ((b * N + tj[:, None]) * H + h) * E + e[None, :],
                    (tj[:, None] < end) & (e[None, :] < E), 0).to(tl.float32)
        if WRITE_BETA:
            beta = tl.load(Beta + bh * N + tj, tj < end, 0)
            v *= beta[:, None]
        out += _dot(score, v, HIGH)
        if lag < last:
            k = tl.load(K + ((b * N + tj[:, None]) * H + h) * D + d[None, :],
                        (tj[:, None] < end) & (d[None, :] < D), 0)
            u = tl.load(U + jt * D + r[:, None] * D + d[None, :], d[None, :] < D, 0)
            x -= _dot(_dot(x, tl.trans(k), HIGH), u, HIGH)
    tl.store(O + voff, out * SCALE, (ti[:, None] < end) & (e[None, :] < E))


@triton.jit(do_not_specialize=["NT"])
def _backward_scan(K, V, Beta, U, Kr, History, DO, GQl, GU, GKr, GK, GV, GB,
                   N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                   NT, C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                   W: tl.constexpr, LAGS: tl.constexpr, SCALE: tl.constexpr,
                   WRITE_BETA: tl.constexpr, HIGH: tl.constexpr,
                   Cu=None, ChunkIndices=None):
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    r, d, e = tl.arange(0, C), tl.arange(0, DK), tl.arange(0, DV)
    start, end, local = _chunk_bounds(block, Cu, ChunkIndices, N, C)
    ti = start + r
    do = tl.load(DO + ((b * N + ti[:, None]) * H + h) * E + e[None, :],
                 (ti[:, None] < end) & (e[None, :] < E), 0).to(tl.float32) * SCALE
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
        score = tl.where(mask, _dot(x, tl.trans(kr), HIGH), 0.)
        v = tl.load(V + ((b * N + tj[:, None]) * H + h) * E + e[None, :],
                    (tj[:, None] < end) & (e[None, :] < E), 0).to(tl.float32)
        veff = v
        if WRITE_BETA:
            beta = tl.load(Beta + bh * N + tj, tj < end, 0)
            veff = v * beta[:, None]
        ds = tl.where(mask, _dot(do, tl.trans(veff), HIGH), 0.)
        gv = _dot(tl.trans(score), do, HIGH)
        if WRITE_BETA:
            tl.atomic_add(GB + jt + r, tl.sum(gv * v, 1), sem="relaxed")
            gv *= beta[:, None]
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
            gx += _dot(g_xk, k, HIGH)
        gx += _dot(ds, kr, HIGH)
    tile = (bh * NT + block) * C
    tl.store(GQl + tile * D + r[:, None] * D + d[None, :], gx, d[None, :] < D)


@triton.jit(do_not_specialize=["NT"])
def _backward_prepare(Q, K, V, Beta, R, DO, GQl, GU, GKr, GK, GV, GB, DQ, DKOut, DVOut, DB,
                      N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
                      NT, C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                      W: tl.constexpr, SCALE: tl.constexpr, WRITE_BETA: tl.constexpr, HIGH: tl.constexpr,
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
    do = tl.load(DO + voff, vvalid, 0).to(tl.float32) * SCALE
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
    a = tl.where(lower, _dot(base, inv, HIGH), 0.)
    bk = beta[:, None] * k
    veff = v
    if WRITE_BETA:
        veff = beta[:, None] * v
    ga = tl.where(mask, _dot(do, tl.trans(veff), HIGH), 0.)
    gv = _dot(tl.trans(tl.where(mask, a, 0.)), do, HIGH)
    if WRITE_BETA:
        gb += tl.sum(gv * v, 1)
        gv *= beta[:, None]
    gv += tl.load(GV + tile * E + r[:, None] * E + e[None, :], e[None, :] < E, 0)
    tl.store(DVOut + voff, gv, vvalid)

    # Ql=Q-A beta K; U=R beta K; Kr=R^T K; A=lower(QK^T)R.
    # In the query adjoint, large terms can cancel (e.g. repeated keys with
    # beta=1/2). Keep this short solve/readout chain in TF32x3, rather than
    # letting independent BF16 roundings leave a spurious query gradient.
    ga = tl.where(lower, ga - _dot(gql, tl.trans(bk), True), 0.)
    gbk = -_dot(tl.trans(a), gql, HIGH) + _dot(tl.trans(inv), gu, HIGH)
    gr = (_dot(tl.trans(base), ga, HIGH) + _dot(gu, tl.trans(bk), HIGH)
          + _dot(k, tl.trans(gkr), HIGH))
    gbase = tl.where(lower, _dot(ga, tl.trans(inv), True), 0.)
    gq = gql + _dot(gbase, k, True)
    gk += _dot(tl.trans(gbase), q, HIGH) + _dot(inv, gkr, HIGH) + beta[:, None] * gbk
    gb += tl.sum(gbk * k, 1)

    # The inverse and its adjoint are the precision-sensitive part.
    gl = -tl.dot(tl.dot(tl.trans(inv), gr, input_precision="tf32x3"),
                 tl.trans(inv), input_precision="tf32x3")
    gl = tl.where(strict, gl, 0.)
    s = tl.where(strict, _dot(k, tl.trans(k), HIGH), 0.)
    gb += tl.sum(gl * s, 1)
    gs = beta[:, None] * gl
    gk += _dot(gs + tl.trans(gs), k, HIGH)
    tl.store(DQ + qoff, gq, qvalid)
    tl.store(DKOut + qoff, gk, qvalid)
    tl.store(DB + bh * N + t, gb, t < end)


def _config(q, v, window_size, chunk_size, scale, write_beta, matmul_precision, metadata=None):
    b, n, h, d = q.shape
    e = v.shape[-1]
    nt = triton.cdiv(n, chunk_size) if metadata is None else metadata.chunks
    max_length = n if metadata is None else metadata.max_seqlen
    w = min(window_size, max_length)
    return dict(N=n, H=h, D=d, E=e, NT=nt, C=chunk_size,
                DK=max(16, triton.next_power_of_2(d)), DV=max(16, triton.next_power_of_2(e)),
                W=w, LAGS=min(triton.cdiv(max_length, chunk_size) - 1, triton.cdiv(w - 1, chunk_size)),
                SCALE=scale, WRITE_BETA=write_beta, HIGH=matmul_precision == "tf32x3")


def _allocate(q, cfg, save):
    b, _, h, d = q.shape
    nt, c, lags = cfg['NT'], cfg['C'], cfg['LAGS']
    kw = dict(device=q.device, dtype=torch.float32)
    dot_kw = dict(device=q.device, dtype=torch.float32 if cfg['HIGH'] else torch.bfloat16)
    inv = torch.empty((b*h, nt, c, c), **kw)
    u, kr = [torch.empty((b*h, nt, c, d), **dot_kw) for _ in range(2)]
    ql = torch.empty((b*h, nt, c, d), **kw)
    # These buffers are only consumed as dot operands, so storing them in
    # the dot's dtype introduces no additional arithmetic rounding.
    history = torch.empty((b*h, nt, lags, c, d) if save else (0,), **dot_kw)
    return inv, u, kr, ql, history


class _SlidingDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, window_size, chunk_size, scale, write_beta, matmul_precision, save, metadata):
        cfg = _config(q, v, window_size, chunk_size, scale, write_beta, matmul_precision, metadata)
        meta = {} if metadata is None else dict(Cu=metadata.cu_seqlens, ChunkIndices=metadata.chunk_indices)
        grid = (cfg['NT'], q.shape[0] * cfg['H'])
        with torch.cuda.device(q.device):
            inv, u, kr, ql, history = _allocate(q, cfg, save)
            out = torch.empty_like(v)
            prep = {key: cfg[key] for key in ('N', 'H', 'D', 'NT', 'C', 'DK', 'HIGH')}
            _prepare[grid](q, k, beta, inv, u, kr, ql, **prep, **meta, num_warps=4, num_stages=1)
            _forward[grid](q, k, v, beta, inv, u, kr, ql, history, out,
                           **cfg, **meta, SAVE=save, num_warps=4, num_stages=1)
        if save:
            ctx.save_for_backward(q, k, v, beta, inv, u, kr, history, *meta.values())
            ctx.cfg = cfg
            ctx.has_varlen = metadata is not None
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        q, k, v, beta, inv, u, kr, history = ctx.saved_tensors[:8]
        meta = {} if not ctx.has_varlen else dict(Cu=ctx.saved_tensors[8], ChunkIndices=ctx.saved_tensors[9])
        cfg = ctx.cfg
        grid = (cfg['NT'], q.shape[0] * cfg['H'])
        with torch.cuda.device(q.device):
            do = do.contiguous()
            shape = u.shape
            kw = dict(device=q.device, dtype=torch.float32)
            gql = torch.empty(shape, **kw)
            gu, gkr, gk = [torch.zeros(shape, **kw) for _ in range(3)]
            gv = torch.zeros((*shape[:-1], cfg['E']), **kw)
            gb = torch.zeros(shape[:-1], **kw)
            dq, dk, dv, db = (torch.empty_like(x) for x in (q, k, v, beta))
            _backward_scan[grid](k, v, beta, u, kr, history, do, gql, gu, gkr, gk, gv, gb,
                                 **cfg, **meta, num_warps=4, num_stages=1)
            prep_cfg = {key: value for key, value in cfg.items() if key != 'LAGS'}
            _backward_prepare[grid](q, k, v, beta, inv, do, gql, gu, gkr, gk, gv, gb,
                                    dq, dk, dv, db, **prep_cfg, **meta, num_warps=4, num_stages=1)
        return dq, dk, dv, db, None, None, None, None, None, None, None


def sliding_window_delta_net(q, k, v, beta, window_size, *, chunk_size=32,
                            scale=1.0, write_beta=False, matmul_precision="bf16", cu_seqlens=None):
    """BF16 sliding-window DeltaNet with first-order autograd.

    Q/K [B,N,H,Dk], V [B,N,H,Dv], beta FP32 [B,H,N]. Returns BF16 V shape.
    Inputs are copied to contiguous storage if needed, preserving gradients.
    Chunk size: 16/32/64. Channel dimensions need not be powers of two.
    scale defaults to 1; beta is already activated; normalization is external.
    The triangular inverse ALWAYS uses TF32x3, regardless of matmul_precision.

    cu_seqlens: optional CPU/CUDA int32/int64 [documents+1], starting at 0,
    ending at N, and nondecreasing. Requires B=1; empty documents are allowed.
    Windows restart at each document boundary; output retains the packed
    layout. First-use metadata validation synchronizes CUDA offsets to host;
    cached metadata is reused across layers. This varlen path does not support
    CUDA Graph capture. N must remain positive (also for packed inputs).
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
    if not isinstance(write_beta, bool):
        raise TypeError("write_beta must be bool")
    if not isinstance(scale, (float, int)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite Python number")
    metadata = prepare_varlen_metadata(cu_seqlens, batch=b, total_tokens=n,
                                       chunk_size=chunk_size, device=q.device)
    save = torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, beta))
    return _SlidingDelta.apply(*(x.contiguous() for x in (q, k, v, beta)), window_size,
                               chunk_size, float(scale), write_beta, matmul_precision, save, metadata)
