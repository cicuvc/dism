"""Fixed true direction: independent Q LSE/labels and K->Q interpolation."""
import torch
import triton
import triton.language as tl
import cu_flash_dism as cu


@triton.jit
def _forward(X, Key, Value, Out, LSE, Top, Label, Tau, beta,
             N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, V: tl.constexpr,
             XB: tl.constexpr, XN: tl.constexpr, XH: tl.constexpr,
             KH: tl.constexpr, KV: tl.constexpr, VH: tl.constexpr, VV: tl.constexpr,
             HAS_TAU: tl.constexpr, LSE_ONLY: tl.constexpr,
             M: tl.constexpr = 64, K: tl.constexpr = 64):
    bh, tile = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    row = tile*M + tl.arange(0, M)
    d = tl.arange(0, D)
    x = tl.load(X + batch*XB + row[:, None]*XN + head*XH + d[None, :], row[:, None] < N, 0)
    maximum = tl.full((M, 1), float('-inf'), tl.float32)
    denominator = tl.full((M, 1), 1., tl.float32)
    label = tl.zeros((M, 1), tl.int32)
    if not LSE_ONLY:
        acc = tl.zeros((M, D), tl.float32)
    for start in range(tl.cdiv(V, K)):
        col = start*K + tl.arange(0, K)
        key = tl.load(Key + head*KH + col[:, None]*KV + d[None, :], col[:, None] < V, 0)
        score = tl.dot(x, tl.trans(key)) * (beta * 1.4426950408889634)
        # Match the original interpolation's vocabulary-tail convention.
        score = tl.where(col[None, :] < V, score, -1e5)
        local_max, local_idx = tl.max(score, 1, return_indices=True, keep_dims=True)
        next_max = tl.maximum(maximum, local_max)
        label = tl.where(local_max > maximum, local_idx + start*K, label)
        p = tl.exp2(score - next_max)
        correction = tl.exp2(maximum - next_max)
        if not LSE_ONLY:
            value = tl.load(Value + head*VH + col[:, None]*VV + d[None, :], col[:, None] < V, 0)
            acc = acc*correction + tl.dot(p.to(value.dtype), value)
        denominator = denominator*correction + tl.sum(p, 1, keep_dims=True)
        maximum = next_max
    if not LSE_ONLY:
        tl.store(Out + ((batch*N + row[:, None])*H + head)*D + d[None, :],
                 acc/denominator, row[:, None] < N)
    offset = bh*N + row[:, None]
    tau = tl.load(Tau + head) if HAS_TAU else 0.
    tl.store(LSE + offset, (tl.log2(denominator) + maximum)*0.6931471805599453 - tau, row[:, None] < N)
    tl.store(Top + offset, 1./denominator, row[:, None] < N)
    tl.store(Label + offset, label, row[:, None] < N)


def emb_fwd_true(q, k, eq, ek, sm_scale=1., tau=None):
    """Same eight-field ordering as emb_fwd_wrapper; unused k_from_q is empty."""
    cu.validate_interpolation(q, k, eq, ek, tau)
    b, n, h, d = q.shape
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    unused = torch.empty(0, device=q.device, dtype=q.dtype)
    lk, lq, tk, tq = [torch.empty((b, h, n), device=q.device, dtype=torch.float32) for _ in range(4)]
    ik, iq = [torch.empty((b, h, n), device=q.device, dtype=torch.int32) for _ in range(2)]
    for x, key, value, lse, top, idx, lse_only in (
            (q, eq, ek, lq, tq, iq, True), (k, ek, eq, lk, tk, ik, False)):
        _forward[(b*h, triton.cdiv(n, 64))](
            x, key, value, out, lse, top, idx, tau, float(sm_scale), n, h, d, eq.shape[1],
            *x.stride()[:3], *key.stride()[:2], *value.stride()[:2],
            tau is not None, lse_only, num_warps=4, num_stages=3)
    return out, unused, lk, lq, tk, tq, ik, iq


@triton.jit
def delta_true(Out, DO, Delta, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
               B: tl.constexpr, M: tl.constexpr = 64):
    bh = tl.program_id(1)
    row = tl.program_id(0)*M + tl.arange(0, M)
    d = tl.arange(0, D)
    offset = (((bh//H)*N + row[:, None])*H + bh%H)*D + d[None, :]
    out = tl.load(Out + offset, row[:, None] < N, 0).to(tl.float32)
    do = tl.load(DO + offset, row[:, None] < N, 0).to(tl.float32)
    tl.store(Delta + bh*N + row, tl.sum(out*do, 1), row < N)
