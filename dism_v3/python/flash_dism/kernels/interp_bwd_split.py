"""FA2-style interpolation backward with global FP32 atomic reduction.

Autotune is process/device/shape/stride specific. Warm up before CUDA Graph
capture. Used by the production backward; no partial-gradient workspace.
"""
import torch
import triton
import triton.language as tl
import cu_flash_dism as cu

from flash_dism.emb_kernel import _interp_bwd_preprocess


def _bench(fn, quantiles):
    return triton.testing.do_bench(fn, warmup=5, rep=20, quantiles=quantiles)


def _prune_splits(configs, named_args, **kwargs):
    # Avoid measuring more partitions than there are token tiles.
    return [c for c in configs if c.kwargs['SPLITS'] <=
            triton.cdiv(named_args['B'] * named_args['N'], c.kwargs['M'])]


_TUNE_KEY = ['B', 'N', 'H', 'V', 'D', 'HAS_TAU', 'DEVICE', 'LSE_ONLY',
             'xb', 'xn', 'xh', 'kh', 'kv', 'vh', 'vv', 'ob', 'on', 'oh']


@triton.autotune(
    configs=[triton.Config({'M': m, 'K': k}, num_warps=4, num_stages=s)
             for m, k, s in [(32, 32, 3), (32, 64, 3), (64, 32, 3),
                             (64, 64, 3), (64, 64, 2)]],
    key=_TUNE_KEY, do_bench=_bench)
@triton.jit
def _dq(X, Key, Value, DO, LSE, DLSE, Delta, DX, Tau,
        B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, V: tl.constexpr,
        D: tl.constexpr, beta, HAS_TAU: tl.constexpr, DEVICE: tl.constexpr,
        xb: tl.constexpr, xn: tl.constexpr, xh: tl.constexpr,
        kh: tl.constexpr, kv: tl.constexpr, vh: tl.constexpr, vv: tl.constexpr,
        ob: tl.constexpr, on: tl.constexpr, oh: tl.constexpr,
        M: tl.constexpr, K: tl.constexpr, LSE_ONLY: tl.constexpr = False):
    row = tl.program_id(0) * M + tl.arange(0, M)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    d = tl.arange(0, D)
    x = tl.load(X + batch * xb + row[:, None] * xn + head * xh + d[None, :],
                row[:, None] < N, 0)
    if not LSE_ONLY:
        do = tl.load(DO + batch * ob + row[:, None] * on + head * oh + d[None, :],
                     row[:, None] < N, 0).to(Key.dtype.element_ty)
    offset = bh * N + row
    lse = tl.load(LSE + offset, row < N, 0)
    if HAS_TAU:
        lse += tl.load(Tau + head)
    lse *= 1.4426950408889634
    dlse = tl.load(DLSE + offset, row < N, 0)
    if not LSE_ONLY:
        delta = tl.load(Delta + offset, row < N, 0)
    acc = tl.zeros((M, D), tl.float32)
    for start in range(tl.cdiv(V, K)):
        col = start * K + tl.arange(0, K)
        key = tl.load(Key + head * kh + col[:, None] * kv + d[None, :], col[:, None] < V, 0)
        if not LSE_ONLY:
            value = tl.load(Value + head * vh + col[:, None] * vv + d[None, :], col[:, None] < V, 0)
        score = tl.dot(x, tl.trans(key))
        score = tl.where((row[:, None] < N) & (col[None, :] < V), score, float('-inf'))
        p = tl.exp2(score * (beta * 1.4426950408889634) - lse[:, None])
        if LSE_ONLY:
            ds = p * dlse[:, None]
        else:
            dp = tl.dot(do, tl.trans(value))
            ds = p * (dp - delta[:, None]) + p * dlse[:, None]
        acc += tl.dot(ds.to(Key.dtype.element_ty), key) * beta
    tl.store(DX + ((batch * N + row[:, None]) * H + head) * D + d[None, :], acc, row[:, None] < N)


@triton.autotune(
    configs=[triton.Config({'M': m, 'K': k, 'SPLITS': splits},
                           num_warps=4, num_stages=3)
             for m, k in [(32, 64), (64, 32), (64, 64)]
             for splits in (1, 4, 8, 16, 32)],
    key=_TUNE_KEY, restore_value=['DK', 'DV'],
    prune_configs_by={'early_config_prune': _prune_splits}, do_bench=_bench)
@triton.jit
def _dkv(X, Key, Value, DO, LSE, DLSE, Delta, DK, DV, Tau,
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
        key_grad += tl.dot(tl.trans(ds).to(Key.dtype.element_ty), x) * beta
        if not LSE_ONLY:
            value_grad += tl.dot(tl.trans(p).to(Key.dtype.element_ty), do)
    offset = (head * V + col[:, None]) * D + d[None, :]
    # Global FP32 atomics only; zeroed outputs, no shared atomics.
    tl.atomic_add(DK + offset, key_grad, col[:, None] < V, sem='relaxed')
    if not LSE_ONLY:
        tl.atomic_add(DV + offset, value_grad, col[:, None] < V, sem='relaxed')


def emb_bwd_split(q, k, q_emb, k_emb, out_q, out_k, olse_q, olse_k,
                  doq, dok, dlq, dlk, sm_scale=1., tau=None, *, autotune=True,
                  splits=16, dq_m=64, dq_v=64, dkv_m=64, dkv_v=32,
                  num_warps=4, num_stages=3, direction_true=False,
                  fuse_dq_tma=False):
    """Implementation of emb_bwd_wrapper, with an explicit diagnostic override.

    FP32 reduction order differs for codebooks. No changed production tolerances,
    forward rounding, hard gradients, beta scaling, or tau gradient ownership.
    Atomic reduction is nondeterministic. Explicit launch parameters are used
    only with autotune=False, for controlled benchmarks and diagnostics.
    direction_true requires zero gradient for the unused Q->K output. That
    output and its dO/delta are not read; Q still contributes its LSE gradient.
    """
    cu.validate_interpolation(q, k, q_emb, k_emb, tau)
    b, n, h, d = q.shape
    vocab = q_emb.shape[1]
    if splits < 1:
        raise ValueError('splits must be positive')
    if dlq is None:
        dlq = torch.zeros((b, h, n), device=q.device, dtype=torch.float32)
    if dlk is None:
        dlk = torch.zeros((b, h, n), device=q.device, dtype=torch.float32)
    cu.validate_interpolation_grads(out_q, out_q if direction_true else out_k,
                                    olse_q, olse_k, doq, dok, dlq, dlk)
    doq, dok, dlq, dlk = [t.contiguous() for t in (doq, dok, dlq, dlk)]
    olse_q, olse_k = olse_q.contiguous(), olse_k.contiguous()
    dq, dk = [torch.empty(t.shape, device=t.device, dtype=torch.float32) for t in (q, k)]
    deq, dek = [torch.empty(t.shape, device=t.device, dtype=torch.float32) for t in (q_emb, k_emb)]
    delta_q = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
    delta_k = delta_q if direction_true else torch.empty_like(delta_q)
    if direction_true:
        from .interp_true import delta_true
        delta_true[(triton.cdiv(n, 64), b*h)](out_q.contiguous(), doq, delta_q, n, h, d, b)
    else:
        _interp_bwd_preprocess[(triton.cdiv(n, 64), b * h)](
            out_q, out_k, doq, dok, delta_q, delta_k, b, h, n,
            out_q.stride(0), out_q.stride(2), out_q.stride(1),
            doq.stride(0), doq.stride(2), doq.stride(1),
            dok.stride(0), dok.stride(2), dok.stride(1), h*n, n, d, 64)
    # Exactly once per invocation, before either attention. In particular, the
    # second attention must retain the first attention's contribution. Autotune
    # restore_value snapshots/restores BOTH outputs around every trial, instead
    # of reset_to_zero which would erase that contribution on a second cache miss.
    deq.zero_()
    dek.zero_()
    if fuse_dq_tma:
        if torch.cuda.get_device_capability(q.device)[0] < 9:
            raise ValueError('TMA interpolation backward requires SM90 or newer')
        from triton.tools.tensor_descriptor import TensorDescriptor
        from .interp_bwd_tma import _dkv_dq_tma
        dq.zero_()
        dk.zero_()
    for side, (x, key, value, do, lse, dlse, delta, dx, dkey, dvalue) in enumerate((
            (q, q_emb, k_emb, dok, olse_k, dlk, delta_k, dq, deq, dek),
            (k, k_emb, q_emb, doq, olse_q, dlq, delta_q, dk, dek, deq))):
        options = dict(LSE_ONLY=direction_true and side == 0)
        common = (b, n, h, vocab, d, float(sm_scale), tau is not None, q.device.index,
                  x.stride(0), x.stride(1), x.stride(2),
                  key.stride(0), key.stride(1), value.stride(0), value.stride(1),
                  do.stride(0), do.stride(1), do.stride(2))
        if fuse_dq_tma:
            descriptor = TensorDescriptor(dx, [b*n, h*d], [h*d, 1], [dkv_m, d])
            arguments = (x, key, value, do, lse, dlse, delta, dkey, dvalue,
                         dx, descriptor, tau, *common)
            if autotune:
                _dkv_dq_tma[lambda meta: (triton.cdiv(vocab, meta['K']), h, meta['SPLITS'])](
                    *arguments, **options)
            else:
                _dkv_dq_tma.fn[(triton.cdiv(vocab, dkv_v), h, splits)](
                    *arguments, dkv_m, dkv_v, splits, num_warps=num_warps,
                    num_stages=num_stages, **options)
        elif autotune:
            _dq[lambda meta: (triton.cdiv(n, meta['M']), b * h)](
                x, key, value, do, lse, dlse, delta, dx, tau, *common, **options)
            _dkv[lambda meta: (triton.cdiv(vocab, meta['K']), h, meta['SPLITS'])](
                x, key, value, do, lse, dlse, delta, dkey, dvalue, tau, *common, **options)
        else:
            _dq.fn[(triton.cdiv(n, dq_m), b * h)](
                x, key, value, do, lse, dlse, delta, dx, tau,
                *common, dq_m, dq_v, num_warps=num_warps, num_stages=num_stages, **options)
            _dkv.fn[(triton.cdiv(vocab, dkv_v), h, splits)](
                x, key, value, do, lse, dlse, delta, dkey, dvalue, tau,
                *common, dkv_m, dkv_v, splits, num_warps=num_warps, num_stages=num_stages, **options)
    return dq, dk, deq, dek
