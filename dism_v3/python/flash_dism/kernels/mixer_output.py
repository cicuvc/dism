"""Two-kernel sigmoid low-rank gate and per-head norm output.

Only the BF16 autocast path is dispatched here. Projection operands/results and
sigmoid output retain the BF16 roundings of the explicit PyTorch implementation;
norm statistics and its affine product use FP32. Parameter gradients preserve
projection BF16-to-parameter casts. No shared-memory atomics are used.
"""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
import triton.testing


def _bench(call, quantiles):
    # All candidates overwrite their complete output, so trials need no reset.
    return triton.testing.do_bench_cudagraph(call, rep=20, quantiles=quantiles)


_DOWN_CONFIGS = [
    triton.Config({'BM': 32, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 16, 'BK': 32}, num_warps=4, num_stages=2),
    triton.Config({'BM': 32, 'BK': 64}, num_warps=4, num_stages=2),
    triton.Config({'BM': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 32, 'BK': 64}, num_warps=8, num_stages=3),
]
_UP_CONFIGS = [
    triton.Config({'BM': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 16}, num_warps=4, num_stages=2),
    triton.Config({'BM': 64}, num_warps=4, num_stages=2),
    triton.Config({'BM': 32}, num_warps=8, num_stages=2),
]
_NORM_BACKWARD_CONFIGS = [
    # Keep the physical partial-gradient layout invariant across trials.
    triton.Config({'BM': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 32}, num_warps=8, num_stages=3),
]


def _autotune_key(kernel, args, kwargs):
    """Mirror Triton's key (shape keys followed by tensor dtypes)."""
    values = {**dict(zip(kernel.arg_names, args)), **kwargs}
    values = {name: value for name, value in values.items() if name in kernel.arg_names}
    return tuple([values[name] for name in kernel.keys if name in values]
                 + [str(value.dtype) for value in values.values() if hasattr(value, 'dtype')])


def _launch(kernel, grid, *args, **kwargs):
    # Eager warmup tunes and populates the normal Triton cache. Capture must
    # never trigger benchmarking/synchronization. A genuinely cold capture
    # uses the original fixed config without claiming it is a tuned winner.
    if torch.cuda.is_current_stream_capturing():
        config = kernel.cache.get(_autotune_key(kernel, args, kwargs), kernel.configs[0])
        return kernel.fn[grid](*args, **kwargs, **config.all_kwargs())
    return kernel[grid](*args, **kwargs)


@triton.autotune(configs=_DOWN_CONFIGS, key=['M', 'WIDTH', 'RANK', 'HAS_BIAS'],
                 do_bench=_bench, cache_results=True)
@triton.jit
def _down_sigmoid(X, W, B, G, M: tl.constexpr, WIDTH: tl.constexpr,
                  RANK: tl.constexpr, HAS_BIAS: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(WIDTH, BK)):
        k = block * BK + kk
        a = tl.load(X + rows[:, None] * WIDTH + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < WIDTH), 0)
        b = tl.load(W + cols[None, :] * WIDTH + k[:, None],
                    (cols[None, :] < RANK) & (k[:, None] < WIDTH), 0).to(tl.bfloat16)
        acc += tl.dot(a, b)
    if HAS_BIAS:
        bias = tl.load(B + cols, cols < RANK, 0).to(tl.bfloat16).to(tl.float32)
        acc += bias[None, :]
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    g = (1.0 / (1.0 + tl.exp(-rounded))).to(tl.bfloat16)
    tl.store(G + rows[:, None] * RANK + cols[None, :], g,
             (rows[:, None] < M) & (cols[None, :] < RANK))


@triton.autotune(configs=_UP_CONFIGS, key=['M', 'H', 'D', 'RANK', 'CENTER', 'HAS_BIAS'],
                 do_bench=_bench, cache_results=True)
@triton.jit
def _up_norm_gate(G, W, B, X, NW, P, Y, M: tl.constexpr, H: tl.constexpr,
                  D: tl.constexpr, RANK: tl.constexpr, EPS: tl.constexpr,
                  SCALE: tl.constexpr, CENTER: tl.constexpr, HAS_BIAS: tl.constexpr,
                  BM: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    dd = tl.arange(0, BD)
    rr = tl.arange(0, BR)
    a = tl.load(G + rows[:, None] * RANK + rr[None, :],
                (rows[:, None] < M) & (rr[None, :] < RANK), 0)
    b = tl.load(W + (head * D + dd[None, :]) * RANK + rr[:, None],
                (dd[None, :] < D) & (rr[:, None] < RANK), 0).to(tl.bfloat16)
    projected = tl.dot(a, b)
    if HAS_BIAS:
        bias = tl.load(B + head * D + dd, dd < D, 0).to(tl.bfloat16).to(tl.float32)
        projected += bias[None, :]
    projected = projected.to(tl.bfloat16)
    offsets = (rows[:, None] * H + head) * D + dd[None, :]
    mask = (rows[:, None] < M) & (dd[None, :] < D)
    raw = tl.load(X + offsets, mask, 0).to(tl.float32)
    if CENTER:
        mean = tl.sum(raw, 1) / D
        raw = tl.where(dd[None, :] < D, raw - mean[:, None], 0.)
    rstd = tl.rsqrt(tl.sum(raw * raw, 1) / D + EPS)
    norm = raw * rstd[:, None]
    weight = tl.load(NW + dd, dd < D, 0).to(tl.float32)
    out = (norm * weight[None, :]) * (projected.to(tl.float32) * SCALE)
    tl.store(P + offsets, projected, mask)
    tl.store(Y + offsets, out.to(tl.bfloat16), mask)


@triton.autotune(configs=_NORM_BACKWARD_CONFIGS, key=['ROWS', 'D', 'CENTER'],
                 do_bench=_bench, cache_results=True)
@triton.jit
def _norm_gate_backward(X, P, NW, DY, DX, DP, PARTIAL,
                        ROWS: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr,
                        SCALE: tl.constexpr, CENTER: tl.constexpr,
                        BM: tl.constexpr, BD: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dd = tl.arange(0, BD)
    offsets = rows[:, None] * D + dd[None, :]
    mask = (rows[:, None] < ROWS) & (dd[None, :] < D)
    value = tl.load(X + offsets, mask, 0).to(tl.float32)
    projected = tl.load(P + offsets, mask, 0).to(tl.float32)
    dy = tl.load(DY + offsets, mask, 0).to(tl.float32)
    weight = tl.load(NW + dd, dd < D, 0).to(tl.float32)
    if CENTER:
        mean = tl.sum(value, 1) / D
        value = tl.where(dd[None, :] < D, value - mean[:, None], 0.)
    inv = tl.rsqrt(tl.sum(value * value, 1) / D + EPS)
    norm = value * inv[:, None]
    dn = dy * (projected * SCALE) * weight[None, :]
    dx = dn - norm * (tl.sum(dn * norm, 1) / D)[:, None]
    if CENTER:
        dx -= (tl.sum(dn, 1) / D)[:, None]
    dx *= inv[:, None]
    dp = ((dy * (norm * weight[None, :])) * SCALE).to(tl.bfloat16)
    dw = tl.sum(dy * (projected * SCALE) * norm, 0)
    tl.store(DX + offsets, dx.to(tl.bfloat16), mask)
    tl.store(DP + offsets, dp, mask)
    tl.store(PARTIAL + tl.program_id(0) * D + dd, dw, dd < D)


@triton.jit
def _reduce_norm_weight(PARTIAL, DW, BLOCKS: tl.constexpr, D: tl.constexpr,
                        BR: tl.constexpr, BD: tl.constexpr):
    dd = tl.program_id(0) * BD + tl.arange(0, BD)
    rr = tl.arange(0, BR)
    acc = tl.full((BR, BD), 0, tl.float32)
    for base in range(tl.cdiv(BLOCKS, BR)):
        rows = base * BR + rr
        acc += tl.load(PARTIAL + rows[:, None] * D + dd[None, :],
                       (rows[:, None] < BLOCKS) & (dd[None, :] < D), 0)
    tl.store(DW + dd, tl.sum(acc, 0), dd < D)


@triton.jit
def _sigmoid_backward(G, DG, DZ, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    dg = tl.load(DG + i, i < N, 0).to(tl.float32)
    # Match the eager sigmoid backward's rounded output operand.
    dz = dg * (1.0 - g) * g
    tl.store(DZ + i, dz.to(tl.bfloat16), i < N)


class _MixerOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, raw, down_weight, down_bias, up_weight, up_bias,
                norm_weight, scale, eps, center):
        width, rank = x.shape[-1], down_weight.shape[0]
        heads, dim = raw.shape[-2:]
        m = x.numel() // width
        g = torch.empty((m, rank), device=x.device, dtype=torch.bfloat16)
        projected = torch.empty_like(raw)
        output = torch.empty_like(raw)
        _launch(_down_sigmoid, lambda meta: (triton.cdiv(m, meta['BM']),),
            x, down_weight, down_bias, g, m, width, rank, down_bias is not None,
            BN=max(16, triton.next_power_of_2(rank)), enable_fp_fusion=False)
        _launch(_up_norm_gate, lambda meta: (triton.cdiv(m, meta['BM']), heads),
            g, up_weight, up_bias, raw, norm_weight, projected, output,
            m, heads, dim, rank, eps, scale, center, up_bias is not None,
            BD=max(16, triton.next_power_of_2(dim)), BR=max(16, triton.next_power_of_2(rank)),
            enable_fp_fusion=False)
        ctx.save_for_backward(x, raw, down_weight, up_weight, norm_weight, g, projected)
        ctx.down_bias = None if down_bias is None else down_bias.dtype
        ctx.up_bias = None if up_bias is None else up_bias.dtype
        ctx.scale, ctx.eps, ctx.center = scale, eps, center
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, raw, down_weight, up_weight, norm_weight, g, projected = ctx.saved_tensors
        width, rank = x.shape[-1], down_weight.shape[0]
        dim = raw.shape[-1]
        m = g.shape[0]
        rows = raw.numel() // dim
        grad_output = grad_output.contiguous()
        draw = torch.empty_like(raw)
        dp = torch.empty_like(raw)
        partial = torch.empty((triton.cdiv(rows, 32), dim), device=x.device, dtype=torch.float32)
        dnorm = torch.empty_like(norm_weight)
        _launch(_norm_gate_backward, lambda meta: (triton.cdiv(rows, meta['BM']),),
            raw, projected, norm_weight, grad_output, draw, dp, partial,
            rows, dim, ctx.eps, ctx.scale, ctx.center,
            BD=triton.next_power_of_2(dim), enable_fp_fusion=False)
        _reduce_norm_weight[(triton.cdiv(dim, 32),)](
            partial, dnorm, partial.shape[0], dim, 128, 32, num_warps=4)
        with torch.autocast('cuda', enabled=False):
            dp2 = dp.view(m, -1)
            wu = up_weight.to(torch.bfloat16)
            wd = down_weight.to(torch.bfloat16)
            dg = dp2 @ wu
            dwu = (dp2.T @ g).to(up_weight.dtype)
            dbu = None if ctx.up_bias is None else dp2.sum(0).to(ctx.up_bias)
            dz = torch.empty_like(g)
            _sigmoid_backward[(triton.cdiv(g.numel(), 256),)](
                g, dg, dz, g.numel(), 256, enable_fp_fusion=False)
            dx = (dz @ wd).view_as(x)
            dwd = (dz.T @ x.view(m, width)).to(down_weight.dtype)
            dbd = None if ctx.down_bias is None else dz.sum(0).to(ctx.down_bias)
        return dx, draw, dwd, dbd, dwu, dbu, dnorm, None, None, None


def supported(x, raw, down_weight, down_bias, up_weight, up_bias, norm_weight):
    if not x.is_cuda or x.dtype != torch.bfloat16 or raw.dtype != torch.bfloat16:
        return False
    if x.ndim < 2 or raw.ndim != x.ndim + 1 or raw.shape[:-2] != x.shape[:-1]:
        return False
    if down_weight.ndim != 2 or up_weight.ndim != 2 or norm_weight.ndim != 1:
        return False
    if not (0 < down_weight.shape[0] <= 128 and 0 < raw.shape[-1] <= 128):
        return False
    if x.numel() == 0 or raw.numel() == 0 or down_weight.shape[1] != x.shape[-1]:
        return False
    if up_weight.shape != (raw.shape[-2] * raw.shape[-1], down_weight.shape[0]):
        return False
    if norm_weight.shape != (raw.shape[-1],):
        return False
    tensors = [x, raw, down_weight, up_weight, norm_weight]
    if down_bias is not None:
        if down_bias.shape != (down_weight.shape[0],):
            return False
        tensors.append(down_bias)
    if up_bias is not None:
        if up_bias.shape != (up_weight.shape[0],):
            return False
        tensors.append(up_bias)
    if not all(t.is_contiguous() and t.device == x.device for t in tensors):
        return False
    if not all(t.dtype in (torch.float32, torch.bfloat16) for t in tensors[2:]):
        return False
    if torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') != torch.bfloat16:
        return False
    # FP32 linear parameters require the same BF16 autocast as the model.
    linear_tensors = [down_weight, up_weight] + [t for t in (down_bias, up_bias) if t is not None]
    if any(t.dtype != torch.bfloat16 for t in linear_tensors):
        if not torch.is_autocast_enabled('cuda') or torch.get_autocast_dtype('cuda') != torch.bfloat16:
            return False
    return True


def fused_mixer_output(x, raw, down_weight, down_bias, up_weight, up_bias,
                       norm_weight, scale, eps, center=True):
    """First-order autograd only; unsupported layouts must use the explicit path."""
    if not supported(x, raw, down_weight, down_bias, up_weight, up_bias, norm_weight):
        raise ValueError('fused mixer output requires contiguous CUDA BF16 operands, rank/norm dimension <=128')
    return _MixerOutput.apply(x, raw, down_weight, down_bias, up_weight, up_bias,
                              norm_weight, float(scale), float(eps), bool(center))
