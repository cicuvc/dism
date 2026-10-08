import math

import torch
import triton
import triton.language as tl

if __package__:
    from .dynamo_utils import mark_cu_seqlens_dynamic
else:
    from dynamo_utils import mark_cu_seqlens_dynamic


def _qknorm_rope_configs():
    return [
        triton.Config(
            {"BLOCK_M": 8, "BLOCK_K": 32},
            num_warps=4, num_stages=3),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_K": 32},
            num_warps=4, num_stages=4),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 32},
            num_warps=8, num_stages=3),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_K": 64},
            num_warps=8, num_stages=3),
    ]


def _linear_bwd_configs():
    return [
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_C": 32, "BLOCK_D": 32},
            num_warps=8, num_stages=3),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_C": 32, "BLOCK_D": 32},
            num_warps=4, num_stages=4),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_C": 64, "BLOCK_D": 32},
            num_warps=8, num_stages=3),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_C": 32, "BLOCK_D": 64},
            num_warps=8, num_stages=3),
    ]


def reference_linear_rmsnorm_rope(
    x, weight, rms_weight, cos, sin, eps=1e-6,
):
    """PyTorch reference using split-half (Llama-style) rotary embedding."""
    num_heads, head_dim = rms_weight.shape
    projected = torch.nn.functional.linear(x.float(), weight.float())
    projected = projected.unflatten(-1, (num_heads, head_dim))
    inv_rms = torch.rsqrt(projected.square().mean(dim=-1, keepdim=True) + eps)
    normalized = projected * inv_rms * rms_weight.float()
    first, second = normalized.chunk(2, dim=-1)
    cos = cos.float()[None, :, None, :]
    sin = sin.float()[None, :, None, :]
    output = torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1)
    return output.to(x.dtype)


def reference_varlen_linear_rmsnorm_rope(
    x, weight, rms_weight, cos, sin, cu_seqlens, eps=1e-6,
):
    outputs = []
    boundaries = cu_seqlens.cpu().tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        length = end - start
        outputs.append(reference_linear_rmsnorm_rope(
            x[:, start:end], weight, rms_weight,
            cos[:length], sin[:length], eps))
    return torch.cat(outputs, dim=1)


@triton.autotune(
    configs=_qknorm_rope_configs(),
    key=["n_seq", "in_channels", "num_heads", "HEAD_DIM",
         "BLOCK_HEADS", "IS_VARLEN"],
    cache_results=True,
)
@triton.jit
def _linear_rmsnorm_rope_fwd(
    x, weight, rms_weight, cos, sin, cu_seqlens, output,
    n_seq, in_channels, num_heads,
    EPS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_HEADS: tl.constexpr,
    BLOCK_K: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)
    block_n: tl.constexpr = BLOCK_HEADS * HEAD_DIM
    half_dim: tl.constexpr = HEAD_DIM // 2

    seq_start = pid_group * n_seq
    seq_len = n_seq
    if IS_VARLEN:
        seq_start = tl.load(cu_seqlens + pid_group)
        seq_len = tl.load(cu_seqlens + pid_group + 1) - seq_start
    local_seq = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    m = seq_start + local_seq
    n = pid_h * block_n + tl.arange(0, block_n)
    m_mask = local_seq < seq_len
    n_mask = n < num_heads * HEAD_DIM

    acc = tl.zeros((BLOCK_M, block_n), tl.float32)
    for k_start in tl.range(0, in_channels, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        x_vals = tl.load(
            x + m[:, None] * in_channels + k[None, :],
            mask=m_mask[:, None] & (k < in_channels)[None, :], other=0.0)
        w_vals = tl.load(
            weight + n[:, None] * in_channels + k[None, :],
            mask=n_mask[:, None] & (k < in_channels)[None, :], other=0.0)
        acc = tl.dot(x_vals, tl.trans(w_vals), acc, input_precision="ieee")

    projected = tl.reshape(
        acc, (BLOCK_M, BLOCK_HEADS, HEAD_DIM), can_reorder=False)
    mean_square = tl.sum(projected * projected, axis=2) / HEAD_DIM
    inv_rms = tl.rsqrt(mean_square + EPS)

    head = pid_h * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    dim = tl.arange(0, HEAD_DIM)
    scale = tl.load(
        rms_weight + head[:, None] * HEAD_DIM + dim[None, :],
        mask=(head < num_heads)[:, None], other=0.0).to(tl.float32)
    normalized = projected * inv_rms[:, :, None] * scale[None, :, :]

    # Split-half RoPE: [x_0..x_half, x_half..x_d] are rotation pairs.
    halves = tl.reshape(
        normalized, (BLOCK_M, BLOCK_HEADS, 2, half_dim),
        can_reorder=False)
    pairs = tl.permute(halves, (0, 1, 3, 2))
    first, second = tl.split(pairs)

    position = local_seq
    rotary_dim = tl.arange(0, half_dim)
    cos_vals = tl.load(
        cos + position[:, None] * half_dim + rotary_dim[None, :],
        mask=m_mask[:, None], other=0.0).to(tl.float32)
    sin_vals = tl.load(
        sin + position[:, None] * half_dim + rotary_dim[None, :],
        mask=m_mask[:, None], other=0.0).to(tl.float32)
    out_first = first * cos_vals[:, None, :] - second * sin_vals[:, None, :]
    out_second = second * cos_vals[:, None, :] + first * sin_vals[:, None, :]
    out_pairs = tl.join(out_first, out_second)
    out_halves = tl.permute(out_pairs, (0, 1, 3, 2))
    out_vals = tl.reshape(out_halves, (BLOCK_M, block_n), can_reorder=False)

    tl.store(
        output + m[:, None] * (num_heads * HEAD_DIM) + n[None, :],
        out_vals.to(output.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=_qknorm_rope_configs(),
    key=["n_seq", "in_channels", "num_heads", "HEAD_DIM",
         "BLOCK_HEADS", "IS_VARLEN"],
    reset_to_zero=["drms_weight"],
    cache_results=True,
)
@triton.jit
def _linear_rmsnorm_rope_bwd_epilogue(
    x, weight, rms_weight, cos, sin, cu_seqlens,
    doutput, dlinear, drms_weight,
    n_seq, in_channels, num_heads,
    EPS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_HEADS: tl.constexpr,
    BLOCK_K: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)
    block_n: tl.constexpr = BLOCK_HEADS * HEAD_DIM
    half_dim: tl.constexpr = HEAD_DIM // 2

    seq_start = pid_group * n_seq
    seq_len = n_seq
    if IS_VARLEN:
        seq_start = tl.load(cu_seqlens + pid_group)
        seq_len = tl.load(cu_seqlens + pid_group + 1) - seq_start
    local_seq = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    m = seq_start + local_seq
    n = pid_h * block_n + tl.arange(0, block_n)
    m_mask = local_seq < seq_len
    n_mask = n < num_heads * HEAD_DIM

    acc = tl.zeros((BLOCK_M, block_n), tl.float32)
    for k_start in tl.range(0, in_channels, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        x_vals = tl.load(
            x + m[:, None] * in_channels + k[None, :],
            mask=m_mask[:, None] & (k < in_channels)[None, :], other=0.0)
        w_vals = tl.load(
            weight + n[:, None] * in_channels + k[None, :],
            mask=n_mask[:, None] & (k < in_channels)[None, :], other=0.0)
        acc = tl.dot(x_vals, tl.trans(w_vals), acc, input_precision="ieee")
    projected = tl.reshape(
        acc, (BLOCK_M, BLOCK_HEADS, HEAD_DIM), can_reorder=False)

    dout = tl.load(
        doutput + m[:, None] * (num_heads * HEAD_DIM) + n[None, :],
        mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
    dout = tl.reshape(dout, (BLOCK_M, BLOCK_HEADS, HEAD_DIM), can_reorder=False)
    dout_halves = tl.reshape(
        dout, (BLOCK_M, BLOCK_HEADS, 2, half_dim), can_reorder=False)
    dout_pairs = tl.permute(dout_halves, (0, 1, 3, 2))
    dout_first, dout_second = tl.split(dout_pairs)

    position = local_seq
    rotary_dim = tl.arange(0, half_dim)
    cos_vals = tl.load(
        cos + position[:, None] * half_dim + rotary_dim[None, :],
        mask=m_mask[:, None], other=0.0).to(tl.float32)
    sin_vals = tl.load(
        sin + position[:, None] * half_dim + rotary_dim[None, :],
        mask=m_mask[:, None], other=0.0).to(tl.float32)
    # Transpose of the rotation used by the forward kernel.
    du_first = dout_first * cos_vals[:, None, :] + dout_second * sin_vals[:, None, :]
    du_second = dout_second * cos_vals[:, None, :] - dout_first * sin_vals[:, None, :]
    du_pairs = tl.join(du_first, du_second)
    du_halves = tl.permute(du_pairs, (0, 1, 3, 2))
    du = tl.reshape(
        du_halves, (BLOCK_M, BLOCK_HEADS, HEAD_DIM), can_reorder=False)

    head = pid_h * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    dim = tl.arange(0, HEAD_DIM)
    head_mask = head < num_heads
    scale = tl.load(
        rms_weight + head[:, None] * HEAD_DIM + dim[None, :],
        mask=head_mask[:, None], other=0.0).to(tl.float32)

    mean_square = tl.sum(projected * projected, axis=2) / HEAD_DIM
    inv_rms = tl.rsqrt(mean_square + EPS)
    normalized = projected * inv_rms[:, :, None]
    drms = tl.sum(du * normalized, axis=0)
    tl.atomic_add(
        drms_weight + head[:, None] * HEAD_DIM + dim[None, :], drms,
        mask=head_mask[:, None])

    scaled_grad = du * scale[None, :, :]
    correction = tl.sum(scaled_grad * normalized, axis=2) / HEAD_DIM
    dz = inv_rms[:, :, None] * (
        scaled_grad - normalized * correction[:, :, None])
    dz = tl.reshape(dz, (BLOCK_M, block_n), can_reorder=False)
    tl.store(
        dlinear + m[:, None] * (num_heads * HEAD_DIM) + n[None, :],
        dz.to(dlinear.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=_linear_bwd_configs(),
    key=["m_size", "in_channels", "out_channels"],
    reset_to_zero=["dweight"],
    cache_results=True,
)
@triton.jit
def _linear_bwd_gemm(
    x, weight, dlinear, dx, dweight,
    m_size, in_channels, out_channels,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_m = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    m_mask = m < m_size
    c_mask = c < in_channels

    x_vals = tl.load(
        x + m[:, None] * in_channels + c[None, :],
        mask=m_mask[:, None] & c_mask[None, :], other=0.0)
    dx_acc = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)

    for d_start in tl.range(0, out_channels, BLOCK_D):
        d = d_start + tl.arange(0, BLOCK_D)
        d_mask = d < out_channels
        dz = tl.load(
            dlinear + m[:, None] * out_channels + d[None, :],
            mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        w_vals = tl.load(
            weight + d[:, None] * in_channels + c[None, :],
            mask=d_mask[:, None] & c_mask[None, :], other=0.0)
        dx_acc = tl.dot(dz, w_vals, dx_acc, input_precision="ieee")
        dw = tl.dot(tl.trans(dz), x_vals, input_precision="ieee")
        tl.atomic_add(
            dweight + d[:, None] * in_channels + c[None, :], dw,
            mask=d_mask[:, None] & c_mask[None, :])

    tl.store(
        dx + m[:, None] * in_channels + c[None, :],
        dx_acc.to(dx.dtype.element_ty),
        mask=m_mask[:, None] & c_mask[None, :])


def _validate_inputs(x, weight, rms_weight, cos, sin):
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, sequence, in_channels]")
    if not x.is_cuda:
        raise ValueError("the fused kernel requires CUDA tensors")
    if x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] == 0:
        raise ValueError("batch, sequence, and in_channels must be nonzero")
    if weight.ndim != 2 or rms_weight.ndim != 2:
        raise ValueError("weight and rms_weight must be two-dimensional")
    num_heads, head_dim = rms_weight.shape
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two and at least 16")
    if weight.shape != (num_heads * head_dim, x.shape[-1]):
        raise ValueError("weight must have shape [num_heads * head_dim, in_channels]")
    if (cos.ndim != 2 or cos.shape[1] != head_dim // 2
            or sin.shape != cos.shape):
        raise ValueError("cos and sin must have shape [positions, head_dim // 2]")
    tensors = (x, weight, rms_weight, cos, sin)
    if any(t.device != x.device for t in tensors):
        raise ValueError("all inputs must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all inputs must be contiguous")
    if any(t.dtype != x.dtype for t in tensors):
        raise ValueError("all floating-point inputs must have the same dtype")
    return num_heads, head_dim


def _launch_config(num_heads, head_dim):
    block_heads = max(1, 128 // head_dim)
    block_heads = min(block_heads, triton.next_power_of_2(num_heads))
    return block_heads


def torch_fwd_linear_rmsnorm_rope(
    x, weight, rms_weight, cos, sin, eps=1e-6,
):
    num_heads, head_dim = _validate_inputs(
        x, weight, rms_weight, cos, sin)
    batch, n_seq, in_channels = x.shape
    if cos.shape[0] != n_seq:
        raise ValueError("fixed-length cos and sin must cover the sequence")
    block_heads = _launch_config(num_heads, head_dim)
    output = torch.empty(
        (batch, n_seq, num_heads, head_dim), dtype=x.dtype, device=x.device)
    _linear_rmsnorm_rope_fwd[lambda meta: (
        triton.cdiv(n_seq, meta["BLOCK_M"]),
        triton.cdiv(num_heads, block_heads), batch
    )](
        x, weight, rms_weight, cos, sin, x, output,
        n_seq, in_channels, num_heads,
        EPS=eps, HEAD_DIM=head_dim, BLOCK_HEADS=block_heads,
        IS_VARLEN=False)
    return output


def torch_bwd_linear_rmsnorm_rope(
    x, weight, rms_weight, cos, sin, doutput, eps=1e-6,
):
    num_heads, head_dim = _validate_inputs(
        x, weight, rms_weight, cos, sin)
    if doutput.shape != (*x.shape[:2], num_heads, head_dim):
        raise ValueError("doutput has an invalid shape")
    if (not doutput.is_contiguous() or doutput.dtype != x.dtype
            or doutput.device != x.device):
        raise ValueError(
            "doutput must be contiguous and have the input dtype and device")

    batch, n_seq, in_channels = x.shape
    if cos.shape[0] != n_seq:
        raise ValueError("fixed-length cos and sin must cover the sequence")
    m_size = batch * n_seq
    out_channels = num_heads * head_dim
    block_heads = _launch_config(num_heads, head_dim)
    # Keep this temporary in the activation dtype. Under BF16 AMP the next
    # GEMM consumes BF16 operands anyway; storing FP32 would double traffic
    # without preserving precision through the tensor-core input conversion.
    dlinear = torch.empty(
        (batch, n_seq, out_channels), dtype=x.dtype, device=x.device)
    drms_weight = torch.zeros_like(rms_weight, dtype=torch.float32)
    _linear_rmsnorm_rope_bwd_epilogue[lambda meta: (
        triton.cdiv(n_seq, meta["BLOCK_M"]),
        triton.cdiv(num_heads, block_heads), batch
    )](
        x, weight, rms_weight, cos, sin, x,
        doutput, dlinear, drms_weight,
        n_seq, in_channels, num_heads,
        EPS=eps, HEAD_DIM=head_dim, BLOCK_HEADS=block_heads,
        IS_VARLEN=False)

    dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight, dtype=torch.float32)
    _linear_bwd_gemm[lambda meta: (
        triton.cdiv(in_channels, meta["BLOCK_C"]),
        triton.cdiv(m_size, meta["BLOCK_M"])
    )](
        x, weight, dlinear, dx, dweight,
        m_size, in_channels, out_channels)
    return dx, dweight, drms_weight


def _varlen_info(x, cu_seqlens, max_seqlen):
    if x.shape[0] != 1:
        raise ValueError("varlen x must have batch size 1")
    if (cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.device != x.device):
        raise ValueError("cu_seqlens must be a contiguous CUDA int tensor")
    if max_seqlen is None:
        if torch.compiler.is_compiling():
            raise ValueError(
                "max_seqlen must be provided when compiling varlen input")
        max_seqlen = int(
            (cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    if max_seqlen < 0:
        raise ValueError("max_seqlen must be nonnegative")
    return cu_seqlens.numel() - 1, int(max_seqlen)


def torch_fwd_varlen_linear_rmsnorm_rope(
    x, weight, rms_weight, cos, sin, cu_seqlens,
    max_seqlen=None, eps=1e-6,
):
    num_heads, head_dim = _validate_inputs(
        x, weight, rms_weight, cos, sin)
    num_groups, max_seqlen = _varlen_info(
        x, cu_seqlens, max_seqlen)
    if cos.shape[0] < max_seqlen:
        raise ValueError("cos and sin must cover max_seqlen positions")
    _, total_tokens, in_channels = x.shape
    output = torch.empty(
        (1, total_tokens, num_heads, head_dim),
        dtype=x.dtype, device=x.device)
    if max_seqlen == 0:
        return output

    block_heads = _launch_config(num_heads, head_dim)
    _linear_rmsnorm_rope_fwd[lambda meta: (
        triton.cdiv(max_seqlen, meta["BLOCK_M"]),
        triton.cdiv(num_heads, block_heads), num_groups
    )](
        x, weight, rms_weight, cos, sin, cu_seqlens, output,
        max_seqlen, in_channels, num_heads,
        EPS=eps, HEAD_DIM=head_dim, BLOCK_HEADS=block_heads,
        IS_VARLEN=True)
    return output


def torch_bwd_varlen_linear_rmsnorm_rope(
    x, weight, rms_weight, cos, sin, doutput, cu_seqlens,
    max_seqlen=None, eps=1e-6,
):
    num_heads, head_dim = _validate_inputs(
        x, weight, rms_weight, cos, sin)
    num_groups, max_seqlen = _varlen_info(
        x, cu_seqlens, max_seqlen)
    if cos.shape[0] < max_seqlen:
        raise ValueError("cos and sin must cover max_seqlen positions")
    _, total_tokens, in_channels = x.shape
    if doutput.shape != (1, total_tokens, num_heads, head_dim):
        raise ValueError("doutput has an invalid shape")
    if (not doutput.is_contiguous() or doutput.dtype != x.dtype
            or doutput.device != x.device):
        raise ValueError(
            "doutput must be contiguous and have the input dtype and device")

    out_channels = num_heads * head_dim
    if max_seqlen == 0:
        return (
            torch.empty_like(x),
            torch.zeros_like(weight, dtype=torch.float32),
            torch.zeros_like(rms_weight, dtype=torch.float32),
        )

    block_heads = _launch_config(num_heads, head_dim)
    dlinear = torch.empty(
        (1, total_tokens, out_channels), dtype=x.dtype, device=x.device)
    drms_weight = torch.zeros_like(rms_weight, dtype=torch.float32)
    _linear_rmsnorm_rope_bwd_epilogue[lambda meta: (
        triton.cdiv(max_seqlen, meta["BLOCK_M"]),
        triton.cdiv(num_heads, block_heads), num_groups
    )](
        x, weight, rms_weight, cos, sin, cu_seqlens,
        doutput, dlinear, drms_weight,
        max_seqlen, in_channels, num_heads,
        EPS=eps, HEAD_DIM=head_dim, BLOCK_HEADS=block_heads,
        IS_VARLEN=True)

    dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight, dtype=torch.float32)
    _linear_bwd_gemm[lambda meta: (
        triton.cdiv(in_channels, meta["BLOCK_C"]),
        triton.cdiv(total_tokens, meta["BLOCK_M"])
    )](
        x, weight, dlinear, dx, dweight,
        total_tokens, in_channels, out_channels)
    return dx, dweight, drms_weight


class LinearRMSNormRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, weight, rms_weight, cos, sin, eps,
        cu_seqlens=None, max_seqlen=None,
    ):
        has_varlen = cu_seqlens is not None
        saved_cu = cu_seqlens if has_varlen else torch.empty(
            0, dtype=torch.int32, device=x.device)
        ctx.save_for_backward(
            x, weight, rms_weight, cos, sin, saved_cu)
        ctx.eps = eps
        ctx.has_varlen = has_varlen
        ctx.max_seqlen = max_seqlen
        if has_varlen:
            return torch_fwd_varlen_linear_rmsnorm_rope(
                x, weight, rms_weight, cos, sin, cu_seqlens,
                max_seqlen, eps)
        return torch_fwd_linear_rmsnorm_rope(
            x, weight, rms_weight, cos, sin, eps)

    @staticmethod
    def backward(ctx, doutput):
        x, weight, rms_weight, cos, sin, saved_cu = ctx.saved_tensors
        if ctx.has_varlen:
            dx, dweight, drms_weight = (
                torch_bwd_varlen_linear_rmsnorm_rope(
                    x, weight, rms_weight, cos, sin,
                    doutput.contiguous(), saved_cu,
                    ctx.max_seqlen, ctx.eps))
        else:
            dx, dweight, drms_weight = torch_bwd_linear_rmsnorm_rope(
                x, weight, rms_weight, cos, sin,
                doutput.contiguous(), ctx.eps)
        return (
            dx,
            dweight.to(weight.dtype),
            drms_weight.to(rms_weight.dtype),
            None,
            None,
            None,
            None,
            None,
        )


def linear_rmsnorm_rope(
    x, weight, rms_weight, cos, sin, eps=1e-6,
    cu_seqlens=None, max_seqlen=None,
):
    if torch.is_autocast_enabled('cuda'):
        x, weight, rms_weight, cos, sin = (
            t.to(torch.bfloat16) for t in (x, weight, rms_weight, cos, sin))
    with torch.autocast('cuda', enabled=False):
        return LinearRMSNormRoPEFunction.apply(
            x, weight, rms_weight, cos, sin, eps,
            cu_seqlens, max_seqlen)


class FusedLinearRMSNormRoPE(torch.nn.Module):
    def __init__(
        self, in_channels, num_heads, head_dim, eps=1e-6,
        device=None, dtype=None,
    ):
        super().__init__()
        if head_dim < 16 or head_dim & (head_dim - 1):
            raise ValueError("head_dim must be a power of two and at least 16")
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(torch.empty(
            num_heads * head_dim, in_channels, **factory_kwargs))
        self.rms_weight = torch.nn.Parameter(torch.ones(
            num_heads, head_dim, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.ones_(self.rms_weight)

    def forward(
        self, x, cos, sin, cu_seqlens=None, max_seqlen=None,
    ):
        return linear_rmsnorm_rope(
            x, self.weight, self.rms_weight, cos, sin, self.eps,
            cu_seqlens, max_seqlen)

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, eps={self.eps}"
        )
