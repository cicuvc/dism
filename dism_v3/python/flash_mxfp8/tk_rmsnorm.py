"""RMSNorm prenorm that emits both MXFP8 activation layouts.

For ``y = x * rsqrt(mean(x^2) + eps) * factor``, this plan produces:

* row-wise ``y[M,K]`` E4M3 and packed 1x32 scales for forward GEMM;
* column-wise ``y[MP,K]`` E4M3 and packed 32x1 scales for wgrad;
* FP32 ``inverse_rms[M]`` for the eventual RMSNorm backward.

Normalized BF16 activations are not materialized in global memory.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _u16_max(x, y):
    return tl.inline_asm_elementwise(
        "max.u16x2 $0, $1, $2;", "=r,r,r", [x, y], tl.uint16, True, 2
    )


@triton.jit
def _u16_clamp_sub(x, amount):
    return tl.inline_asm_elementwise(
        "max.u16x2 $0, $1, $2; sub.u32 $0, $0, $2;",
        "=r,r,r",
        [x, amount],
        tl.uint16,
        True,
        2,
    )


@triton.jit
def _online_scale(x: tl.tensor):
    """Quantize a BF16 tile with one E8M0 scale per contiguous K32."""
    tl.static_assert(x.dtype == tl.bfloat16)
    rows: tl.constexpr = x.shape[0]
    cols: tl.constexpr = x.shape[1]
    x = tl.reshape(x, (rows, cols // 32, 32))
    exponents = tl.inline_asm_elementwise(
        "and.b32 $0, $1, 0x7F807F80;", "=r,r", [x], tl.uint16, True, 2
    )
    scale_u16 = tl.reduce(exponents, axis=-1, combine_fn=_u16_max)
    # E4M3 emax is 8. Bias by 7 to leave one binary exponent of headroom
    # while using the upper safe E4M3 range.
    scale_u16 = _u16_clamp_sub(
        scale_u16,
        tl.full(scale_u16.shape, 0x0380, tl.uint16),
    )
    reciprocal = tl.inline_asm_elementwise(
        "sub.u32 $0, 0x7F007F00, $1; and.b32 $0, $0, 0x7F807F80;",
        "=r,r",
        [scale_u16],
        tl.bfloat16,
        True,
        2,
    )
    quantized = tl.cast(
        x * reciprocal[:, :, None], tl.float8e4nv
    ).reshape(rows, cols)
    return quantized, tl.cast(scale_u16 >> 7, tl.uint8)


@triton.jit
def _inverse_rms(
    x,
    inverse_rms,
    M,
    K,
    x_stride,
    eps,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    values = tl.load(
        x + row * x_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / K
    tl.store(inverse_rms + row, tl.rsqrt(mean_square + eps))


@triton.jit
def _rmsnorm_dual_quant(
    x,
    factor,
    inverse_rms,
    row_fp8,
    row_scale,
    column_fp8,
    column_scale,
    M,
    K,
    MP,
    MP16,
    SP,
    x_stride,
    factor_stride,
    row_stride,
    column_stride,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * 32 + tl.arange(0, 32)
    cols = col_block * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    values = tl.load(
        x + rows[:, None] * x_stride + cols[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    inv = tl.load(
        inverse_rms + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    scale_factor = tl.load(
        factor + cols * factor_stride,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    normalized = (
        values.to(tl.float32)
        * inv[:, None]
        * scale_factor[None, :]
    ).to(tl.bfloat16)

    row_q, row_s = _online_scale(normalized)
    tl.store(
        row_fp8 + rows[:, None] * row_stride + cols[None, :],
        row_q,
        mask=row_mask[:, None] & col_mask[None, :],
    )
    # BLOCK_K is 128, exactly four K32 scale blocks per packed word.
    row_word = tl.sum(
        row_s.to(tl.uint32)
        << (8 * tl.arange(0, 4))[None, :],
        axis=1,
    )
    tl.store(
        row_scale + col_block * MP16 + rows,
        row_word,
        mask=rows < MP16,
    )

    column_q_t, column_s = _online_scale(tl.trans(normalized))
    tl.store(
        column_fp8 + rows[:, None] * column_stride + cols[None, :],
        tl.trans(column_q_t),
        mask=col_mask[None, :],
    )
    column_values = tl.reshape(column_s, (BLOCK_K,))
    tl.store(
        column_scale
        + ((row_block // 4) * SP + cols) * 4
        + (row_block % 4),
        column_values,
        mask=cols < SP,
    )


class RMSNormDualQuantPlan:
    """Reusable fixed-shape prenorm and two-direction quantization plan."""

    def __init__(self, x, factor, *, eps=1e-6):
        if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
            raise ValueError("Expected CUDA BF16 X[M,K]")
        if factor.ndim != 1 or factor.shape[0] != x.shape[1]:
            raise ValueError("Expected elementwise factor[K]")
        if factor.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Factor must be FP32 or BF16")
        if not factor.is_cuda or factor.device != x.device:
            raise ValueError("Expected factor on the input device")
        m, k = x.shape
        if k % 32:
            raise ValueError("K must be a multiple of 32")
        self.x = x
        self.factor = factor
        self.eps = float(eps)
        self.m, self.k = m, k
        self.mp = (m + 127) // 128 * 128
        self.mp16 = (m + 15) // 16 * 16
        self.sp = (k + 15) // 16 * 16
        self.inverse_rms = torch.empty(m, device=x.device, dtype=torch.float32)
        self.row_fp8 = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
        self.row_scale = torch.zeros(
            ((k + 127) // 128, self.mp16),
            device=x.device,
            dtype=torch.int32,
        )
        self.column_fp8 = torch.empty(
            (self.mp, k), device=x.device, dtype=torch.float8_e4m3fn
        )
        self.column_scale = torch.zeros(
            (self.mp // 128, self.sp),
            device=x.device,
            dtype=torch.int32,
        )

    def rebind(self, x, factor):
        if x.shape != (self.m, self.k) or x.dtype != torch.bfloat16:
            raise ValueError("Expected BF16 X with the plan shape")
        if factor.shape != self.factor.shape or factor.dtype != self.factor.dtype:
            raise ValueError("Expected factor with the plan shape and dtype")
        if (
            not x.is_cuda
            or not x.is_contiguous()
            or x.device != self.inverse_rms.device
            or factor.device != x.device
            or not factor.is_contiguous()
        ):
            raise ValueError("Expected contiguous tensors on one CUDA device")
        self.x, self.factor = x, factor
        return self

    def __call__(self):
        _inverse_rms[(self.m,)](
            self.x,
            self.inverse_rms,
            self.m,
            self.k,
            self.x.stride(0),
            self.eps,
            BLOCK_K=triton.next_power_of_2(self.k),
        )
        _rmsnorm_dual_quant[(self.mp // 32, triton.cdiv(self.k, 128))](
            self.x,
            self.factor,
            self.inverse_rms,
            self.row_fp8,
            self.row_scale,
            self.column_fp8,
            self.column_scale.view(torch.uint8),
            self.m,
            self.k,
            self.mp,
            self.mp16,
            self.sp,
            self.x.stride(0),
            self.factor.stride(0),
            self.k,
            self.k,
            BLOCK_K=128,
            num_warps=4,
        )
        return (
            self.row_fp8,
            self.row_scale,
            self.column_fp8,
            self.column_scale,
            self.inverse_rms,
        )


@triton.jit
def _rmsnorm_dot(
    dy,
    x,
    factor,
    dot,
    M,
    K,
    dy_stride,
    x_stride,
    factor_stride,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    dy_values = tl.load(
        dy + row * dy_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x_values = tl.load(
        x + row * x_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale_factor = tl.load(
        factor + cols * factor_stride,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(dot + row, tl.sum(dy_values * x_values * scale_factor, axis=0))


@triton.jit
def _rmsnorm_backward(
    dy,
    x,
    factor,
    inverse_rms,
    dot,
    dx,
    factor_partial,
    M,
    K,
    dy_stride,
    x_stride,
    dx_stride,
    factor_stride,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * 32 + tl.arange(0, 32)
    cols = col_block * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    mask = row_mask[:, None] & col_mask[None, :]
    dy_values = tl.load(
        dy + rows[:, None] * dy_stride + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x_values = tl.load(
        x + rows[:, None] * x_stride + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale_factor = tl.load(
        factor + cols * factor_stride,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    inverse = tl.load(
        inverse_rms + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    dy_scaled = dy_values * scale_factor[None, :]
    correction = (inverse * inverse / K) * tl.load(dot + rows, mask=row_mask, other=0.0)
    grad_x = inverse[:, None] * (
        dy_scaled - correction[:, None] * x_values
    )
    tl.store(
        dx + rows[:, None] * dx_stride + cols[None, :],
        grad_x.to(tl.bfloat16),
        mask=mask,
    )
    partial = tl.sum(dy_values * x_values * inverse[:, None], axis=0)
    tl.store(
        factor_partial + row_block * K + cols,
        partial,
        mask=col_mask,
    )


@triton.jit
def _reduce_factor_grad(
    factor_partial,
    factor_grad,
    row_blocks,
    K,
    BLOCK: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < K
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for row_block in range(0, row_blocks):
        accumulator += tl.load(
            factor_partial + row_block * K + cols,
            mask=mask,
            other=0.0,
        )
    tl.store(factor_grad + cols, accumulator, mask=mask)


class RMSNormBackwardPlan:
    """Reusable backward from BF16 dY to BF16 dX and FP32 dFactor."""

    def __init__(self, dy, x, factor, inverse_rms):
        if (
            dy.shape != x.shape
            or dy.ndim != 2
            or dy.dtype != torch.bfloat16
            or x.dtype != torch.bfloat16
        ):
            raise ValueError("Expected BF16 dY and X with equal [M,K] shapes")
        if factor.ndim != 1 or factor.shape[0] != x.shape[1]:
            raise ValueError("Expected factor[K]")
        if inverse_rms.shape != (x.shape[0],) or inverse_rms.dtype != torch.float32:
            raise ValueError("Expected FP32 inverse_rms[M]")
        if not x.is_cuda or any(
            tensor.device != x.device
            for tensor in [dy, factor, inverse_rms]
        ):
            raise ValueError("Expected all tensors on one CUDA device")
        self.dy, self.x, self.factor = dy, x, factor
        self.factor_shape, self.factor_dtype = factor.shape, factor.dtype
        self.inverse_rms = inverse_rms
        self.inverse_shape = inverse_rms.shape
        self.m, self.k = x.shape
        self.device = x.device
        self.row_blocks = (self.m + 31) // 32
        self.dx = torch.empty_like(x)
        self.dot = torch.empty(self.m, device=x.device, dtype=torch.float32)
        self.factor_grad = torch.empty(self.k, device=x.device, dtype=torch.float32)
        self.factor_partial = torch.empty(
            (self.row_blocks, self.k), device=x.device, dtype=torch.float32
        )

    def rebind(self, dy, x, factor, inverse_rms):
        if (
            dy.shape != (self.m, self.k)
            or x.shape != (self.m, self.k)
            or dy.dtype != torch.bfloat16
            or x.dtype != torch.bfloat16
            or factor.shape != self.factor_shape
            or factor.dtype != self.factor_dtype
            or inverse_rms.shape != self.inverse_shape
            or inverse_rms.dtype != torch.float32
        ):
            raise ValueError("Invalid RMSNorm backward rebind tensors")
        if any(
            tensor.device != self.dx.device or not tensor.is_contiguous()
            for tensor in [dy, x, factor, inverse_rms]
        ):
            raise ValueError("Expected contiguous tensors on one CUDA device")
        self.dy, self.x, self.factor = dy, x, factor
        self.inverse_rms = inverse_rms
        return self

    def allocate_workspace(self):
        self.dot = torch.empty(self.m, device=self.device, dtype=torch.float32)
        self.dx = torch.empty(
            (self.m, self.k), device=self.device, dtype=torch.bfloat16
        )
        self.factor_grad = torch.empty(
            self.k, device=self.device, dtype=torch.float32
        )
        self.factor_partial = torch.empty(
            (self.row_blocks, self.k), device=self.device, dtype=torch.float32
        )
        return self

    def release_workspace(self):
        self.dy = self.x = self.factor = self.inverse_rms = None
        self.dx = self.dot = self.factor_grad = self.factor_partial = None

    def __call__(self):
        _rmsnorm_dot[(self.m,)](
            self.dy,
            self.x,
            self.factor,
            self.dot,
            self.m,
            self.k,
            self.dy.stride(0),
            self.x.stride(0),
            self.factor.stride(0),
            BLOCK_K=triton.next_power_of_2(self.k),
        )
        _rmsnorm_backward[(self.row_blocks, triton.cdiv(self.k, 128))](
            self.dy,
            self.x,
            self.factor,
            self.inverse_rms,
            self.dot,
            self.dx,
            self.factor_partial,
            self.m,
            self.k,
            self.dy.stride(0),
            self.x.stride(0),
            self.dx.stride(0),
            self.factor.stride(0),
            BLOCK_K=128,
            num_warps=4,
        )
        block = 256
        _reduce_factor_grad[(triton.cdiv(self.k, block),)](
            self.factor_partial,
            self.factor_grad,
            self.row_blocks,
            self.k,
            BLOCK=block,
            num_warps=4,
        )
        return self.dx, self.factor_grad


@triton.jit
def _inverse_rms_add(
    x,
    residual,
    inverse_rms,
    sum_out,
    M,
    K,
    x_stride,
    residual_stride,
    sum_stride,
    eps,
    WRITE_SUM: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    x_values = tl.load(
        x + row * x_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual_values = tl.load(
        residual + row * residual_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    summed = x_values + residual_values
    mean_square = tl.sum(summed * summed, axis=0) / K
    tl.store(inverse_rms + row, tl.rsqrt(mean_square + eps))
    if WRITE_SUM:
        tl.store(
            sum_out + row * sum_stride + cols,
            summed.to(tl.bfloat16),
            mask=mask,
        )


@triton.jit
def _add_rmsnorm_output(
    x,
    residual,
    factor,
    inverse_rms,
    y_bf16,
    row_fp8,
    row_scale,
    column_fp8,
    column_scale,
    M,
    K,
    MP,
    MP16,
    SP,
    x_stride,
    residual_stride,
    factor_stride,
    y_stride,
    row_stride,
    column_stride,
    Y_MODE: tl.constexpr,
    HAS_FACTOR: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * 32 + tl.arange(0, 32)
    cols = col_block * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    mask = row_mask[:, None] & col_mask[None, :]
    summed = (
        tl.load(
            x + rows[:, None] * x_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        + tl.load(
            residual + rows[:, None] * residual_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    )
    inverse = tl.load(
        inverse_rms + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_FACTOR:
        scale_factor = tl.load(
            factor + cols * factor_stride,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        normalized = summed * inverse[:, None] * scale_factor[None, :]
    else:
        normalized = summed * inverse[:, None]
    normalized_bf16 = normalized.to(tl.bfloat16)
    if Y_MODE == 1 or Y_MODE == 3:
        tl.store(
            y_bf16 + rows[:, None] * y_stride + cols[None, :],
            normalized_bf16,
            mask=mask,
        )
    if Y_MODE == 2 or Y_MODE == 3:
        row_q, row_s = _online_scale(normalized_bf16)
        tl.store(
            row_fp8 + rows[:, None] * row_stride + cols[None, :],
            row_q,
            mask=mask,
        )
        row_word = tl.sum(
            row_s.to(tl.uint32)
            << (8 * tl.arange(0, 4))[None, :],
            axis=1,
        )
        tl.store(
            row_scale + col_block * MP16 + rows,
            row_word,
            mask=rows < MP16,
        )
        column_q_t, column_s = _online_scale(tl.trans(normalized_bf16))
        tl.store(
            column_fp8 + rows[:, None] * column_stride + cols[None, :],
            tl.trans(column_q_t),
            mask=col_mask[None, :],
        )
        column_values = tl.reshape(column_s, (BLOCK_K,))
        tl.store(
            column_scale
            + ((row_block // 4) * SP + cols) * 4
            + (row_block % 4),
            column_values,
            mask=cols < SP,
        )


class FusedAddRMSNormPlan:
    """``RMSNorm(x + residual)`` with compile-time output modes."""

    Y_NONE = 0
    Y_BF16 = 1
    Y_MXFP8 = 2
    Y_BF16_MXFP8 = 3
    Y_MODES = {
        "none": Y_NONE,
        "bf16": Y_BF16,
        "mxfp8": Y_MXFP8,
        "bf16_mxfp8": Y_BF16_MXFP8,
    }

    def __init__(
        self,
        x,
        residual,
        factor=None,
        *,
        eps=1e-6,
        write_sum=False,
        y_mode="none",
    ):
        if (
            x.ndim != 2
            or x.shape != residual.shape
            or x.dtype != torch.bfloat16
            or residual.dtype != torch.bfloat16
            or not x.is_cuda
            or residual.device != x.device
        ):
            raise ValueError("Expected equal CUDA BF16 X and residual[M,K]")
        if factor is not None:
            if factor.ndim != 1 or factor.shape[0] != x.shape[1]:
                raise ValueError("Expected factor[K]")
            if factor.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("Factor must be FP32 or BF16")
            if not factor.is_cuda or factor.device != x.device:
                raise ValueError("Expected factor on the input device")
        if y_mode not in self.Y_MODES:
            raise ValueError(
                "y_mode must be 'none', 'bf16', 'mxfp8', or 'bf16_mxfp8'"
            )
        m, k = x.shape
        if k % 32:
            raise ValueError("K must be a multiple of 32")
        self.x, self.residual, self.factor = x, residual, factor
        self.eps = float(eps)
        self.m, self.k = m, k
        self.mp = (m + 127) // 128 * 128
        self.mp16 = (m + 15) // 16 * 16
        self.sp = (k + 15) // 16 * 16
        self.y_mode = self.Y_MODES[y_mode]
        self.write_sum = bool(write_sum)
        self.inverse_rms = torch.empty(m, device=x.device, dtype=torch.float32)
        self.sum_out = (
            torch.empty_like(x) if self.write_sum else None
        )
        self.y_bf16 = (
            torch.empty_like(x)
            if self.y_mode in (self.Y_BF16, self.Y_BF16_MXFP8)
            else None
        )
        if self.y_mode in (self.Y_MXFP8, self.Y_BF16_MXFP8):
            self.row_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            self.row_scale = torch.zeros(
                ((k + 127) // 128, self.mp16),
                device=x.device,
                dtype=torch.int32,
            )
            self.column_fp8 = torch.empty(
                (self.mp, k), device=x.device, dtype=torch.float8_e4m3fn
            )
            self.column_scale = torch.zeros(
                (self.mp // 128, self.sp),
                device=x.device,
                dtype=torch.int32,
            )
        else:
            self.row_fp8 = self.row_scale = None
            self.column_fp8 = self.column_scale = None

    def rebind(self, x, residual, factor=None):
        if (
            x.shape != (self.m, self.k)
            or residual.shape != x.shape
            or x.dtype != torch.bfloat16
            or residual.dtype != torch.bfloat16
        ):
            raise ValueError("Expected equal BF16 X and residual with the plan shape")
        if (factor is None) != (self.factor is None):
            raise ValueError("Factor presence must match the plan")
        if factor is not None and (
            factor.shape != self.factor.shape
            or factor.dtype != self.factor.dtype
        ):
            raise ValueError("Factor shape or dtype changed")
        tensors = [x, residual] + ([factor] if factor is not None else [])
        if any(
            tensor.device != self.inverse_rms.device or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError("Expected contiguous tensors on one CUDA device")
        self.x, self.residual, self.factor = x, residual, factor
        return self

    def __call__(self):
        sum_ptr = self.sum_out if self.sum_out is not None else self.x
        _inverse_rms_add[(self.m,)](
            self.x,
            self.residual,
            self.inverse_rms,
            sum_ptr,
            self.m,
            self.k,
            self.x.stride(0),
            self.residual.stride(0),
            self.sum_out.stride(0) if self.sum_out is not None else 0,
            self.eps,
            WRITE_SUM=self.write_sum,
            BLOCK_K=triton.next_power_of_2(self.k),
        )
        if self.y_mode != self.Y_NONE:
            factor = self.factor if self.factor is not None else self.x
            factor_stride = self.factor.stride(0) if self.factor is not None else 0
            y_ptr = self.y_bf16 if self.y_bf16 is not None else self.x
            row_ptr = self.row_fp8 if self.row_fp8 is not None else self.x
            row_scale_ptr = self.row_scale if self.row_scale is not None else self.x
            column_ptr = self.column_fp8 if self.column_fp8 is not None else self.x
            column_scale_ptr = (
                self.column_scale.view(torch.uint8)
                if self.column_scale is not None
                else self.x
            )
            _add_rmsnorm_output[
                (self.mp // 32, triton.cdiv(self.k, 128))
            ](
                self.x,
                self.residual,
                factor,
                self.inverse_rms,
                y_ptr,
                row_ptr,
                row_scale_ptr,
                column_ptr,
                column_scale_ptr,
                self.m,
                self.k,
                self.mp,
                self.mp16,
                self.sp,
                self.x.stride(0),
                self.residual.stride(0),
                factor_stride,
                self.y_bf16.stride(0) if self.y_bf16 is not None else 0,
                self.k,
                self.k,
                Y_MODE=self.y_mode,
                HAS_FACTOR=self.factor is not None,
                BLOCK_K=128,
                num_warps=4,
            )
        return (
            self.inverse_rms,
            self.sum_out,
            self.y_bf16,
            self.row_fp8,
            self.row_scale,
            self.column_fp8,
            self.column_scale,
        )


@triton.jit
def _fused_add_dot(
    dy,
    x,
    residual,
    factor,
    dot,
    K,
    dy_stride,
    x_stride,
    residual_stride,
    factor_stride,
    HAS_FACTOR: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    dy_values = tl.load(
        dy + row * dy_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    summed = (
        tl.load(
            x + row * x_stride + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        + tl.load(
            residual + row * residual_stride + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    )
    if HAS_FACTOR:
        scale_factor = tl.load(
            factor + cols * factor_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dot_value = tl.sum(dy_values * summed * scale_factor, axis=0)
    else:
        dot_value = tl.sum(dy_values * summed, axis=0)
    tl.store(dot + row, dot_value)


@triton.jit
def _fused_add_rmsnorm_backward(
    dy,
    x,
    residual,
    factor,
    inverse_rms,
    dot,
    dsum,
    dx,
    factor_partial,
    M,
    K,
    dy_stride,
    x_stride,
    residual_stride,
    dx_stride,
    factor_stride,
    HAS_FACTOR: tl.constexpr,
    HAS_DSUM: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * 32 + tl.arange(0, 32)
    cols = col_block * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    mask = row_mask[:, None] & col_mask[None, :]
    dy_values = tl.load(
        dy + rows[:, None] * dy_stride + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    summed = (
        tl.load(
            x + rows[:, None] * x_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        + tl.load(
            residual + rows[:, None] * residual_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    )
    inverse = tl.load(
        inverse_rms + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_FACTOR:
        scale_factor = tl.load(
            factor + cols * factor_stride,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        dy_scaled = dy_values * scale_factor[None, :]
    else:
        dy_scaled = dy_values
    dot_value = tl.load(dot + rows, mask=row_mask, other=0.0)
    correction = (inverse * inverse / K) * dot_value
    grad_sum = inverse[:, None] * (
        dy_scaled - correction[:, None] * summed
    )
    if HAS_DSUM:
        grad_sum += tl.load(
            dsum + rows[:, None] * dx_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(
        dx + rows[:, None] * dx_stride + cols[None, :],
        grad_sum.to(tl.bfloat16),
        mask=mask,
    )
    if HAS_FACTOR:
        partial = tl.sum(dy_values * summed * inverse[:, None], axis=0)
        tl.store(
            factor_partial + row_block * K + cols,
            partial,
            mask=col_mask,
        )


@triton.jit
def _reduce_factor_grad_bf16(
    factor_partial,
    factor_grad,
    row_blocks,
    K,
    BLOCK: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < K
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for row_block in range(0, row_blocks):
        accumulator += tl.load(
            factor_partial + row_block * K + cols,
            mask=mask,
            other=0.0,
        )
    tl.store(factor_grad + cols, accumulator.to(tl.bfloat16), mask=mask)


class FusedAddRMSNormBackwardPlan:
    """BF16 backward for ``RMSNorm(x + residual)``."""

    def __init__(self, dy, x, residual, factor, inverse_rms, *, dsum=None):
        if (
            dy.shape != x.shape
            or residual.shape != x.shape
            or dy.dtype != torch.bfloat16
            or x.dtype != torch.bfloat16
            or residual.dtype != torch.bfloat16
        ):
            raise ValueError("Expected BF16 dY, X, residual with equal shapes")
        if factor is not None:
            if factor.ndim != 1 or factor.shape[0] != x.shape[1]:
                raise ValueError("Expected factor[K]")
            if factor.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("Factor must be FP32 or BF16")
        if inverse_rms.shape != (x.shape[0],) or inverse_rms.dtype != torch.float32:
            raise ValueError("Expected FP32 inverse_rms[M]")
        if dsum is not None and (
            dsum.shape != x.shape
            or dsum.dtype != torch.bfloat16
            or not dsum.is_contiguous()
            or dsum.device != x.device
        ):
            raise ValueError("Expected contiguous BF16 dsum[M,K]")
        self.dy, self.x, self.residual = dy, x, residual
        self.factor, self.inverse_rms, self.dsum = factor, inverse_rms, dsum
        self.m, self.k = x.shape
        self.device = x.device
        self.has_factor = factor is not None
        self.factor_shape = None if factor is None else factor.shape
        self.factor_dtype = None if factor is None else factor.dtype
        self.inverse_shape = inverse_rms.shape
        self.row_blocks = (self.m + 31) // 32
        self.dot = torch.empty(self.m, device=x.device, dtype=torch.float32)
        self.dx = torch.empty_like(x)
        self.dresidual = self.dx
        self.factor_grad = (
            torch.empty(self.k, device=x.device, dtype=torch.bfloat16)
            if factor is not None
            else None
        )
        self.factor_partial = (
            torch.empty(
                (self.row_blocks, self.k),
                device=x.device,
                dtype=torch.float32,
            )
            if factor is not None
            else None
        )

    def rebind(self, dy, x, residual, factor, inverse_rms, *, dsum=None):
        if (
            dy.shape != (self.m, self.k)
            or x.shape != (self.m, self.k)
            or residual.shape != (self.m, self.k)
            or dy.dtype != torch.bfloat16
            or x.dtype != torch.bfloat16
            or residual.dtype != torch.bfloat16
            or inverse_rms.shape != self.inverse_shape
            or inverse_rms.dtype != torch.float32
        ):
            raise ValueError("Invalid fused RMSNorm backward rebind tensors")
        if (factor is None) == self.has_factor:
            raise ValueError("Factor presence must match the plan")
        if factor is not None and (
            factor.shape != self.factor_shape or factor.dtype != self.factor_dtype
        ):
            raise ValueError("Factor shape or dtype changed")
        if dsum is not None and (
            dsum.shape != x.shape
            or dsum.dtype != torch.bfloat16
            or not dsum.is_contiguous()
        ):
            raise ValueError("Expected contiguous BF16 dsum[M,K]")
        tensors = [dy, x, residual] + (
            [factor] if factor is not None else []
        ) + ([dsum] if dsum is not None else []) + [inverse_rms]
        if any(
            tensor.device != self.dx.device or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError("Expected contiguous tensors on one CUDA device")
        self.dy, self.x, self.residual = dy, x, residual
        self.factor, self.inverse_rms, self.dsum = factor, inverse_rms, dsum
        return self

    def allocate_workspace(self):
        device = self.device
        self.dot = torch.empty(self.m, device=device, dtype=torch.float32)
        self.dx = torch.empty(
            (self.m, self.k), device=device, dtype=torch.bfloat16
        )
        self.dresidual = self.dx
        self.factor_grad = (
            torch.empty(self.k, device=device, dtype=torch.bfloat16)
            if self.has_factor
            else None
        )
        self.factor_partial = (
            torch.empty((self.row_blocks, self.k), device=device, dtype=torch.float32)
            if self.has_factor
            else None
        )
        return self

    def release_workspace(self):
        self.dy = self.x = self.residual = None
        self.factor = self.inverse_rms = self.dsum = None
        self.dot = self.dx = self.dresidual = None
        self.factor_grad = self.factor_partial = None

    def __call__(self):
        factor = self.factor if self.factor is not None else self.x
        factor_stride = self.factor.stride(0) if self.factor is not None else 0
        _fused_add_dot[(self.m,)](
            self.dy,
            self.x,
            self.residual,
            factor,
            self.dot,
            self.k,
            self.dy.stride(0),
            self.x.stride(0),
            self.residual.stride(0),
            factor_stride,
            HAS_FACTOR=self.factor is not None,
            BLOCK_K=triton.next_power_of_2(self.k),
        )
        partial = self.factor_partial if self.factor_partial is not None else self.dot
        dsum = self.dsum if self.dsum is not None else self.dy
        _fused_add_rmsnorm_backward[
            (self.row_blocks, triton.cdiv(self.k, 128))
        ](
            self.dy,
            self.x,
            self.residual,
            factor,
            self.inverse_rms,
            self.dot,
            dsum,
            self.dx,
            partial,
            self.m,
            self.k,
            self.dy.stride(0),
            self.x.stride(0),
            self.residual.stride(0),
            self.dx.stride(0),
            factor_stride,
            HAS_FACTOR=self.factor is not None,
            HAS_DSUM=self.dsum is not None,
            BLOCK_K=128,
            num_warps=4,
        )
        if self.factor_grad is not None:
            block = 256
            _reduce_factor_grad_bf16[(triton.cdiv(self.k, block),)](
                self.factor_partial,
                self.factor_grad,
                self.row_blocks,
                self.k,
                BLOCK=block,
                num_warps=4,
            )
        return self.dx, self.dresidual, self.factor_grad
