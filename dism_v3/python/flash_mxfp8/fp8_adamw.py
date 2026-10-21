"""AdamW variants with configurable optimizer-state dtypes."""

import torch
import triton
import triton.language as tl


@triton.jit
def _adamw_kernel(
    parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    n_elements,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    PARAMETER_FP32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    p = tl.load(parameter + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gradient + offsets, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(exp_avg + offsets, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(exp_avg_sq + offsets, mask=mask, other=0.0).to(tl.float32)

    m_fp32 = beta1 * m + (1.0 - beta1) * g
    v_fp32 = beta2 * v + (1.0 - beta2) * g * g
    m_bf16 = m_fp32.to(tl.bfloat16)
    v_bf16 = v_fp32.to(tl.bfloat16)
    m_rounded = m_bf16.to(tl.float32)
    v_rounded = v_bf16.to(tl.float32)

    step_size = lr / bias_correction1
    denominator = tl.sqrt(v_rounded / bias_correction2) + eps
    updated = (
        p * (1.0 - lr * weight_decay)
        - step_size * m_rounded / denominator
    )
    if PARAMETER_FP32:
        tl.store(parameter + offsets, updated, mask=mask)
    else:
        tl.store(parameter + offsets, updated.to(tl.bfloat16), mask=mask)
    tl.store(exp_avg + offsets, m_bf16, mask=mask)
    tl.store(exp_avg_sq + offsets, v_bf16, mask=mask)


@triton.jit
def _adamw_mxfp8_kernel(
    parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    weight_fp8,
    weight_scale,
    n,
    k,
    parameter_stride,
    gradient_stride,
    exp_avg_stride,
    exp_avg_sq_stride,
    weight_fp8_stride,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_n = tl.program_id(0)
    block_k = tl.program_id(1)
    num_k_blocks = tl.cdiv(k, BLOCK_K)
    rows = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (rows[:, None] < n) & (cols[None, :] < k)
    offsets = rows[:, None] * parameter_stride + cols[None, :]

    p = tl.load(parameter + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(
        gradient + rows[:, None] * gradient_stride + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    m = tl.load(
        exp_avg + rows[:, None] * exp_avg_stride + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        exp_avg_sq + rows[:, None] * exp_avg_sq_stride + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    m_fp32 = beta1 * m + (1.0 - beta1) * g
    v_fp32 = beta2 * v + (1.0 - beta2) * g * g
    m_bf16 = m_fp32.to(tl.bfloat16)
    v_bf16 = v_fp32.to(tl.bfloat16)
    m_rounded = m_bf16.to(tl.float32)
    v_rounded = v_bf16.to(tl.float32)

    step_size = lr / bias_correction1
    denominator = tl.sqrt(v_rounded / bias_correction2) + eps
    updated = (
        p * (1.0 - lr * weight_decay)
        - step_size * m_rounded / denominator
    )
    tl.store(parameter + offsets, updated, mask=mask)
    tl.store(
        exp_avg + rows[:, None] * exp_avg_stride + cols[None, :],
        m_bf16,
        mask=mask,
    )
    tl.store(
        exp_avg_sq + rows[:, None] * exp_avg_sq_stride + cols[None, :],
        v_bf16,
        mask=mask,
    )

    rounded_bf16 = updated.to(tl.bfloat16)
    exponent = rounded_bf16.to(tl.uint16, bitcast=True) & 0x7F80
    maximum_exponent = tl.max(exponent).to(tl.uint32)
    scale_bits = tl.maximum(maximum_exponent, 0x0380) - 0x0380
    scale_exponent = scale_bits >> 7
    reciprocal_bits = tl.where(
        scale_exponent == 254,
        0x00400000,
        (254 - scale_exponent) << 23,
    ).to(tl.uint32)
    reciprocal = reciprocal_bits.to(tl.float32, bitcast=True)
    quantized = tl.cast(
        rounded_bf16.to(tl.float32) * reciprocal,
        tl.float8e4nv,
    )
    fp8_offsets = rows[:, None] * weight_fp8_stride + cols[None, :]
    tl.store(weight_fp8 + fp8_offsets, quantized, mask=mask)
    tl.store(
        weight_scale + block_n * num_k_blocks + block_k,
        scale_exponent.to(tl.uint8),
    )


class FusedAdamWMXFP8(torch.optim.Optimizer):
    """AdamW with BF16 moments, FP32 master weights, and fused MXFP8 output.

    Quantized parameters are registered with :meth:`bind_fp8_parameter`.
    Unregistered BF16 or FP32 parameters use the same AdamW update without a
    quantization epilogue. Hyperparameters remain standard per-group values.
    """

    _fused_fp8_sync = True

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self._fp8_outputs = {}

    def bind_fp8_parameter(
        self,
        parameter: torch.Tensor,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
    ):
        """Attach persistent MXFP8 outputs to an FP32 master parameter."""
        if not any(
            parameter is candidate
            for group in self.param_groups
            for candidate in group["params"]
        ):
            raise ValueError("parameter does not belong to this optimizer")
        if (
            parameter.ndim != 2
            or parameter.dtype != torch.float32
            or not parameter.is_cuda
            or not parameter.is_contiguous()
        ):
            raise ValueError("Expected contiguous CUDA FP32 parameter[N,K]")
        n, k = parameter.shape
        padded_k = (k + 15) // 16 * 16
        if k % 32:
            raise ValueError("quantized parameters require K divisible by 32")
        if (
            weight_fp8.shape != (n, k)
            or weight_fp8.dtype != torch.float8_e4m3fn
            or weight_fp8.device != parameter.device
            or weight_fp8.stride(1) != 1
            or weight_fp8.stride(0) != padded_k
            or weight_fp8.data_ptr() % 16
        ):
            raise ValueError("Invalid MXFP8 weight output")
        required = weight_fp8.storage_offset() + n * padded_k
        if weight_fp8.untyped_storage().nbytes() < required:
            raise ValueError("FP8 output must include the padded TMA row tail")
        expected_scales = ((n + 31) // 32, (k + 31) // 32)
        if (
            weight_scale.shape != expected_scales
            or weight_scale.dtype != torch.uint8
            or weight_scale.device != parameter.device
            or not weight_scale.is_contiguous()
        ):
            raise ValueError("Invalid MXFP8 block scales")
        if padded_k != k:
            with torch.no_grad():
                padded = weight_fp8.as_strided(
                    (n, padded_k),
                    (padded_k, 1),
                    weight_fp8.storage_offset(),
                )
                padded[:, k:].zero_()
        self._fp8_outputs[parameter] = (weight_fp8, weight_scale)
        return self

    def _parameter_state(self, parameter):
        state = self.state[parameter]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(
                parameter, dtype=torch.bfloat16
            )
            state["exp_avg_sq"] = torch.zeros_like(
                parameter, dtype=torch.bfloat16
            )
        if (
            state["exp_avg"].dtype != torch.bfloat16
            or state["exp_avg_sq"].dtype != torch.bfloat16
        ):
            raise TypeError("FusedAdamWMXFP8 requires BF16 moment storage")
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.dtype not in (torch.float32, torch.bfloat16):
                    raise TypeError(
                        f"expected FP32 or BF16 parameter, got {parameter.dtype}"
                    )
                if parameter.grad.dtype != parameter.dtype:
                    raise TypeError(
                        f"expected {parameter.dtype} gradient, got "
                        f"{parameter.grad.dtype}"
                    )
                if parameter.grad.is_sparse:
                    raise RuntimeError("sparse gradients are not supported")
                if not parameter.is_contiguous() or not parameter.grad.is_contiguous():
                    raise ValueError("fused AdamW requires contiguous tensors")

                state = self._parameter_state(parameter)
                state["step"] += 1
                step = state["step"]
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                binding = self._fp8_outputs.get(parameter)
                if binding is not None:
                    if parameter.dtype != torch.float32 or parameter.ndim != 2:
                        raise TypeError(
                            "MXFP8 optimizer outputs require FP32 [N,K] parameters"
                        )
                    weight_fp8, weight_scale = binding
                    n, k = parameter.shape
                    _adamw_mxfp8_kernel[
                        (triton.cdiv(n, 32), triton.cdiv(k, 32))
                    ](
                        parameter,
                        parameter.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        weight_fp8,
                        weight_scale,
                        n,
                        k,
                        parameter.stride(0),
                        parameter.grad.stride(0),
                        state["exp_avg"].stride(0),
                        state["exp_avg_sq"].stride(0),
                        weight_fp8.stride(0),
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        bias_correction1,
                        bias_correction2,
                        BLOCK_N=32,
                        BLOCK_K=32,
                        num_warps=8,
                    )
                else:
                    block = 256
                    _adamw_kernel[
                        (triton.cdiv(parameter.numel(), block),)
                    ](
                        parameter,
                        parameter.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        parameter.numel(),
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        bias_correction1,
                        bias_correction2,
                        PARAMETER_FP32=parameter.dtype == torch.float32,
                        BLOCK=block,
                        num_warps=8,
                    )
        return loss


class AdamWStateDtype(torch.optim.Optimizer):
    """AdamW with configurable parameter and moment dtypes.

    Updates are computed in FP32. Output parameters and moments may be BF16
    to reduce persistent optimizer and model memory.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        moment1_dtype: torch.dtype = torch.float32,
        moment2_dtype: torch.dtype = torch.float32,
    ):
        if moment1_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"unsupported moment1 dtype: {moment1_dtype}")
        if moment2_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"unsupported moment2 dtype: {moment2_dtype}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self.moment1_dtype = moment1_dtype
        self.moment2_dtype = moment2_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.dtype not in (torch.float32, torch.bfloat16):
                    raise TypeError(
                        f"expected FP32 or BF16 parameter, got {parameter.dtype}"
                    )
                if parameter.grad.dtype != parameter.dtype:
                    raise TypeError(
                        f"expected {parameter.dtype} gradient, got "
                        f"{parameter.grad.dtype}"
                    )
                if parameter.grad.is_sparse:
                    raise RuntimeError("sparse gradients are not supported")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        dtype=self.moment1_dtype,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter,
                        dtype=self.moment2_dtype,
                    )

                state["step"] += 1
                step = state["step"]
                grad = parameter.grad.float()

                moment1_fp32 = state["exp_avg"].float()
                moment1_fp32.lerp_(grad, 1.0 - beta1)
                state["exp_avg"].copy_(moment1_fp32)

                moment2_fp32 = state["exp_avg_sq"].float()
                moment2_fp32.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1.0 - beta2,
                )
                state["exp_avg_sq"].copy_(moment2_fp32)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denominator = (
                    state["exp_avg_sq"].float()
                    .div(bias_correction2)
                    .sqrt()
                    .add_(eps)
                )

                if parameter.dtype == torch.float32:
                    parameter.mul_(1.0 - lr * weight_decay)
                    parameter.addcdiv_(
                        state["exp_avg"].float(),
                        denominator,
                        value=-step_size,
                    )
                else:
                    parameter_fp32 = parameter.float()
                    parameter_fp32.mul_(1.0 - lr * weight_decay)
                    parameter_fp32.addcdiv_(
                        state["exp_avg"].float(),
                        denominator,
                        value=-step_size,
                    )
                    parameter.copy_(parameter_fp32)

        return loss


def optimizer_state_nbytes(optimizer: torch.optim.Optimizer) -> int:
    """Return persistent tensor bytes held in an optimizer's state."""
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def optimizer_state_dtypes(
    optimizer: torch.optim.Optimizer,
) -> tuple[set[torch.dtype], set[torch.dtype]]:
    moment1_dtypes = set()
    moment2_dtypes = set()
    for state in optimizer.state.values():
        if "exp_avg" in state:
            moment1_dtypes.add(state["exp_avg"].dtype)
        if "exp_avg_sq" in state:
            moment2_dtypes.add(state["exp_avg_sq"].dtype)
    return moment1_dtypes, moment2_dtypes
