"""Shared runtime helpers for the pure-MXFP8 CUDA kernels."""

from collections.abc import Iterable, Iterator
import os
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
BUILD_ROOT = Path(
    os.environ.get(
        "DISM_MXFP8_BUILD_ROOT",
        ROOT / "_libs",
    )
)
FLAGS = [
    "-std=c++20",
    "-gencode=arch=compute_120a,code=sm_120a",
    "-DKITTENS_RTX_BLACKWELL",
    f"-I{ROOT / 'third_party/thunderkittens/include'}",
    "--expt-relaxed-constexpr",
    "--extended-lambda",
    "-O3",
    "-lineinfo",
]

_weight_quant_library = None


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 weights to E4M3 with one E8M0 scale per 32x32 block.

    The pre-round to BF16 preserves the numerical contract of the retired
    online path while the CUDA quantizer performs the scale reduction.
    """
    global _weight_quant_library

    if weight.ndim != 2 or weight.dtype != torch.float32 or not weight.is_cuda:
        raise ValueError("Expected CUDA FP32 weight[N,K]")
    if _weight_quant_library is None:
        from .tk_weight_quantize import QuantLibrary

        _weight_quant_library = QuantLibrary()
    plan = _weight_quant_library.plan(weight, round_bf16=True)
    return plan()


def _is_fp8_sync_module(module: nn.Module) -> bool:
    return callable(getattr(module, "refresh_fp8_weights", None)) and callable(
        getattr(module, "_sync_if_needed", None)
    )


def iter_fp8_modules(module: nn.Module) -> Iterator[nn.Module]:
    """Yield outermost modules that own synchronized MXFP8 caches."""
    if _is_fp8_sync_module(module):
        yield module
        return
    for child in module.children():
        yield from iter_fp8_modules(child)


def refresh_fp8_modules(module: nn.Module) -> None:
    for fp8_module in iter_fp8_modules(module):
        fp8_module.refresh_fp8_weights()


def mark_fp8_modules_dirty(module: nn.Module) -> None:
    for fp8_module in iter_fp8_modules(module):
        fp8_module.mark_fp8_dirty()


def bind_fp8_optimizer(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """Refresh every MXFP8 cache once after an optimizer step."""

    bind_parameter = getattr(optimizer, "bind_fp8_parameter", None)
    fused_quantization = callable(bind_parameter)
    if fused_quantization:
        bound = set()
        for submodule in module.modules():
            bindings = getattr(submodule, "fp8_parameter_bindings", None)
            if not callable(bindings):
                continue
            for parameter, weight_fp8, weight_scale in bindings():
                parameter_id = id(parameter)
                if parameter_id not in bound:
                    bind_parameter(parameter, weight_fp8, weight_scale)
                    bound.add(parameter_id)

    def post_step_hook(optimizer, args, kwargs):
        if fused_quantization:
            for submodule in module.modules():
                mark_synced = getattr(submodule, "mark_fp8_synced", None)
                if callable(mark_synced):
                    mark_synced()
        else:
            refresh_fp8_modules(module)

    return optimizer.register_step_post_hook(post_step_hook)


def step_fp8_optimizers(
    module: nn.Module,
    optimizers: Iterable[torch.optim.Optimizer],
) -> None:
    for optimizer in optimizers:
        optimizer.step()
    refresh_fp8_modules(module)
