"""Reproduce the large-M MXFP8 down-projection forward error.

The down projection's input activation quantizer corrupts its output for
sufficiently large M. The GEMM itself remains correct when the same GEMM is
given an equivalent input quantized manually in PyTorch.
"""

import argparse

import torch
from torch.nn import functional as F

from flash_mxfp8.tk_mxfp8_down import MXFP8DownProjection
from flash_mxfp8.tk_mxfp8_gemm import Library as GemmLibrary, pack_a_scales


def quantize_rows(activation):
    """Quantize BF16 [M,K] rows to E4M3 with E8M0 scales over each K32."""
    m, k = activation.shape
    if k % 32:
        raise ValueError("K must be divisible by 32")
    bits = activation.contiguous().view(torch.uint16).to(torch.int32)
    max_exponent = (bits & 0x7F80).reshape(m, k // 32, 32).amax(-1)
    # E4M3 exponent offset is 7; this matches shift_e8m0 in the CUDA kernel.
    scale_codes = ((max_exponent >> 7).clamp_min(7) - 7).to(torch.uint8)
    scales = torch.ldexp(
        torch.ones_like(scale_codes, dtype=torch.float32),
        scale_codes.to(torch.int32) - 127,
    )
    quantized = (
        activation.float().reshape(m, k // 32, 32) / scales[..., None]
    ).to(torch.float8_e4m3fn).reshape(m, k)
    return quantized, pack_a_scales(scale_codes)


def metrics(actual, reference):
    actual = actual.detach().float().flatten()
    reference = reference.detach().float().flatten()
    difference = actual - reference
    return {
        "cosine": float(
            torch.dot(actual, reference)
            / (actual.norm() * reference.norm()).clamp_min(1e-30)
        ),
        "relative_l2": float(
            difference.norm() / reference.norm().clamp_min(1e-30)
        ),
    }


def run(tokens, seed):
    torch.manual_seed(seed)
    down = MXFP8DownProjection(
        3968,
        512,
        device="cuda",
        forward_config=67,
        dx_config=17,
        dw_config=12,
        raster_group=13,
    )
    activation = torch.randn(
        tokens, 3968, device="cuda", dtype=torch.bfloat16
    )

    module_output = down(activation)
    quantized, scale = quantize_rows(activation)
    manual_gemm = GemmLibrary().plan(
        quantized,
        down._weight_fp8,
        scale,
        down._weight_scale,
        config=67,
        raster_group=13,
    )
    manual_output = manual_gemm()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference = F.linear(activation, down.weight)

    return {
        "tokens": tokens,
        "module_forward": metrics(module_output, reference),
        "manual_input_quantization_then_gemm": metrics(
            manual_output, reference
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 3072])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    for tokens in args.tokens:
        result = run(tokens, args.seed + tokens)
        print(
            f"M={tokens}: "
            f"module(cos={result['module_forward']['cosine']:.6f}, "
            f"rel={result['module_forward']['relative_l2']:.6f}), "
            "manual_q+gemm"
            f"(cos={result['manual_input_quantization_then_gemm']['cosine']:.6f}, "
            f"rel={result['manual_input_quantization_then_gemm']['relative_l2']:.6f})"
        )


if __name__ == "__main__":
    main()
