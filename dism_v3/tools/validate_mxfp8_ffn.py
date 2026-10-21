"""Compare MXFP8 and BF16 SwiGLU outputs and gradients on identical weights."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from flash_mxfp8 import MXFP8SwiGLU
from flash_mxfp8.tk_mxfp8_down import MXFP8DownProjection


CUDA = torch.device("cuda")


class BF16SwiGLU(nn.Module):
    """The reference SwiGLU computation used by the BF16 training path."""

    def __init__(self, gate, up, down):
        super().__init__()
        self.gate_weight = nn.Parameter(gate.detach().clone())
        self.up_weight = nn.Parameter(up.detach().clone())
        self.down_weight = nn.Parameter(down.detach().clone())

    def forward(self, x):
        gate = F.linear(x, self.gate_weight)
        up = F.linear(x, self.up_weight)
        return F.linear(F.silu(gate) * up, self.down_weight)


def error_metrics(actual, reference):
    actual = actual.detach().float()
    reference = reference.detach().float()
    difference = actual - reference
    actual_norm = torch.linalg.vector_norm(actual)
    reference_norm = torch.linalg.vector_norm(reference)
    difference_norm = torch.linalg.vector_norm(difference)
    denominator = (actual_norm * reference_norm).clamp_min(1e-30)
    return {
        "relative_l2": float(difference_norm / reference_norm.clamp_min(1e-30)),
        "cosine": float(torch.sum(actual * reference) / denominator),
        "reference_rms": float(torch.sqrt(torch.mean(reference.square()))),
        "actual_rms": float(torch.sqrt(torch.mean(actual.square()))),
        "error_rms": float(torch.sqrt(torch.mean(difference.square()))),
        "max_abs_error": float(difference.abs().max()),
        "max_abs_reference": float(reference.abs().max()),
    }


def e8m0_values(codes):
    codes = codes.to(torch.int32)
    return torch.ldexp(
        torch.ones_like(codes, dtype=torch.float32),
        codes - 127,
    )


def dequantize_row_major(values, scales, m, n):
    groups = (n + 127) // 128
    padded_m = (m + 15) // 16 * 16
    raw = scales.view(torch.uint8).reshape(groups, padded_m, 4)
    codes = raw.permute(1, 0, 2).reshape(padded_m, groups * 4)[:m]
    codes = codes[:, :(n + 31) // 32].repeat_interleave(32, dim=1)[:, :n]
    return values.float() * e8m0_values(codes)


def dequantize_column_major(values, scales, m, n):
    padded_m = (m + 127) // 128 * 128
    groups = padded_m // 128
    raw = scales.view(torch.uint8).reshape(groups, n, 4)
    codes = raw.permute(1, 0, 2).reshape(n, groups * 4)
    codes = codes.T.repeat_interleave(32, dim=0)[:m]
    return values[:, :m].T.float() * e8m0_values(codes)


def load_weights(checkpoint, layer):
    state = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    prefix = f"layers.{layer}.mlp."
    return tuple(
        state[prefix + name].to(device=CUDA, dtype=torch.float32)
        for name in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
    )


def random_weights(width, hidden, seed):
    generator = torch.Generator(device=CUDA).manual_seed(seed)
    shapes = ((hidden, width), (hidden, width), (width, hidden))
    return tuple(
        torch.randn(shape, device=CUDA, dtype=torch.float32, generator=generator)
        * (1.0 / math.sqrt(shape[1]))
        for shape in shapes
    )


def run_case(label, weights, tokens, input_rms, seed, config, raster_group):
    gate, up, down_weight = weights
    width, hidden = gate.shape[1], gate.shape[0]
    mxfp8 = MXFP8SwiGLU(
        width,
        hidden,
        width,
        device=CUDA,
        forward_config=config[0],
        dx_config=config[1],
        wgrad_config=config[2],
        down_forward_config=config[3],
        down_dx_config=config[4],
        down_dw_config=config[5],
        raster_group=raster_group,
    )
    reference = BF16SwiGLU(gate, up, down_weight).to(CUDA)
    with torch.no_grad():
        mxfp8.gate_weight.copy_(gate)
        mxfp8.up_weight.copy_(up)
        mxfp8.down_proj.weight.copy_(down_weight)
    mxfp8.refresh_fp8_weights()

    generator = torch.Generator(device=CUDA).manual_seed(seed)
    x = torch.randn(
        (tokens, width), device=CUDA, dtype=torch.bfloat16, generator=generator
    )
    x.mul_(input_rms)
    dy = torch.randn(
        (tokens, width), device=CUDA, dtype=torch.bfloat16, generator=generator
    )
    dy.mul_(1.0 / math.sqrt(width))

    mxfp8.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    reference_input = x.detach().clone().requires_grad_()
    mxfp8_input = x.detach().clone().requires_grad_()
    mxfp8_output = mxfp8(mxfp8_input)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference_gate = F.linear(reference_input, reference.gate_weight)
        reference_up = F.linear(reference_input, reference.up_weight)
        reference_output = reference(reference_input)
        down_input = F.silu(reference_gate) * reference_up

    down = MXFP8DownProjection(
        hidden,
        width,
        device=CUDA,
        forward_config=config[3],
        dx_config=config[4],
        dw_config=config[5],
        raster_group=raster_group,
    )
    with torch.no_grad():
        down.weight.copy_(down_weight)
    down.refresh_fp8_weight()
    down_mxfp8_input = down_input.detach().clone().requires_grad_()
    down_reference_input = down_input.detach().clone().requires_grad_()
    down_reference_weight = down_weight.detach().clone().requires_grad_()
    mxfp8_down_output = down(down_mxfp8_input)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference_down_output = F.linear(
            down_reference_input, down_reference_weight
        )
    mxfp8_down_output.backward(dy)
    reference_down_output.backward(dy)

    from flash_mxfp8.tk_silu_bwd import SiluBwdLibrary, SiluBwdQuantizeLibrary

    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden_gradient = torch.matmul(dy, reference.down_weight)
    elementwise_backward = SiluBwdLibrary().plan(
        hidden_gradient, 0.5 * reference_gate, reference_up
    )
    actual_dgate, actual_dup = elementwise_backward()
    quantized_backward = SiluBwdQuantizeLibrary().plan(
        hidden_gradient, 0.5 * reference_gate, reference_up
    )
    quantized_backward()
    reference_gate_for_grad = reference_gate.detach().requires_grad_()
    reference_up_for_grad = reference_up.detach().requires_grad_()
    reference_dgate, reference_dup = torch.autograd.grad(
        F.silu(reference_gate_for_grad) * reference_up_for_grad,
        (reference_gate_for_grad, reference_up_for_grad),
        grad_outputs=hidden_gradient,
    )
    mxfp8_output.backward(dy)
    reference_output.backward(dy)

    forward_state = next(reversed(mxfp8._forward_states.values()))
    actual_gate = 2.0 * forward_state.gate_linear.float()
    actual_up = forward_state.up_linear.float()
    gradients = {
        "dx": (mxfp8_input.grad, reference_input.grad),
        "gate_weight": (mxfp8.gate_weight.grad, reference.gate_weight.grad),
        "up_weight": (mxfp8.up_weight.grad, reference.up_weight.grad),
        "down_weight": (
            mxfp8.down_proj.weight.grad,
            reference.down_weight.grad,
        ),
    }
    report = {
        "label": label,
        "tokens": tokens,
        "input_rms": input_rms,
        "output": error_metrics(mxfp8_output, reference_output),
        "gate": error_metrics(actual_gate, reference_gate),
        "up": error_metrics(actual_up, reference_up),
        "down_projection": {
            "output": error_metrics(mxfp8_down_output, reference_down_output),
            "dx": error_metrics(
                down_mxfp8_input.grad, down_reference_input.grad
            ),
            "weight": error_metrics(
                down.weight.grad, down_reference_weight.grad
            ),
        },
        "elementwise": {
            "dgate": error_metrics(actual_dgate, reference_dgate),
            "dup": error_metrics(actual_dup, reference_dup),
        },
        "elementwise_quantized": {
            "dgate_row": error_metrics(
                dequantize_row_major(
                    quantized_backward.gate_row,
                    quantized_backward.gate_row_scale,
                    tokens,
                    hidden,
                ),
                reference_dgate,
            ),
            "dgate_column": error_metrics(
                dequantize_column_major(
                    quantized_backward.gate_col,
                    quantized_backward.gate_col_scale,
                    tokens,
                    hidden,
                ),
                reference_dgate,
            ),
            "dup_row": error_metrics(
                dequantize_row_major(
                    quantized_backward.up_row,
                    quantized_backward.up_row_scale,
                    tokens,
                    hidden,
                ),
                reference_dup,
            ),
        },
        "gradients": {
            name: error_metrics(actual, expected)
            for name, (actual, expected) in gradients.items()
        },
    }
    del mxfp8, reference, down, x, dy, reference_input, mxfp8_input
    del mxfp8_output, reference_output, gradients
    del down_input, down_mxfp8_input, down_reference_input
    del down_reference_weight, mxfp8_down_output, reference_down_output
    del elementwise_backward, quantized_backward, actual_dgate, actual_dup
    del reference_gate_for_grad, reference_up_for_grad
    del reference_dgate, reference_dup
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192, 16384])
    parser.add_argument("--input-rms", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--config",
        type=int,
        nargs=6,
        default=[8, 12, 14, 67, 17, 12],
        metavar=("GATE_FWD", "GATE_DX", "GATE_DW",
                 "DOWN_FWD", "DOWN_DX", "DOWN_DW"),
    )
    parser.add_argument("--raster-group", type=int, default=13)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.checkpoint is not None:
        weights = load_weights(args.checkpoint, args.layer)
        weight_source = {
            "kind": "checkpoint",
            "path": str(args.checkpoint),
            "layer": args.layer,
        }
    else:
        weights = random_weights(512, 3968, args.seed)
        weight_source = {"kind": "random", "seed": args.seed}

    cases = []
    for tokens in args.tokens:
        for input_rms in args.input_rms:
            label = f"tokens{tokens}-rms{input_rms:g}"
            cases.append(
                run_case(
                    label,
                    weights,
                    tokens,
                    input_rms,
                    args.seed + tokens + round(input_rms * 1000),
                    args.config,
                    args.raster_group,
                )
            )
    report = {
        "gpu": torch.cuda.get_device_name(),
        "weights": weight_source,
        "config": args.config,
        "raster_group": args.raster_group,
        "cases": cases,
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
