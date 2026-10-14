"""Small, sanitizer-friendly driver for the fused value-residual kernels.

Intended for ``compute-sanitizer --tool memcheck|racecheck|synccheck``: it
covers BF16/FP32, DV 32/48/64/1, a non-contiguous head layout and a
non-contiguous upstream gradient with a tiny grid, so the run stays short.
"""

import torch

from flash_dism.kernels.value_residual import (
    torch_bwd_value_residual,
    torch_fwd_value_residual,
)


def strided(shape, strides, dtype, base):
    out = torch.empty_strided(shape, strides, dtype=dtype, device="cuda")
    out.copy_(base)
    return out


def case(batch, seq, heads, dim, dtype, noncontiguous=False):
    v = torch.randn(batch, seq, heads, dim, device="cuda", dtype=dtype)
    v_first = torch.randn_like(v)
    gate = torch.randn(batch, seq, heads, device="cuda", dtype=dtype)
    dy = torch.randn_like(v)
    if noncontiguous:
        v = strided(v.shape, (seq * heads * dim * 2, heads * dim * 2, dim * 2, 1),
                    dtype, v)
        dy = strided(dy.shape, (seq * heads * dim * 2, heads * dim * 2, dim * 2, 1),
                     dtype, dy)
        gate = strided(gate.shape, (seq * heads * 3, heads * 3, 3), dtype, gate)
    out = torch_fwd_value_residual(v, v_first, gate)
    torch.cuda.synchronize()
    dv, dvfirst, dgate = torch_bwd_value_residual(v, v_first, gate, dy)
    torch.cuda.synchronize()
    return (out.sum() + dv.sum() + dvfirst.sum() + dgate.sum()).item()


def main():
    torch.manual_seed(0)
    values = [
        sum(case(2, 5, 3, 64, torch.bfloat16) for _ in range(2)),
        case(1, 9, 5, 48, torch.float32),
        case(2, 4, 3, 32, torch.bfloat16, noncontiguous=True),
        case(1, 3, 2, 1, torch.float32),
    ]
    # Last-axis slicing and scalar-expanded dout were absent in the original
    # smoke test; neither may be treated as a unit-stride vector.
    for dtype in (torch.bfloat16, torch.float32):
        v = torch.randn(1, 7, 3, 128, device='cuda', dtype=dtype)[..., ::2]
        vf = torch.randn(1, 7, 64, 3, device='cuda', dtype=dtype).transpose(-1, -2)
        gate = torch.randn(1, 7, 3, device='cuda', dtype=dtype)
        torch_fwd_value_residual(v, vf, gate)
        for dy in (v, torch.ones((), device='cuda', dtype=dtype).expand_as(v)):
            torch_bwd_value_residual(v, vf, gate, dy)
    torch.cuda.synchronize()
    print([round(value, 3) for value in values])
    print("OK")


if __name__ == "__main__":
    main()
