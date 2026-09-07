"""Core backward primitives; B1/B2/B3 and autograd are not connected yet."""
from functools import lru_cache
from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load


@lru_cache(None)
def _extension():
    source=Path(__file__).resolve().parent/"csrc"
    return load(name="dism_v2_backward_sm120a",
        sources=[str(source/f) for f in ("backward_bindings.cpp","core_bwd.cu")],
        extra_cflags=["-O2","-std=c++20"],
        extra_cuda_cflags=["-O3","-std=c++20","-lineinfo",
            "-gencode=arch=compute_120a,code=sm_120a","--ptxas-options=-v"],
        verbose=os.environ.get("DISM_VERBOSE_BUILD")=="1")


def delta(dout,out):
    """FP32 rowwise dot(dO,O), BF16 contiguous [B,H,N,DV] inputs.

    Uses the stored, rounded forward O as in the planned FlashAttention-style
    backward. This is not an exact derivative of BF16 rounding. No RNG consumed.
    """
    if torch.is_grad_enabled() and (dout.requires_grad or out.requires_grad):
        raise NotImplementedError("higher-order backward is not implemented")
    return _extension().delta(dout,out)
