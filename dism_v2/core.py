"""Initial sm120 core forward, independent of the legacy dism interfaces.

Accepts already selected, BF16 interpolation operands. Only fixed directions
and hard_prob=0/1 are implemented. No autograd, RNG, varlen or graph promise yet.
"""
from functools import lru_cache
from pathlib import Path
import os

import torch
from torch.utils.cpp_extension import load


@lru_cache(None)
def _extension():
    root = Path(__file__).resolve().parent.parent
    glx = Path(os.environ.get("GLX_ROOT", "/home/cicuvc/cs/projects/glx"))
    return load(
        name="dism_v2_core_sm120a",
        sources=[str(root / "dism_v2/csrc" / f) for f in ("bindings.cpp", "core_fwd.cu")],
        extra_include_paths=[str(root / "include"), str(glx / "include")],
        extra_cflags=["-O2", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo", "--extended-lambda",
                          "--expt-relaxed-constexpr", "-gencode=arch=compute_120a,code=sm_120a",
                          "--ptxas-options=-v"],
        extra_ldflags=["-lcuda"], verbose=os.environ.get("DISM_VERBOSE_BUILD") == "1",
    )


def forward(a, b, v, lse, tau, q_label, k_label, *, sm_scale, direction,
            hard_prob=0.0, return_debug=False):
    """Return (BF16 O, FP32 log2 normalizer); optional checkpoint diagnostics.

    direction='q_from_k': a=q, b=q_from_k, lse=q_lse.
    direction='k_from_q': a=k_from_q, b=k, lse=k_lse.
    BF16 interpolation conversion is the caller's explicit responsibility.
    All input tensors must be contiguous. Checkpoint diagnostics are O(N^2/32),
    never full logM/W/P. Current implementation computes the full key range.
    """
    if direction not in ("q_from_k", "k_from_q"):
        raise NotImplementedError("only fixed directions are implemented")
    if not isinstance(hard_prob, (int, float)) or hard_prob not in (0.0, 1.0):
        raise NotImplementedError("mixed RNG and probability broadcasting are not implemented yet")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (a,b,v,lse,tau)):
        raise NotImplementedError("core backward is not implemented")
    if not isinstance(sm_scale, (int, float)):
        raise TypeError("sm_scale must be a scalar")
    out, norm, summary, boundary = _extension().forward(
        a,b,v,lse,tau,q_label,k_label,float(sm_scale),direction=="k_from_q",bool(hard_prob))
    return (out,norm,summary,boundary) if return_debug else (out,norm)
