"""Initial sm120 core forward, independent of the legacy dism interfaces.

Accepts already selected, BF16 interpolation operands and warp-local row RNG.
No autograd, random direction, varlen or graph support yet.
"""
from functools import lru_cache
from pathlib import Path
import os
import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RowRNGState:
    seed: int
    offset: int
    shape: tuple[int, int, int]
    direction: str
    hard_prob: float


def forward(a, b, v, lse, tau, q_label, k_label, *, sm_scale, direction,
            hard_prob=0.0, return_debug=False, generator=None, rng_state=None,
            return_rng_state=False):
    """Return (BF16 O, FP32 log2 normalizer); optional checkpoint diagnostics.

    direction='q_from_k': a=q, b=q_from_k, lse=q_lse.
    direction='k_from_q': a=k_from_q, b=k, lse=k_lse.
    BF16 interpolation conversion is the caller's explicit responsibility.
    All input tensors must be contiguous. Checkpoint diagnostics are O(N^2/32),
    never full logM/W/P. Current implementation computes the full key range.
    Mixed scalar probability reserves four Philox words per row subsequence
    from the CUDA generator. Endpoints and explicit replay consume nothing.
    return_rng_state appends replay metadata to the usual result tuple.
    """
    if direction not in ("q_from_k", "k_from_q"):
        raise NotImplementedError("only fixed directions are implemented")
    if not isinstance(hard_prob, (int, float)):
        raise NotImplementedError("probability broadcasting is not implemented yet")
    if not math.isfinite(hard_prob) or not 0 <= hard_prob <= 1:
        raise ValueError("hard_prob must be finite and in [0,1]")
    if generator is not None and rng_state is not None:
        raise ValueError("generator and rng_state are mutually exclusive")
    if rng_state is not None:
        if not isinstance(rng_state, RowRNGState):
            raise TypeError("rng_state must be RowRNGState")
        if (rng_state.shape, rng_state.direction, rng_state.hard_prob) != (tuple(a.shape[:3]), direction, hard_prob):
            raise ValueError("replay shape, direction and probability must match")
        if not (0 <= rng_state.seed < 2**64 and 0 <= rng_state.offset < 2**64 and rng_state.offset % 4 == 0):
            raise ValueError("invalid replay seed/offset")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (a,b,v,lse,tau)):
        raise NotImplementedError("core backward is not implemented")
    if not isinstance(sm_scale, (int, float)):
        raise TypeError("sm_scale must be a scalar")
    tensors, seed, offset = _extension().forward(
        a,b,v,lse,tau,q_label,k_label,float(sm_scale),direction=="k_from_q",float(hard_prob),
        generator, None if rng_state is None else (rng_state.seed, rng_state.offset))
    result = tuple(tensors) if return_debug else tuple(tensors[:2])
    state = RowRNGState(seed, offset, tuple(a.shape[:3]), direction, float(hard_prob))
    return (*result, state) if return_rng_state else result
