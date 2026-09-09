"""Initial sm120 core forward, independent of the legacy dism interfaces.

Accepts already selected, BF16 interpolation operands and warp-local row RNG.
Supports fixed or per-call random direction. No autograd, varlen or graph support yet.
"""
from functools import lru_cache
from pathlib import Path
import os
import math
from dataclasses import dataclass

import torch
from torch.utils.cpp_extension import load
from .kernel_config import LSE_SUFFIX, LSE_FLAGS


@lru_cache(None)
def _extension():
    root = Path(__file__).resolve().parent.parent
    glx = Path(os.environ.get("GLX_ROOT", "/home/cicuvc/cs/projects/glx"))
    return load(
        name="dism_v2_core_sm120a" + LSE_SUFFIX,
        sources=[str(root / "dism_v2/csrc" / f) for f in ("bindings.cpp", "core_fwd.cu")],
        extra_include_paths=[str(root / "include"), str(glx / "include")],
        extra_cflags=["-O2", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "--extended-lambda",
                          "--expt-relaxed-constexpr", "-gencode=arch=compute_120a,code=sm_120a",
                          "--ptxas-options=-v"] + LSE_FLAGS
                          + (["-lineinfo"] if os.environ.get("DISM_LINEINFO", "0") == "1" else []),
        extra_ldflags=["-lcuda"], verbose=os.environ.get("DISM_VERBOSE_BUILD") == "1",
    )


@dataclass(frozen=True)
class RowRNGState:
    seed: int
    offset: int
    shape: tuple[int, int, int]
    direction: str
    hard_prob: float


@dataclass(frozen=True)
class ScanBoundaries:
    """FP32 log2 W: vertical [B,H,paddedN/16,paddedN], horizontal /64.

    Last axis is query for vertical, key for horizontal. Padding transports
    diagonal state using affine identity; it is not an attention weight.
    """
    vertical: torch.Tensor
    horizontal: torch.Tensor


def forward(a, b, v, lse, tau, q_label, k_label, *, sm_scale, direction,
            hard_prob=0.0, return_debug=False, generator=None, rng_state=None,
            return_rng_state=False, save_boundaries=False, hard_bits=None):
    """Return (BF16 O, FP32 log2 normalizer); optional checkpoint diagnostics.

    direction='q_from_k': a=q, b=q_from_k, lse=q_lse.
    direction='k_from_q': a=k_from_q, b=k, lse=k_lse.
    direction='random': a=(q,k_from_q), b=(q_from_k,k), lse=(q_lse,k_lse).
    Alternatively use forward_interpolated to supply both directions by name.
    BF16 interpolation conversion is the caller's explicit responsibility.
    All input tensors must be contiguous. Checkpoint diagnostics are O(N^2/32),
    never full logM/W/P. Current implementation computes the full key range.
    Mixed scalar probability reserves four Philox words per row subsequence
    from the CUDA generator. Endpoints and explicit replay consume nothing.
    return_rng_state appends replay metadata to the usual result tuple.
    save_boundaries appends ScanBoundaries before optional RNG metadata.
    """
    if direction not in ("q_from_k", "k_from_q", "random"):
        raise ValueError("unknown direction")
    alternative = None
    if direction == "random":
        if any(not isinstance(x, (tuple,list)) or len(x)!=2 for x in (a,b,lse)):
            raise ValueError("random direction requires two operands for each of a, b, lse")
        alternative = [a[1],b[1],lse[1]]
        a,b,lse = a[0],b[0],lse[0]
    if not isinstance(hard_prob, (int, float)):
        raise NotImplementedError("probability broadcasting is not implemented yet")
    if not math.isfinite(hard_prob) or not 0 <= hard_prob <= 1:
        raise ValueError("hard_prob must be finite and in [0,1]")
    if generator is not None and rng_state is not None:
        raise ValueError("generator and rng_state are mutually exclusive")
    if rng_state is not None:
        if not isinstance(rng_state, RowRNGState):
            raise TypeError("rng_state must be RowRNGState")
        if (rng_state.shape != tuple(a.shape[:3]) or rng_state.hard_prob != hard_prob or
            rng_state.direction not in ("q_from_k", "k_from_q") or
            (direction != "random" and rng_state.direction != direction)):
            raise ValueError("replay shape, direction and probability must match")
        if not (0 <= rng_state.seed < 2**64 and 0 <= rng_state.offset < 2**64 and rng_state.offset % 4 == 0):
            raise ValueError("invalid replay seed/offset")
    grad_inputs = (a,b,v,lse,tau,*(alternative or []))
    if torch.is_grad_enabled() and any(x.requires_grad for x in grad_inputs):
        raise NotImplementedError("core backward is not implemented")
    if not isinstance(sm_scale, (int, float)):
        raise TypeError("sm_scale must be a scalar")
    if rng_state is not None and direction == "random":
        if rng_state.direction == "k_from_q":
            a,b,lse = alternative
        alternative = None
        direction = rng_state.direction
    tensors, seed, offset, column_lse = _extension().forward(
        a,b,v,lse,tau,q_label,k_label,float(sm_scale),direction=="k_from_q",float(hard_prob),
        generator, None if rng_state is None else (rng_state.seed, rng_state.offset), alternative,
        save_boundaries, hard_bits)
    result = tuple(tensors[:4]) if return_debug else tuple(tensors[:2])
    if save_boundaries:
        result = (*result, ScanBoundaries(*tensors[4:6]))
    state = RowRNGState(seed, offset, tuple(a.shape[:3]),
                        "k_from_q" if column_lse else "q_from_k", float(hard_prob))
    return (*result, state) if return_rng_state else result


def forward_interpolated(q, k, v, tau, interpolation, *, direction="random", **kwargs):
    """Core forward from a prepared InterpolationResult; does not run embedding.

    Both interpolated operands must already be BF16. No implicit conversion.
    A random direction is chosen once for the entire call, including all B/H.
    """
    if direction == "q_from_k":
        a,b,lse = q,interpolation.q_from_k,interpolation.q_lse
    elif direction == "k_from_q":
        a,b,lse = interpolation.k_from_q,k,interpolation.k_lse
    elif direction == "random":
        a,b,lse = (q,interpolation.k_from_q),(interpolation.q_from_k,k),\
                  (interpolation.q_lse,interpolation.k_lse)
    else:
        raise ValueError("unknown direction")
    return forward(a,b,v,lse,tau,interpolation.q_index,interpolation.k_index,
                   direction=direction,**kwargs)
