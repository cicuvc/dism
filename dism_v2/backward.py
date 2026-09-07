"""Core delta, dV and affine summary/passing; B3 and autograd are not connected."""
from functools import lru_cache
from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load
from .core import RowRNGState, ScanBoundaries


@lru_cache(None)
def _extension():
    source=Path(__file__).resolve().parent/"csrc"
    return load(name="dism_v2_backward_sm120a",
        sources=[str(source/f) for f in ("backward_bindings.cpp","core_bwd.cu","core_dv.cu")],
        extra_include_paths=[str(source.parents[1]/"include"),
            str(Path(os.environ.get("GLX_ROOT","/home/cicuvc/cs/projects/glx"))/"include")],
        extra_cflags=["-O2","-std=c++20"],
        extra_cuda_cflags=["-O3","-std=c++20","-lineinfo","--extended-lambda","--expt-relaxed-constexpr",
            "-gencode=arch=compute_120a,code=sm_120a","--ptxas-options=-v"],
        extra_ldflags=["-lcuda"],
        verbose=os.environ.get("DISM_VERBOSE_BUILD")=="1")


def delta(dout,out):
    """FP32 rowwise dot(dO,O), BF16 contiguous [B,H,N,DV] inputs.

    Uses the stored, rounded forward O as in the planned FlashAttention-style
    backward. This is not an exact derivative of BF16 rounding. No RNG consumed.
    """
    if torch.is_grad_enabled() and (dout.requires_grad or out.requires_grad):
        raise NotImplementedError("higher-order backward is not implemented")
    return _extension().delta(dout,out)


def value_gradient(a,b,dout,lse,tau,q_label,k_label,normalizer,boundaries,*,sm_scale,rng_state,
                   v=None,delta=None):
    """Compute FP32 dV using selected A/B/LSE and saved forward states.

    Replays the forward direction and row RNG without new generator consumption.
    FP32 P is split into BF16 high/residual for two Tensor Core products; dV
    accumulation is FP32, no atomics or global W/P. No autograd yet.
    Caller must supply unchanged operands, scale and states from the same forward.
    Supplying both v and FP32 delta returns (dV, affine32_summary, G32_boundary).
    """
    if not isinstance(rng_state,RowRNGState) or not isinstance(boundaries,ScanBoundaries):
        raise TypeError("saved RowRNGState and ScanBoundaries required")
    if rng_state.shape!=tuple(a.shape[:3]) or rng_state.direction not in ("q_from_k","k_from_q"):
        raise ValueError("replay shape/direction mismatch")
    if not (0<=rng_state.seed<2**64 and 0<=rng_state.offset<2**64 and rng_state.offset%4==0):
        raise ValueError("invalid replay seed/offset")
    if not isinstance(sm_scale,(int,float)):
        raise TypeError("sm_scale must be a scalar")
    if (v is None)!=(delta is None):
        raise ValueError("v and delta must be supplied together")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (a,b,dout,lse,tau,normalizer,boundaries.vertical,boundaries.horizontal,*(() if v is None else (v,delta)))):
        raise NotImplementedError("higher-order backward is not implemented")
    result = _extension().value_gradient(a,b,dout,lse,tau,q_label,k_label,normalizer,
        boundaries.vertical,boundaries.horizontal,float(sm_scale),rng_state.direction=="k_from_q",
        rng_state.hard_prob,rng_state.seed,rng_state.offset,v,delta)
    return tuple(result) if v is not None else result[0]
