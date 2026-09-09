"""Core delta, dV, affine summary/passing and operand gradients; no autograd yet."""
from functools import lru_cache
from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load
from .core import RowRNGState, ScanBoundaries
from .kernel_config import LSE_SUFFIX, LSE_FLAGS


@lru_cache(None)
def _extension():
    source=Path(__file__).resolve().parent/"csrc"
    return load(name="dism_v2_backward_sm120a" + LSE_SUFFIX,
        sources=[str(source/f) for f in ("backward_bindings.cpp","core_bwd.cu","core_dv.cu","core_dv_ws.cu","core_ab.cu","core_ab_ws.cu","core_tau.cu")],
        extra_include_paths=[str(source.parents[1]/"include"),
            str(Path(os.environ.get("GLX_ROOT","/home/cicuvc/cs/projects/glx"))/"include")],
        extra_cflags=["-O2","-std=c++20"],
        extra_cuda_cflags=["-O3","-std=c++20","-lineinfo","--extended-lambda","--expt-relaxed-constexpr",
            "-gencode=arch=compute_120a,code=sm_120a","--ptxas-options=-v"] + LSE_FLAGS,
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
                   v=None,delta=None,warp_specialized=False,hard_bits=None):
    """Compute FP32 dV using selected A/B/LSE and saved forward states.

    Replays the forward direction and row RNG without new generator consumption.
    The single-warp baseline splits FP32 P into BF16 high/residual for two
    Tensor Core products. The experimental WS path uses one BF16 P product
    (known long-sequence precision tradeoff). dV accumulation is FP32,
    no atomics or global W/P. No autograd yet.
    Caller must supply unchanged operands, scale and states from the same forward.
    Supplying both v and FP32 delta returns (dV, affine32_summary, G32_boundary).
    warp_specialized=True selects the experimental12-warp/double-buffer path.
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
    if warp_specialized and v is None:
        raise ValueError("warp_specialized requires v and delta")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (a,b,dout,lse,tau,normalizer,boundaries.vertical,boundaries.horizontal,*(() if v is None else (v,delta)))):
        raise NotImplementedError("higher-order backward is not implemented")
    result = _extension().value_gradient(a,b,dout,lse,tau,q_label,k_label,normalizer,
        boundaries.vertical,boundaries.horizontal,float(sm_scale),rng_state.direction=="k_from_q",
        rng_state.hard_prob,rng_state.seed,rng_state.offset,v,delta,warp_specialized,hard_bits)
    return tuple(result) if v is not None else result[0]


def operand_gradient(a,b,v,dout,lse,tau,q_label,k_label,normalizer,delta,boundaries,g_boundary,*,sm_scale,rng_state,warp_specialized=True,hard_bits=None):
    """B3: FP32 (dA,dB,dLSE,drtau), no global G; defaults to 12-warp WS.

    Requires G32 boundaries from value_gradient(v=...,delta=...).
    Replays RNG; BF16 Gsoft MMA; dA uses TMA atomic add and is nondeterministic.
    dLSE [B,H,N] is for the selected direction's LSE; drtau [H] sums batches.
    Scalar gradients reduce FP32 G before BF16 conversion, with no extra scale.
    No autograd integration yet. D128 spill accepted for validation.
    warp_specialized=False selects the single-warp32-key diagnostic baseline.
    """
    if not isinstance(rng_state,RowRNGState) or not isinstance(boundaries,ScanBoundaries):
        raise TypeError("saved RowRNGState and ScanBoundaries required")
    if rng_state.shape!=tuple(a.shape[:3]) or rng_state.direction not in ("q_from_k","k_from_q"):
        raise ValueError("replay shape/direction mismatch")
    if not (0<=rng_state.seed<2**64 and 0<=rng_state.offset<2**64 and rng_state.offset%4==0):
        raise ValueError("invalid replay seed/offset")
    if not isinstance(sm_scale,(int,float)):
        raise TypeError("sm_scale must be a scalar")
    if torch.is_grad_enabled() and any(x.requires_grad for x in
            (a,b,v,dout,lse,tau,normalizer,delta,boundaries.vertical,boundaries.horizontal,g_boundary)):
        raise NotImplementedError("higher-order backward is not implemented")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("CUDA Graph capture is not supported")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("operand_gradient uses nondeterministic TMA atomic reduction")
    return tuple(_extension().operand_gradient(a,b,v,dout,lse,tau,q_label,k_label,normalizer,delta,
        boundaries.vertical,boundaries.horizontal,g_boundary,float(sm_scale),rng_state.direction=="k_from_q",
        rng_state.hard_prob,rng_state.seed,rng_state.offset,warp_specialized,hard_bits))
