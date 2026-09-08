"""Experimental CUDA embedding forward, with a single-warp diagnostic baseline.

No autograd here. The production end-to-end path still uses Triton embedding.
"""
from functools import lru_cache
from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load


@lru_cache(None)
def _extension():
    src = Path(__file__).resolve().parent / "csrc"
    return load(
        name="dism_v2_embedding_sm120a",
        sources=[str(src / n) for n in ("embedding_bindings.cpp", "embedding_fwd.cu")],
        extra_include_paths=[str(src.parents[1] / "include")],
        extra_cflags=["-O2", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo", "--extended-lambda",
                          "--expt-relaxed-constexpr", "-gencode=arch=compute_120a,code=sm_120a",
                          "--ptxas-options=-v"],
        extra_ldflags=["-lcuda"],
        verbose=os.environ.get("DISM_VERBOSE_BUILD") == "1")


def forward(q, k, q_voc, k_voc, sm_scale=1.0, *, warp_specialized=True, block_v=64):
    """Return the same eight outputs/order as emb_fwd_wrapper.

    Defaults to fused 12-warp/two-slot CUDA forward. Set warp_specialized=False
    for two independent single-warp FA launches. End-to-end voc_dism still
    defaults to Triton embedding; opt in with embedding_backend="cuda" there.
    block_v=128 is an experimental WS option for D=32/64 only.
    """
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, q_voc, k_voc)):
        raise NotImplementedError("CUDA embedding forward has no autograd wrapper yet")
    return tuple(_extension().forward(q, k, q_voc, k_voc, float(sm_scale), warp_specialized, block_v))


@lru_cache(None)
def _backward_extension():
    src=Path(__file__).resolve().parent/"csrc"
    return load(name="dism_v2_embedding_backward_sm120a",
        sources=[str(src/n) for n in ("embedding_bwd_bindings.cpp","embedding_bwd.cu")],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[str(src.parents[1]/"include")],
        extra_cflags=["-O2","-std=c++20"],
        extra_cuda_cflags=["-O3","-std=c++20","-lineinfo","--extended-lambda",
                          "--expt-relaxed-constexpr","-gencode=arch=compute_120a,code=sm_120a",
                          "--ptxas-options=-v"],
        verbose=os.environ.get("DISM_VERBOSE_BUILD")=="1")


def backward(q,k,q_voc,k_voc,out,k_lse,q_lse,u,dlse,*,direction,sm_scale=1.,return_preprocess=False,warp_specialized=False):
    """Dism-specific sparse embedding backward, currently single-warp baseline.

    out/U belong to the selected interpolation; dlse to the OPPOSITE branch.
    U and dlse are FP32. Returns FP32 (dq,dk,dq_voc,dk_voc), embedding terms only.
    Uses saved BF16 output for FP32 delta before BF16 U packing. No RNG.
    """
    if direction not in ("q_from_k","k_from_q"):
        raise ValueError("backward requires the saved fixed direction")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,q_voc,k_voc,out,k_lse,q_lse,u,dlse)):
        raise NotImplementedError("higher-order embedding backward is not implemented")
    if direction=="q_from_k":
        raw=_backward_extension().backward(k,q,k_voc,q_voc,out,u,k_lse,q_lse,dlse,float(sm_scale),warp_specialized)
        result=(raw[1],raw[0],raw[3],raw[2])
    else:
        raw=_backward_extension().backward(q,k,q_voc,k_voc,out,u,q_lse,k_lse,dlse,float(sm_scale),warp_specialized)
        result=tuple(raw[:4])
    return result+tuple(raw[4:]) if return_preprocess else result
