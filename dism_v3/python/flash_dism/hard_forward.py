"""Explicit hard-only CUDA forward; not registered as a training autograd path."""
import torch
import cu_flash_dism as cu

from .summary import summarize_hard, hard_chunk_scan


@torch.no_grad()
def hard_log_weights(tau, max_length):
    """Fresh FP32 log2 geometric-sum table [H,max_length+1], finite per-head tau."""
    return cu.hard_log_weights(tau, max_length)


@torch.no_grad()
def hard_readout(sq, sk, v, idx_q, idx_k, hard, boundary, log_weights, *, layout=None, ctas=0):
    """Fused hard matching/normalization/readout with integer incoming boundaries.

    SQ/SK/V: contiguous BF16 BNHD; labels: int32 BHN; hard: sequence bool.
    Returns BF16 output BNHD and FP32 log2 normalizer BHN. Soft sequences are
    skipped (zero output/statistics); no vocabulary embedding or soft attention
    is computed. Boundary and table must belong to this invocation's labels/tau.
    """
    return cu.hard_readout((sq,sk,v,idx_q,idx_k,hard),boundary,log_weights,
                           None if layout is None else layout.table,ctas)


@torch.no_grad()
def forward_hard(sq, sk, v, idx_q, idx_k, hard, tau, *, layout=None, ctas=0, save_state=False):
    """Complete standalone hard contribution: summary, passing, LUT and readout.

    Fixed N and packed document boundaries are 256-aligned. Packed flags are
    sampled per original equal-length training row/head, not per document.
    This explicit forward has no autograd registration or backward dispatch.
    Direction is irrelevant for hard matching. No checkpoint/RNG state changes.
    """
    summary = summarize_hard(idx_q,idx_k,hard,layout=layout,ctas=ctas)
    boundary = hard_chunk_scan(summary,layout=layout,heads=sq.shape[2])
    max_length = sq.shape[1] if layout is None else max(layout.lengths)
    lut = hard_log_weights(tau,max_length)
    output,normalizer = hard_readout(sq,sk,v,idx_q,idx_k,hard,boundary,lut,layout=layout,ctas=ctas)
    if save_state:
        return output,normalizer,dict(summary=summary,boundary=boundary,log_weights=lut)
    return output,normalizer
