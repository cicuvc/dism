"""GPU metadata and final reduction for token/document-balanced MXFP8 CE.

A document is a contiguous segment in cu_seqlens, covering loss positions.
Empty/all-ignored documents do not contribute to the document denominator.
All counts remain on device, including during CUDA Graph replay.
"""

import torch
import triton
import triton.language as tl

from .mxfp8_ce import _validate_target, _validate_ignore_index


@triton.jit
def _document_weights(Target, CU, Weight, Counts, M: tl.constexpr,
                      D: tl.constexpr, IGNORE: tl.constexpr, B: tl.constexpr):
    doc = tl.program_id(0)
    begin = tl.load(CU + doc).to(tl.int64)
    end = tl.load(CU + doc + 1).to(tl.int64)
    tl.device_assert((begin >= 0) & (end >= begin) & (end <= M), "invalid cu_seqlens interval")
    tl.device_assert((doc != 0) | (begin == 0), "cu_seqlens must start at zero")
    tl.device_assert((doc != D - 1) | (end == M), "cu_seqlens must end at token count")
    # Bound addresses even on invalid metadata before a device assertion traps.
    begin = tl.minimum(tl.maximum(begin, 0), M)
    end = tl.minimum(tl.maximum(end, begin), M)
    offsets = tl.arange(0, B)
    counts = tl.full((B,), 0, tl.int32)
    for start in range(begin, end, B):
        rows = start + offsets
        y = tl.load(Target + rows, rows < end, IGNORE)
        counts += ((rows < end) & (y != IGNORE)).to(tl.int32)
    count = tl.sum(counts.to(tl.int64), 0)
    tl.store(Counts + doc, count)
    inv = 1. / tl.maximum(count, 1).to(tl.float32)
    for start in range(begin, end, B):
        rows = start + offsets
        y = tl.load(Target + rows, rows < end, IGNORE)
        tl.store(Weight + rows, tl.where(y != IGNORE, inv, 0.), rows < end)


@triton.jit
def _loss_partial(Losses, Target, DocCounts, Sums, Counts, Out, Inv,
                  M: tl.constexpr, D: tl.constexpr, DOCUMENT: tl.constexpr,
                  IGNORE: tl.constexpr, B: tl.constexpr, SINGLE: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    loss = tl.load(Losses + rows, rows < M, 0)
    total = tl.sum(loss, 0)
    if DOCUMENT:
        n = tl.load(DocCounts + rows, rows < D, 0)
        count = tl.sum((n > 0).to(tl.int64), 0)
    else:
        y = tl.load(Target + rows, rows < M, IGNORE)
        count = tl.sum(((rows < M) & (y != IGNORE)).to(tl.int64), 0)
    if SINGLE:
        inv = 1. / tl.maximum(count, 1).to(tl.float32)
        tl.store(Out, total * inv)
        tl.store(Inv, inv)
    else:
        tl.store(Sums + tl.program_id(0), total)
        tl.store(Counts + tl.program_id(0), count)


@triton.jit
def _loss_finish(Sums, Counts, Out, Inv, N: tl.constexpr, B: tl.constexpr):
    # Bounded register footprint even for a large number of partial reductions.
    offsets = tl.arange(0, B)
    sums = tl.full((B,), 0., tl.float32)
    counts = tl.full((B,), 0, tl.int64)
    for start in range(0, N, B):
        i = start + offsets
        sums += tl.load(Sums + i, i < N, 0)
        counts += tl.load(Counts + i, i < N, 0)
    inv = 1. / tl.maximum(tl.sum(counts, 0), 1).to(tl.float32)
    tl.store(Out, tl.sum(sums, 0) * inv)
    tl.store(Inv, inv)


def validate_cu_seqlens(cu_seqlens, device):
    if (cu_seqlens is None or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or
            cu_seqlens.dtype not in (torch.int32, torch.int64) or
            cu_seqlens.device != device or not cu_seqlens.is_contiguous()):
        raise ValueError("Expected contiguous CUDA int32/int64 cu_seqlens[D+1] with D >= 1")


def document_weights(target, cu_seqlens, *, ignore_index=-100):
    """Return FP32 [M] inverse-valid-length weights and int64 [D] valid counts.

    Nondecreasing CUDA cu_seqlens must start at 0 and end at M; repeated
    boundaries represent empty documents. Content checks run on the device.
    """
    m = target.numel()
    if m <= 0 or not target.is_cuda:
        raise ValueError("Expected nonempty CUDA target")
    _validate_target(target, m, target.device)
    _validate_ignore_index(ignore_index)
    validate_cu_seqlens(cu_seqlens, target.device)
    d = cu_seqlens.numel() - 1
    weight = torch.empty(m, device=target.device, dtype=torch.float32)
    counts = torch.empty(d, device=target.device, dtype=torch.int64)
    with torch.cuda.device(target.device):
        _document_weights[(d,)](target, cu_seqlens, weight, counts, m, d,
                                ignore_index, min(1024, triton.next_power_of_2(m)),
                                num_warps=4, debug=True)
    return weight, counts


def reduce_balanced_loss(losses, target, *, balance, doc_counts=None, ignore_index=-100):
    """Normalize the sum and return its device scalar gradient multiplier.

    For document balance, losses must already include inverse document lengths.
    The common normalization is applied to saved dX/dW only in backward.
    """
    if balance not in ("token", "document"):
        raise ValueError("Expected token or document balance")
    if (losses.ndim != 1 or losses.numel() <= 0 or losses.dtype != torch.float32 or
            not losses.is_cuda or not losses.is_contiguous()):
        raise ValueError("Expected nonempty contiguous CUDA FP32 losses[M]")
    m = losses.numel()
    _validate_target(target, m, losses.device)
    _validate_ignore_index(ignore_index)
    d = 0
    if balance == "document":
        if (doc_counts is None or doc_counts.ndim != 1 or doc_counts.numel() <= 0 or
                doc_counts.dtype != torch.int64 or doc_counts.device != losses.device or
                not doc_counts.is_contiguous()):
            raise ValueError("Expected contiguous int64 document counts[D]")
        d = doc_counts.numel()
    block = min(8192, triton.next_power_of_2(max(m, d)))
    parts = triton.cdiv(max(m, d), block)
    out = torch.empty((), device=losses.device, dtype=torch.float32)
    inv = torch.empty_like(out)
    sums = counts = None
    if parts > 1:
        sums = torch.empty(parts, device=losses.device, dtype=torch.float32)
        counts = torch.empty(parts, device=losses.device, dtype=torch.int64)
    with torch.cuda.device(losses.device):
        _loss_partial[(parts,)](losses, target, doc_counts, sums, counts, out, inv,
                               m, d, balance == "document", ignore_index, block,
                               parts == 1, num_warps=8)
        if parts > 1:
            _loss_finish[(1,)](sums, counts, out, inv, parts,
                              min(1024, triton.next_power_of_2(parts)))
    return out, inv
