"""Chunked fused linear cross-entropy with softcap and document-balanced loss.

Route ported from the chunked GEMM formulation of FLA's fused linear cross
entropy (``fla/modules/fused_linear_cross_entropy.py``, MIT License,
Copyright (c) 2023-2025 Songlin Yang, Yu Zhang, itself adapted from
Liger-Kernel's ``fused_linear_cross_entropy``): the token dimension is split
into chunks, each chunk's logits are materialised with one library GEMM, the
chunk CE kernels overwrite that same buffer with dlogits, and two library GEMMs
produce dHidden and dWeight with the per-chunk dWeight partial sums accumulated
in FP32.  No FLA source is vendored and no FLA private helper is imported; the
in-place logits->dlogits reuse, the chunked library GEMMs and the chunked FP32
dWeight accumulation are the inherited ideas.  Unlike the reference, ignored
rows are not short-circuited: their logits are already materialised, so the CE
kernels mask them and only the final loss reduction counts them.

Deliberate differences from that reference route, required by the task
contract in ``docs/FUSED_CE_TASK.md``:

* ``softcap`` is applied to the logits used by the log-sum-exp, the target
  logit *and* the softmax gradient, and the backward multiplies by
  ``1 - tanh(z / c) ** 2`` in the chunk kernel.
* per-token weights implement ``cu_seqlens`` document balancing
  (``1 / (D * n_d)`` over documents with at least one valid label) or the
  valid-token mean when ``cu_seqlens`` is absent.  All counting and reduction
  stays on the device: no per-chunk ``.item()``, D2H copy or host-device
  synchronisation.
* the upstream gradient is applied to the saved gradients at backward time
  instead of scaling them in place, so an arbitrary upstream value and repeated
  backward passes over one graph stay correct.

dtypes (contract, see ``docs/FUSED_CE_TASK.md``):

* chunk logits: one library GEMM per chunk writing ``logits_dtype``; BF16
  output is a BF16 rounding of an FP32 accumulation, FP32 output is exact for
  the given inputs.  The buffer is never the full ``[tokens, vocab]`` logits.
* chunk dlogits: written back into the logits buffer with the same dtype, so
  the softmax gradient carries one rounding to ``logits_dtype``.
* dHidden: library GEMM ``dlogits @ weight`` in ``logits_dtype``, rounded to
  ``logits_dtype``; saved as-is and cast to the input dtype after the upstream
  multiply.
* dWeight: library GEMM ``dlogits.T @ hidden`` returns ``logits_dtype`` (FP32
  accumulation, ``logits_dtype`` output rounding) and each chunk partial is
  then added into one FP32 ``[vocab, channels]`` accumulator.  A BF16 chunk
  partial added into an FP32 buffer is *not* the same as an FP32 GEMM output;
  the FP32 ``logits_dtype`` mode is the way to obtain the unrounded partials.

The loss is returned normalised (document mean, or valid-token mean) and the
unnormalised numerator plus the matching count are returned alongside it so a
caller can renormalise across accumulation microbatches and ranks.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# Launch configuration.  Chunk-row count is a per-call argument; these block
# sizes are defaults for the kernels that consume one chunk.
_CE_BLOCK_M = 32
_CE_STATS_BLOCK_V = 128
_CE_DLOGITS_BLOCK_V = 128
_CE_ROW_STATS_BLOCK_M = 16
_CE_ROW_STATS_BLOCK_N = 64
_CE_REDUCE_BLOCK = 1024
_CE_ACCUMULATE_BLOCK = 4096
_CE_DEFAULT_CHUNK_BYTES = 64 << 20

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


def _promote_dtype(dtype):
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "hidden_states and weight must be torch.bfloat16 or torch.float32")
    return dtype


def _resolve_logits_dtype(dtype, logits_dtype):
    """Validate the requested chunk-buffer dtype against the input dtype."""
    if logits_dtype is None:
        return dtype
    if logits_dtype not in _SUPPORTED_DTYPES:
        raise TypeError("logits_dtype must be torch.bfloat16 or torch.float32")
    if logits_dtype == torch.bfloat16 and dtype != torch.bfloat16:
        raise ValueError(
            "logits_dtype=torch.bfloat16 would round FP32 inputs; use "
            "logits_dtype=torch.float32")
    return logits_dtype


@triton.jit
def _accumulate_fp32(dst, src, numel, BLOCK: tl.constexpr):
    """``dst += src`` in FP32, one pass over an existing FP32 accumulator."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    total = tl.load(dst + offsets, mask=mask, other=0.0) + tl.load(
        src + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(dst + offsets, total, mask=mask)


@triton.jit
def _chunk_partial_stats(
    logits, labels, partial_max, partial_sum, partial_target,
    rows, partial_stride, row_offset, vocab_size, softcap,
    IGNORE_INDEX: tl.constexpr, HAS_SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """Per-(row, vocab-block) capped max, shifted sum-exp and target logit."""
    pid_m = tl.program_id(0)
    pid_v = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    m_mask = m < rows
    v_mask = v < vocab_size
    tile = m_mask[:, None] & v_mask[None, :]
    z = tl.load(
        logits + m[:, None] * vocab_size + v[None, :],
        mask=tile, other=0.0).to(tl.float32)
    if HAS_SOFTCAP:
        cap = softcap.to(tl.float32)
        z = cap * libdevice.tanh(z / cap)
    target = tl.load(labels + row_offset + m, mask=m_mask, other=IGNORE_INDEX)
    hit = v[None, :] == target[:, None]
    block_target = tl.sum(tl.where(hit, z, 0.0), axis=1)
    z = tl.where(tile, z, float("-inf"))
    block_max = tl.max(z, axis=1)
    block_sum = tl.sum(tl.exp(z - block_max[:, None]), axis=1)
    offset = m * partial_stride + pid_v
    tl.store(partial_max + offset, block_max, mask=m_mask)
    tl.store(partial_sum + offset, block_sum, mask=m_mask)
    tl.store(partial_target + offset, block_target, mask=m_mask)


@triton.jit
def _chunk_row_stats(
    labels, lse, nll, partial_max, partial_sum, partial_target,
    rows, partial_stride, row_offset,
    IGNORE_INDEX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Merge the (row, vocab-block) partials into one row LSE and NLL."""
    pid_m = tl.program_id(0)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m < rows
    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    target = tl.zeros((BLOCK_M,), tl.float32)
    for j0 in tl.range(0, partial_stride, BLOCK_N):
        j = j0 + tl.arange(0, BLOCK_N)
        j_mask = j < partial_stride
        stat_mask = m_mask[:, None] & j_mask[None, :]
        block_max = tl.load(
            partial_max + m[:, None] * partial_stride + j[None, :],
            mask=stat_mask, other=float("-inf"))
        block_sum = tl.load(
            partial_sum + m[:, None] * partial_stride + j[None, :],
            mask=stat_mask, other=0.0)
        block_target = tl.load(
            partial_target + m[:, None] * partial_stride + j[None, :],
            mask=stat_mask, other=0.0)
        new_max = tl.maximum(running_max, tl.max(block_max, axis=1))
        running_sum = (
            running_sum * tl.exp(running_max - new_max)
            + tl.sum(block_sum * tl.exp(block_max - new_max[:, None]), axis=1))
        running_max = new_max
        target += tl.sum(block_target, axis=1)
    row_lse = running_max + tl.log(running_sum)
    label = tl.load(labels + row_offset + m, mask=m_mask, other=IGNORE_INDEX)
    valid = m_mask & (label != IGNORE_INDEX)
    tl.store(lse + m, row_lse, mask=m_mask)
    tl.store(
        nll + row_offset + m,
        tl.where(valid, row_lse - target, 0.0), mask=m_mask)


@triton.jit
def _chunk_dlogits(
    logits, labels, lse, scale,
    rows, row_offset, vocab_size, softcap,
    IGNORE_INDEX: tl.constexpr, HAS_SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr,
):
    """Overwrite one chunk's logits tile with the chunk's softmax gradient."""
    pid_m = tl.program_id(0)
    pid_v = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m < rows
    v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v < vocab_size
    tile = m_mask[:, None] & v_mask[None, :]
    label = tl.load(labels + row_offset + m, mask=m_mask, other=IGNORE_INDEX)
    valid = m_mask & (label != IGNORE_INDEX)
    row_lse = tl.load(lse + m, mask=m_mask, other=0.0)
    z = tl.load(
        logits + m[:, None] * vocab_size + v[None, :],
        mask=tile, other=0.0).to(tl.float32)
    grad_cap = 1.0
    if HAS_SOFTCAP:
        cap = softcap.to(tl.float32)
        tanh_logits = libdevice.tanh(z / cap)
        z = cap * tanh_logits
        grad_cap = 1.0 - tanh_logits * tanh_logits
    row_scale = tl.load(scale + row_offset + m, mask=m_mask, other=0.0)
    grad = tl.exp(z - row_lse[:, None])
    grad -= (v[None, :] == label[:, None]).to(tl.float32)
    grad = grad * (grad_cap * row_scale[:, None])
    grad = tl.where(valid[:, None] & v_mask[None, :], grad, 0.0)
    tl.store(
        logits + m[:, None] * vocab_size + v[None, :], grad, mask=tile)


@triton.jit
def _valid_token_count(
    labels, count_out, num_tokens,
    IGNORE_INDEX: tl.constexpr, BLOCK: tl.constexpr,
):
    total = tl.zeros((), tl.int32)
    for start in tl.range(0, num_tokens, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < num_tokens
        row_labels = tl.load(labels + offsets, mask=mask, other=IGNORE_INDEX)
        total += tl.sum((mask & (row_labels != IGNORE_INDEX)).to(tl.int32),
                        axis=0)
    tl.store(count_out, total)


@triton.jit
def _token_scale(
    labels, count_ptr, scale, num_tokens,
    IGNORE_INDEX: tl.constexpr, NORMALIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if NORMALIZE:
        count = tl.load(count_ptr).to(tl.float32)
        weight = tl.where(count > 0, 1.0 / count, 0.0)
    else:
        weight = 1.0
    for start in tl.range(0, num_tokens, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < num_tokens
        row_labels = tl.load(labels + offsets, mask=mask, other=IGNORE_INDEX)
        valid = mask & (row_labels != IGNORE_INDEX)
        tl.store(scale + offsets, tl.where(valid, weight, 0.0), mask=mask)


@triton.jit
def _document_valid_counts(
    labels, cu_seqlens, counts, num_documents,
    IGNORE_INDEX: tl.constexpr, BLOCK: tl.constexpr,
):
    document = tl.program_id(0)
    start = tl.load(cu_seqlens + document)
    end = tl.load(cu_seqlens + document + 1)
    total = tl.zeros((), tl.int32)
    for offset_start in tl.range(0, end - start, BLOCK):
        offsets = start + offset_start + tl.arange(0, BLOCK)
        mask = offsets < end
        row_labels = tl.load(labels + offsets, mask=mask, other=IGNORE_INDEX)
        total += tl.sum((mask & (row_labels != IGNORE_INDEX)).to(tl.int32),
                        axis=0)
    tl.store(counts + document, total)


@triton.jit
def _nonzero_count(counts, count_out, num_documents, BLOCK: tl.constexpr):
    total = tl.zeros((), tl.int32)
    for start in tl.range(0, num_documents, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < num_documents
        values = tl.load(counts + offsets, mask=mask, other=0)
        total += tl.sum((values > 0).to(tl.int32), axis=0)
    tl.store(count_out, total)


@triton.jit
def _document_scale(
    cu_seqlens, counts, valid_ptr, scale, num_documents,
    NORMALIZE: tl.constexpr, BLOCK: tl.constexpr,
):
    document = tl.program_id(0)
    start = tl.load(cu_seqlens + document)
    end = tl.load(cu_seqlens + document + 1)
    count = tl.load(counts + document).to(tl.float32)
    if NORMALIZE:
        total = tl.load(valid_ptr).to(tl.float32)
        divisor = total * count
        weight = tl.where((count > 0) & (total > 0), 1.0 / divisor, 0.0)
    else:
        weight = tl.where(count > 0, 1.0 / count, 0.0)
    for offset_start in tl.range(0, end - start, BLOCK):
        offsets = start + offset_start + tl.arange(0, BLOCK)
        mask = offsets < end
        tl.store(scale + offsets, weight, mask=mask)


@triton.jit
def _token_nll_sum(
    nll, labels, sum_out, num_tokens,
    IGNORE_INDEX: tl.constexpr, BLOCK: tl.constexpr,
):
    total = tl.zeros((), tl.float32)
    for start in tl.range(0, num_tokens, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < num_tokens
        row_labels = tl.load(labels + offsets, mask=mask, other=IGNORE_INDEX)
        valid = mask & (row_labels != IGNORE_INDEX)
        values = tl.load(nll + offsets, mask=mask, other=0.0)
        total += tl.sum(tl.where(valid, values, 0.0), axis=0)
    tl.store(sum_out, total)


@triton.jit
def _document_nll_mean(
    nll, cu_seqlens, counts, document_loss, num_documents, BLOCK: tl.constexpr,
):
    document = tl.program_id(0)
    start = tl.load(cu_seqlens + document)
    end = tl.load(cu_seqlens + document + 1)
    total = tl.zeros((), tl.float32)
    for offset_start in tl.range(0, end - start, BLOCK):
        offsets = start + offset_start + tl.arange(0, BLOCK)
        mask = offsets < end
        total += tl.sum(tl.load(nll + offsets, mask=mask, other=0.0), axis=0)
    count = tl.load(counts + document).to(tl.float32)
    mean = tl.where(count > 0, total / count, 0.0)
    tl.store(document_loss + document, mean)


@triton.jit
def _document_mean_sum(
    document_loss, counts, sum_out, count_out, num_documents,
    BLOCK: tl.constexpr,
):
    total = tl.zeros((), tl.float32)
    valid = tl.zeros((), tl.int32)
    for start in tl.range(0, num_documents, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < num_documents
        values = tl.load(document_loss + offsets, mask=mask, other=0.0)
        counts_block = tl.load(counts + offsets, mask=mask, other=0)
        total += tl.sum(values, axis=0)
        valid += tl.sum((counts_block > 0).to(tl.int32), axis=0)
    tl.store(sum_out, total)
    tl.store(count_out, valid)


@triton.jit
def _finalize_loss(sum_ptr, count_ptr, loss_ptr, NORMALIZE: tl.constexpr):
    total = tl.load(sum_ptr)
    if NORMALIZE:
        count = tl.load(count_ptr).to(tl.float32)
        total = tl.where(count > 0, total / count, 0.0)
    tl.store(loss_ptr, total)


def _validate_softcap(softcap):
    if softcap is None:
        return None
    if isinstance(softcap, bool) or not isinstance(softcap, (int, float)):
        raise TypeError("softcap must be None or a finite positive number")
    softcap = float(softcap)
    if not 0.0 < softcap < float("inf"):
        raise ValueError("softcap must be finite and greater than zero")
    return softcap


def _validate_inputs(hidden_states, weight, labels, ignore_index):
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise TypeError("ignore_index must be an integer")
    if not (hidden_states.is_cuda and weight.is_cuda and labels.is_cuda):
        raise ValueError("hidden_states, weight, and labels must be CUDA tensors")
    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape [batch, sequence, channels]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [vocab_size, channels]")
    if labels.ndim != 2 or labels.shape != hidden_states.shape[:2]:
        raise ValueError("labels must have shape [batch, sequence]")
    dtype = _promote_dtype(hidden_states.dtype)
    if weight.dtype != dtype:
        raise TypeError("hidden_states and weight must share one dtype")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError("labels must use torch.int32 or torch.int64")
    if not (hidden_states.is_contiguous() and weight.is_contiguous()
            and labels.is_contiguous()):
        raise ValueError("hidden_states, weight, and labels must be contiguous")
    if not (hidden_states.device == weight.device == labels.device):
        raise ValueError("all inputs must be on the same CUDA device")
    batch, sequence, in_channels = hidden_states.shape
    vocab_size, weight_channels = weight.shape
    if min(batch, sequence, in_channels, vocab_size) <= 0:
        raise ValueError("all input dimensions must be nonzero")
    if weight_channels != in_channels:
        raise ValueError("weight.shape[1] must equal hidden_states.shape[2]")
    # A device-to-host label check would break the Dynamo graph.  Keep the
    # diagnostic in eager mode only, like the existing fused entry.
    if not torch.compiler.is_compiling():
        invalid = ((labels != ignore_index)
                   & ((labels < 0) | (labels >= vocab_size)))
        if bool(invalid.any().item()):
            raise ValueError(
                "every label must equal ignore_index or lie in [0, vocab_size)")
    return batch, sequence, in_channels, vocab_size


def _varlen_documents(hidden_states, cu_seqlens, max_seqlen):
    if hidden_states.shape[0] != 1:
        raise ValueError("varlen hidden_states must have batch size 1")
    if (cu_seqlens is None or cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 2 or not cu_seqlens.is_contiguous()
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.device != hidden_states.device):
        raise ValueError(
            "cu_seqlens must be a contiguous CUDA int32 or int64 tensor")
    total_tokens = hidden_states.shape[1]
    if not torch.compiler.is_compiling():
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if (int(cu_seqlens[0].item()) != 0
                or int(cu_seqlens[-1].item()) != total_tokens
                or bool((lengths < 0).any().item())):
            raise ValueError(
                "cu_seqlens must be nondecreasing, start at zero, and end at "
                "total_tokens")
        actual_max = int(lengths.max().item())
        if max_seqlen is not None and (
                isinstance(max_seqlen, bool)
                or not isinstance(max_seqlen, int)
                or max_seqlen < actual_max):
            raise ValueError("max_seqlen must cover every packed document")
    return cu_seqlens.numel() - 1


def _chunk_row_count(total_tokens, vocab_size, chunk_rows, itemsize):
    if chunk_rows is not None:
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int):
            raise TypeError("chunk_rows must be a positive integer or None")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be a positive integer")
        return min(total_tokens, chunk_rows)
    # Default buffer budget, rounded down to a power of two.  Smaller chunks
    # keep the chunk buffer inside L2 for the CE passes, which measured faster
    # than the extra GEMM efficiency of larger chunks on the RTX 5090 fixture.
    budget = max(1, _CE_DEFAULT_CHUNK_BYTES // (vocab_size * itemsize))
    max_rows = max(1, 1 << (budget.bit_length() - 1))
    return min(total_tokens, max_rows)


def _row_scale_token(labels_flat, total_tokens, ignore_index, normalize, device):
    count = torch.empty((), device=device, dtype=torch.int32)
    scale = torch.empty(total_tokens, device=device, dtype=torch.float32)
    _valid_token_count[(1,)](
        labels_flat, count, total_tokens,
        IGNORE_INDEX=ignore_index, BLOCK=_CE_REDUCE_BLOCK, num_warps=4)
    _token_scale[(1,)](
        labels_flat, count, scale, total_tokens,
        IGNORE_INDEX=ignore_index, NORMALIZE=normalize,
        BLOCK=_CE_REDUCE_BLOCK, num_warps=4)
    return scale, count


def _row_scale_document(
    labels_flat, cu_seqlens, num_documents, ignore_index, normalize, device,
):
    counts = torch.empty(num_documents, device=device, dtype=torch.int32)
    valid = torch.empty((), device=device, dtype=torch.int32)
    scale = torch.empty(labels_flat.numel(), device=device, dtype=torch.float32)
    _document_valid_counts[(num_documents,)](
        labels_flat, cu_seqlens, counts, num_documents,
        IGNORE_INDEX=ignore_index, BLOCK=_CE_REDUCE_BLOCK, num_warps=4)
    _nonzero_count[(1,)](
        counts, valid, num_documents, BLOCK=_CE_REDUCE_BLOCK, num_warps=4)
    _document_scale[(num_documents,)](
        cu_seqlens, counts, valid, scale, num_documents,
        NORMALIZE=normalize, BLOCK=_CE_REDUCE_BLOCK, num_warps=4)
    return scale, counts, valid


def _loss_token(nll, labels_flat, total_tokens, count, ignore_index, normalize):
    total = torch.empty((), device=nll.device, dtype=torch.float32)
    loss = torch.empty((), device=nll.device, dtype=torch.float32)
    _token_nll_sum[(1,)](
        nll, labels_flat, total, total_tokens,
        IGNORE_INDEX=ignore_index, BLOCK=_CE_REDUCE_BLOCK, num_warps=4)
    _finalize_loss[(1,)](total, count, loss, NORMALIZE=normalize, num_warps=1)
    return loss, total


def _loss_document(nll, cu_seqlens, counts, num_documents, device, normalize):
    document_loss = torch.empty(num_documents, device=device, dtype=torch.float32)
    total = torch.empty((), device=device, dtype=torch.float32)
    count = torch.empty((), device=device, dtype=torch.int32)
    loss = torch.empty((), device=device, dtype=torch.float32)
    _document_nll_mean[(num_documents,)](
        nll, cu_seqlens, counts, document_loss, num_documents,
        BLOCK=_CE_REDUCE_BLOCK, num_warps=4)
    _document_mean_sum[(1,)](
        document_loss, counts, total, count, num_documents,
        BLOCK=_CE_REDUCE_BLOCK, num_warps=4)
    _finalize_loss[(1,)](total, count, loss, NORMALIZE=normalize, num_warps=1)
    return loss, total, count, document_loss


class ChunkedCEState:
    """Forward products: normalised loss, raw numerator, count, gradients."""

    __slots__ = ("loss", "mean_sum", "count", "dhidden", "dweight",
                 "reduction", "document_loss")

    def __init__(self, loss, mean_sum, count, dhidden, dweight, reduction,
                 document_loss=None):
        self.loss = loss
        self.mean_sum = mean_sum
        self.count = count
        self.dhidden = dhidden
        self.dweight = dweight
        self.reduction = reduction
        self.document_loss = document_loss


@torch.no_grad()
def chunked_ce_forward(
    hidden_states, weight, labels, ignore_index=-1, softcap=None,
    cu_seqlens=None, max_seqlen=None, chunk_rows=None, logits_dtype=None,
    normalize=True,
):
    """Chunked fused linear CE forward plus the precomputed local gradients.

    Returns a :class:`ChunkedCEState`.  ``loss`` is the normalised objective;
    ``mean_sum``/``count`` are its unnormalised numerator (sum of document
    means, or sum of valid-token losses) and the matching denominator
    (documents with at least one valid label, or valid tokens).  With
    ``normalize=False`` the per-token gradient weights drop the global
    ``1 / count`` factor, so a caller can scale ``loss`` by the global
    normaliser gathered across microbatches and ranks before backward and
    obtain exactly ``sum(document means) / total valid documents``.
    """
    batch, sequence, in_channels, vocab_size = _validate_inputs(
        hidden_states, weight, labels, ignore_index)
    softcap = _validate_softcap(softcap)
    dtype = hidden_states.dtype
    buffer_dtype = _resolve_logits_dtype(dtype, logits_dtype)
    total_tokens = batch * sequence
    has_softcap = softcap is not None
    softcap_value = softcap if has_softcap else 1.0
    device = hidden_states.device

    if cu_seqlens is not None:
        num_documents = _varlen_documents(hidden_states, cu_seqlens, max_seqlen)
        if num_documents == 0:
            raise ValueError("cu_seqlens must describe at least one document")
        reduction = "document_mean"
    else:
        if max_seqlen is not None:
            raise ValueError("max_seqlen requires cu_seqlens")
        num_documents = 0
        reduction = "token_mean"

    itemsize = torch.empty((), dtype=buffer_dtype).element_size()
    rows_per_chunk = _chunk_row_count(
        total_tokens, vocab_size, chunk_rows, itemsize)
    block_m = _CE_BLOCK_M
    stats_block_v = min(_CE_STATS_BLOCK_V, triton.next_power_of_2(vocab_size))
    dlogits_block_v = min(
        _CE_DLOGITS_BLOCK_V, triton.next_power_of_2(vocab_size))
    stats_n_v = triton.cdiv(vocab_size, stats_block_v)
    dlogits_n_v = triton.cdiv(vocab_size, dlogits_block_v)

    hidden_flat = hidden_states.reshape(-1, in_channels)
    labels_flat = labels.reshape(-1)
    # GEMM operands follow the chunk-buffer dtype.  ``.to`` is a no-op when the
    # dtypes already match, so only the FP32 validation mode upcasts.
    gemm_hidden = hidden_flat.to(buffer_dtype)
    gemm_weight = weight.to(buffer_dtype)
    weight_t = gemm_weight.t()

    lse = torch.empty(rows_per_chunk, device=device, dtype=torch.float32)
    nll = torch.empty(total_tokens, device=device, dtype=torch.float32)
    partial_shape = (rows_per_chunk, stats_n_v)
    partial_max = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_sum = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_target = torch.empty(
        partial_shape, device=device, dtype=torch.float32)

    dhidden = torch.empty(
        (total_tokens, in_channels), device=device, dtype=buffer_dtype)
    dweight = torch.zeros(
        (vocab_size, in_channels), device=device, dtype=torch.float32)

    if reduction == "document_mean":
        scale, counts, valid_documents = _row_scale_document(
            labels_flat, cu_seqlens, num_documents, ignore_index, normalize,
            device)
    else:
        scale, token_count = _row_scale_token(
            labels_flat, total_tokens, ignore_index, normalize, device)

    for row_start in range(0, total_tokens, rows_per_chunk):
        rows = min(rows_per_chunk, total_tokens - row_start)
        gemm_hidden_chunk = gemm_hidden[row_start:row_start + rows]
        chunk_logits = torch.mm(gemm_hidden_chunk, weight_t)
        _chunk_partial_stats[(triton.cdiv(rows, block_m), stats_n_v)](
            chunk_logits, labels_flat, partial_max, partial_sum, partial_target,
            rows, stats_n_v, row_start, vocab_size, softcap_value,
            IGNORE_INDEX=ignore_index, HAS_SOFTCAP=has_softcap,
            BLOCK_M=block_m, BLOCK_V=stats_block_v, num_warps=4)
        _chunk_row_stats[(triton.cdiv(rows, _CE_ROW_STATS_BLOCK_M),)](
            labels_flat, lse, nll, partial_max, partial_sum, partial_target,
            rows, stats_n_v, row_start,
            IGNORE_INDEX=ignore_index,
            BLOCK_M=_CE_ROW_STATS_BLOCK_M, BLOCK_N=_CE_ROW_STATS_BLOCK_N,
            num_warps=4)
        _chunk_dlogits[(triton.cdiv(rows, block_m), dlogits_n_v)](
            chunk_logits, labels_flat, lse, scale,
            rows, row_start, vocab_size, softcap_value,
            IGNORE_INDEX=ignore_index, HAS_SOFTCAP=has_softcap,
            BLOCK_M=block_m, BLOCK_V=dlogits_block_v, num_warps=4)
        dhidden[row_start:row_start + rows] = torch.mm(
            chunk_logits, gemm_weight)
        # One BF16 chunk partial per chunk is added into the FP32 accumulator by
        # our own kernel: the accumulation dtype and the accumulator lifetime
        # then stay explicit and identical in eager and compiled execution.
        chunk_dweight = torch.mm(chunk_logits.t(), gemm_hidden_chunk)
        _accumulate_fp32[(triton.cdiv(dweight.numel(), _CE_ACCUMULATE_BLOCK),)](
            dweight, chunk_dweight, dweight.numel(),
            BLOCK=_CE_ACCUMULATE_BLOCK, num_warps=4)

    if reduction == "document_mean":
        loss, mean_sum, count, document_loss = _loss_document(
            nll, cu_seqlens, counts, num_documents, device, normalize)
    else:
        loss, mean_sum = _loss_token(
            nll, labels_flat, total_tokens, token_count, ignore_index,
            normalize)
        count = token_count
        document_loss = None
    return ChunkedCEState(
        loss, mean_sum, count, dhidden, dweight, reduction, document_loss)


def chunked_ce_backward(state, dloss, hidden_dtype, weight_dtype):
    """Apply the upstream scalar to the saved gradients, without mutation."""
    dloss = dloss.to(torch.float32)
    dhidden = (state.dhidden.to(torch.float32) * dloss).to(hidden_dtype)
    dweight = (state.dweight * dloss).to(weight_dtype)
    return dhidden, dweight


def fused_ce_chunked_stats(
    hidden_states, weight, labels, ignore_index=-1, softcap=None,
    cu_seqlens=None, max_seqlen=None, chunk_rows=None, logits_dtype=None,
    normalize=True,
):
    """Forward-only ``(loss, numerator, count)``; the accumulation interface."""
    state = chunked_ce_forward(
        hidden_states, weight, labels, ignore_index, softcap, cu_seqlens,
        max_seqlen, chunk_rows, logits_dtype, normalize)
    return state.loss, state.mean_sum, state.count


class ChunkedCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx, hidden_states, weight, labels, ignore_index=-1, softcap=None,
        cu_seqlens=None, max_seqlen=None, chunk_rows=None, logits_dtype=None,
    ):
        state = chunked_ce_forward(
            hidden_states, weight, labels, ignore_index, softcap, cu_seqlens,
            max_seqlen, chunk_rows, logits_dtype)
        ctx.save_for_backward(state.dhidden, state.dweight)
        ctx.hidden_dtype = hidden_states.dtype
        ctx.weight_dtype = weight.dtype
        ctx.hidden_shape = hidden_states.shape
        return state.loss

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dloss):
        dhidden, dweight = ctx.saved_tensors
        state = ChunkedCEState(None, None, None, dhidden, dweight, None)
        dhidden_grad, dweight_grad = chunked_ce_backward(
            state, dloss.contiguous(), ctx.hidden_dtype, ctx.weight_dtype)
        return (dhidden_grad.reshape(ctx.hidden_shape), dweight_grad) + (None,) * 7


def fused_ce_chunked(
    hidden_states, weight, labels, ignore_index=-1, softcap=None,
    cu_seqlens=None, max_seqlen=None, chunk_rows=None, logits_dtype=None,
):
    """Fused linear + CE with optional softcap and ``cu_seqlens`` balancing."""
    return ChunkedCrossEntropyFunction.apply(
        hidden_states, weight, labels, ignore_index, softcap, cu_seqlens,
        max_seqlen, chunk_rows, logits_dtype)
