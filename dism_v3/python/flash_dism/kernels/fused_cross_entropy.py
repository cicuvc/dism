import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

if __package__:
    from .dynamo_utils import mark_cu_seqlens_dynamic
else:
    from dynamo_utils import mark_cu_seqlens_dynamic


# Static launch parameters for the first implementation.  Keeping these in one
# place makes it straightforward to add autotuning after the numerical and
# memory behaviour has been established on representative model shapes.
_FWD_BM = 16
_FWD_BV = 64
_FWD_BK = 32
_REDUCE_BLOCK = 256
_DLOGITS_BM = 16
_DLOGITS_BV = 64
_DLOGITS_BK = 32
_DX_BM = 32
_DX_BC = 64
_DX_BV = 32
_DW_BV = 32
_DW_BC = 64
_DW_BR = 32
_DEFAULT_DLOGITS_BYTES = 64 * 1024 * 1024


def reference_fused_cross_entropy(
    hidden_states, weight, labels, ignore_index=-100, softcap=None,
):
    """FP32 reference for the scalar, valid-token mean loss."""
    logits = torch.nn.functional.linear(
        hidden_states.float(), weight.float())
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    losses = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten().long(),
        ignore_index=ignore_index, reduction="none")
    valid_count = (labels != ignore_index).sum()
    return losses.sum() / valid_count.clamp_min(1)


def reference_fused_cross_entropy_per_position(
    hidden_states, weight, labels, ignore_index=-100, softcap=None,
):
    """FP32 reference for the valid-batch mean at each sequence position."""
    logits = torch.nn.functional.linear(
        hidden_states.float(), weight.float())
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    flat_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten().long(),
        ignore_index=ignore_index, reduction="none")
    losses = flat_loss.view_as(labels)
    valid = labels != ignore_index
    counts = valid.sum(dim=0)
    return torch.where(
        counts > 0,
        losses.sum(dim=0) / counts.clamp_min(1),
        torch.zeros_like(losses[0]),
    )


def reference_varlen_fused_cross_entropy(
    hidden_states, weight, labels, cu_seqlens,
    ignore_index=-100, softcap=None,
):
    """FP32 sequence-balanced reference for packed variable-length inputs."""
    logits = torch.nn.functional.linear(
        hidden_states.float(), weight.float())
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    losses = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten().long(),
        ignore_index=ignore_index, reduction="none")
    document_losses = []
    boundaries = cu_seqlens.cpu().tolist()
    flat_labels = labels.flatten()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        valid = flat_labels[start:end] != ignore_index
        if bool(valid.any().item()):
            document_losses.append(losses[start:end][valid].mean())
    if not document_losses:
        return losses.sum() * 0.0
    return torch.stack(document_losses).mean()


def reference_varlen_fused_cross_entropy_per_position(
    hidden_states, weight, labels, cu_seqlens, max_seqlen=None,
    ignore_index=-100, softcap=None,
):
    """FP32 valid-document mean at each local sequence position."""
    logits = torch.nn.functional.linear(
        hidden_states.float(), weight.float())
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    losses = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten().long(),
        ignore_index=ignore_index, reduction="none")
    boundaries = cu_seqlens.cpu().tolist()
    lengths = [end - start for start, end in zip(
        boundaries[:-1], boundaries[1:])]
    if max_seqlen is None:
        max_seqlen = max(lengths, default=0)
    flat_labels = labels.flatten()
    output = []
    for position in range(max_seqlen):
        values = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            row = start + position
            if row < end and flat_labels[row] != ignore_index:
                values.append(losses[row])
        output.append(
            torch.stack(values).mean() if values
            else losses.sum() * 0.0)
    return torch.stack(output) if output else losses.new_empty((0,))


@triton.jit
def _linear_cross_entropy_fwd(
    x, weight, labels, lse, nll,
    m_size, in_channels, vocab_size, softcap,
    IGNORE_INDEX: tl.constexpr, HAS_SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m < m_size
    label = tl.load(labels + m, mask=m_mask, other=IGNORE_INDEX)
    valid = m_mask & (label != IGNORE_INDEX)

    neg_inf = -float("inf")
    running_max = tl.full((BLOCK_M,), neg_inf, tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    target = tl.zeros((BLOCK_M,), tl.float32)

    for v_start in tl.range(0, vocab_size, BLOCK_V):
        v = v_start + tl.arange(0, BLOCK_V)
        v_mask = v < vocab_size
        logits = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)

        for k_start in tl.range(0, in_channels, BLOCK_K):
            k = k_start + tl.arange(0, BLOCK_K)
            k_mask = k < in_channels
            x_vals = tl.load(
                x + m[:, None] * in_channels + k[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w_vals = tl.load(
                weight + v[:, None] * in_channels + k[None, :],
                mask=v_mask[:, None] & k_mask[None, :], other=0.0)
            logits = tl.dot(
                x_vals, tl.trans(w_vals), logits,
                input_precision="ieee")

        if HAS_SOFTCAP:
            cap = softcap.to(tl.float32)
            logits = cap * libdevice.tanh(logits / cap)
        target += tl.sum(
            tl.where(v[None, :] == label[:, None], logits, 0.0), axis=1)
        logits = tl.where(
            m_mask[:, None] & v_mask[None, :], logits, neg_inf)
        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(running_max, block_max)
        running_sum = (
            running_sum * tl.exp(running_max - new_max)
            + tl.sum(tl.exp(logits - new_max[:, None]), axis=1)
        )
        running_max = new_max

    row_lse = running_max + tl.log(running_sum)
    row_nll = tl.where(valid, row_lse - target, 0.0)
    tl.store(lse + m, row_lse, mask=m_mask)
    tl.store(nll + m, row_nll, mask=m_mask)


@triton.jit
def _mean_loss_reduce(
    nll, labels, output, valid_count, m_size,
    IGNORE_INDEX: tl.constexpr, BLOCK: tl.constexpr,
):
    total = tl.zeros((), tl.float32)
    count = tl.zeros((), tl.int32)
    for start in tl.range(0, m_size, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < m_size
        values = tl.load(nll + offsets, mask=mask, other=0.0)
        row_labels = tl.load(
            labels + offsets, mask=mask, other=IGNORE_INDEX)
        valid = mask & (row_labels != IGNORE_INDEX)
        total += tl.sum(tl.where(valid, values, 0.0), axis=0)
        count += tl.sum(valid.to(tl.int32), axis=0)
    loss = tl.where(count > 0, total / count.to(tl.float32), 0.0)
    tl.store(output, loss)
    tl.store(valid_count, count)


@triton.jit
def _per_position_reduce(
    nll, labels, output, batch, sequence,
    IGNORE_INDEX: tl.constexpr, BLOCK: tl.constexpr,
):
    position = tl.program_id(0)
    total = tl.zeros((), tl.float32)
    count = tl.zeros((), tl.int32)
    for b_start in tl.range(0, batch, BLOCK):
        b = b_start + tl.arange(0, BLOCK)
        mask = b < batch
        offsets = b * sequence + position
        values = tl.load(nll + offsets, mask=mask, other=0.0)
        row_labels = tl.load(
            labels + offsets, mask=mask, other=IGNORE_INDEX)
        valid = mask & (row_labels != IGNORE_INDEX)
        total += tl.sum(tl.where(valid, values, 0.0), axis=0)
        count += tl.sum(valid.to(tl.int32), axis=0)
    value = tl.where(count > 0, total / count.to(tl.float32), 0.0)
    tl.store(output + position, value)


@triton.jit
def _document_loss_reduce(
    nll, labels, cu_seqlens, document_loss, document_valid,
    token_inv_count, num_documents,
    IGNORE_INDEX: tl.constexpr, BLOCK: tl.constexpr,
):
    document = tl.program_id(0)
    start = tl.load(cu_seqlens + document)
    end = tl.load(cu_seqlens + document + 1)
    length = end - start
    total = tl.zeros((), tl.float32)
    count = tl.zeros((), tl.int32)
    for offset_start in tl.range(0, length, BLOCK):
        offset = offset_start + tl.arange(0, BLOCK)
        mask = offset < length
        row = start + offset
        row_labels = tl.load(
            labels + row, mask=mask, other=IGNORE_INDEX)
        valid = mask & (row_labels != IGNORE_INDEX)
        values = tl.load(nll + row, mask=valid, other=0.0)
        total += tl.sum(values, axis=0)
        count += tl.sum(valid.to(tl.int32), axis=0)

    has_valid = count > 0
    inv_count = tl.where(has_valid, 1.0 / count.to(tl.float32), 0.0)
    tl.store(document_loss + document, total * inv_count)
    tl.store(document_valid + document, has_valid.to(tl.int32))

    # Save the per-document component of the backward normalization.  The
    # global 1 / valid_document_count factor is applied by dlogits later.
    for offset_start in tl.range(0, length, BLOCK):
        offset = offset_start + tl.arange(0, BLOCK)
        mask = offset < length
        row = start + offset
        row_labels = tl.load(
            labels + row, mask=mask, other=IGNORE_INDEX)
        valid = mask & (row_labels != IGNORE_INDEX)
        tl.store(
            token_inv_count + row,
            tl.where(valid, inv_count, 0.0), mask=mask)


@triton.jit
def _document_mean_reduce(
    document_loss, document_valid, output, valid_document_count,
    num_documents, BLOCK: tl.constexpr,
):
    total = tl.zeros((), tl.float32)
    count = tl.zeros((), tl.int32)
    for start in tl.range(0, num_documents, BLOCK):
        document = start + tl.arange(0, BLOCK)
        mask = document < num_documents
        valid = tl.load(document_valid + document, mask=mask, other=0)
        values = tl.load(document_loss + document, mask=mask, other=0.0)
        total += tl.sum(tl.where(valid != 0, values, 0.0), axis=0)
        count += tl.sum((valid != 0).to(tl.int32), axis=0)
    loss = tl.where(count > 0, total / count.to(tl.float32), 0.0)
    tl.store(output, loss)
    tl.store(valid_document_count, count)


@triton.jit
def _varlen_per_position_reduce(
    nll, labels, cu_seqlens, output, num_documents, max_seqlen,
    IGNORE_INDEX: tl.constexpr, BLOCK: tl.constexpr,
):
    position = tl.program_id(0)
    total = tl.zeros((), tl.float32)
    count = tl.zeros((), tl.int32)
    for start_document in tl.range(0, num_documents, BLOCK):
        document = start_document + tl.arange(0, BLOCK)
        document_mask = document < num_documents
        start = tl.load(
            cu_seqlens + document, mask=document_mask, other=0)
        end = tl.load(
            cu_seqlens + document + 1, mask=document_mask, other=0)
        row = start + position
        exists = document_mask & (row < end)
        row_labels = tl.load(
            labels + row, mask=exists, other=IGNORE_INDEX)
        valid = exists & (row_labels != IGNORE_INDEX)
        values = tl.load(nll + row, mask=valid, other=0.0)
        total += tl.sum(values, axis=0)
        count += tl.sum(valid.to(tl.int32), axis=0)
    value = tl.where(count > 0, total / count.to(tl.float32), 0.0)
    tl.store(output + position, value)


@triton.jit
def _cross_entropy_dlogits(
    x, weight, labels, lse, dloss, valid_count, token_inv_count,
    dlogits,
    row_start, chunk_rows, in_channels, vocab_size, softcap,
    IGNORE_INDEX: tl.constexpr, HAS_SOFTCAP: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_v = tl.program_id(1)
    local_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = local_m < chunk_rows
    v_mask = v < vocab_size
    m = row_start + local_m

    logits = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)
    for k_start in tl.range(0, in_channels, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        k_mask = k < in_channels
        x_vals = tl.load(
            x + m[:, None] * in_channels + k[None, :],
            mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        w_vals = tl.load(
            weight + v[:, None] * in_channels + k[None, :],
            mask=v_mask[:, None] & k_mask[None, :], other=0.0)
        logits = tl.dot(
            x_vals, tl.trans(w_vals), logits,
            input_precision="ieee")

    softcap_grad = 1.0
    if HAS_SOFTCAP:
        cap = softcap.to(tl.float32)
        tanh_logits = libdevice.tanh(logits / cap)
        logits = cap * tanh_logits
        softcap_grad = 1.0 - tanh_logits * tanh_logits

    label = tl.load(labels + m, mask=row_mask, other=IGNORE_INDEX)
    row_lse = tl.load(lse + m, mask=row_mask, other=0.0)
    upstream = tl.load(dloss).to(tl.float32)
    count = tl.load(valid_count).to(tl.float32)
    scale = tl.where(count > 0, upstream / count, 0.0)
    if IS_VARLEN:
        scale *= tl.load(
            token_inv_count + m, mask=row_mask, other=0.0)
    valid = row_mask & (label != IGNORE_INDEX)
    grad = tl.exp(logits - row_lse[:, None])
    grad -= (v[None, :] == label[:, None]).to(tl.float32)
    grad *= softcap_grad
    if IS_VARLEN:
        grad = tl.where(
            valid[:, None] & v_mask[None, :],
            grad * scale[:, None], 0.0)
    else:
        grad = tl.where(
            valid[:, None] & v_mask[None, :], grad * scale, 0.0)
    tl.store(
        dlogits + local_m[:, None] * vocab_size + v[None, :], grad,
        mask=row_mask[:, None] & v_mask[None, :])


@triton.jit
def _dhidden_gemm(
    dlogits, weight, dhidden,
    row_start, chunk_rows, in_channels, vocab_size,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    local_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    m_mask = local_m < chunk_rows
    c_mask = c < in_channels
    acc = tl.zeros((BLOCK_M, BLOCK_C), tl.float32)
    for v_start in tl.range(0, vocab_size, BLOCK_V):
        v = v_start + tl.arange(0, BLOCK_V)
        v_mask = v < vocab_size
        dl = tl.load(
            dlogits + local_m[:, None] * vocab_size + v[None, :],
            mask=m_mask[:, None] & v_mask[None, :], other=0.0)
        w = tl.load(
            weight + v[:, None] * in_channels + c[None, :],
            mask=v_mask[:, None] & c_mask[None, :], other=0.0)
        acc = tl.dot(dl, w, acc, input_precision="ieee")
    tl.store(
        dhidden + (row_start + local_m[:, None]) * in_channels + c[None, :],
        acc, mask=m_mask[:, None] & c_mask[None, :])


@triton.jit
def _dweight_gemm_accumulate(
    dlogits, x, dweight,
    row_start, chunk_rows, in_channels, vocab_size,
    BLOCK_V: tl.constexpr, BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_c = tl.program_id(1)
    v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    v_mask = v < vocab_size
    c_mask = c < in_channels
    acc = tl.zeros((BLOCK_V, BLOCK_C), tl.float32)
    for r_start in tl.range(0, chunk_rows, BLOCK_R):
        r = r_start + tl.arange(0, BLOCK_R)
        r_mask = r < chunk_rows
        dl = tl.load(
            dlogits + r[:, None] * vocab_size + v[None, :],
            mask=r_mask[:, None] & v_mask[None, :], other=0.0)
        x_vals = tl.load(
            x + (row_start + r[:, None]) * in_channels + c[None, :],
            mask=r_mask[:, None] & c_mask[None, :], other=0.0)
        acc = tl.dot(tl.trans(dl), x_vals, acc, input_precision="ieee")
    old = tl.load(
        dweight + v[:, None] * in_channels + c[None, :],
        mask=v_mask[:, None] & c_mask[None, :], other=0.0)
    tl.store(
        dweight + v[:, None] * in_channels + c[None, :], old + acc,
        mask=v_mask[:, None] & c_mask[None, :])


def _validate_inputs(hidden_states, weight, labels, ignore_index):
    if not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer")
    if not (hidden_states.is_cuda and weight.is_cuda and labels.is_cuda):
        raise ValueError("hidden_states, weight, and labels must be CUDA tensors")
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, channels]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [vocab_size, channels]")
    if labels.ndim != 2 or labels.shape != hidden_states.shape[:2]:
        raise ValueError("labels must have shape [batch, sequence]")
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("hidden_states and weight must both use torch.bfloat16")
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
    # A device-to-host label check introduces a Dynamo graph break. Keep the
    # stricter diagnostic in eager mode; compiled training assumes the same
    # documented label invariant as torch.nn.functional.cross_entropy.
    if not torch.compiler.is_compiling():
        invalid = ((labels != ignore_index)
                   & ((labels < 0) | (labels >= vocab_size)))
        if bool(invalid.any().item()):
            raise ValueError(
                "every label must equal ignore_index or lie in [0, vocab_size)")
    return batch, sequence, in_channels, vocab_size


def _validate_softcap(softcap):
    if softcap is None:
        return None
    if isinstance(softcap, bool) or not isinstance(softcap, (int, float)):
        raise TypeError("softcap must be None or a finite positive number")
    softcap = float(softcap)
    if not 0.0 < softcap < float('inf'):
        raise ValueError("softcap must be finite and greater than zero")
    return softcap


def _varlen_info(hidden_states, cu_seqlens, max_seqlen):
    if hidden_states.shape[0] != 1:
        raise ValueError("varlen hidden_states must have batch size 1")
    if (cu_seqlens is None or cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 2 or not cu_seqlens.is_contiguous()
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.device != hidden_states.device):
        raise ValueError(
            "cu_seqlens must be a contiguous CUDA int32 or int64 tensor")
    total_tokens = hidden_states.shape[1]
    if torch.compiler.is_compiling():
        if max_seqlen is None:
            raise ValueError(
                "max_seqlen must be provided when compiling varlen input")
        if not isinstance(max_seqlen, int) or max_seqlen < 0:
            raise ValueError("max_seqlen must be a nonnegative integer")
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if (int(cu_seqlens[0].item()) != 0
                or int(cu_seqlens[-1].item()) != total_tokens
                or bool((lengths < 0).any().item())):
            raise ValueError(
                "cu_seqlens must be nondecreasing, start at zero, and end at total_tokens")
        actual_max = int(lengths.max().item())
        if max_seqlen is None:
            max_seqlen = actual_max
        if (isinstance(max_seqlen, bool)
                or not isinstance(max_seqlen, int)
                or max_seqlen < actual_max):
            raise ValueError("max_seqlen must cover every packed document")
    return cu_seqlens.numel() - 1, max_seqlen


def _forward_rows(hidden_states, weight, labels, ignore_index, softcap):
    batch, sequence, in_channels, vocab_size = _validate_inputs(
        hidden_states, weight, labels, ignore_index)
    m_size = batch * sequence
    lse = torch.empty(m_size, device=hidden_states.device, dtype=torch.float32)
    nll = torch.empty_like(lse)
    has_softcap = softcap is not None
    softcap_value = softcap if has_softcap else 1.0
    _linear_cross_entropy_fwd[(triton.cdiv(m_size, _FWD_BM),)](
        hidden_states, weight, labels, lse, nll,
        m_size, in_channels, vocab_size, softcap_value,
        IGNORE_INDEX=ignore_index, HAS_SOFTCAP=has_softcap,
        BLOCK_M=_FWD_BM, BLOCK_V=_FWD_BV, BLOCK_K=_FWD_BK,
        num_warps=4, num_stages=3)
    return lse, nll


def torch_fwd_fused_cross_entropy(
    hidden_states, weight, labels, ignore_index=-100, softcap=None,
):
    """Return scalar loss plus the saved state required by the backward."""
    batch, sequence, _, _ = _validate_inputs(
        hidden_states, weight, labels, ignore_index)
    softcap = _validate_softcap(softcap)
    lse, nll = _forward_rows(
        hidden_states, weight, labels, ignore_index, softcap)
    loss = torch.empty((), device=hidden_states.device, dtype=torch.float32)
    valid_count = torch.empty(
        (), device=hidden_states.device, dtype=torch.int32)
    _mean_loss_reduce[(1,)](
        nll, labels, loss, valid_count, batch * sequence,
        IGNORE_INDEX=ignore_index, BLOCK=_REDUCE_BLOCK,
        num_warps=4)
    return loss, lse, valid_count


def torch_fwd_varlen_fused_cross_entropy(
    hidden_states, weight, labels, cu_seqlens,
    max_seqlen=None, ignore_index=-100, softcap=None,
):
    """Return sequence-balanced loss and state for packed backward."""
    _validate_inputs(hidden_states, weight, labels, ignore_index)
    num_documents, max_seqlen = _varlen_info(
        hidden_states, cu_seqlens, max_seqlen)
    softcap = _validate_softcap(softcap)
    lse, nll = _forward_rows(
        hidden_states, weight, labels, ignore_index, softcap)
    total_tokens = hidden_states.shape[1]
    document_loss = torch.empty(
        num_documents, device=hidden_states.device, dtype=torch.float32)
    document_valid = torch.empty(
        num_documents, device=hidden_states.device, dtype=torch.int32)
    token_inv_count = torch.empty(
        total_tokens, device=hidden_states.device, dtype=torch.float32)
    _document_loss_reduce[(num_documents,)](
        nll, labels, cu_seqlens, document_loss, document_valid,
        token_inv_count, num_documents,
        IGNORE_INDEX=ignore_index, BLOCK=_REDUCE_BLOCK,
        num_warps=4)
    loss = torch.empty((), device=hidden_states.device, dtype=torch.float32)
    valid_document_count = torch.empty(
        (), device=hidden_states.device, dtype=torch.int32)
    _document_mean_reduce[(1,)](
        document_loss, document_valid, loss, valid_document_count,
        num_documents, BLOCK=_REDUCE_BLOCK, num_warps=4)
    return loss, lse, valid_document_count, token_inv_count


@torch.no_grad()
def fused_cross_entropy_per_position(
    hidden_states, weight, labels, ignore_index=-100, softcap=None,
    cu_seqlens=None, max_seqlen=None,
):
    """Return the valid batch/document mean at each position; forward only."""
    batch, sequence, _, _ = _validate_inputs(
        hidden_states, weight, labels, ignore_index)
    softcap = _validate_softcap(softcap)
    _, nll = _forward_rows(
        hidden_states, weight, labels, ignore_index, softcap)
    if cu_seqlens is not None:
        num_documents, max_seqlen = _varlen_info(
            hidden_states, cu_seqlens, max_seqlen)
        output = torch.empty(
            max_seqlen, device=hidden_states.device, dtype=torch.float32)
        if max_seqlen > 0:
            _varlen_per_position_reduce[(max_seqlen,)](
                nll, labels, cu_seqlens, output,
                num_documents, max_seqlen,
                IGNORE_INDEX=ignore_index, BLOCK=_REDUCE_BLOCK,
                num_warps=4)
        return output
    if max_seqlen is not None:
        raise ValueError("max_seqlen requires cu_seqlens")
    output = torch.empty(
        sequence, device=hidden_states.device, dtype=torch.float32)
    _per_position_reduce[(sequence,)](
        nll, labels, output, batch, sequence,
        IGNORE_INDEX=ignore_index, BLOCK=_REDUCE_BLOCK,
        num_warps=4)
    return output


def _chunk_rows(m_size, vocab_size, chunk_size):
    if chunk_size is not None:
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        return min(m_size, chunk_size)
    max_elements = _DEFAULT_DLOGITS_BYTES // torch.bfloat16.itemsize
    rows = max(1, max_elements // vocab_size)
    return min(m_size, rows)


def _backward_impl(
    hidden_states, weight, labels, lse, valid_count, token_inv_count,
    dloss, ignore_index, chunk_size, softcap, is_varlen,
):
    batch, sequence, in_channels, vocab_size = _validate_inputs(
        hidden_states, weight, labels, ignore_index)
    softcap = _validate_softcap(softcap)
    has_softcap = softcap is not None
    softcap_value = softcap if has_softcap else 1.0
    m_size = batch * sequence
    if (lse.shape != (m_size,) or lse.dtype != torch.float32
            or lse.device != hidden_states.device or not lse.is_contiguous()):
        raise ValueError("lse must be a contiguous CUDA FP32 tensor of shape [B*N]")
    if (valid_count.shape != () or valid_count.dtype != torch.int32
            or valid_count.device != hidden_states.device):
        raise ValueError("valid_count must be a scalar CUDA int32 tensor")
    if is_varlen and (
            token_inv_count.shape != (m_size,)
            or token_inv_count.dtype != torch.float32
            or token_inv_count.device != hidden_states.device
            or not token_inv_count.is_contiguous()):
        raise ValueError(
            "token_inv_count must be contiguous CUDA FP32 with shape [total_tokens]")
    if (dloss.numel() != 1 or dloss.device != hidden_states.device
            or dloss.dtype != torch.float32):
        raise ValueError("dloss must be a scalar CUDA FP32 tensor")

    rows_per_chunk = _chunk_rows(m_size, vocab_size, chunk_size)
    dlogits = torch.empty(
        (rows_per_chunk, vocab_size), device=hidden_states.device,
        dtype=torch.bfloat16)
    dhidden = torch.empty_like(hidden_states)
    dweight = torch.zeros_like(weight, dtype=torch.float32)

    for row_start in range(0, m_size, rows_per_chunk):
        rows = min(rows_per_chunk, m_size - row_start)
        _cross_entropy_dlogits[(
            triton.cdiv(rows, _DLOGITS_BM),
            triton.cdiv(vocab_size, _DLOGITS_BV),
        )](
            hidden_states, weight, labels, lse, dloss, valid_count,
            token_inv_count, dlogits,
            row_start, rows, in_channels, vocab_size,
            softcap_value,
            IGNORE_INDEX=ignore_index, HAS_SOFTCAP=has_softcap,
            IS_VARLEN=is_varlen,
            BLOCK_M=_DLOGITS_BM, BLOCK_V=_DLOGITS_BV,
            BLOCK_K=_DLOGITS_BK, num_warps=4, num_stages=3)
        _dhidden_gemm[(
            triton.cdiv(rows, _DX_BM),
            triton.cdiv(in_channels, _DX_BC),
        )](
            dlogits, weight, dhidden,
            row_start, rows, in_channels, vocab_size,
            BLOCK_M=_DX_BM, BLOCK_C=_DX_BC, BLOCK_V=_DX_BV,
            num_warps=8, num_stages=3)
        _dweight_gemm_accumulate[(
            triton.cdiv(vocab_size, _DW_BV),
            triton.cdiv(in_channels, _DW_BC),
        )](
            dlogits, hidden_states, dweight,
            row_start, rows, in_channels, vocab_size,
            BLOCK_V=_DW_BV, BLOCK_C=_DW_BC, BLOCK_R=_DW_BR,
            num_warps=8, num_stages=3)
    return dhidden, dweight


def torch_bwd_fused_cross_entropy(
    hidden_states, weight, labels, lse, valid_count, dloss,
    ignore_index=-100, chunk_size=None, softcap=None,
):
    """Fixed-length backward exposed separately for numerical testing."""
    return _backward_impl(
        hidden_states, weight, labels, lse, valid_count, lse,
        dloss, ignore_index, chunk_size, softcap, False)


def torch_bwd_varlen_fused_cross_entropy(
    hidden_states, weight, labels, lse, valid_document_count,
    token_inv_count, dloss, ignore_index=-100, chunk_size=None,
    softcap=None,
):
    """Packed sequence-balanced backward exposed for numerical testing."""
    return _backward_impl(
        hidden_states, weight, labels, lse, valid_document_count,
        token_inv_count, dloss, ignore_index, chunk_size, softcap, True)


class FusedCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx, hidden_states, weight, labels, ignore_index=-100,
        chunk_size=None, softcap=None, cu_seqlens=None,
        max_seqlen=None,
    ):
        if chunk_size is not None and (
                not isinstance(chunk_size, int) or chunk_size <= 0):
            raise ValueError("chunk_size must be a positive integer")
        has_varlen = cu_seqlens is not None
        if has_varlen:
            loss, lse, valid_count, token_inv_count = (
                torch_fwd_varlen_fused_cross_entropy(
                    hidden_states, weight, labels, cu_seqlens,
                    max_seqlen, ignore_index, softcap))
        else:
            if max_seqlen is not None:
                raise ValueError("max_seqlen requires cu_seqlens")
            loss, lse, valid_count = torch_fwd_fused_cross_entropy(
                hidden_states, weight, labels, ignore_index, softcap)
            token_inv_count = torch.empty(
                0, device=hidden_states.device, dtype=torch.float32)
        ctx.save_for_backward(
            hidden_states, weight, labels, lse, valid_count,
            token_inv_count)
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        ctx.softcap = softcap
        ctx.has_varlen = has_varlen
        return loss

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dloss):
        (hidden_states, weight, labels, lse, valid_count,
         token_inv_count) = ctx.saved_tensors
        if ctx.has_varlen:
            dhidden, dweight = torch_bwd_varlen_fused_cross_entropy(
                hidden_states, weight, labels, lse, valid_count,
                token_inv_count, dloss.contiguous(), ctx.ignore_index,
                ctx.chunk_size, ctx.softcap)
        else:
            dhidden, dweight = torch_bwd_fused_cross_entropy(
                hidden_states, weight, labels, lse, valid_count,
                dloss.contiguous(), ctx.ignore_index, ctx.chunk_size,
                ctx.softcap)
        return dhidden, dweight, None, None, None, None, None, None


def fused_cross_entropy(
    hidden_states, weight, labels, ignore_index=-100, chunk_size=None,
    softcap=None, cu_seqlens=None, max_seqlen=None,
):
    """Fused linear and CE, with sequence-balanced mean for packed input."""
    return FusedCrossEntropyFunction.apply(
        hidden_states, weight, labels, ignore_index, chunk_size, softcap,
        cu_seqlens, max_seqlen)


@torch.no_grad()
def fused_cross_entropy_unreduced(
    hidden_states, weight, labels, ignore_index=-100, softcap=None,
):
    """Forward-only per-token NLL; ignored tokens return zero, without logits."""
    softcap = _validate_softcap(softcap)
    _, nll = _forward_rows(hidden_states, weight, labels, ignore_index, softcap)
    return nll.view_as(labels)
