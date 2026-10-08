"""Opaque CUDA/Triton boundaries for Dynamo, with explicit saved-state autograd.

Packed checkpoints have data-dependent sizes. Compile with
torch._dynamo.config.capture_dynamic_output_shape_ops=True; no padded workspace
or backward forward-recomputation is introduced.
"""
from typing import Optional
import torch
import cu_flash_dism as cu
from .emb_kernel import emb_fwd_wrapper, emb_bwd_wrapper
from .kernels.voc_glue import (prepare_embedding, select_operands, split_interpolation_gradients,
                               merge_token_gradients, cast_vocabulary_gradients)


@torch.library.custom_op('flash_dism::conv_forward', mutates_args=())
def conv_forward(x: torch.Tensor, weight: torch.Tensor, kernel: torch.Tensor,
                 boundaries: Optional[torch.Tensor], maximum: Optional[int]) -> torch.Tensor:
    from .kernels.conv1d import torch_fwd_short_conv_silu
    return torch_fwd_short_conv_silu(x, weight, kernel, boundaries, maximum)


@conv_forward.register_fake
def _conv_fake(x, weight, kernel, boundaries, maximum):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@torch.library.custom_op('flash_dism::conv_backward', mutates_args=())
def conv_backward(x: torch.Tensor, weight: torch.Tensor, kernel: torch.Tensor,
                  boundaries: Optional[torch.Tensor], maximum: Optional[int],
                  dout: torch.Tensor) -> list[torch.Tensor]:
    from .kernels.conv1d import torch_bwd_short_conv_silu
    gradients = torch_bwd_short_conv_silu(x, weight, kernel, dout.contiguous(), boundaries, maximum)
    return [g.to(t.dtype).contiguous() for g, t in zip(gradients, (x, weight, kernel))]


@conv_backward.register_fake
def _conv_backward_fake(x, weight, kernel, boundaries, maximum, dout):
    return [t.new_empty(t.shape) for t in (x, weight, kernel)]


def _conv_setup(ctx, inputs, output):
    x, weight, kernel, boundaries, ctx.maximum = inputs
    ctx.packed = boundaries is not None
    ctx.save_for_backward(x, weight, kernel, *([boundaries] if ctx.packed else []))


def _conv_backward(ctx, dout):
    saved = ctx.saved_tensors
    return (*conv_backward(*saved[:3], saved[3] if ctx.packed else None, ctx.maximum, dout), None, None)


conv_forward.register_autograd(_conv_backward, setup_context=_conv_setup)


@torch.library.custom_op('flash_dism::validate_pack', mutates_args=())
def validate_pack(boundaries: torch.Tensor, tokens: int, maximum: int) -> torch.Tensor:
    values = cu.make_varlen_layout(boundaries, tokens)
    if maximum < max(values[1], default=0):
        raise ValueError('max_seqlen must be >= longest document')
    return boundaries.clone()


@validate_pack.register_fake
def _validate_fake(boundaries, tokens, maximum):
    return torch.empty_like(boundaries)


def _table(boundaries, tokens):
    return None if boundaries is None else cu.make_varlen_layout(boundaries, tokens)[0]


@torch.library.custom_op('flash_dism::prepare_pack_flat', mutates_args=())
def prepare_pack(boundaries: torch.Tensor, tokens: int, maximum: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One validation/readback shared by compiled layers and their backwards.

    Explicit tensor outputs keep ownership within this invocation/autograd graph;
    no pointer-based cache of mutable cu_seqlens is used.
    """
    values = cu.make_varlen_layout(boundaries, tokens)
    if maximum < max(values[1], default=0):
        raise ValueError('max_seqlen must be >= longest document')
    # Flat transport avoids AOT's size-one specialization when one document
    # would otherwise change a [documents,7] stride/contiguity guard.
    return boundaries.clone(), values[0].flatten()


@prepare_pack.register_fake
def _prepare_pack_fake(boundaries, tokens, maximum):
    return (torch.empty_like(boundaries),
            torch.empty(((boundaries.numel()-1)*7,), device='cpu', dtype=torch.int64))


# Version the internal operator identity: saved interpolants changed from
# BHND to BNHD, so old Inductor disk-cache output contracts must not be reused.
@torch.library.custom_op('flash_dism::voc_forward_bnhd', mutates_args=())
def voc_forward(q: torch.Tensor, k: torch.Tensor, sq: torch.Tensor, sk: torch.Tensor,
                v: torch.Tensor, eq: torch.Tensor, ek: torch.Tensor, tau: torch.Tensor,
                direction: torch.Tensor, hard: torch.Tensor,
                boundaries: Optional[torch.Tensor], layout: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
    cu.prepare_embedding(q, k, sq, sk, v, eq, ek, tau, direction, hard, False)
    query, key, qe, ke = prepare_embedding(q, k, eq, ek)
    qfk, kfq, lk, lq, _, _, ik, iq = emb_fwd_wrapper(query, key, qe, ke, 1., tau.detach())
    qv, kv = select_operands(q, k, qfk, kfq, direction)
    output, norm, state = cu.core_forward(
        (qv, kv, sq, sk, v, lq.transpose(1, 2), lk.transpose(1, 2),
         iq, ik, direction, hard, tau), layout.view(-1, 7) if layout is not None else _table(boundaries, q.shape[1]),
        0, True, False, True)
    # All auxiliary results own newly computed storage; no inputs alias outputs.
    return [output, qfk, kfq, lk, lq, ik, iq,
            state['vertical'], state['operands'][12], norm]


@voc_forward.register_fake
def _forward_fake(q, k, sq, sk, v, eq, ek, tau, direction, hard, boundaries, layout=None):
    b, n, h, d = q.shape
    vector = lambda: q.new_empty((b, n, h, d))
    scalar = lambda dtype: q.new_empty((b, h, n), dtype=dtype)
    if boundaries is None:
        vertical = q.new_empty((b, h, (n-1)//16, n), dtype=torch.float32)
        boundary = q.new_empty((b, h, (n-1)//32, n), dtype=torch.float32)
    else:
        ctx = torch.library.get_ctx()
        vertical = q.new_empty((ctx.new_dynamic_size(),), dtype=torch.float32)
        boundary = q.new_empty((ctx.new_dynamic_size(),), dtype=torch.float32)
    return [torch.empty_like(v), vector(), vector(), scalar(torch.float32),
            scalar(torch.float32), scalar(torch.int32), scalar(torch.int32),
            vertical, boundary, scalar(torch.float32)]


@torch.library.custom_op('flash_dism::voc_backward_bnhd', mutates_args=())
def voc_backward(inputs: list[torch.Tensor], saved: list[torch.Tensor],
                 boundaries: Optional[torch.Tensor], dout: torch.Tensor,
                 layout: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
    q, k, sq, sk, v, eq, ek, tau, direction, hard = inputs
    output, qfk, kfq, lk, lq, ik, iq, vertical, boundary, norm = saved
    cu.prepare_embedding(q, k, sq, sk, v, eq, ek, tau, direction, hard, False)
    query, key, qe, ke = prepare_embedding(q, k, eq, ek)
    qv, kv = select_operands(q, k, qfk, kfq, direction)
    raw = (qv, kv, sq, sk, v, lq.transpose(1, 2), lk.transpose(1, 2), iq, ik, direction, hard, tau)
    # Forward already absorbed tau. Do not allocate zeros or subtract it again.
    prepared = list(cu.prepare_operands(raw, boundaries is not None, True))
    prepared.append(boundary)
    core_norm = norm.flatten() if boundaries is not None else norm
    g, _ = cu.core_backward(prepared, vertical, output, core_norm, dout.to(torch.bfloat16).contiguous(),
                            layout.view(-1, 7) if layout is not None else _table(boundaries, q.shape[1]), 0, False, False)
    doq, dok = split_interpolation_gradients(g['q_vec'], g['k_vec'], direction)
    dq, dk, dqe, dke = emb_bwd_wrapper(query, key, qe, ke, qfk, kfq, lk, lq,
        doq, dok, g['k_lse'].transpose(1, 2).float(), g['q_lse'].transpose(1, 2).float(), 1., tau.detach())
    dq, dk, dsq = merge_token_gradients(g['q_vec'], g['k_vec'], dq, dk, g['sq_vec'], direction)
    dqe, dke = cast_vocabulary_gradients(dqe, dke, eq, ek)
    gradients = [dq, dk, dsq, g['sk_vec'].to(sk.dtype), g['v'].to(v.dtype),
                 dqe, dke, g['rtau'].to(tau.dtype)]
    return [grad.contiguous() for grad in gradients]


@voc_backward.register_fake
def _backward_fake(inputs, saved, boundaries, dout, layout=None):
    return [t.new_empty(t.shape) for t in inputs[:8]]


def _setup(ctx, inputs, output):
    ctx.packed = inputs[10] is not None
    ctx.has_layout = inputs[11] is not None
    ctx.save_for_backward(*inputs[:10], *output,
                          *([inputs[10]] if ctx.packed else []),
                          *([inputs[11]] if ctx.has_layout else []))
    ctx.mark_non_differentiable(*output[1:])


def _backward(ctx, grads):
    saved = ctx.saved_tensors
    gradients = voc_backward(list(saved[:10]), list(saved[10:20]),
                             saved[20] if ctx.packed else None, grads[0],
                             saved[20 + int(ctx.packed)] if ctx.has_layout else None)
    # Dispatcher omits trailing default=None arguments, while setup_context
    # receives defaults restored. Match the actual autograd input arity.
    return (*gradients, None, None, None, None)[:len(ctx.needs_input_grad)]


voc_forward.register_autograd(_backward, setup_context=_setup)
