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
def _voc_forward_impl(q: torch.Tensor, k: torch.Tensor, sq: torch.Tensor, sk: torch.Tensor,
                v: torch.Tensor, eq: torch.Tensor, ek: torch.Tensor, tau: torch.Tensor,
                beta: torch.Tensor, direction: torch.Tensor, hard: torch.Tensor,
                boundaries: Optional[torch.Tensor], layout: Optional[torch.Tensor],
                need_backward: bool, direction_true=False) -> list[torch.Tensor]:
    from .hard_training import default_hard_planner, layout_from_table
    from .hard_forward import forward_hard
    b = float(beta)
    cu.prepare_embedding(q, k, sq, sk, v, eq, ek, tau, direction, hard, False)
    query, key, qe, ke = prepare_embedding(q, k, eq, ek)
    if direction_true:
        from .kernels.interp_true import emb_fwd_true
        qfk, kfq, lk, lq, _, _, ik, iq = emb_fwd_true(query, key, qe, ke, b, tau.detach())
    else:
        qfk, kfq, lk, lq, _, _, ik, iq = emb_fwd_wrapper(query, key, qe, ke, b, tau.detach())
    # The core score must be beta*raw - LSE_beta; scale the interpolated side.
    if direction_true:
        qv, kv = q.contiguous(), qfk if b == 1.0 else qfk*b
    else:
        qsk, ksk = (qfk, kfq) if b == 1.0 else (qfk * b, kfq * b)
        qv, kv = select_operands(q, k, qsk, ksk, direction)
    table = layout.view(-1, 7) if layout is not None else _table(boundaries, q.shape[1])
    pack = layout_from_table(table, q.shape[1])
    ticket = (default_hard_planner(q.device).submit(iq, ik, hard, tau, layout=pack)
              if need_backward and q.shape[1] else None)
    output, norm, state = cu.core_forward(
        (qv, kv, sq, sk, v, lq.transpose(1, 2), lk.transpose(1, 2),
         iq, ik, direction, hard, tau), table,
        0, True, False, True, direction_true)
    combined = output + forward_hard(sq.contiguous(),sk.contiguous(),v.contiguous(),iq,ik,
                                     hard.contiguous(),tau.contiguous(),layout=pack)[0] if q.shape[1] else output.clone()
    # All auxiliary results own newly computed storage; no inputs alias outputs.
    # qfk/kfq are saved UNSCALED so the backward delta uses E_beta with dL/dE.
    return [combined, qfk, kfq, lk, lq, ik, iq,
            state['vertical'], state['operands'][12], norm, output, cu.hold_training_plan(ticket)]


@torch.library.custom_op('flash_dism::voc_forward_hybrid_v1', mutates_args=())
def _voc_forward(q: torch.Tensor, k: torch.Tensor, sq: torch.Tensor, sk: torch.Tensor,
                 v: torch.Tensor, eq: torch.Tensor, ek: torch.Tensor, tau: torch.Tensor,
                 beta: torch.Tensor, direction: torch.Tensor, hard: torch.Tensor,
                 boundaries: Optional[torch.Tensor], layout: Optional[torch.Tensor],
                 need_backward: bool) -> list[torch.Tensor]:
    # Compatibility for explicit tensor-beta callers, not the model default.
    return _voc_forward_impl(q,k,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout,need_backward)


@torch.library.custom_op('flash_dism::voc_forward_host_beta_v2', mutates_args=())
def _voc_forward_host(q: torch.Tensor, k: torch.Tensor, sq: torch.Tensor, sk: torch.Tensor,
                      v: torch.Tensor, eq: torch.Tensor, ek: torch.Tensor, tau: torch.Tensor,
                      beta: float, direction: torch.Tensor, hard: torch.Tensor,
                      boundaries: Optional[torch.Tensor], layout: Optional[torch.Tensor],
                      need_backward: bool, direction_true: bool = False) -> list[torch.Tensor]:
    return _voc_forward_impl(q,k,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout,need_backward,direction_true)


@_voc_forward_host.register_fake
@_voc_forward.register_fake
def _forward_fake(q, k, sq, sk, v, eq, ek, tau, beta, direction, hard, boundaries, layout, need_backward, direction_true=False):
    b, n, h, d = q.shape
    vector = lambda: q.new_empty((b, n, h, d))
    scalar = lambda dtype: q.new_empty((b, h, n), dtype=dtype)
    if boundaries is None and layout is None:
        vertical = q.new_empty((b, h, (n-1)//16, n), dtype=torch.float32)
        boundary = q.new_empty((b, h, (n-1)//32, n), dtype=torch.float32)
    else:
        ctx = torch.library.get_ctx()
        vertical = q.new_empty((ctx.new_dynamic_size(),), dtype=torch.float32)
        boundary = q.new_empty((ctx.new_dynamic_size(),), dtype=torch.float32)
    return [torch.empty_like(v), vector(), q.new_empty((0,)) if direction_true else vector(), scalar(torch.float32),
            scalar(torch.float32), scalar(torch.int32), scalar(torch.int32),
            vertical, boundary, scalar(torch.float32), torch.empty_like(v),
            torch.empty((1,), dtype=torch.int64, device='cpu')]


def _voc_backward_impl(inputs: list[torch.Tensor], saved: list[torch.Tensor],
                 boundaries: Optional[torch.Tensor], dout: torch.Tensor,
                 layout: Optional[torch.Tensor] = None, direction_true=False) -> list[torch.Tensor]:
    q, k, sq, sk, v, eq, ek, tau, beta, direction, hard = inputs
    b = float(beta)
    _, qfk, kfq, lk, lq, ik, iq, vertical, boundary, norm, output, handle = saved
    cu.prepare_embedding(q, k, sq, sk, v, eq, ek, tau, direction, hard, False)
    query, key, qe, ke = prepare_embedding(q, k, eq, ek)
    if direction_true:
        qv, kv = q.contiguous(), qfk if b == 1.0 else qfk*b
    else:
        qsk, ksk = (qfk, kfq) if b == 1.0 else (qfk * b, kfq * b)
        qv, kv = select_operands(q, k, qsk, ksk, direction)
    raw = (qv, kv, sq, sk, v, lq.transpose(1, 2), lk.transpose(1, 2), iq, ik, direction, hard, tau)
    # Forward already absorbed tau. Do not allocate zeros or subtract it again.
    packed = boundaries is not None or layout is not None
    prepared = list(cu.prepare_operands(raw, packed, True))
    prepared.append(boundary)
    core_norm = norm.flatten() if packed else norm
    g, _ = cu.core_backward(prepared, vertical, output, core_norm, dout.to(torch.bfloat16).contiguous(),
                            layout.view(-1, 7) if layout is not None else _table(boundaries, q.shape[1]),
                            0, False, False, direction_true)
    ticket = cu.get_training_plan(handle)
    if ticket is not None:
        from .hard_backward import backward_hard_recompute
        from .hard_training import layout_from_table
        do = dout.to(torch.bfloat16).contiguous()
        hg = ticket.backward(sq, sk, v, do, fallback=lambda: backward_hard_recompute(
            sq,sk,v,do,iq,ik,hard,tau,layout=layout_from_table(
                layout.view(-1,7) if layout is not None else _table(boundaries,q.shape[1]),q.shape[1])))
        for name in ('sq_vec', 'sk_vec', 'v', 'rtau'):
            g[name] = g[name].float() + hg[name]
    if direction_true:
        # Q output does not exist; its backward reads only dLSE, not this placeholder.
        doq, dok = g['k_vec'].to(torch.bfloat16).contiguous(), q
        from .kernels.interp_bwd_split import emb_bwd_split
        backward = lambda *args: emb_bwd_split(*args, direction_true=True)
    else:
        doq, dok = split_interpolation_gradients(g['q_vec'], g['k_vec'], direction)
        backward = emb_bwd_wrapper
    # doq/dok are dL/d(beta*E); the kernel wants dL/dE with out=E, so multiply by beta.
    dq, dk, dqe, dke = backward(query, key, qe, ke, qfk, kfq, lk, lq,
        doq if direction_true and b == 1.0 else doq * b, dok if direction_true else dok * b,
        g['k_lse'].transpose(1, 2).float(), g['q_lse'].transpose(1, 2).float(), b, tau.detach())
    dq, dk, dsq = merge_token_gradients(g['q_vec'], g['k_vec'], dq, dk, g['sq_vec'], direction)
    dqe, dke = cast_vocabulary_gradients(dqe, dke, eq, ek)
    gradients = [dq, dk, dsq, g['sk_vec'].to(sk.dtype), g['v'].to(v.dtype),
                 dqe, dke, g['rtau'].to(tau.dtype)]
    return [grad.contiguous() for grad in gradients]


@torch.library.custom_op('flash_dism::voc_backward_hybrid_v1', mutates_args=())
def voc_backward(inputs: list[torch.Tensor], saved: list[torch.Tensor],
                 boundaries: Optional[torch.Tensor], dout: torch.Tensor,
                 layout: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
    return _voc_backward_impl(inputs, saved, boundaries, dout, layout)


@torch.library.custom_op('flash_dism::voc_backward_host_beta_v2', mutates_args=())
def _voc_backward_host(inputs: list[torch.Tensor], beta: float, saved: list[torch.Tensor],
                       boundaries: Optional[torch.Tensor], dout: torch.Tensor,
                       layout: Optional[torch.Tensor], direction_true: bool = False) -> list[torch.Tensor]:
    return _voc_backward_impl([*inputs[:8], beta, *inputs[8:]], saved, boundaries, dout, layout,direction_true)


@_voc_backward_host.register_fake
def _backward_host_fake(inputs, beta, saved, boundaries, dout, layout,direction_true=False):
    return [t.new_empty(t.shape) for t in inputs[:8]]


@voc_backward.register_fake
def _backward_fake(inputs, saved, boundaries, dout, layout=None):
    return [t.new_empty(t.shape) for t in inputs[:8]]


def _setup(ctx, inputs, output):
    ctx.packed = inputs[11] is not None
    ctx.has_layout = inputs[12] is not None
    ctx.save_for_backward(*inputs[:11], *output,
                          *([inputs[11]] if ctx.packed else []),
                          *([inputs[12]] if ctx.has_layout else []))
    ctx.mark_non_differentiable(*output[1:])


def _backward(ctx, grads):
    saved = ctx.saved_tensors
    gradients = voc_backward(list(saved[:11]), list(saved[11:23]),
                             saved[23] if ctx.packed else None, grads[0],
                             saved[23 + int(ctx.packed)] if ctx.has_layout else None)
    # Dispatcher omits trailing default=None arguments, while setup_context
    # receives defaults restored. Match the actual autograd input arity.
    return (*gradients, None, None, None, None, None, None)[:len(ctx.needs_input_grad)]


_voc_forward.register_autograd(_backward, setup_context=_setup)


def _setup_host(ctx, inputs, output):
    ctx.beta = inputs[8]  # Immutable forward-time host value, not a live schedule buffer.
    ctx.direction_true = inputs[14]
    ctx.packed = inputs[11] is not None
    ctx.has_layout = inputs[12] is not None
    ctx.save_for_backward(*inputs[:8], *inputs[9:11], *output,
                          *([inputs[11]] if ctx.packed else []),
                          *([inputs[12]] if ctx.has_layout else []))
    ctx.mark_non_differentiable(*output[1:])


def _backward_host(ctx, grads):
    saved = ctx.saved_tensors
    gradients = _voc_backward_host(list(saved[:10]), ctx.beta, list(saved[10:22]),
                                   saved[22] if ctx.packed else None, grads[0],
                                   saved[22 + int(ctx.packed)] if ctx.has_layout else None,ctx.direction_true)
    return (*gradients, None, None, None, None, None, None, None)[:len(ctx.needs_input_grad)]


_voc_forward_host.register_autograd(_backward_host, setup_context=_setup_host)


def voc_forward(q, k, sq, sk, v, eq, ek, tau, beta, direction, hard, boundaries, layout=None, *, direction_true=False):
    need_backward = torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,sq,sk,v,eq,ek,tau))
    entry = _voc_forward if isinstance(beta, torch.Tensor) else _voc_forward_host
    if direction_true:
        if isinstance(beta, torch.Tensor):
            raise ValueError('fixed true direction requires a host scalar beta')
        # Explicit semantic override, never a GPU all()/item() dispatch decision.
        direction = torch.ones_like(direction)
        return entry(q,k,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout,need_backward,True)
    return entry(q,k,sq,sk,v,eq,ek,tau,beta,direction,hard,boundaries,layout,need_backward)


@torch.library.custom_op('flash_dism::core_forward_hybrid_v2', mutates_args=())
def _core_forward(raw: list[torch.Tensor], table: Optional[torch.Tensor], ctas: int,
                  absorbed: bool, need_backward: bool) -> list[torch.Tensor]:
    from .hard_training import default_hard_planner, layout_from_table
    from .hard_forward import forward_hard
    q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau = raw
    pack = layout_from_table(table,q.shape[1])
    cu.prepare_operands(raw,table is not None,absorbed)
    iq_c,ik_c=iq.to(torch.int32).contiguous(),ik.to(torch.int32).contiguous()
    ticket = (default_hard_planner(q.device).submit(iq_c,ik_c,hard,tau,layout=pack)
              if need_backward and q.shape[1] else None)
    out,norm,state = cu.core_forward(raw,table,ctas,True,False,absorbed,False,False)
    combined = out+forward_hard(sq.contiguous(),sk.contiguous(),v.contiguous(),iq_c,ik_c,
                                hard.contiguous(),tau.contiguous(),layout=pack,ctas=ctas)[0] if q.shape[1] else out.clone()
    return [combined,state['vertical'],state['operands'][12],out,norm,cu.hold_training_plan(ticket)]


@_core_forward.register_fake
def _core_fake(raw,table,ctas,absorbed,need_backward):
    q,v=raw[0],raw[4]
    b,n,h,_=q.shape
    if table is None:
        vertical=q.new_empty((b,h,(n-1)//16,n),dtype=torch.float32)
        boundary=q.new_empty((b,h,(n-1)//32,n),dtype=torch.float32)
    else:
        ctx=torch.library.get_ctx()
        vertical=q.new_empty((ctx.new_dynamic_size(),),dtype=torch.float32)
        boundary=q.new_empty((ctx.new_dynamic_size(),),dtype=torch.float32)
    return [torch.empty_like(v),vertical,boundary,torch.empty_like(v),
            q.new_empty((b,h,n),dtype=torch.float32),torch.empty(1,device='cpu',dtype=torch.int64)]


@torch.library.custom_op('flash_dism::core_backward_hybrid_v2', mutates_args=())
def _core_backward(raw: list[torch.Tensor], saved: list[torch.Tensor], table: Optional[torch.Tensor],
                   ctas: int, absorbed: bool, dout: torch.Tensor) -> list[torch.Tensor]:
    _,vertical,boundary,out,norm,handle=saved
    prepared=list(cu.prepare_operands(raw,table is not None,absorbed))
    prepared.append(boundary)
    dout=dout.to(torch.bfloat16).contiguous()
    g,_=cu.core_backward(prepared,vertical,out,norm.flatten() if table is not None else norm,
                         dout,table,ctas,False,False)
    ticket=cu.get_training_plan(handle)
    if ticket is not None:
        from .hard_backward import backward_hard_recompute
        from .hard_training import layout_from_table
        hg=ticket.backward(raw[2],raw[3],raw[4],dout,fallback=lambda: backward_hard_recompute(
            raw[2],raw[3],raw[4],dout,raw[7],raw[8],raw[10],raw[11],
            layout=layout_from_table(table,raw[0].shape[1]),ctas=ctas))
        for name in ('sq_vec','sk_vec','v','rtau'):
            g[name]=g[name].float()+hg[name]
    return [g[name].to(raw[i].dtype).contiguous() for name,i in
            zip(('q_vec','k_vec','sq_vec','sk_vec','v','q_lse','k_lse','rtau'),(0,1,2,3,4,5,6,11))]


@_core_backward.register_fake
def _core_backward_fake(raw,saved,table,ctas,absorbed,dout):
    return [torch.empty_like(raw[i]) for i in (0,1,2,3,4,5,6,11)]


def _core_setup(ctx,inputs,output):
    raw,table,ctx.ctas,ctx.absorbed,_=inputs
    ctx.packed=table is not None
    ctx.save_for_backward(*raw,*output,*([table] if ctx.packed else []))
    ctx.mark_non_differentiable(*output[1:])


def _core_autograd(ctx,grads):
    saved=ctx.saved_tensors
    g=_core_backward(list(saved[:12]),list(saved[12:18]),saved[18] if ctx.packed else None,
                      ctx.ctas,ctx.absorbed,grads[0])
    return ([*g[:7],None,None,None,None,g[7]],None,None,None,None)[:len(ctx.needs_input_grad)]


_core_forward.register_autograd(_core_autograd,setup_context=_core_setup)


def core_forward(raw,table=None,ctas=0,absorbed=False):
    need_backward=torch.is_grad_enabled() and any(raw[i].requires_grad for i in (0,1,2,3,4,5,6,11))
    return _core_forward(list(raw),table,ctas,absorbed,need_backward)[0]
