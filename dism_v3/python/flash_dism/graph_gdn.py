"""FLA's training kernels with caller-owned chunk indices for graph replay.

This adapter preserves FLA's l2norm, forward, saved tensors and backward. It
only replaces the CPU chunk-index builder. cu_seqlens must be a fresh tensor
per capture invocation (the nanochat forward clones it), so FLA's offset cache
cannot supply a stale tensor computed outside the captured graph.
"""
import torch
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd, chunk_gated_delta_rule_bwd,
)
from fla.utils import autocast_custom_fwd, autocast_custom_bwd, input_guard


class _PreparedGDN(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, beta, boundaries, chunks):
        q, qr = l2norm_fwd(q)
        k, kr = l2norm_fwd(k)
        ctx.scale = q.shape[-1] ** -0.5
        g, out, a, _, initial = chunk_gated_delta_rule_fwd(
            q=q, k=k, v=v, g=g, beta=beta, scale=ctx.scale,
            initial_state=None, output_final_state=False,
            cu_seqlens=boundaries, chunk_indices=chunks,
            cp_context=None, transpose_state_layout=False,
        )
        ctx.save_for_backward(q, qr, k, kr, v, g, beta, a, initial, boundaries, chunks)
        return out.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dout):
        q, qr, k, kr, v, g, beta, a, initial, boundaries, chunks = ctx.saved_tensors
        dq, dk, dv, db, dg, _ = chunk_gated_delta_rule_bwd(
            q=q, k=k, v=v, g=g, beta=beta, A=a, scale=ctx.scale,
            initial_state=initial, do=dout, dht=None,
            cu_seqlens=boundaries, chunk_indices=chunks,
            cp_context=None, transpose_state_layout=False,
        )
        return (l2norm_bwd(q, qr, dq).to(q), l2norm_bwd(k, kr, dk).to(k),
                dv.to(v), dg.to(g), db.to(beta), None, None)


prepared_gdn = _PreparedGDN.apply
