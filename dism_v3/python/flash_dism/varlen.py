"""Packed batch1 varlen infrastructure; independent from fixed-length kernels."""
from dataclasses import dataclass

import torch
import cu_flash_dism as cu


@dataclass(frozen=True)
class VarlenLayout:
    """Reusable validated sequence layout.

    Construction synchronizes a CUDA cu_seqlens once to allocate exact ragged
    capacities. Reuse across layers/steps with the same pack. CUDA Graph capture
    is not supported yet: integer task/document tables are uploaded per call.
    Every boundary, including the terminal T, must be256-token aligned.
    Empty documents are allowed. No private aligned input copies are made.
    The table is internal CPU metadata, not a user-editable device descriptor.
    """
    table: torch.Tensor
    lengths: tuple[int, ...]
    tokens: int
    padded_tokens: int
    forward_elements: int
    vertical_elements: int
    backward_elements: int
    max_length: int = 0  # Nonzero only for an explicitly prepared graph capacity.

    @classmethod
    def for_graph(cls, table, total_tokens):
        maximum = cu.graph_layout_maximum(table)
        if not maximum:
            raise ValueError('expected an owned CUDA Graph layout')
        return cls(table, tuple(table[:, 1].tolist()), total_tokens, total_tokens,
                   total_tokens*((maximum-1)//32),
                   total_tokens*((maximum-1)//16),
                   total_tokens*(maximum//32), maximum)

    @classmethod
    def from_cu_seqlens(cls, cu_seqlens, total_tokens):
        values=cu.make_varlen_layout(cu_seqlens,total_tokens)
        return cls(values[0],tuple(values[1]),*values[2:])

    def pack(self, tensor, *, vectors):
        """Diagnostic compatibility only; production never packs tensors."""
        if vectors:
            return tensor.squeeze(0)
        return cu.varlen_pack(tensor.contiguous(),self.table,False)

    def checkpoint_bytes(self, heads):
        return 4*heads*(self.forward_elements+self.vertical_elements+3*self.backward_elements)


def stage_operands(layout,q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau):
    """Native validation, raw-LSE absorption and head-major metadata staging."""
    return tuple(cu.prepare_operands(
        (q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau),True))


@torch.no_grad()
def forward_varlen(q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,
                   cu_seqlens=None,*,layout=None,save_state=False,fp32_output=False,_lse_preabsorbed=False,
                   direction_true=False):
    """Native three-stage varlen forward; inputs and output retain batch1.

    Raw natural-log LSE follows fixed-length semantics. Reusing layout avoids
    repeating the cu_seqlens host synchronization. Returned normalizer is
    [1,H,T]; internal statistics use the same head-major layout.
    fp32_output=True selects the separately compiled FP32 output instance.
    hard is bool [training_rows,H], with equal-length original training rows
    covering T. Documents cannot cross those row boundaries. Hard rows return
    zero output/norm and zero gradients; no hard attention is computed here.
    direction_true is a host assertion that every direction is true; it selects
    specialized kernels without inspecting GPU flags. False retains the general
    path (including all-true tensors). Backward reuses the saved selection.
    """
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('varlen metadata uploads do not support CUDA Graph capture yet')
    if layout is None:
        if cu_seqlens is None:
            raise ValueError('provide cu_seqlens or a preconstructed layout')
        layout=VarlenLayout.from_cu_seqlens(cu_seqlens,q.shape[1])
    elif cu_seqlens is not None:
        raise ValueError('provide either layout or cu_seqlens, not both')
    output,lse2,state=cu.core_forward(
        (q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau),
        layout.table,0,save_state,fp32_output,_lse_preabsorbed,direction_true)
    if save_state:
        state['layout']=layout
        state['direction_true']=direction_true
        return output,lse2,state
    return output,lse2


@torch.no_grad()
def backward_varlen(state,dout,*,fp32_output=False,return_diagnostics=False):
    """Native B1/B2/B3, returning packed gradients for the eight core operands."""
    layout=state['layout']
    if dout.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('varlen descriptor uploads do not support CUDA Graph capture yet')
    gradients,details=cu.core_backward(
        state['operands'],state['vertical'],state['output'],state['normalizer'],dout,
        layout.table,0,fp32_output,return_diagnostics,state.get('direction_true',False))
    return (gradients,details) if return_diagnostics else gradients


class _VarlenCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau,layout,absorbed):
        out,_,state=forward_varlen(q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau,
                                   layout=layout,save_state=True,_lse_preabsorbed=absorbed)
        ctx.layout=layout
        ctx.save_for_backward(*state['operands'],state['vertical'],out,state['normalizer'],layout.table)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,dout):
        saved=ctx.saved_tensors
        state=dict(layout=ctx.layout,operands=saved[:13],vertical=saved[13],
                   output=saved[14],normalizer=saved[15])
        g=backward_varlen(state,dout.to(torch.bfloat16))
        return (g['q_vec'],g['k_vec'],g['sq_vec'],g['sk_vec'],g['v'],g['q_lse'],g['k_lse'],
                None,None,None,None,g['rtau'],None,None)


def dism_core_varlen(q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,
                     cu_seqlens=None,*,layout=None,_lse_preabsorbed=False):
    """Differentiable packed v3 core; first derivatives and runtime R/D/DV dispatch."""
    if torch.compiler.is_compiling():
        from .compiler import core_forward, prepare_pack
        if layout is not None and cu_seqlens is not None:
            raise ValueError('provide either layout or cu_seqlens, not both')
        if layout is None:
            if cu_seqlens is None:
                raise ValueError('provide cu_seqlens or layout')
            _, table = prepare_pack(cu_seqlens,q.shape[1],q.shape[1])
            table = table.view(-1,7)
        else:
            table = layout.table
        return core_forward((q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau),
                             table=table,absorbed=_lse_preabsorbed)
    if layout is None:
        if cu_seqlens is None:
            raise ValueError('provide cu_seqlens or layout')
        layout=VarlenLayout.from_cu_seqlens(cu_seqlens,q.shape[1])
    elif cu_seqlens is not None:
        raise ValueError('provide either layout or cu_seqlens, not both')
    from .hard_training import dism_core_hybrid
    return dism_core_hybrid(q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,
                            layout=layout,_lse_preabsorbed=_lse_preabsorbed)


def voc_dism_varlen(q,k,sq,sk,v,q_vocab,k_vocab,rtau,cu_seqlens=None,*,direction,hard,layout=None,beta=1.0,hard_planner=None):
    """Existing Triton embedding interpolation followed by native varlen core.

    Vocabulary activation and sq/sk activation belong to the caller. Direction
    remains [1,H], not one independent direction per document. Hard flags are
    [training_rows,H], inherited by documents within each equal-length row.
    hard_planner overrides the automatic eager runtime. beta is the
    vocabulary-softmax temperature (1.0 reproduces the unscaled score).
    """
    if torch.compiler.is_compiling() and hard_planner is None:
        if layout is not None and cu_seqlens is not None:
            raise ValueError('provide either layout or cu_seqlens, not both')
        if layout is None and cu_seqlens is None:
            raise ValueError('provide cu_seqlens or layout')
        from .compiler import voc_forward
        return voc_forward(q,k,sq,sk,v,q_vocab,k_vocab,rtau,
                            beta,direction,hard,
                            cu_seqlens,None if layout is None else layout.table.flatten())[0]
    from .emb_kernel import interp_embeddings
    if layout is None:
        if cu_seqlens is None:
            raise ValueError('provide cu_seqlens or layout')
        layout=VarlenLayout.from_cu_seqlens(cu_seqlens,q.shape[1])
    elif cu_seqlens is not None:
        raise ValueError('provide either layout or cu_seqlens, not both')
    query,key,eq,ek=cu.prepare_embedding(q,k,sq,sk,v,q_vocab,k_vocab,rtau,direction,hard)
    heads=q.shape[2]
    if not layout.tokens:
        lse=torch.empty((1,0,heads),dtype=torch.float32,device=q.device)
        labels=torch.empty((1,heads,0),dtype=torch.int32,device=q.device)
        out=dism_core_varlen(q,k,sq,sk,v,lse,lse,labels,labels,direction,hard,rtau,layout=layout)
        return out+(eq.float().sum()*0+ek.float().sum()*0).to(out.dtype)
    qfk,kfq,lk,lq,ik,iq=interp_embeddings(
        query,key,eq,ek,beta,rtau.detach())
    qv,kv=cu.select_embedding(q,k,qfk,kfq,direction)
    if hard_planner is not None:
        from .hard_training import dism_core_hybrid
        return dism_core_hybrid(qv,kv,sq,sk,v,lq.transpose(1,2),lk.transpose(1,2),iq,ik,
                                direction,hard,rtau,planner=hard_planner,layout=layout,
                                _lse_preabsorbed=True)
    return dism_core_varlen(qv,kv,sq,sk,v,lq.transpose(1,2),lk.transpose(1,2),iq,ik,
                           direction,hard,rtau,layout=layout,_lse_preabsorbed=True)
