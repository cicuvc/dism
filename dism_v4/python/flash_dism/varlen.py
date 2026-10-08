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


def stage_operands(layout,q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,gate_delta=None):
    """Native validation, raw-LSE absorption and head-major metadata staging."""
    return tuple(cu.prepare_operands(
        (q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau)+(() if gate_delta is None else (gate_delta,)),True))


@torch.no_grad()
def forward_varlen(q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,
                   cu_seqlens=None,*,layout=None,save_state=False,fp32_output=False,_lse_preabsorbed=False,gate_delta=None):
    """Native three-stage varlen forward; inputs and output retain batch1.

    Raw natural-log LSE follows fixed-length semantics. Reusing layout avoids
    repeating the cu_seqlens host synchronization. Returned normalizer is
    [1,H,T]; internal statistics use the same head-major layout.
    fp32_output=True selects the separately compiled FP32 output instance.
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
        (q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau)+(() if gate_delta is None else (gate_delta,)),
        layout.table,0,save_state,fp32_output,_lse_preabsorbed)
    if save_state:
        state['layout']=layout
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
        layout.table,0,fp32_output,return_diagnostics)
    return (gradients,details) if return_diagnostics else gradients


class _VarlenCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau,layout,absorbed,gate_delta):
        out,_,state=forward_varlen(q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau,
                                   layout=layout,save_state=True,_lse_preabsorbed=absorbed,gate_delta=gate_delta)
        ctx.operand_count=len(state["operands"])
        ctx.layout=layout
        ctx.save_for_backward(*state['operands'],state['vertical'],out,state['normalizer'],layout.table)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,dout):
        saved=ctx.saved_tensors
        n=ctx.operand_count
        state=dict(layout=ctx.layout,operands=saved[:n],vertical=saved[n],
                   output=saved[n+1],normalizer=saved[n+2])
        g=backward_varlen(state,dout.to(torch.bfloat16))
        return (g['q_vec'],g['k_vec'],g['sq_vec'],g['sk_vec'],g['v'],g['q_lse'],g['k_lse'],
                None,None,None,None,g['rtau'],None,None,g.get('gate_delta'))


def dism_core_varlen(q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,
                     cu_seqlens=None,*,layout=None,_lse_preabsorbed=False,gate_delta=None):
    """Differentiable packed v3/v4 core; optional FP32 gate_delta is [1,H,T]."""
    if layout is None:
        if cu_seqlens is None:
            raise ValueError('provide cu_seqlens or layout')
        layout=VarlenLayout.from_cu_seqlens(cu_seqlens,q.shape[1])
    elif cu_seqlens is not None:
        raise ValueError('provide either layout or cu_seqlens, not both')
    return _VarlenCore.apply(q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,layout,_lse_preabsorbed,gate_delta)


def voc_dism_varlen(q,k,sq,sk,v,q_vocab,k_vocab,rtau,cu_seqlens=None,*,direction,hard,layout=None,gate_delta=None):
    """Existing Triton embedding interpolation followed by native varlen core.

    Vocabulary activation and sq/sk activation belong to the caller. Direction
    remains [1,H], not one independent direction per document. Hard flags are
    explicit [1,H,T], as in the current fixed-length v3 API.
    """
    from .emb_kernel import EmbInterpFunction
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
        out=dism_core_varlen(q,k,sq,sk,v,lse,lse,labels,labels,direction,hard,rtau,layout=layout,gate_delta=gate_delta)
        return out+(eq.float().sum()*0+ek.float().sum()*0).to(out.dtype)
    qfk,kfq,lk,lq,_,_,ik,iq=EmbInterpFunction.apply(
        query,key,eq,ek,1.,rtau.detach())
    qv,kv=cu.select_embedding(q,k,qfk,kfq,direction)
    return dism_core_varlen(qv,kv,sq,sk,v,lq.transpose(1,2),lk.transpose(1,2),iq,ik,
                           direction,hard,rtau,layout=layout,_lse_preabsorbed=True,gate_delta=gate_delta)
