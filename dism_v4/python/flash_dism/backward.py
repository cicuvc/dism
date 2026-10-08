"""Shape-selected R/D/DV backward, with saved padded forward checkpoints.

Key-owned dV/dsk/dk/dk_lse default to BF16; cross-CTA atomic gradients remain FP32.
This low-level API is explicit replay, not a new production RNG/varlen interface.
"""
import torch
import cu_flash_dism
from .forward import forward_core


@torch.no_grad()
def backward_core(state,dout,*,ctas=0,fp32_output=False,return_diagnostics=False):
    """Native preparation/dispatch; BF16 key gradients, FP32 atomic gradients."""
    gradients,details=cu_flash_dism.core_backward(
        state['operands'],state['vertical'],state['output'],state['lse2'],dout,
        None,ctas,fp32_output,return_diagnostics)
    return (gradients,details) if return_diagnostics else gradients


class _Core(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau,ctas,absorbed,gate_delta):
        out,norm,state=forward_core(q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau,
                                    ctas=ctas,save_state=True,_lse_preabsorbed=absorbed,gate_delta=gate_delta)
        ctx.save_for_backward(*state['operands'],state['vertical'],out,norm)
        ctx.operand_count=len(state["operands"])
        ctx.n=q.shape[1]
        ctx.ctas=ctas
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,dout):
        saved=ctx.saved_tensors
        n=ctx.operand_count
        state=dict(operands=saved[:n],vertical=saved[n],output=saved[n+1],
                   lse2=saved[n+2],n=ctx.n)
        g=backward_core(state,dout.to(torch.bfloat16),ctas=ctx.ctas)
        return (g['q_vec'],g['k_vec'],g['sq_vec'],g['sk_vec'],g['v'],g['q_lse'],g['k_lse'],
                None,None,None,None,g['rtau'],None,None,g.get('gate_delta'))


def dism_core(q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,*,ctas=0,_lse_preabsorbed=False,gate_delta=None):
    """Differentiable v3/v4 core. q_lse/k_lse are raw natural-log statistics.

    Optional gate_delta is FP32 [B,H,N], nonnegative -log(g), with gradients.
    Zero recovers v3; +inf resets recurrence.
    Explicit hard/direction replay; sq/sk have already been activated. First
    derivatives only. FP32 atomic gradients cast to BF16 only at the upstream
    BF16 autograd input boundary, not during cross-CTA accumulation.
    """
    return _Core.apply(q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,ctas,_lse_preabsorbed,gate_delta)
