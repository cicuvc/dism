"""Aligned v3 forward; validation, preparation and dispatch live in C++."""
import cu_flash_dism as cu


def forward_core(q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,
                 *,ctas=0,save_state=False,fp32_output=False,_lse_preabsorbed=False):
    """BF16 operands [B,N,H,C], N>0 divisible by256; no automatic padding.

    LSE is raw natural-log [B,N,H]. SQ/SK are already activated.
    Returns O, log2-normalizer, and optionally saved backward state.
    """
    out,norm,state=cu.core_forward(
        (q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau),
        None,ctas,save_state,fp32_output,_lse_preabsorbed)
    return (out,norm,state) if save_state else (out,norm)
