"""Aligned v3/v4 forward; validation, preparation and dispatch live in C++."""
import cu_flash_dism as cu


def forward_core(q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau,
                 *,ctas=0,save_state=False,fp32_output=False,_lse_preabsorbed=False,gate_delta=None):
    """BF16 operands [B,N,H,C], N>0 divisible by256; no automatic padding.

    LSE is raw natural-log [B,N,H]. SQ/SK are already activated.
    Optional gate_delta is nonnegative natural-log attenuation, FP32 [B,H,N].
    None or zero gives v3; +inf resets the diagonal recurrence.
    Returns O, log2-normalizer, and optionally saved backward state.
    """
    out,norm,state=cu.core_forward(
        (q,k,sq,sk,v,q_lse,k_lse,idx_q,idx_k,direction,hard,rtau)+(() if gate_delta is None else (gate_delta,)),
        None,ctas,save_state,fp32_output,_lse_preabsorbed)
    return (out,norm,state) if save_state else (out,norm)
