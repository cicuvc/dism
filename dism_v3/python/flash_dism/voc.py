"""v3 training wrapper using the existing Triton embedding interpolation."""
import torch
import cu_flash_dism
from .backward import dism_core


def voc_dism(q,k,sq,sk,v,q_vocab,k_vocab,rtau,*,direction,hard,ctas=0):
    """BF16 [B,N,H,D] token inputs, already-activated [B,N,H,R] sq/sk.

    Vocabularies are [H,V,D] or shared [V,D], with caller-owned activation.
    R/D/DV select a compiled configuration automatically.
    Explicit direction [B,H] and hard [B,H,N] are replayed by forward/backward.
    No extra scale: like dism_ref, score is q_vec @ k_vec - LSE + rtau.
    """
    from .emb_kernel import EmbInterpFunction
    query,key,eq,ek=cu_flash_dism.prepare_embedding(
        q,k,sq,sk,v,q_vocab,k_vocab,rtau,direction,hard)
    q_from_k,k_from_q,k_lse,q_lse,_,_,idx_k,idx_q=EmbInterpFunction.apply(
        query,key,eq,ek,1.,rtau.detach())
    q_vec,k_vec=cu_flash_dism.select_embedding(q,k,q_from_k,k_from_q,direction)
    return dism_core(q_vec,k_vec,sq,sk,v,q_lse.transpose(1,2),k_lse.transpose(1,2),
                     idx_q,idx_k,direction,hard,rtau,ctas=ctas,_lse_preabsorbed=True)
