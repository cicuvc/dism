"""v3 training wrapper using the existing Triton embedding interpolation."""
import torch
import cu_flash_dism
from .backward import dism_core


def voc_dism(q,k,sq,sk,v,q_vocab,k_vocab,rtau,*,direction,hard,ctas=0,beta=1.0,hard_planner=None):
    """BF16 [B,N,H,D] token inputs, already-activated [B,N,H,R] sq/sk.

    Vocabularies are [H,V,D] or shared [V,D], with caller-owned activation.
    R/D/DV select a compiled configuration automatically.
    Explicit direction [B,H] and hard [B,H] are replayed by forward/backward.
    Without hard_planner, CUDA computes only soft rows (hard zero placeholders).
    An explicit AsyncHardPlanner enables complete eager hard/soft training.
    beta is the vocabulary-softmax temperature: beta=1 reproduces the unscaled
    ``q_vec @ k_vec - LSE + rtau``; larger beta sharpens the soft match gate
    toward the hard argmax indicator (see docs/SOFT_HARD_RELAXATION.md).
    """
    from .emb_kernel import interp_embeddings
    query,key,eq,ek=cu_flash_dism.prepare_embedding(
        q,k,sq,sk,v,q_vocab,k_vocab,rtau,direction,hard)
    q_from_k,k_from_q,k_lse,q_lse,idx_k,idx_q=interp_embeddings(
        query,key,eq,ek,beta,rtau.detach())
    q_vec,k_vec=cu_flash_dism.select_embedding(q,k,q_from_k,k_from_q,direction)
    if hard_planner is not None:
        from .hard_training import dism_core_hybrid
        return dism_core_hybrid(q_vec,k_vec,sq,sk,v,q_lse.transpose(1,2),k_lse.transpose(1,2),
                                idx_q,idx_k,direction,hard,rtau,planner=hard_planner,
                                ctas=ctas,_lse_preabsorbed=True)
    return dism_core(q_vec,k_vec,sq,sk,v,q_lse.transpose(1,2),k_lse.transpose(1,2),
                     idx_q,idx_k,direction,hard,rtau,ctas=ctas,_lse_preabsorbed=True)
