"""Training forward for product-quantised DISM vocabularies."""

import torch

from .pq_vocab import factorized_interp


@torch.compiler.disable
def pq_voc_forward(q, k, sq, sk, v, q_vocab, k_vocab, tau, beta, direction,
                   hard, boundaries=None, layout=None, *, sizes):
    """Run grouped interpolation and feed the unchanged DISM core.

    ``factorized_interp`` returns already beta-scaled interpolants and natural
    log-sum-exp values. The core receives those LSEs pre-absorbed, exactly like
    the dense vocabulary path.
    """
    import cu_flash_dism
    from .compiler import _table, core_forward

    qfk, kfq, k_lse, q_lse, idx_k, idx_q = factorized_interp(
        q, k, q_vocab, k_vocab, beta, tau.detach(), sizes=sizes)
    q_vec, k_vec = cu_flash_dism.select_embedding(
        q, k, qfk, kfq, direction)
    table = layout.view(-1, 7) if layout is not None else _table(boundaries, q.shape[1])
    return core_forward(
        (q_vec, k_vec, sq, sk, v,
         q_lse.transpose(1, 2), k_lse.transpose(1, 2),
         idx_q, idx_k, direction, hard, tau),
        table=table, absorbed=True)
