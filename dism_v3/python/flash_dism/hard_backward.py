"""Explicit fixed-length hard quadratic CUDA backward prototype (not default)."""
import torch
import cu_flash_dism as cu


@torch.no_grad()
def backward_hard(sq, sk, v, dout, idx_q, idx_k, hard, normalizer, state, *,
                  layout=None, ctas=0, _graph_choice=None):
    """Return dSQ(FP32), dSK/dV(BF16), d(tau)(FP32), for sequence-hard rows.

    Requires forward_hard(..., save_state=True) state from the same inputs.
    Optional layout enables document-task execution without stacking packed
    slices. Soft rows return zeros. No embedding/LSE/direction gradients.
    tau is the natural-log coefficient, before any outer softplus chain rule.
    Neither this function nor forward_hard registers an autograd path.
    """
    return cu.hard_backward(
        (sq,sk,v,dout,idx_q,idx_k,hard),
        state['boundary'],
        state['log_weights'],
        normalizer,
        ctas,
        None if layout is None else layout.table,
        _graph_choice,
    )


@torch.no_grad()
def backward_hard_recompute(sq, sk, v, dout, iq, ik, hard, tau, *, layout=None, ctas=0, _graph_choice=None):
    """Fallback with no retained quadratic state or CPU/GPU metadata readback.

    Packed execution consumes the original tensors and document task table
    directly. Forward-state recomputation is included in fallback cost. Hard
    flags still belong to original training rows.
    """
    from .hard_forward import forward_hard
    n=sq.shape[1]
    if layout is not None:
        if sq.shape[0]!=1 or hard.ndim!=2 or n%hard.shape[0] or layout.tokens!=n:
            raise ValueError('invalid packed fallback rows')
        row_n=n//hard.shape[0]
        if any(
            start//row_n!=(start+length-1)//row_n
            for start,length,*_ in layout.table.tolist()
            if length
        ):
            raise ValueError('document crosses original hard row')
    vectors,labels=((sq,sk,v,dout),(iq,ik))
    a,b,c,do = [x.contiguous() for x in vectors]
    q,k = [x.to(torch.int32).contiguous() for x in labels]
    _,norm,state = forward_hard(
        a,b,c,q,k,hard.contiguous(),tau.contiguous(),
        layout=layout,ctas=ctas,save_state=True,_graph_choice=_graph_choice,
    )
    result=backward_hard(
        a,b,c,do,q,k,hard.contiguous(),norm,state,
        layout=layout,ctas=ctas,_graph_choice=_graph_choice,
    )
    return dict(zip(('sq_vec','sk_vec','v','rtau'),result))
