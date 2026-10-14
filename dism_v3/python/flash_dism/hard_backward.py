"""Explicit fixed-length hard quadratic CUDA backward prototype (not default)."""
import torch
import cu_flash_dism as cu


@torch.no_grad()
def backward_hard(sq, sk, v, dout, idx_q, idx_k, hard, normalizer, state, *, ctas=0):
    """Return dSQ(FP32), dSK/dV(BF16), d(tau)(FP32), for sequence-hard rows.

    Requires fixed-length forward_hard(..., save_state=True) state from the
    same inputs. Packed documents are intentionally unsupported in this first
    prototype. Soft rows return zeros. No embedding/LSE/direction gradients.
    tau is the natural-log coefficient, before any outer softplus chain rule.
    Neither this function nor forward_hard registers an autograd path.
    """
    return cu.hard_backward((sq,sk,v,dout,idx_q,idx_k,hard),state['boundary'],
                            state['log_weights'],normalizer,ctas)


@torch.no_grad()
def backward_hard_recompute(sq, sk, v, dout, iq, ik, hard, tau, *, layout=None, ctas=0):
    """Fallback with no retained quadratic state or CPU/GPU metadata readback.

    Packed prototype buckets documents by length (no padding/cross-document
    attention). Packing, forward-state recomputation and scattering are included
    in fallback cost. Hard flags still belong to original training rows.
    """
    from .hard_forward import forward_hard
    def run(vectors, labels, flags):
        a,b,c,do = [x.contiguous() for x in vectors]
        q,k = [x.to(torch.int32).contiguous() for x in labels]
        _,norm,state = forward_hard(a,b,c,q,k,flags.contiguous(),tau.contiguous(),
                                     save_state=True,ctas=ctas)
        return backward_hard(a,b,c,do,q,k,flags.contiguous(),norm,state,ctas=ctas)
    if layout is None:
        result=run((sq,sk,v,dout),(iq,ik),hard)
    else:
        n=sq.shape[1]
        if sq.shape[0]!=1 or hard.ndim!=2 or n%hard.shape[0] or layout.tokens!=n:
            raise ValueError('invalid packed fallback rows')
        row_n=n//hard.shape[0]
        groups={}
        cursor=0
        for start,length,*_ in layout.table.tolist():
            if start!=cursor or length<0 or start%256 or length%256:
                raise ValueError('invalid aligned fallback layout')
            cursor+=length
            if length:
                if start//row_n!=(start+length-1)//row_n:
                    raise ValueError('document crosses original hard row')
                groups.setdefault(length,[]).append(start)
        if cursor!=n:
            raise ValueError('documents must cover packed input')
        result=[torch.zeros_like(x,dtype=torch.float32) for x in (sq,sk,v,tau)]
        for length,starts in groups.items():
            vectors=[torch.stack([x[0,s:s+length] for s in starts]) for x in (sq,sk,v,dout)]
            labels=[torch.stack([x[0,:,s:s+length] for s in starts]) for x in (iq,ik)]
            flags=torch.stack([hard[s//row_n] for s in starts])
            grads=run(vectors,labels,flags)
            for target,source in zip(result[:3],grads[:3]):
                for i,s in enumerate(starts): target[0,s:s+length].copy_(source[i])
            result[3].add_(grads[3])
    return dict(zip(('sq_vec','sk_vec','v','rtau'),result))
