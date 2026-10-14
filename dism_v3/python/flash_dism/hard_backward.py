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
