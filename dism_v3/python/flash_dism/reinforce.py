"""Opt-in sampled categorical hard path for the DISM selector (REINFORCE).

This is not used by the production training/inference path. A caller wraps a
forward in ``collect()``; then every DISM attention samples its Q/K symbols
categorically at the current vocabulary temperature beta, records the sampled
log-probabilities (and entropies) for a score-function objective, and runs the
core with all-hard indices. The hard core ignores Q/K vector scores and LSE, so
q/k are detached there: the selector gradient flows only through the recorded
log-probabilities (and the soft pathwise branch, which the caller keeps).

``beta`` is the same vocabulary-softmax temperature as the soft branch, so the
sampled policy is the soft distribution softmax(beta * q @ vocab^T).
"""
from contextlib import contextmanager
import torch

_active = False
_logps = []          # list of (logp[1,H,N], entropy[1,H,N]) tensors


@contextmanager
def collect():
    global _active, _logps
    if _active:
        raise RuntimeError('reinforce.collect() is not reentrant')
    _active = True
    _logps = []
    try:
        yield _logps
    finally:
        _active = False
        _logps = []


def enabled():
    return _active


class Categorical(torch.autograd.Function):
    """Sample symbols and expose d logp / d entropy w.r.t. q and the vocab."""

    CHUNK = 1024

    @staticmethod
    def forward(ctx, x, vocab, beta):
        # x: [1,N,H,D] bf16, vocab: [H,V,D] fp32. logits = beta * x @ vocab^T.
        if x.ndim != 4 or x.shape[0] != 1 or vocab.ndim != 3:
            raise ValueError('Categorical expects x [1,N,H,D] and vocab [H,V,D]')
        _, n, h, d = x.shape
        v = vocab.shape[1]
        idx = torch.empty((1, h, n), device=x.device, dtype=torch.int32)
        lp = torch.empty((1, h, n), device=x.device, dtype=torch.float32)
        ent = torch.empty((1, h, n), device=x.device, dtype=torch.float32)
        with torch.autocast(x.device.type, enabled=False):
            e = vocab.float()
            for start in range(0, n, Categorical.CHUNK):
                end = min(start + Categorical.CHUNK, n)
                xx = x[0, start:end].transpose(0, 1).float()      # [H,c,D]
                logits = beta * (xx @ e.transpose(1, 2))          # [H,c,V]
                logp = logits.log_softmax(-1)
                p = logp.exp()
                ids = torch.multinomial(p.reshape(-1, v), 1).reshape(h, end - start)
                idx[0, :, start:end] = ids.to(torch.int32)
                lp[0, :, start:end] = logp.gather(-1, ids[..., None])[..., 0]
                ent[0, :, start:end] = -(p * logp).sum(-1)
        ctx.save_for_backward(x, vocab, idx, ent)
        ctx.beta = float(beta)
        ctx.mark_non_differentiable(idx)
        return idx, lp, ent

    @staticmethod
    def backward(ctx, _didx, dlp, dent):
        x, vocab, idx, ent = ctx.saved_tensors
        _, n, h, d = x.shape
        v = vocab.shape[1]
        beta = ctx.beta
        dq = torch.zeros_like(x, dtype=torch.float32)
        de = torch.zeros_like(vocab, dtype=torch.float32)
        with torch.autocast(x.device.type, enabled=False):
            e = vocab.float()
            for start in range(0, n, Categorical.CHUNK):
                end = min(start + Categorical.CHUNK, n)
                xx = x[0, start:end].transpose(0, 1).float()
                logits = beta * (xx @ e.transpose(1, 2))
                logp = logits.log_softmax(-1)
                p = logp.exp()
                ids = idx[0, :, start:end].long()[..., None]
                # d logp(z) / d logits = onehot(z) - p
                dl = -p
                dl = dl.scatter_add(-1, ids, torch.ones_like(ids, dtype=dl.dtype))
                dl = dl * dlp[0, :, start:end, None].float()
                # d H / d logits = -p (log p + H)
                if dent is not None:
                    H = ent[0, :, start:end, None]
                    dl = dl + (-p * (logp + H)) * dent[0, :, start:end, None].float()
                g = dl * beta
                dq[0, start:end] = (g @ e).transpose(0, 1)
                de += g.transpose(1, 2) @ xx
        return dq.to(x.dtype), de.to(vocab.dtype), None


def sampled_voc(q, k, sq, sk, v, eq, ek, tau, *, direction, hard, layout=None, beta=1.0):
    """Sample Q/K labels, run the all-hard core, and record logp/entropy."""
    from .varlen import dism_core_varlen
    from .backward import dism_core
    b = float(beta)
    iq, lpq, entq = Categorical.apply(q, eq, b)
    ik, lpk, entk = Categorical.apply(k, ek, b)
    _logps.append((lpq, entq))
    _logps.append((lpk, entk))
    zeros = torch.zeros((1, q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    hard_all = torch.ones_like(hard)
    qd, kd = q.detach(), k.detach()
    if layout is not None:
        return dism_core_varlen(qd, kd, sq, sk, v, zeros, zeros, iq, ik,
                                direction, hard_all, tau, layout=layout, _lse_preabsorbed=False)
    return dism_core(qd, kd, sq, sk, v, zeros, zeros, iq, ik,
                     direction, hard_all, tau, _lse_preabsorbed=False)
