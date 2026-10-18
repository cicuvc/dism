"""Opt-in fixed-true QB: delayed global prices and consistent soft/hard selectors.

Prices are logit offsets (including beta's units), not parameters. Updates use
all ranks' sampled scores from a completed forward; eval never updates them.
The chunked interpolation recomputes probabilities in backward to bound memory.
"""
import math
import torch
from torch import nn
import torch.distributed as dist


def validate_qb(q_strength, k_strength, iterations, interval, samples, rate):
    if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in (q_strength, k_strength)):
        raise ValueError('qb_q_strength and qb_k_strength must be finite and in [0,1]')
    if any(type(x) is not int or x <= 0 for x in (iterations, interval, samples)):
        raise ValueError('QB iterations, update interval and samples must be positive integers')
    if not math.isfinite(float(rate)) or not 0 < float(rate) <= 1:
        raise ValueError('qb_update_rate must be in (0,1]')


@torch.no_grad()
def fit_prices(scores, initial, iterations):
    """Batched hinge-dual coordinate minimizers; finite iterations need not balance.

    scores [H,N,V], initial [H,V]. Fractional capacity N/V is supported.
    No gradient or RNG is consumed, including in tied/degenerate cases.
    """
    h, n, v = scores.shape
    if n < 1 or v < 2:
        raise ValueError('QB needs at least one sample and two codes')
    price = initial.clone()
    price -= price.mean(-1, keepdim=True)
    capacity = n / v
    rank = math.ceil(capacity)
    best = price.clone()
    best_dual = (scores-price[:, None]).amax(-1).mean(-1)+price.mean(-1)
    for _ in range(iterations):
        alpha = (scores-price[:, None]).topk(2, dim=-1).values.mean(-1)
        columns = (scores-alpha[..., None]).topk(min(rank+1, n), dim=1).values
        price = columns[:, rank-1]
        if n % v == 0:
            price = (price+columns[:, rank])*.5
        price = price-price.mean(-1, keepdim=True)
        dual = (scores-price[:, None]).amax(-1).mean(-1)+price.mean(-1)
        better = dual < best_dual
        best = torch.where(better[:, None], price, best)
        best_dual = torch.minimum(dual, best_dual)
    return best


def _work(x):
    return x.double() if x.dtype == torch.float64 else x.float()


class PricedInterpolation(torch.autograd.Function):
    """True-direction interpolation, Q log-normalizer and E_K[Q price]."""
    # Tokens per step of the eager chunk loop. Larger chunks cut the Python
    # iteration count (n/CHUNK per layer per microbatch) at the cost of holding
    # bigger [b,h,chunk,V] fp32 intermediates.
    import os as _os
    CHUNK = int(_os.environ.get('DISM_QB_CHUNK', '4096'))

    @staticmethod
    def forward(ctx, q, k, eq, ek, aq, ak, beta):
        b, n, h, d = q.shape
        eq, ek = eq.contiguous(), ek.contiguous()
        out = torch.empty_like(q)
        lq = torch.empty((b, n, h), device=q.device, dtype=torch.float64 if q.dtype == torch.float64 else torch.float32)
        mean_price = torch.empty_like(lq)
        iq = torch.empty((b, h, n), device=q.device, dtype=torch.int32)
        ik = torch.empty_like(iq)
        with torch.autocast(q.device.type, enabled=False):
            qe, ke = _work(eq), _work(ek)
            for start in range(0, n, PricedInterpolation.CHUNK):
                end = min(start+PricedInterpolation.CHUNK, n)
                qs, ks = _work(q[:, start:end]).transpose(1, 2), _work(k[:, start:end]).transpose(1, 2)
                zq = beta*(qs @ qe.transpose(-1, -2))-aq[:, None, :]
                zk = beta*(ks @ ke.transpose(-1, -2))-ak[:, None, :]
                pk = zk.softmax(-1)
                out[:, start:end] = (pk @ qe).transpose(1, 2)
                lq[:, start:end] = zq.logsumexp(-1).transpose(1, 2)
                mean_price[:, start:end] = (pk*aq[:, None, :]).sum(-1).transpose(1, 2)
                iq[:, :, start:end] = zq.argmax(-1)
                ik[:, :, start:end] = zk.argmax(-1)
        ctx.save_for_backward(q, k, eq, ek, aq, ak)
        ctx.beta = beta
        ctx.mark_non_differentiable(iq, ik)
        return out, lq, mean_price, iq, ik

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do, dlq, dm, _diq, _dik):
        q, k, eq, ek, aq, ak = ctx.saved_tensors
        beta = ctx.beta
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        deq, dek = torch.zeros_like(_work(eq)), torch.zeros_like(_work(ek))
        with torch.autocast(q.device.type, enabled=False):
            qe, ke = _work(eq), _work(ek)
            for start in range(0, q.shape[1], PricedInterpolation.CHUNK):
                end = min(start+PricedInterpolation.CHUNK, q.shape[1])
                qs, ks = _work(q[:, start:end]).transpose(1, 2), _work(k[:, start:end]).transpose(1, 2)
                pq = (beta*(qs @ qe.transpose(-1, -2))-aq[:, None, :]).softmax(-1)
                pk = (beta*(ks @ ke.transpose(-1, -2))-ak[:, None, :]).softmax(-1)
                go = _work(do[:, start:end]).transpose(1, 2)
                gp = go @ qe.transpose(-1, -2) + dm[:, start:end].transpose(1, 2)[..., None]*aq[:, None, :]
                gk = pk*(gp-(pk*gp).sum(-1, keepdim=True))
                gq = pq*dlq[:, start:end].transpose(1, 2)[..., None]
                dq[:, start:end] = (beta*(gq @ qe)).transpose(1, 2)
                dk[:, start:end] = (beta*(gk @ ke)).transpose(1, 2)
                deq += (pk.transpose(-1, -2) @ go + beta*(gq.transpose(-1, -2) @ qs)).sum(0)
                dek += (beta*(gk.transpose(-1, -2) @ ks)).sum(0)
        return dq, dk, deq.to(eq.dtype), dek.to(ek.dtype), None, None, None


class QuantileBalance(nn.Module):
    def __init__(self, heads, vocab, *, q_strength=0., k_strength=0., iterations=4,
                 update_interval=16, samples=2048, update_rate=.1):
        super().__init__()
        validate_qb(q_strength, k_strength, iterations, update_interval, samples, update_rate)
        if vocab < 2:
            raise ValueError('QB requires at least two codes')
        self.q_strength, self.k_strength = float(q_strength), float(k_strength)
        self.iterations, self.update_interval = iterations, update_interval
        self.samples, self.update_rate = samples, float(update_rate)
        # Host-only evaluation knob. Training always uses the stored
        # strengths; inference may attenuate both selector biases equally.
        self.eval_factor = 1.0
        self.register_buffer('q_price', torch.zeros(heads, vocab, dtype=torch.float32))
        self.register_buffer('k_price', torch.zeros(heads, vocab, dtype=torch.float32))
        self.register_buffer('forward_count', torch.zeros((), dtype=torch.int64))
        self.register_buffer('update_count', torch.zeros((), dtype=torch.int64))

    def set_eval_factor(self, factor):
        """Scale both Q and K price biases during evaluation only."""
        factor = float(factor)
        if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
            raise ValueError('QB eval factor must be finite and in [0,1]')
        self.eval_factor = factor

    def active_strengths(self):
        """Return the strengths used by the current training/eval forward."""
        factor = 1.0 if self.training else self.eval_factor
        return self.q_strength*factor, self.k_strength*factor

    @torch.no_grad()
    def reset_parameters(self):
        for buffer in self.buffers():
            buffer.zero_()

    @torch.no_grad()
    def _update(self, q, k, eq, ek, beta, valid_mask=None):
        # Global-rank concatenation, not averaging locally fitted prices.
        # Fixed-size per-rank samples keep collective order/shapes deterministic.
        n = q.shape[0]*q.shape[1]
        eligible = (torch.arange(n, device=q.device) if valid_mask is None
                    else valid_mask.reshape(-1).nonzero().flatten())
        count = torch.tensor([eligible.numel()], device=q.device, dtype=torch.int64)
        distributed = dist.is_available() and dist.is_initialized()
        counts = [count]
        if distributed:
            counts = [torch.empty_like(count) for _ in range(dist.get_world_size())]
            dist.all_gather(counts, count)
        nonempty = [int(c) > 0 for c in counts]
        if not any(nonempty):
            return
        ids = (eligible[torch.linspace(0, eligible.numel()-1, self.samples, device=q.device).long()]
               if eligible.numel() else torch.zeros(self.samples, device=q.device, dtype=torch.long))
        for x, e, price, strength in ((q, eq, self.q_price, self.q_strength),
                                      (k, ek, self.k_price, self.k_strength)):
            if not strength:
                continue
            xx = x.reshape(n, x.shape[2], x.shape[3])[ids].transpose(0, 1).float()
            scores = beta*(xx @ e.float().transpose(-1, -2))
            if distributed:
                gathered = [torch.empty_like(scores) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, scores.contiguous())
                scores = torch.cat([s for s, keep in zip(gathered, nonempty) if keep], dim=1)
            fitted = fit_prices(scores, price, self.iterations)
            price.lerp_(fitted, self.update_rate)
            price.sub_(price.mean(-1, keepdim=True))
        self.update_count.add_(1)

    @torch.compiler.disable
    def forward(self, q, k, sq, sk, v, eq, ek, tau, beta, direction, hard,
                boundaries=None, layout=None, *, direction_true=False, valid_mask=None):
        if not direction_true or not bool(direction.all()):
            raise ValueError('QB only supports explicit fixed direction=true')
        if q.shape[1] == 0:
            raise ValueError('QB does not support empty batches')
        from .compiler import core_forward, _table
        h = q.shape[2]
        eq = eq.to(torch.bfloat16).expand(h, -1, -1)
        ek = ek.to(torch.bfloat16).expand(h, -1, -1)
        # Own immutable snapshots: updating buffers below cannot change backward.
        q_strength, k_strength = self.active_strengths()
        aq = self.q_price.detach().clone()*q_strength
        ak = self.k_price.detach().clone()*k_strength
        with torch.autocast(q.device.type, enabled=False):
            out, lq, m, iq, ik = PricedInterpolation.apply(q, k, eq, ek, aq, ak, float(beta))
            table = layout.view(-1, 7) if layout is not None else _table(boundaries, q.shape[1])
            result = core_forward((q, (out*beta).to(q.dtype), sq, sk, v,
                                   lq-tau.detach(), m, iq, ik, direction, hard, tau),
                                  table=table, absorbed=True, qb_key_bias=True)
            if self.training and torch.is_grad_enabled():
                self.forward_count.add_(1)
                if int(self.forward_count) % self.update_interval == 0:
                    self._update(q.detach(), k.detach(), eq.detach(), ek.detach(), float(beta), valid_mask)
        return result
