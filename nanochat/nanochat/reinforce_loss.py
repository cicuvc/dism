"""Warm-started score-function regularization for the DISM selector.

Opt-in helper for continued training. The soft pathwise loss stays the primary
objective; a small, advantage-standardized REINFORCE term corrects the residual
hard/argmax bias, optionally with an entropy floor that prevents the policy
collapse that killed the from-scratch runs.

Estimator (per optimizer step, one soft and one sampled-hard forward):

    A       = standardize_per_document( causal_return(hard_tokens - soft_tokens) )
    policy  = mean_symbol( log p(z) * A )
    loss    = soft_loss + lam * policy + hard_weight * hard_loss
              (+ eta * relu(entropy_floor - mean_symbol(H(pi)))   if eta > 0)

soft_tokens are the detached deterministic-soft per-token NLL (baseline);
hard_loss/tokens come from the sampled categorical hard path. Keeping the soft
pathwise term is the key robustness difference from the earlier failures, which
used no soft gradient at all.
"""
import torch


def causal_returns(delta, cu):
    """Per-token suffix sum of delta from each token to the end of its document."""
    flat = delta.detach().reshape(-1).double()
    positions = torch.arange(flat.numel(), device=flat.device)
    doc = torch.bucketize(positions, cu[1:].long(), right=True)
    prefix = torch.cat((flat.new_zeros(1), flat.cumsum(0)))
    return (prefix[cu[1:].long()][doc] - prefix[:-1]).float()


def standardize_per_document(adv, cu):
    out = torch.zeros_like(adv)
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        if end <= start:
            continue
        seg = adv[start:end]
        out[start:end] = (seg - seg.mean()) / (seg.std(unbiased=False) + 1e-6)
    return out


def reinforce_loss(model, x, y, cu, seg, *, lam=0.0, eta=0.0, entropy_floor=4.0,
                   hard_weight=0.0):
    from flash_dism.reinforce import collect
    counter = model.rng_counter.detach().clone()
    model.rng_counter.add_(1)
    # Primary: differentiable soft pathwise loss + detached soft per-token baseline.
    soft_loss, soft_tokens = model(x, y, cu_seqlens=cu, segment_ids=seg,
                                   hard_mode=False, paired_counter=counter,
                                   loss_reduction='reinforce')
    # Secondary: sampled categorical hard path with recorded log-probabilities.
    with collect() as scores:
        hard_loss, hard_tokens = model(x, y, cu_seqlens=cu, segment_ids=seg,
                                       hard_mode=True, paired_counter=counter,
                                       loss_reduction='reinforce')
    adv = standardize_per_document(causal_returns(hard_tokens - soft_tokens, cu), cu)
    nsym = max(1, sum(lp.numel() for lp, _ in scores))
    policy = sum((lp * adv[None, None, :]).sum() for lp, _ in scores) / nsym
    ent = sum(e.sum() for _, e in scores) / nsym
    loss = soft_loss + lam * policy + hard_weight * hard_loss
    if eta and entropy_floor > 0:
        loss = loss + eta * torch.relu(entropy_floor - ent)
    metrics = {
        "soft": float(soft_loss.detach()),
        "hard": float(hard_loss.detach()),
        "policy": float(policy.detach()),
        "entropy": float(ent.detach()),
        "advantage_rms": float(adv.square().mean().sqrt()),
        "n_symbols": nsym,
    }
    return loss, metrics
