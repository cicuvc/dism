"""Optional training diagnostics for DISM nanochat runs.

These helpers only measure; they do not alter forward/backward math, optimizer
state or RNG counters.

- residual RMS: pre-hook on each block's final norm captures the residual stream
  (the pre-norm sum), which post-norm keeps unnormalized. One max-over-tokens
  value per layer.
- optimizer update RMS: per parameter RMS of (param_after - param_before), then
  one max per parameter group / layer.
- DISM q/k codebook usage: eval-mode eager forward through the blocks only,
  argmax/entropy of the softmax selection over each codebook.
"""

import torch


# -----------------------------------------------------------------------------
# Residual-stream RMS (one max value per layer)

def register_residual_rms_probe(model):
    """Attach pre-hooks to every block's ``mlp_norm``.

    Returns ``(handles, buffer)`` with ``buffer`` a CUDA tensor of shape
    ``[n_layer]``; ``buffer[i]`` is the max-over-tokens RMS of the residual
    stream entering layer ``i``'s final norm.
    """
    layers = model.layers
    device = next(model.parameters()).device
    buffer = torch.zeros(len(layers), device=device, dtype=torch.float32)
    handles = []
    for index, layer in enumerate(layers):
        def hook(module, args, index=index):
            residual = args[0]
            buffer[index] = residual.detach().float().pow(2).mean(-1).sqrt().max()
        handles.append(layer.mlp_norm.register_forward_pre_hook(hook))
    return handles, buffer


def remove_probe(handles):
    for handle in handles:
        handle.remove()


# -----------------------------------------------------------------------------
# Optimizer update RMS (one max value per parameter group / layer)

def snapshot_parameters(model):
    """Clone trainable parameters before an optimizer step."""
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _group_key(name):
    parts = name.split(".")
    if parts[0] == "layers" and len(parts) > 1 and parts[1].isdigit():
        return "layer" + parts[1].zfill(2)
    return parts[0]


def update_rms_max_per_group(model, snapshot):
    """Max per-parameter RMS of the applied update, grouped by layer/module."""
    grouped = {}
    for name, param in model.named_parameters():
        before = snapshot.get(name)
        if before is None:
            continue
        rms = (param.detach() - before).float().pow(2).mean().sqrt()
        grouped.setdefault(_group_key(name), []).append(rms)
    return {key: torch.stack(values).max().item() for key, values in grouped.items()}


# -----------------------------------------------------------------------------
# DISM q/k codebook usage

@torch.no_grad()
def probe_qk_vocab_usage(model, input_ids, cu_seqlens, max_seqlen, chunk_size=4096):
    """Measure per-layer q/k codebook selection on one packed batch.

    Runs an eager eval-mode forward through the embedding and blocks only (the
    fused LM head is skipped). Selection for the ``q`` codebook is
    ``argmax(softmax(q @ q_vocab.T))``; likewise for ``k``. Returns
    ``{layer_index: {"q": stats, "k": stats}}`` with ``counts`` a CPU list of
    vocab-size selection frequencies plus entropy (normalized), max_share and
    coverage.
    """
    if not hasattr(model, "layers"):
        return {}
    config = model.config
    device = input_ids.device
    heads = config.n_head
    vocab_size = config.qk_vocab_size
    was_training = model.training
    results = {}
    model.eval()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.embedding(input_ids.reshape(1, -1))
            for index, layer in enumerate(model.layers):
                attn = layer.attn
                if getattr(attn, "is_pure_gdn", False):
                    hidden = layer(hidden, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, use_cache=False)
                    continue
                length = hidden.shape[1]
                q = attn.q_conv(hidden, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                k = attn.k_conv(hidden, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                q = q.reshape(1, length, heads, attn.head_dim).float()
                k = k.reshape(1, length, heads, attn.head_dim).float()
                stats = {}
                for tag, vec, table in (("q", q, attn.q_vocab), ("k", k, attn.k_vocab)):
                    # Per-head counts: q_vocab/k_vocab are [heads, V, D] with independent
                    # per-head codebooks unless vocab_share_heads, so entries must not
                    # be pooled across heads.
                    counts = torch.zeros((heads, vocab_size), device=device, dtype=torch.float64)
                    entropy_sum = 0.0
                    max_sum = 0.0
                    for start in range(0, length, chunk_size):
                        stop = min(start + chunk_size, length)
                        scores = torch.einsum("bnhd,hvd->bnhv", vec[:, start:stop], table.float())
                        probs = torch.softmax(scores, dim=-1).reshape(-1, vocab_size)
                        idx = probs.argmax(-1).reshape(-1, heads)
                        for head in range(heads):
                            counts[head] += torch.bincount(idx[:, head], minlength=vocab_size).double()
                        entropy_sum += float(-(probs * (probs + 1e-12).log()).sum(-1).mean()) * (stop - start)
                        max_sum += float(probs.max(-1).values.mean()) * (stop - start)
                    used = counts > 0
                    stats[tag] = {
                        # Average per-head selection-frequency distribution [V].
                        "counts": counts.mean(0).cpu().tolist(),
                        # Utilization of the actual codebook parameters: fraction of the
                        # heads*V rows that are the argmax at least once.
                        "coverage": float(used.float().mean()),
                        # Per-head coverage, and the head-pooled variant (meaningful only
                        # when the codebook is shared across heads).
                        "coverage_per_head": used.float().mean(1).cpu().tolist(),
                        "coverage_pooled": float((counts.sum(0) > 0).float().mean()),
                        "entropy": entropy_sum / length / torch.log(torch.tensor(float(vocab_size))).item(),
                        "max_share": max_sum / length,
                    }
                results[index] = stats
                hard = torch.ones((1, heads, length), dtype=torch.bool, device=device)
                direction = torch.ones((1, heads), dtype=torch.bool, device=device)
                hidden = layer(hidden, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                               hard=hard, direction=direction, use_cache=False)
    finally:
        model.train(was_training)
    return results
