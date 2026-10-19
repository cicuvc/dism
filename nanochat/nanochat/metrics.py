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
        # Capture a scalar view, not a Python layer index: an index specializes
        # the compiled block once per layer and exhausts Dynamo's cache limit.
        def hook(module, args, slot=buffer[index]):
            residual = args[0]
            if (residual.is_cuda and residual.is_contiguous() and residual.numel()
                    and residual.shape[-1] <= 4096
                    and residual.numel() // residual.shape[-1] <= 1048576):
                from nanochat.metrics_kernels import residual_rms_max
                slot.copy_(residual_rms_max(residual.detach()))
            else:
                slot.copy_(residual.detach().float().pow(2).mean(-1).sqrt().max())
        handles.append(layer.mlp_norm.register_forward_pre_hook(hook))
    return handles, buffer


def remove_probe(handles):
    for handle in handles:
        handle.remove()


# -----------------------------------------------------------------------------
# Optimizer update RMS (one max value per parameter group / layer)

def snapshot_parameters(model):
    """Clone trainable parameters before an optimizer step."""
    params = {
        name: param.detach()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    if not params:
        return {}
    # Multi-tensor copy batches CUDA launches, preserving independent storage
    # and named_parameters' default deduplication of tied parameters.
    return dict(zip(params, torch._foreach_clone(list(params.values()))))


def _group_key(name):
    parts = name.split(".")
    if parts[0] == "layers" and len(parts) > 1 and parts[1].isdigit():
        return "layer" + parts[1].zfill(2)
    return parts[0]


def update_rms_max_per_group(model, snapshot):
    """Max per-parameter RMS of the applied update, grouped by layer/module."""
    grouped = {}
    pairs = [(name, param.detach(), snapshot[name])
             for name, param in model.named_parameters() if name in snapshot]
    if pairs and all(p.is_cuda and p.dtype == torch.float32 and
                     before.dtype == p.dtype and before.device == p.device
                     for _, p, before in pairs):
        # Keep subtraction in parameter dtype. The FP32 production path can
        # use multi-tensor L2 reductions without per-parameter cast launches.
        deltas = torch._foreach_sub([p for _, p, _ in pairs],
                                   [b for _, _, b in pairs])
        norms = torch._foreach_norm(deltas, 2)
        rms_values = torch._foreach_div(norms, [p.numel() ** 0.5 for _, p, _ in pairs])
        for (name, _, _), rms in zip(pairs, rms_values):
            grouped.setdefault(_group_key(name), []).append(rms)
    else:
        for name, param, before in pairs:
            rms = (param - before).float().pow(2).mean().sqrt()
            grouped.setdefault(_group_key(name), []).append(rms)
    if not grouped:
        return {}
    # Exactly one device-to-host readback rather than one synchronization per
    # layer. Group maxima are over parameter RMS, not a pooled group norm.
    by_device = {}
    for key, group in grouped.items():
        value = torch.stack(group).max()
        by_device.setdefault(value.device, []).append((key, value))
    result = {}
    for items in by_device.values():
        values = torch.stack([v for _, v in items]).tolist()
        result.update(zip([k for k, _ in items], values))
    return {k: result[k] for k in grouped}


# -----------------------------------------------------------------------------
# DISM q/k codebook usage

@torch.no_grad()
def probe_qk_vocab_usage(model, input_ids, cu_seqlens, max_seqlen, chunk_size=4096):
    """Measure per-layer q/k codebook selection on one packed batch.

    Runs an eager eval-mode forward through the embedding and blocks only (the
    fused LM head is skipped). Selection for the ``q`` codebook is
    ``argmax(softmax(q @ q_vocab.T))``; likewise for ``k``. Enabled QB instead
    uses the current beta and frozen side-specific prices. Returns
    ``{layer_index: {"q": stats, "k": stats}}`` with ``counts`` a CPU list of
    vocab-size selection frequencies plus entropy (normalized), max_share and
    coverage. Traversal follows the all-hard model evaluation path, including
    the first GDN value reused by later value-residual layers. Hard flags refer
    to the original packed rows, not tokens or document boundaries.
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
            conv_seq_idx = None
            if config.conv_impl == 'linear_fla' and config.conv_backend == 'cuda':
                from nanochat.models.dism.conv import packed_conv_sequence_ids
                conv_seq_idx = packed_conv_sequence_ids(cu_seqlens, hidden.shape[1], device)
            v_first = None
            value_residual = getattr(config, "value_residual", False)
            for index, layer in enumerate(model.layers):
                attn = layer.attn
                if getattr(attn, "is_pure_mixer", False):
                    if value_residual and v_first is None:
                        from nanochat.models.dism.model import _mixer_value_first
                        v_first = _mixer_value_first(
                            attn, hidden, cu_seqlens, max_seqlen, conv_seq_idx)
                        hidden = layer(hidden, cu_seqlens=cu_seqlens,
                                       max_seqlen=max_seqlen, use_cache=False, conv_seq_idx=conv_seq_idx)
                    else:
                        hidden = layer(hidden, cu_seqlens=cu_seqlens,
                                       max_seqlen=max_seqlen, use_cache=False,
                                       v_first=v_first if value_residual else None, conv_seq_idx=conv_seq_idx)
                    continue
                length = hidden.shape[1]
                conv_options = {'seq_idx': conv_seq_idx} if getattr(attn.q_conv, 'supports_seq_idx', False) else {}
                q = attn.q_conv(hidden, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **conv_options)
                k = attn.k_conv(hidden, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **conv_options)
                # kv_group_size>1 gives the key side fewer heads than the query
                # side, and fewer codebook rows, so the two axes differ.
                kv_heads = int(getattr(attn, "kv_heads", heads))
                q = q.reshape(1, length, heads, attn.head_dim).float()
                k = k.reshape(1, length, kv_heads, attn.head_dim).float()
                if getattr(attn, "vocab_transvq", False):
                    # Effective codebook is C' = P_phi(C), not the frozen base.
                    with torch.autocast("cuda", enabled=False):
                        table_q = attn.q_vocab_map(attn.q_vocab.float())
                        table_k = attn.k_vocab_map(attn.k_vocab.float())
                else:
                    table_q, table_k = attn.q_vocab, attn.k_vocab
                stats = {}
                pq_sizes = tuple(getattr(attn, 'pq_vocab_sizes', ()))
                for tag, vec, table, rows in (("q", q, table_q, heads),
                                              ("k", k, table_k, kv_heads)):
                    # Per-head counts: q_vocab/k_vocab are [heads, V, D] with independent
                    # per-head codebooks unless vocab_share_heads, so entries must not
                    # be pooled across heads.
                    counts = torch.zeros((rows, vocab_size), device=device, dtype=torch.float64)
                    entropy_sum = 0.0
                    max_sum = 0.0
                    for start in range(0, length, chunk_size):
                        stop = min(start + chunk_size, length)
                        qb = getattr(attn, 'qb', None)
                        if pq_sizes:
                            from flash_dism.pq_vocab import strides
                            group_dim = attn.head_dim // len(pq_sizes)
                            code = torch.arange(vocab_size, device=device)
                            full_logp = None
                            idx_flat = torch.zeros(
                                (vec.shape[0], stop-start, rows),
                                device=device, dtype=torch.long)
                            for group, (size, step) in enumerate(zip(
                                    pq_sizes, strides(pq_sizes))):
                                scores = model.vocab_temperature * torch.einsum(
                                    "bnhd,hvd->bnhv",
                                    vec[:, start:stop, :, group*group_dim:(group+1)*group_dim],
                                    table[:, group, :size].float())
                                logp = torch.log_softmax(scores, dim=-1)
                                positions = (code // step) % size
                                contribution = logp[..., positions]
                                full_logp = (contribution if full_logp is None
                                             else full_logp + contribution)
                                idx_flat += logp.argmax(-1) * step
                            probs = full_logp.exp().reshape(-1, vocab_size)
                            idx = idx_flat.reshape(-1, rows)
                        else:
                            scores = torch.einsum("bnhd,hvd->bnhv", vec[:, start:stop], table.float())
                            if qb is not None:
                                price = qb.q_price if tag == 'q' else qb.k_price[::attn.kv_group_size]
                                strength = qb.q_strength if tag == 'q' else qb.k_strength
                                with torch.autocast('cuda', enabled=False):
                                    scores = model.vocab_temperature * torch.einsum(
                                        'bnhd,hvd->bnhv', vec[:, start:stop].float(),
                                        table.to(torch.bfloat16).float()) - strength*price
                            probs = torch.softmax(scores, dim=-1).reshape(-1, vocab_size)
                            idx = probs.argmax(-1).reshape(-1, rows)
                        for head in range(rows):
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
                    # Usage statistics of the *hard* (argmax) assignment, which is
                    # what a capacity target actually constrains. max_share above
                    # is the soft probability; max_load is the count share, and
                    # capacity_residual is the worst per-code deviation from the
                    # uniform capacity 1/V.
                    share = counts / counts.sum(1, keepdim=True).clamp_min(1.0)
                    usage_h = -(share * (share + 1e-12).log()).sum(1)
                    norm_log = torch.log(torch.tensor(float(vocab_size),
                                                      device=counts.device, dtype=counts.dtype))
                    stats[tag]["active_codes"] = float(used.float().sum(1).mean())
                    stats[tag]["usage_entropy"] = float((usage_h / norm_log).mean())
                    stats[tag]["effective_symbols"] = float(usage_h.exp().mean())
                    stats[tag]["max_load"] = float(share.max(1).values.mean())
                    stats[tag]["capacity_residual"] = float(
                        (share - 1.0 / vocab_size).abs().max(1).values.mean())
                    if qb is not None:
                        price = qb.q_price if tag == 'q' else qb.k_price[::attn.kv_group_size]
                        stats[tag]["price_abs_mean"] = float(price.abs().mean())
                        stats[tag]["price_std"] = float(price.std())
                results[index] = stats
                hard = torch.ones((input_ids.shape[0], heads), dtype=torch.bool, device=device)
                direction = torch.ones((1, heads), dtype=torch.bool, device=device)
                hidden = layer(hidden, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                               hard=hard, direction=direction, use_cache=False,
                               v_first=v_first if value_residual else None,
                               vocab_temperature=model.vocab_temperature, conv_seq_idx=conv_seq_idx,
                               direction_true=getattr(attn, 'qb', None) is not None)
    finally:
        model.train(was_training)
    return results
