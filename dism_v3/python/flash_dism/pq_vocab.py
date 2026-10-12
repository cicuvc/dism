"""Product-quantised (factorised) DISM codebook.

Replaces one V-way vocabulary softmax over D dims with ``groups`` independent
S-way softmaxes over D/groups dims each, where V = S**groups. The identities

    q . E_v            = sum_g q_g . E^g_{v_g}
    logsumexp_v(...)   = sum_g logsumexp_{v_g}(...)

mean that the interpolated embedding is just the concatenation of the per-group
blocks and the match LSE is the *sum* of the per-group LSEs. The DISM core
computes ``q_vec . k_vec - LSE``, and the dot product already sums over
concatenated dims, so **the core and its kernels are reused unchanged**.

The hard label is the bijection ``v = sum_g v_g * S**(groups-1-g)``, so
``idx_q == idx_k`` still means "every group agrees": exact hard matching is
preserved even though the soft match distribution is restricted to the product
family.

Codebook layout expected by :func:`factorized_interp`: ``[H, groups, S, dg]``
with ``dg = D // groups``. Group ``g`` matches the query's dim block
``[g*dg, (g+1)*dg)`` against its own ``S`` codewords; the concatenation of the
per-group interpolations reconstructs the full ``D``-dim embedding.
"""
import torch

from .emb_kernel import interp_embeddings


def split_codebook(vocab, groups):
    """[H, groups, S, dg] -> list of ``groups`` [H, S, dg] codebooks."""
    if vocab.dim() != 4 or vocab.shape[1] != groups:
        raise ValueError('codebook must be [H, groups, S, dg]')
    return [vocab[:, g] for g in range(groups)]


def factorized_interp(query, key, q_emb, k_emb, beta=1.0, tau=None, *, groups):
    """Drop-in replacement for ``interp_embeddings`` with a factorised codebook.

    Returns the same tuple ``(q_from_k, k_from_q, k_lse, q_lse, idx_k, idx_q)``.
    """
    if q_emb.dim() != 4 or q_emb.shape[1] != groups:
        raise ValueError('codebook must be [H, groups, S, dg]')
    H, _, S, dg = q_emb.shape
    D = dg * groups
    if query.shape[-1] != D:
        raise ValueError('query last dim must equal groups * dg')
    # The interpolation kernel requires BF16 codebooks (prepare_embedding does
    # exactly this cast); keep it here so callers can pass FP32 parameters.
    cast = lambda t: t if t.dtype == torch.bfloat16 else t.to(torch.bfloat16)
    q_groups = [cast(t) for t in split_codebook(q_emb, groups)]
    k_groups = [cast(t) for t in split_codebook(k_emb, groups)]
    qfk, kfq, idx_q, idx_k = [], [], None, None
    lse_q = lse_k = None
    for g in range(groups):
        sl = slice(g * dg, (g + 1) * dg)
        a, b, lk, lq, ik, iq = interp_embeddings(
            query[..., sl], key[..., sl], q_groups[g], k_groups[g], beta, tau)
        qfk.append(a)
        kfq.append(b)
        lse_k = lk if lse_k is None else lse_k + lk
        lse_q = lq if lse_q is None else lse_q + lq
        w = S ** (groups - 1 - g)
        idx_q = iq * w if idx_q is None else idx_q + iq * w
        idx_k = ik * w if idx_k is None else idx_k + ik * w
    return (torch.cat(qfk, dim=-1), torch.cat(kfq, dim=-1),
            lse_k, lse_q, idx_k, idx_q)


def expand_codebook(vocab, groups):
    """Build the equivalent full ``[H, S**groups, D]`` block codebook.

    Entry v = (v_1, ..., v_g) becomes ``concat_g(E^g_{v_g})``; the factorised
    path is numerically identical to running the original kernel on this tensor.
    Used by the equivalence test.
    """
    H, _, S, dg = vocab.shape
    D = dg * groups
    parts = []
    for g in range(groups):
        blk = vocab[:, g]                                     # [H, S, dg]
        shape = [H] + [1] * g + [S] + [1] * (groups - 1 - g) + [dg]
        parts.append(blk.reshape(shape).expand(H, *([S] * groups), dg))
    return torch.cat(parts, dim=-1).reshape(H, S ** groups, D)


if __name__ == '__main__':
    """Exactness check: a block-structured full codebook must give identical
    interpolations through the factorised path, as judged by the repo's own FP32
    reference interpolator; the BF16 Triton kernel differs only by logsumexp
    accumulation order."""
    from .emb_kernel import _voc_interp

    torch.manual_seed(0)
    B, N, H, D, S, G = 1, 256, 2, 64, 16, 2
    dg, dev = D // G, 'cuda'
    q = torch.randn(B, N, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    qv = torch.randn(H, G, S, dg, device=dev, dtype=torch.float32)
    kv = torch.randn_like(qv)
    bf = lambda t: t.to(torch.bfloat16)
    EQ, EK = bf(expand_codebook(qv, G)), bf(expand_codebook(kv, G))
    qg = [bf(t) for t in split_codebook(qv, G)]
    kg = [bf(t) for t in split_codebook(kv, G)]

    print(f"factorised V={G * S} as {G}x{S} (dg={dg}) vs block-structured full codebook")
    # _voc_interp(x, key_voc, val_voc): q side is (x=query, key=k_emb, val=q_emb)
    for name, x, Ga, Gb in (('q_from_k', k, qg, kg), ('k_from_q', q, kg, qg)):
        full = _voc_interp(x, bf(expand_codebook(Gb_code := kv if Ga is qg else qv, G)), bf(expand_codebook(Ga_code := qv if Ga is qg else kv, G)), 1.0)
        parts = [_voc_interp(x[..., g * dg:(g + 1) * dg], Gb[g], Ga[g], 1.0) for g in range(G)]
        out = torch.cat([t[0] for t in parts], dim=-1)
        lse = sum(t[1] for t in parts)
        idx = sum(t[2] * (S ** (G - 1 - g)) for g, t in enumerate(parts))
        rel_out = (full[0] - out).abs().max().item() / max(full[0].abs().max().item(), 1e-9)
        print(f"  {name:9} out rel-err={rel_out:.2e}   lse abs-err={(full[1] - lse).abs().max().item():.2e}"
              f" (scale {full[1].abs().max().item():.1f})   idx exact={bool((full[2] == idx).all())}")
    print("  -> identities hold to FP32 precision; BF16 kernel order differs ~1e0 on LSE")
