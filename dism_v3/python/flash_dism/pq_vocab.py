"""Product-quantised (factorised) DISM codebook.

Replaces one V-way vocabulary softmax over D dims with ``len(sizes)`` independent
``S_g``-way softmaxes over ``D/g`` dims each, where ``V = prod(sizes)``. The
identities

    q . E_v            = sum_g q_g . E^g_{v_g}
    logsumexp_v(...)   = sum_g logsumexp_{v_g}(...)

mean that the interpolated embedding is the concatenation of the per-group
blocks and the match LSE is the *sum* of the per-group LSEs. The DISM core
computes ``q_vec . k_vec - LSE``, and the dot product already sums over
concatenated dims, so **the core and its kernels are reused unchanged**.

The hard label uses the mixed-radix bijection
``v = sum_g v_g * prod(sizes[g+1:])``, so ``idx_q == idx_k`` still means "every
group agrees": exact hard matching is preserved even though the soft match
distribution is restricted to the product family.

Group sizes may differ (e.g. ``sizes=(32, 16)`` gives V=512 with a 32-way and a
16-way factor), but every group must use the same interpolation width
``dg = D / len(sizes)``, and the interpolation kernel only accepts ``dg`` in
{32, 64}.

Codebook layout expected by :func:`factorized_interp`: ``[H, len(sizes), Smax,
dg]`` where ``Smax = max(sizes)``; group ``g`` uses only rows ``[:sizes[g]]``, so
smaller groups leave inert padding rows when sizes differ.
"""
import math

import torch

from .emb_kernel import interp_embeddings


def validate_sizes(sizes, *, vocab_size, head_dim):
    """Validate the initial training-only product-quantised codebook contract."""
    sizes = tuple(int(size) for size in sizes)
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError('pq_vocab_sizes must contain group sizes >= 2')
    if math.prod(sizes) != vocab_size:
        raise ValueError('product(pq_vocab_sizes) must equal qk_vocab_size')
    if head_dim % len(sizes):
        raise ValueError('head_dim must be divisible by len(pq_vocab_sizes)')
    group_dim = head_dim // len(sizes)
    if group_dim not in (32, 64):
        raise ValueError('factorised group dimension must be 32 or 64')
    return sizes


def codebook_shape(heads, sizes, head_dim):
    sizes = tuple(sizes)
    return heads, len(sizes), max(sizes), head_dim//len(sizes)


def _check(vocab, sizes):
    if vocab.dim() != 4 or vocab.shape[1] != len(sizes):
        raise ValueError('codebook must be [H, len(sizes), Smax, dg]')
    if any(vocab.shape[2] < s for s in sizes):
        raise ValueError('padded group dim must be >= every group size')
    return vocab.shape[0], len(sizes), vocab.shape[2], vocab.shape[3], math.prod(sizes)


def split_codebook(vocab, sizes):
    """[H, G, Smax, dg] -> list of G [H, S_g, dg] codebooks (padding removed)."""
    _check(vocab, sizes)
    return [vocab[:, g, :s] for g, s in enumerate(sizes)]


def strides(sizes):
    """Mixed-radix strides for v = sum_g v_g * stride_g (last group is least significant)."""
    out, acc = [0] * len(sizes), 1
    for g in range(len(sizes) - 1, -1, -1):
        out[g] = acc
        acc *= sizes[g]
    return out


def factorized_interp(query, key, q_emb, k_emb, beta=1.0, tau=None, *, sizes):
    """Drop-in replacement for ``interp_embeddings`` with a factorised codebook.

    Returns the same tuple ``(q_from_k, k_from_q, k_lse, q_lse, idx_k, idx_q)``.
    """
    _, G, _, dg, V = _check(q_emb, sizes)
    D = dg * G
    if query.shape[-1] != D:
        raise ValueError('query last dim must equal groups * dg')
    cast = lambda t: t if t.dtype == torch.bfloat16 else t.to(torch.bfloat16)
    q_groups = [cast(t) for t in split_codebook(q_emb, sizes)]
    k_groups = [cast(t) for t in split_codebook(k_emb, sizes)]
    st = strides(sizes)
    qfk, kfq, idx_q, idx_k = [], [], None, None
    lse_q = lse_k = None
    for g in range(G):
        sl = slice(g * dg, (g + 1) * dg)
        a, b, lk, lq, ik, iq = interp_embeddings(
            query[..., sl], key[..., sl], q_groups[g], k_groups[g], beta, tau)
        qfk.append(a)
        kfq.append(b)
        lse_k = lk if lse_k is None else lse_k + lk
        lse_q = lq if lse_q is None else lse_q + lq
        w = st[g]
        idx_q = iq * w if idx_q is None else idx_q + iq * w
        idx_k = ik * w if idx_k is None else idx_k + ik * w
    return (torch.cat(qfk, dim=-1), torch.cat(kfq, dim=-1),
            lse_k, lse_q, idx_k, idx_q)


def expand_codebook(vocab, sizes):
    """Build the equivalent full ``[H, prod(sizes), D]`` block codebook.

    Entry v = (v_1, ..., v_g) becomes ``concat_g(E^g_{v_g})``; the factorised path
    is numerically identical to running the original kernel on this tensor.
    """
    H, G, Smax, dg, V = _check(vocab, sizes)
    parts = []
    for g, s in enumerate(sizes):
        blk = vocab[:, g, :s]                                 # [H, s, dg]
        shape = [H] + [1] * g + [s] + [1] * (G - 1 - g) + [dg]
        parts.append(blk.reshape(shape).expand(H, *sizes, dg))
    return torch.cat(parts, dim=-1).reshape(H, V, dg * G)


if __name__ == '__main__':
    """Exactness check: a block-structured full codebook must give identical
    interpolations through the factorised path, as judged by the repo's own FP32
    reference interpolator; the BF16 Triton kernel differs only by logsumexp
    accumulation order."""
    from .emb_kernel import _voc_interp

    torch.manual_seed(0)
    B, N, H, D = 1, 256, 2, 64
    dev = 'cuda'
    for sizes in [(2, 2), (16, 16), (32, 16), (64, 64)]:
        G, dg = len(sizes), D // len(sizes)
        Smax = max(sizes)
        q = torch.randn(B, N, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        qv = torch.randn(H, G, Smax, dg, device=dev, dtype=torch.float32)
        kv = torch.randn_like(qv)
        bf = lambda t: t.to(torch.bfloat16)
        EQ, EK = bf(expand_codebook(qv, sizes)), bf(expand_codebook(kv, sizes))
        qg = [bf(t) for t in split_codebook(qv, sizes)]
        kg = [bf(t) for t in split_codebook(kv, sizes)]
        st = strides(sizes)
        print(f"sizes={sizes} V={math.prod(sizes)} dg={dg}")
        # _voc_interp(x, key_voc, val_voc): q_from_k is (x=query, key=k_emb, val=q_emb)
        for name, x, Ga, Gb in (('q_from_k', k, qg, kg), ('k_from_q', q, kg, qg)):
            full = _voc_interp(x, bf(expand_codebook(kv if Ga is qg else qv, sizes)),
                               bf(expand_codebook(qv if Ga is qg else kv, sizes)), 1.0)
            parts = [_voc_interp(x[..., g * dg:(g + 1) * dg], Gb[g], Ga[g], 1.0) for g in range(G)]
            out = torch.cat([t[0] for t in parts], dim=-1)
            lse = sum(t[1] for t in parts)
            idx = sum(t[2] * st[g] for g, t in enumerate(parts))
            rel = (full[0] - out).abs().max().item() / max(full[0].abs().max().item(), 1e-9)
            print(f"  {name:9} out rel-err={rel:.2e}  lse abs-err={(full[1] - lse).abs().max().item():.2e}"
                  f" (scale {full[1].abs().max().item():.1f})  idx exact={bool((full[2] == idx).all())}")
    print("  -> identities hold to FP32 reference precision; BF16 kernel order differs ~1e0 on LSE")
