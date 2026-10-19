"""Per-head role expansion for compact q/k vocabulary parameterizations."""

import torch


def expand_qk_vocab(q_vocab, k_vocab, pairing, *, heads, kv_heads):
    """Return per-head ``(q_table, k_table)`` tensors for a compact vocabulary."""
    if not pairing:
        return q_vocab, k_vocab
    if pairing != 'swap_halves':
        raise ValueError(f'unsupported vocabulary head pairing {pairing!r}')
    if heads % 2 or kv_heads % 2 or q_vocab.ndim != k_vocab.ndim:
        raise ValueError('swap_halves requires even Q/K head counts and matching ranks')
    if q_vocab.shape[0] != heads//2 or k_vocab.shape[0] != kv_heads//2:
        raise ValueError('swap_halves expects exactly half as many Q and K tables as heads')
    # Heads [0,H/2) use (Q,K); heads [H/2,H) use (K,Q). Both roles remain
    # distinct within every head, unlike shareqk.
    return (torch.cat((q_vocab, k_vocab), dim=0),
            torch.cat((k_vocab, q_vocab), dim=0))
