"""Correctness-first v4 reference with hard history and all-symbol query reads.

This is a dense, eager autograd oracle for small inputs, not the bounded-memory
blockwise implementation in DESIGN.md. Autograd retains candidate intermediates.
No v3 extension, model state, sampling, or global RNG is used.
"""

from typing import Literal

import torch
from torch import Tensor

KeyMode = Literal["hard", "soft", "jensen"]
_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _check_floats(named: dict[str, Tensor]) -> torch.dtype:
    device = next(iter(named.values())).device
    for name, value in named.items():
        if value.dtype not in _FLOATS:
            raise ValueError(f"{name} must have a floating-point dtype")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    return torch.float64 if any(x.dtype == torch.float64 for x in named.values()) else torch.float32


def _sequence(
    q_logits: Tensor,
    k_logits: Tensor,
    v: Tensor,
    tau: Tensor,
    key_mode: KeyMode,
    sq: Tensor | None,
    sk: Tensor | None,
    query_reset: Tensor,
) -> Tensor:
    # Work internally in BHNC; every batch row is a separate sequence.
    logp = q_logits.permute(0, 2, 1, 3).log_softmax(-1)
    keys = k_logits.permute(0, 2, 1, 3)
    iq, ik = logp.argmax(-1), keys.argmax(-1)
    batch, heads, length, vocab = logp.shape
    if key_mode == "hard":
        symbols = torch.arange(vocab, device=logp.device)
        logk = torch.zeros_like(keys).masked_fill(ik[..., None] != symbols, -torch.inf)
    else:
        logk = keys.log_softmax(-1)
    key_probability = logk.exp() if key_mode == "jensen" else None
    values = v.permute(0, 2, 1, 3)
    query_readout = None if sq is None else sq.permute(0, 2, 1, 3)
    key_readout = None if sk is None else sk.permute(0, 2, 1, 3)
    previous = logp.new_full((batch, heads, length), -torch.inf)
    empty_column = logp.new_full((batch, heads, 1), -torch.inf)
    fallback = logp.new_zeros((batch, heads, 1, vocab))
    positions = torch.arange(length, device=logp.device)
    rows = []
    for i in range(length):
        shifted = torch.cat((empty_column, previous[..., :-1]), dim=-1)
        shifted = shifted.masked_fill(query_reset[:, i, None, None], -torch.inf)
        # log A = tau + log(1 + H_previous). Exact logaddexp, not thresholded softplus.
        loga = tau[None, :, None] + torch.logaddexp(torch.zeros_like(shifted), shifted)
        logw = loga[..., :i + 1, None] + logk[..., :i + 1, :]
        # Include fallback BEFORE logsumexp: an empty candidate has finite logden=0
        # and zero derivatives, rather than the all--inf logsumexp backward NaN.
        logden = torch.logsumexp(torch.cat((fallback, logw), dim=-2), dim=-2)
        if key_mode == "jensen":
            # Jensen on the FULL normalized coefficient, including its dynamic
            # candidate denominator. History remains hard in this mode too.
            logu = logp[..., i, :] - logden
            logb = (key_probability[..., :i + 1, :] * logu[..., None, :]).sum(-1)
            pair_weight = (loga[..., :i + 1] + logb).exp()
        else:
            candidate_weight = (logw - logden[..., None, :]).exp()
            pair_weight = (candidate_weight * logp[..., i, None, :].exp()).sum(-1)
        if query_readout is not None:
            similarity = (query_readout[..., i, None, :] * key_readout[..., :i + 1, :]).sum(-1)
            pair_weight = pair_weight * similarity
        rows.append((pair_weight[..., None] * values[..., :i + 1, :]).sum(-2))
        matches = (iq[..., i, None] == ik) & (positions <= i)
        previous = loga.masked_fill(~matches, -torch.inf)
    return torch.stack(rows, dim=2).permute(0, 2, 1, 3).contiguous()


def dism_v4_ref(
    q_logits: Tensor,
    k_logits: Tensor,
    v: Tensor,
    tau: Tensor,
    *,
    key_mode: KeyMode = "hard",
    sq: Tensor | None = None,
    sk: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    query_reset: Tensor | None = None,
) -> Tensor:
    """Read every current Q symbol, normalize each candidate, then mix by p(Q).

    Args:
        q_logits, k_logits: [B,N,H,C] unnormalized symbol logits; all entries finite.
        v: [B,N,H,Dv] values. No SQ/SK or output normalization by default.
        tau: [H] log match multiplier, differentiable (negative/zero/positive).
        key_mode: "hard" uses one-hot K for the current read; "soft" uses its
            full softmax and exact arithmetic candidate mixing; "jensen" bounds
            the complete normalized read coefficient, not just the numerator.
            ALL modes commit argmax Q/K to history. Ties select the first index.
        sq, sk: optional [B,N,H,R] already-activated signed readout features;
            provide both. They affect only the numerator. No implicit scaling.
        cu_seqlens: optional int32/int64 [segments+1], for B=1 packed sequences.
            Must start at 0, end at N, and increase strictly. Each segment has
            independent query/key history and causal attention. No alignment limit.
        query_reset: optional bool [B,N]; clears Q matching history BEFORE that
            row, but keeps earlier keys readable. Not document isolation.

    Returns:
        [B,N,H,Dv], float64 if any floating input is float64, otherwise float32.
        Gradients do not pass through argmax history. In hard mode K logits have
        no differentiable use; autograd may return None for their gradient.

    This reference can retain O(B H N^2 C) candidate activations in backward.
    Python loops and validation synchronize devices; this is not graph-capturable.
    """
    if q_logits.ndim != 4 or q_logits.shape != k_logits.shape or min(q_logits.shape) <= 0:
        raise ValueError("q_logits/k_logits must have equal nonempty [B,N,H,C] shapes")
    batch, length, heads, _ = q_logits.shape
    if v.ndim != 4 or v.shape[:3] != (batch, length, heads) or v.shape[-1] <= 0:
        raise ValueError("v must have nonempty [B,N,H,Dv] shape")
    if tau.shape != (heads,):
        raise ValueError("tau must have shape [H]")
    if key_mode not in ("hard", "soft", "jensen"):
        raise ValueError("key_mode must be hard, soft, or jensen")
    if (sq is None) != (sk is None):
        raise ValueError("provide both sq and sk, or neither")
    named = dict(q_logits=q_logits, k_logits=k_logits, v=v, tau=tau)
    if sq is not None:
        if sq.ndim != 4 or sq.shape != sk.shape or sq.shape[:3] != (batch, length, heads) or sq.shape[-1] <= 0:
            raise ValueError("sq/sk must have equal nonempty [B,N,H,R] shapes")
        named.update(sq=sq, sk=sk)
    dtype = _check_floats(named)
    if query_reset is None:
        query_reset = torch.zeros((batch, length), device=q_logits.device, dtype=torch.bool)
    if query_reset.shape != (batch, length) or query_reset.dtype != torch.bool or query_reset.device != q_logits.device:
        raise ValueError("query_reset must be bool [B,N] on the input device")
    boundaries = [0, length]
    if cu_seqlens is not None:
        if batch != 1 or cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens requires B=1 and an int32/int64 vector")
        boundaries = cu_seqlens.tolist()
        if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != length or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
            raise ValueError("cu_seqlens must strictly increase from 0 to N")
    q_logits, k_logits, v, tau = (x.to(dtype) for x in (q_logits, k_logits, v, tau))
    sq = None if sq is None else sq.to(dtype)
    sk = None if sk is None else sk.to(dtype)
    outputs = []
    for begin, end in zip(boundaries, boundaries[1:]):
        outputs.append(_sequence(
            q_logits[:, begin:end], k_logits[:, begin:end], v[:, begin:end], tau,
            key_mode, None if sq is None else sq[:, begin:end],
            None if sk is None else sk[:, begin:end], query_reset[:, begin:end],
        ))
    return torch.cat(outputs, dim=1)


def dism_v4_from_embeddings(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    v: Tensor,
    tau: Tensor,
    *,
    key_mode: KeyMode = "hard",
    sq: Tensor | None = None,
    sk: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    query_reset: Tensor | None = None,
) -> Tensor:
    """Per-head codebook adapter: q/k [B,N,H,D], weights [H,C,D].

    No implicit sqrt(D) scaling, activation, temperature, gating, or RNG update.
    The resulting logits are passed to dism_v4_ref with the same semantics.
    """
    if q.ndim != 4 or q.shape != k.shape or min(q.shape) <= 0:
        raise ValueError("q/k must have equal nonempty [B,N,H,D] shapes")
    if q_weight.ndim != 3 or q_weight.shape != k_weight.shape or q_weight.shape[0] != q.shape[2] or q_weight.shape[2] != q.shape[3] or q_weight.shape[1] <= 0:
        raise ValueError("q_weight/k_weight must have equal nonempty [H,C,D] shapes")
    named = dict(q=q, k=k, q_weight=q_weight, k_weight=k_weight, v=v, tau=tau)
    if sq is not None:
        named["sq"] = sq
    if sk is not None:
        named["sk"] = sk
    dtype = _check_floats(named)
    q_logits = torch.einsum("bnhd,hcd->bnhc", q.to(dtype), q_weight.to(dtype))
    k_logits = torch.einsum("bnhd,hcd->bnhc", k.to(dtype), k_weight.to(dtype))
    return dism_v4_ref(
        q_logits, k_logits, v, tau, key_mode=key_mode, sq=sq, sk=sk,
        cu_seqlens=cu_seqlens, query_reset=query_reset,
    )
