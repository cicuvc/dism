"""Readable reference for the random-direction, hard-probability Dism path.

Formulation
===========

For query token ``i``, key token ``j``, and vocabulary label ``r``, define

    Q[i,r] = sm_scale * <q[i], q_voc[r]>
    K[j,r] = sm_scale * <k[j], k_voc[r]>
    pq[i]   = softmax(Q[i])
    pk[j]   = softmax(K[j])

The exact collision log-probability (not materialized here) is

    C[i,j] = log sum_r pq[i,r] pk[j,r]
           = logsumexp_r(Q[i,r] + K[j,r]) - LQ[i] - LK[j].

Embedding interpolation gives two Jensen lower bounds without materializing
the vocabulary dimension in the N x N score matrix:

    q_from_k[j] = sum_r pk[j,r] q_voc[r]
    A[i,j]       = sm_scale * <q[i], q_from_k[j]> - LQ[i]

    k_from_q[i] = sum_r pq[i,r] k_voc[r]
    B[i,j]      = sm_scale * <k[j], k_from_q[i]> - LK[j].

Indeed, A <= C and B <= C by Jensen's inequality.  ``direction="random"``
draws one global coin per call and uses either A or B for every score.  This
matches ``mix="random"`` in dism.py; it is deliberately not a per-row coin.

The hard score uses the top-1 labels

    hard[i,j] = rtau                       if argmax(Q[i]) == argmax(K[j])
                -infinity                  otherwise.

One Bernoulli variable is then drawn for every (batch, head, query row), and
is shared by all keys in that row:

    logM[i,:] = hard[i,:]  with probability hard_prob
                soft[i,:]  with probability 1 - hard_prob,
    soft = rtau + (A or B).

Thus hard rows give no gradient to q/k/q_voc/k_voc through label selection,
while v and the downstream computation remain trainable.  Soft rows retain
the dense embedding-interpolation gradient.

The causal Dism recurrence is

    W[i,0] = logM[i,0]
    W[i,j] = logM[i,j] + softplus(W[i-1,j-1]),  i,j > 0,

followed by a causal mask.  Output normalization includes a fixed fallback
KV pair with log-score 0 and value 0:

    out[i] = sum_{j<=i} exp(W[i,j]) v[j]
             / (1 + sum_{j<=i} exp(W[i,j])).

Interaction with emb_kernel.py
==============================

``build_interpolation(..., backend="kernel")`` calls

    EmbInterpFunction.apply(q, k, q_voc, k_voc, sm_scale)

and converts its positional tuple into ``InterpolationResult``.  The exact
return order used by emb_kernel is documented in ``build_interpolation``.
Keeping this as a separate stage is useful while writing the Dism kernel:

1. use ``backend="torch"`` for a fully transparent oracle;
2. use ``backend="kernel"`` to test the new Dism kernel behind the existing
   embedding kernel;
3. pass both a precomputed ``interpolation=...`` and an explicit
   ``hard_mask=...`` to compare several Dism kernels against precisely the
   same interpolation outputs and row decisions.

The reference defaults to FP32 score computation even when q/k/vocab tables
are BF16.  The embedding-kernel backend naturally retains its BF16
interpolated embeddings, which are promoted to FP32 before the N x N scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


Direction = Literal["random", "q_from_k", "k_from_q"]
Backend = Literal["torch", "kernel"]


@dataclass(frozen=True)
class InterpolationResult:
    """Named version of the eight values returned by ``EmbInterpFunction``.

    All tokenwise tensors use shape ``[B,H,N,...]``.

    ``q_from_k`` is ``softmax(K) @ q_voc`` and is indexed by key token.
    ``k_from_q`` is ``softmax(Q) @ k_voc`` and is indexed by query token.
    The remaining names refer to the logits that produced them, rather than
    to the positional variable names inside emb_kernel.py.
    """

    q_from_k: torch.Tensor       # [B,H,N,D]
    k_from_q: torch.Tensor       # [B,H,N,D]
    k_lse: torch.Tensor          # [B,H,N], logsumexp(K)
    q_lse: torch.Tensor          # [B,H,N], logsumexp(Q)
    k_top_prob: torch.Tensor     # [B,H,N], max softmax(K)
    q_top_prob: torch.Tensor     # [B,H,N], max softmax(Q)
    k_index: torch.Tensor        # [B,H,N], argmax(K)
    q_index: torch.Tensor        # [B,H,N], argmax(Q)


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rtau: torch.Tensor,
    q_voc: torch.Tensor,
    k_voc: torch.Tensor,
) -> tuple[int, int, int, int]:
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError(f"q/k must have equal [B,H,N,D] shapes, got {q.shape} and {k.shape}")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError(f"v must be [B,H,N,DV] matching q/k, got {v.shape}")
    if q_voc.ndim != 3 or k_voc.shape != q_voc.shape:
        raise ValueError(
            f"q_voc/k_voc must have equal [H,V,D] shapes, got {q_voc.shape} and {k_voc.shape}"
        )
    batch, heads, length, dimension = q.shape
    if q_voc.shape[0] != heads or q_voc.shape[2] != dimension:
        raise ValueError(f"vocabulary shape {q_voc.shape} is incompatible with q shape {q.shape}")
    if rtau.shape != (heads,):
        raise ValueError(f"rtau must have shape [H]={heads}, got {rtau.shape}")
    tensors = (q, k, v, rtau, q_voc, k_voc)
    if any(t.device != q.device for t in tensors):
        raise ValueError("q, k, v, rtau, q_voc, and k_voc must be on the same device")
    return batch, heads, length, dimension


def interpolation_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    q_voc: torch.Tensor,
    k_voc: torch.Tensor,
    sm_scale: float = 1.0,
) -> InterpolationResult:
    """Pure PyTorch embedding interpolation with FP32 internal computation."""

    qf, kf = q.float(), k.float()
    qvf, kvf = q_voc.float(), k_voc.float()
    q_logits = torch.einsum("bhnd,hvd->bhnv", qf, qvf) * float(sm_scale)
    k_logits = torch.einsum("bhnd,hvd->bhnv", kf, kvf) * float(sm_scale)

    q_prob = torch.softmax(q_logits, dim=-1)
    k_prob = torch.softmax(k_logits, dim=-1)
    q_from_k = torch.einsum("bhnv,hvd->bhnd", k_prob, qvf)
    k_from_q = torch.einsum("bhnv,hvd->bhnd", q_prob, kvf)
    q_lse = torch.logsumexp(q_logits, dim=-1)
    k_lse = torch.logsumexp(k_logits, dim=-1)
    q_top_prob, q_index = q_prob.max(dim=-1)
    k_top_prob, k_index = k_prob.max(dim=-1)
    return InterpolationResult(
        q_from_k=q_from_k,
        k_from_q=k_from_q,
        k_lse=k_lse,
        q_lse=q_lse,
        k_top_prob=k_top_prob,
        q_top_prob=q_top_prob,
        k_index=k_index,
        q_index=q_index,
    )


def build_interpolation(
    q: torch.Tensor,
    k: torch.Tensor,
    q_voc: torch.Tensor,
    k_voc: torch.Tensor,
    sm_scale: float = 1.0,
    backend: Backend = "torch",
) -> InterpolationResult:
    """Build interpolation data using PyTorch or ``emb_kernel.py``.

    ``EmbInterpFunction`` returns, in its actual positional order,

        q_from_k, k_from_q, k_lse, q_lse,
        k_top_prob, q_top_prob, k_index, q_index.

    The apparently crossed order is intentional and matches the assignment in
    ``dism.py``.  The custom backward consumes gradients from ``q_from_k``,
    ``k_from_q``, ``k_lse``, and ``q_lse``.  Gradients of top probabilities and
    integer indices are ignored, as required by the hard branch.
    """

    if backend == "torch":
        return interpolation_ref(q, k, q_voc, k_voc, sm_scale)
    if backend != "kernel":
        raise ValueError(f"unknown interpolation backend: {backend!r}")

    # Lazy import keeps the CPU reference usable without Triton/FlashAttention.
    from emb_kernel import EmbInterpFunction

    values = EmbInterpFunction.apply(
        q,
        k,
        q_voc.to(dtype=q.dtype),
        k_voc.to(dtype=k.dtype),
        float(sm_scale),
    )
    return InterpolationResult(*values)


def _choose_direction(
    direction: Direction,
    device: torch.device,
    generator: torch.Generator | None,
) -> Literal["q_from_k", "k_from_q"]:
    if direction == "q_from_k" or direction == "k_from_q":
        return direction
    if direction != "random":
        raise ValueError(f"unknown soft direction: {direction!r}")
    coin = torch.randint(0, 2, (), device=device, generator=generator)
    return "q_from_k" if int(coin.item()) == 0 else "k_from_q"


def _broadcast_hard_prob(
    hard_prob: float | torch.Tensor,
    shape: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    probability = torch.as_tensor(hard_prob, dtype=torch.float32, device=device)
    if probability.shape == shape[:-1]:
        probability = probability.unsqueeze(-1)
    try:
        probability = torch.broadcast_to(probability, shape)
    except RuntimeError as error:
        raise ValueError(
            f"hard_prob shape {tuple(probability.shape)} is not broadcastable to {shape}"
        ) from error
    if torch.any((probability < 0) | (probability > 1)):
        raise ValueError("hard_prob must lie in [0,1]")
    return probability


def dism_recurrence(log_m: torch.Tensor) -> torch.Tensor:
    """Materialize causal recurrence scores ``W`` from ``log_m``.

    Args:
        log_m: ``[B,H,N,N]`` pair scores.
    Returns:
        ``[B,H,N,N]`` with the upper triangle set to ``-inf``.
    """

    if log_m.ndim != 4 or log_m.shape[-1] != log_m.shape[-2]:
        raise ValueError(f"log_m must be square [B,H,N,N], got {log_m.shape}")
    length = log_m.shape[-1]
    rows: list[torch.Tensor] = []
    previous: torch.Tensor | None = None
    for row_index in range(length):
        if previous is None:
            current = log_m[:, :, row_index, :]
        else:
            shifted = F.pad(previous[..., :-1], (1, 0), value=float("-inf"))
            current = log_m[:, :, row_index, :] + F.softplus(shifted)
        rows.append(current)
        previous = current
    scores = torch.stack(rows, dim=-2)
    causal = torch.ones((length, length), dtype=torch.bool, device=log_m.device).tril()
    return scores.masked_fill(~causal, float("-inf"))


def normalize_with_zero_fallback(scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply causal scores to values with the fixed (score=0, value=0) fallback."""

    zmax = scores.amax(dim=-1, keepdim=True).clamp_min(0.0).detach()
    weight = torch.exp(scores - zmax)
    fallback_weight = torch.exp(-zmax)
    numerator = torch.matmul(weight.to(dtype=v.dtype), v)
    denominator = weight.sum(dim=-1, keepdim=True) + fallback_weight
    return numerator / denominator


def voc_dism_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rtau: torch.Tensor,
    q_voc: torch.Tensor,
    k_voc: torch.Tensor,
    *,
    hard_prob: float | torch.Tensor,
    sm_scale: float = 1.0,
    direction: Direction = "random",
    generator: torch.Generator | None = None,
    hard_mask: torch.Tensor | None = None,
    interpolation_backend: Backend = "torch",
    interpolation: InterpolationResult | None = None,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
    """Reference implementation of random-direction + row-hard-prob Dism.

    ``hard_prob`` is a scalar or tensor broadcastable to ``[B,H,N,1]``.  A
    single hard/soft decision is shared by every key in a query row.  Set it to
    zero for all-soft training and one for the hard inference path.  Supplying
    a boolean ``hard_mask`` bypasses Bernoulli sampling, which lets a kernel and
    this reference consume exactly the same row decisions.

    Passing ``interpolation`` bypasses both interpolation backends.  This is
    useful for testing a Dism kernel independently from ``emb_kernel.py``.
    """

    batch, heads, length, _ = _check_inputs(q, k, v, rtau, q_voc, k_voc)
    if interpolation is None:
        interpolation = build_interpolation(
            q, k, q_voc, k_voc, sm_scale=sm_scale, backend=interpolation_backend
        )

    qf, kf = q.float(), k.float()
    q_from_k = interpolation.q_from_k.float()
    k_from_q = interpolation.k_from_q.float()
    # A[i,j] = E_{r~pk[j]} Q[i,r] - LQ[i].
    score_q_from_k = (
        torch.einsum("bhid,bhjd->bhij", qf, q_from_k) * float(sm_scale)
        - interpolation.q_lse.float().unsqueeze(-1)
    )
    # B[i,j] = E_{r~pq[i]} K[j,r] - LK[j].
    score_k_from_q = (
        torch.einsum("bhjd,bhid->bhij", kf, k_from_q) * float(sm_scale)
        - interpolation.k_lse.float().unsqueeze(-2)
    )
    selected_direction = _choose_direction(direction, q.device, generator)
    soft_score = (
        score_q_from_k if selected_direction == "q_from_k" else score_k_from_q
    )

    tau = rtau.float().reshape(1, heads, 1, 1)
    soft_log_m = tau + soft_score
    labels_match = interpolation.q_index.long().unsqueeze(-1) == interpolation.k_index.long().unsqueeze(-2)
    hard_log_m = torch.where(
        labels_match,
        tau.expand(batch, heads, length, length),
        torch.full((), float("-inf"), dtype=torch.float32, device=q.device),
    )

    probability = _broadcast_hard_prob(
        hard_prob, (batch, heads, length, 1), q.device
    )
    if hard_mask is None:
        random_rows = torch.rand(
            (batch, heads, length, 1),
            dtype=torch.float32,
            device=q.device,
            generator=generator,
        )
        use_hard = random_rows < probability
    else:
        use_hard = torch.as_tensor(hard_mask, dtype=torch.bool, device=q.device)
        if use_hard.shape == (batch, heads, length):
            use_hard = use_hard.unsqueeze(-1)
        try:
            use_hard = torch.broadcast_to(use_hard, (batch, heads, length, 1))
        except RuntimeError as error:
            raise ValueError(
                f"hard_mask shape {tuple(use_hard.shape)} is not broadcastable to "
                f"{(batch, heads, length, 1)}"
            ) from error
    log_m = torch.where(use_hard, hard_log_m, soft_log_m)
    scores = dism_recurrence(log_m)
    output = normalize_with_zero_fallback(scores, v)

    if not return_aux:
        return output
    auxiliary: dict[str, torch.Tensor | str] = {
        "direction": selected_direction,
        "use_hard": use_hard,
        "hard_prob": probability,
        "log_m": log_m,
        "hard_log_m": hard_log_m,
        "soft_log_m": soft_log_m,
        "score_q_from_k": score_q_from_k,
        "score_k_from_q": score_k_from_q,
        "scores": scores,
        "q_index": interpolation.q_index,
        "k_index": interpolation.k_index,
    }
    return output, auxiliary


def _smoke_test() -> None:
    torch.manual_seed(7)
    batch, heads, length, dimension, value_dimension, vocabulary = 2, 3, 7, 5, 4, 11
    q = torch.randn(batch, heads, length, dimension, requires_grad=True)
    k = torch.randn(batch, heads, length, dimension, requires_grad=True)
    v = torch.randn(batch, heads, length, value_dimension, requires_grad=True)
    q_voc = torch.randn(heads, vocabulary, dimension, requires_grad=True)
    k_voc = torch.randn(heads, vocabulary, dimension, requires_grad=True)
    rtau = torch.randn(heads, requires_grad=True)

    # Both interpolation scores must be Jensen lower bounds on the exact
    # collision log-probability.
    interpolation = interpolation_ref(q, k, q_voc, k_voc)
    q_logits = torch.einsum("bhnd,hvd->bhnv", q, q_voc)
    k_logits = torch.einsum("bhnd,hvd->bhnv", k, k_voc)
    exact = torch.logsumexp(
        q_logits.unsqueeze(-2) + k_logits.unsqueeze(-3), dim=-1
    ) - torch.logsumexp(q_logits, dim=-1).unsqueeze(-1) \
      - torch.logsumexp(k_logits, dim=-1).unsqueeze(-2)
    bound_a = torch.einsum("bhid,bhjd->bhij", q, interpolation.q_from_k) \
        - interpolation.q_lse.unsqueeze(-1)
    bound_b = torch.einsum("bhjd,bhid->bhij", k, interpolation.k_from_q) \
        - interpolation.k_lse.unsqueeze(-2)
    assert bool(torch.all(bound_a <= exact + 2e-6))
    assert bool(torch.all(bound_b <= exact + 2e-6))

    soft, soft_aux = voc_dism_ref(
        q, k, v, rtau, q_voc, k_voc,
        hard_prob=0.0, direction="q_from_k", return_aux=True,
    )
    assert not bool(soft_aux["use_hard"].any())
    soft.square().mean().backward()
    assert q.grad is not None and q_voc.grad is not None and v.grad is not None

    for tensor in (q, k, v, q_voc, k_voc, rtau):
        tensor.grad = None
    hard, hard_aux = voc_dism_ref(
        q, k, v, rtau, q_voc, k_voc,
        hard_prob=1.0, direction="q_from_k", return_aux=True,
    )
    assert bool(hard_aux["use_hard"].all())
    hard.square().mean().backward()
    assert v.grad is not None
    for tensor in (q, k, q_voc, k_voc):
        assert tensor.grad is None or bool(torch.count_nonzero(tensor.grad) == 0)

    # An explicit row mask makes the stochastic mixing interface reproducible
    # for reference-vs-kernel comparisons, independent of RNG state.
    row_mask = (torch.arange(length) % 2 == 0).reshape(1, 1, length, 1)
    masked_a = voc_dism_ref(
        q, k, v, rtau, q_voc, k_voc,
        hard_prob=0.5, direction="q_from_k", hard_mask=row_mask,
        interpolation=interpolation,
    )
    masked_b = voc_dism_ref(
        q, k, v, rtau, q_voc, k_voc,
        hard_prob=0.5, direction="q_from_k", hard_mask=row_mask,
        interpolation=interpolation,
    )
    torch.testing.assert_close(masked_a, masked_b)
    print("dism_ref smoke test: OK")


if __name__ == "__main__":
    _smoke_test()
