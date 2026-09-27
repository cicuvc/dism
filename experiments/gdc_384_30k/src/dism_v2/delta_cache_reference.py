"""Correctness reference for the Delta-Cache recurrence and its WY solve.

Tensor convention
-----------------
K, Q:   [B, H, L, Dk]
V:      [B, H, L, Dv]
lam:    [B, H, L]
gamma:  [B, H, L]
S0:     [B, H, Dv, Dk] (optional; omitted means zero)

The cache value stored at position m is the vector actually read/removed by
the Delta term, S^{m-1} K_m (pre-update).  Consequently both the ordinary
Delta dependency and the cache-state expansion reuse one strict-lower A.
Readout remains post-update: O_n = S^n Q_n.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def _check_inputs(
    K: torch.Tensor,
    V: torch.Tensor,
    Q: torch.Tensor,
    lam: torch.Tensor,
    gamma: torch.Tensor,
    S0: Optional[torch.Tensor],
) -> Tuple[int, int, int, int, int]:
    if K.ndim != 4:
        raise ValueError(f"K must have shape [B,H,L,Dk], got {tuple(K.shape)}")
    B, H, L, Dk = K.shape
    if V.ndim != 4 or V.shape[:3] != (B, H, L):
        raise ValueError("V must have shape [B,H,L,Dv] matching K")
    Dv = V.shape[-1]
    if Q.shape != K.shape:
        raise ValueError("Q must have the same shape as K")
    if lam.shape != (B, H, L) or gamma.shape != (B, H, L):
        raise ValueError("lam and gamma must both have shape [B,H,L]")
    if S0 is not None and S0.shape != (B, H, Dv, Dk):
        raise ValueError("S0 must have shape [B,H,Dv,Dk]")
    tensors = (K, V, Q, lam, gamma) + (() if S0 is None else (S0,))
    if any(x.device != K.device for x in tensors):
        raise ValueError("all inputs must be on the same device")
    if any(x.dtype != K.dtype for x in tensors):
        raise ValueError("all inputs must have the same dtype")
    if not K.is_floating_point():
        raise ValueError("inputs must be floating-point tensors")
    return B, H, L, Dk, Dv


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    # CPU triangular solves do not support every low-precision dtype, and a
    # correctness reference should accumulate fp16/bf16 inputs in fp32.
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def safe_strict_causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Softmax over source m < target n, returning an all-zero row at n=0.

    `scores[..., n, m]` has target/query position n on the row axis and
    source/key position m on the column axis.  This implementation never
    applies softmax to an all-`-inf` row, so the empty first row contains no
    NaNs.  Computation is stable under a per-row common-mode shift.
    """
    if scores.ndim < 2 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("scores must end in a square [L,L] matrix")
    L = scores.shape[-1]
    valid = torch.tril(
        torch.ones(L, L, dtype=torch.bool, device=scores.device), diagonal=-1
    )
    masked = scores.masked_fill(~valid, -torch.inf)
    row_max = masked.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    weights = torch.exp(masked - row_max)
    denom = weights.sum(dim=-1, keepdim=True)
    # `weights` is exactly zero on an empty row. clamp_min only prevents 0/0.
    tiny = torch.finfo(scores.dtype).tiny
    return weights / denom.clamp_min(tiny)


def delta_cache_recurrent(
    K: torch.Tensor,
    V: torch.Tensor,
    Q: torch.Tensor,
    lam: torch.Tensor,
    gamma: torch.Tensor,
    S0: Optional[torch.Tensor] = None,
    *,
    sm_scale: float = 1.0,
    return_intermediates: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[
    torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]
]:
    """Literal sequential recurrence; use as the ground-truth oracle.

    At position n:

      w[n,m] = softmax_m(<Q_n,K_m>), m < n
      cache_n = sum_{m<n} w[n,m] (S^{m-1} K_m)
      U_n = lam_n (V_n - S^{n-1} K_n) + gamma_n cache_n
      S^n = S^{n-1} + U_n K_n^T
      O_n = S^n Q_n

    Returns final state [B,H,Dv,Dk] and O [B,H,L,Dv].
    """
    B, H, L, Dk, Dv = _check_inputs(K, V, Q, lam, gamma, S0)
    out_dtype = K.dtype
    wd = _work_dtype(out_dtype)
    K, V, Q, lam, gamma = (x.to(wd) for x in (K, V, Q, lam, gamma))
    if S0 is None:
        S = torch.zeros(B, H, Dv, Dk, dtype=wd, device=K.device)
    else:
        S = S0.to(wd).clone()

    if sm_scale <= 0:
        raise ValueError("sm_scale must be positive")
    scores = torch.einsum("bhnd,bhmd->bhnm", Q, K)
    scores = torch.einsum("bhnd,bhmd->bhnm", Q, K) * sm_scale
    W = safe_strict_causal_softmax(scores)
    outputs, coeffs, cache_values = [], [], []

    for n in range(L):
        if n == 0:
            cache_read = torch.zeros(B, H, Dv, dtype=wd, device=K.device)
        else:
            history = torch.stack(cache_values, dim=2)  # [B,H,n,Dv]
            cache_read = torch.einsum("bhm,bhmv->bhv", W[:, :, n, :n], history)

        erased = torch.einsum("bhvd,bhd->bhv", S, K[:, :, n])
        U_n = (
            lam[:, :, n, None] * (V[:, :, n] - erased)
            + gamma[:, :, n, None] * cache_read
        )
        S = S + torch.einsum("bhv,bhd->bhvd", U_n, K[:, :, n])
        O_n = torch.einsum("bhvd,bhd->bhv", S, Q[:, :, n])

        # Persist the vector read before the update: S^{n-1} K_n.
        cache_values.append(erased)
        coeffs.append(U_n)
        outputs.append(O_n)

    O = torch.stack(outputs, dim=2).to(out_dtype)
    S_out = S.to(out_dtype)
    if return_intermediates:
        aux = {
            "scores_qk": scores.to(out_dtype),
            "attention": W.to(out_dtype),
            "U": torch.stack(coeffs, dim=2).to(out_dtype),
            "cache_values": torch.stack(cache_values, dim=2).to(out_dtype),
        }
        return S_out, O, aux
    return S_out, O


def delta_cache_wy(
    K: torch.Tensor,
    V: torch.Tensor,
    Q: torch.Tensor,
    lam: torch.Tensor,
    gamma: torch.Tensor,
    S0: Optional[torch.Tensor] = None,
    *,
    sm_scale: float = 1.0,
    return_intermediates: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[
    torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]
]:
    """Parallel correctness reference using the PDF's triangular WY system.

    The two time axes always use [target, source] order:

      A[target=n, source=l] = <K_n, K_l>
      C[target=n, source=m] = <Q_n, K_m>

    Both Delta and the cached pre-update value S^{m-1}K_m use strict-lower
    A_lt.  This deliberately avoids keeping a second inclusive A matrix.
    """
    B, H, L, Dk, Dv = _check_inputs(K, V, Q, lam, gamma, S0)
    out_dtype = K.dtype
    wd = _work_dtype(out_dtype)
    K, V, Q, lam, gamma = (x.to(wd) for x in (K, V, Q, lam, gamma))
    if S0 is None:
        S_init = torch.zeros(B, H, Dv, Dk, dtype=wd, device=K.device)
    else:
        S_init = S0.to(wd)

    lower_strict = torch.tril(
        torch.ones(L, L, dtype=torch.bool, device=K.device), diagonal=-1
    )
    # A[n,l] = <K_n,K_l>.  A_lt is reused for both S^{n-1}K_n and
    # the cached pre-update value S^{m-1}K_m.
    A = torch.einsum("bhnd,bhld->bhnl", K, K)
    A_lt = A.masked_fill(~lower_strict, 0)

    # This is the direction required by the recurrence.  The PDF's displayed
    # definition C_nl=<Q_l,K_n> is transposed relative to its later exp(C_nm).
    if sm_scale <= 0:
        raise ValueError("sm_scale must be positive")
    C = torch.einsum("bhnd,bhmd->bhnm", Q, K)
    scores = C * sm_scale
    W = safe_strict_causal_softmax(C)
    W = safe_strict_causal_softmax(scores)

    # D[n] = S0 K_n and E[n] = S0 Q_n.
    D = torch.einsum("bhvd,bhnd->bhnv", S_init, K)
    E = torch.einsum("bhvd,bhnd->bhnv", S_init, Q)

    # U = lam(V-D-A_lt U) + gamma W(D+A_lt U)
    WA = torch.matmul(W, A_lt)
    eye = torch.eye(L, dtype=wd, device=K.device).view(1, 1, L, L)
    system = (
        eye
        + lam[..., :, None] * A_lt
        - gamma[..., :, None] * WA
    )
    rhs = (
        lam[..., None] * (V - D)
        + gamma[..., None] * torch.matmul(W, D)
    )
    U = torch.linalg.solve_triangular(
        system, rhs, upper=False, unitriangular=True
    )

    S_out = S_init + torch.einsum("bhlv,bhld->bhvd", U, K)

    # Post-update readout: R[n,l]=<Q_n,K_l>, l<=n.
    # Unlike the pre-update cache, post-update readout includes l=n.
    R_le = C.masked_fill(~torch.tril(torch.ones_like(C, dtype=torch.bool)), 0)
    O = E + torch.matmul(R_le, U)

    if return_intermediates:
        cache_values = D + torch.matmul(A_lt, U)
        aux = {
            "A": A.to(out_dtype),
            "A_strict": A_lt.to(out_dtype),
            "C_qk": C.to(out_dtype),
            "attention_scores": scores.to(out_dtype),
            "attention": W.to(out_dtype),
            "W_A_strict": WA.to(out_dtype),
            "D_S0K": D.to(out_dtype),
            "E_S0Q": E.to(out_dtype),
            "system": system.to(out_dtype),
            "rhs": rhs.to(out_dtype),
            "U": U.to(out_dtype),
            "cache_values": cache_values.to(out_dtype),
            "readout_gram": R_le.to(out_dtype),
        }
        return S_out.to(out_dtype), O.to(out_dtype), aux
    return S_out.to(out_dtype), O.to(out_dtype)


def _self_test() -> None:
    """Compare WY against the literal recurrence, including both S0 cases."""
    torch.manual_seed(7)
    B, H, L, Dk, Dv = 2, 3, 7, 5, 4
    dtype = torch.float64
    K = torch.randn(B, H, L, Dk, dtype=dtype) / Dk**0.5
    Q = torch.randn(B, H, L, Dk, dtype=dtype) / Dk**0.5
    V = torch.randn(B, H, L, Dv, dtype=dtype)
    lam = torch.sigmoid(torch.randn(B, H, L, dtype=dtype))
    gamma = 0.25 * torch.tanh(torch.randn(B, H, L, dtype=dtype))

    for label, S0 in (
        ("S0=0", None),
        ("general S0", torch.randn(B, H, Dv, Dk, dtype=dtype) / Dk**0.5),
    ):
        S_seq, O_seq, a_seq = delta_cache_recurrent(
            K, V, Q, lam, gamma, S0, return_intermediates=True
        )
        S_wy, O_wy, a_wy = delta_cache_wy(
            K, V, Q, lam, gamma, S0, return_intermediates=True
        )
        torch.testing.assert_close(S_wy, S_seq, rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(O_wy, O_seq, rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(
            a_wy["cache_values"], a_seq["cache_values"], rtol=2e-11, atol=2e-11
        )
        assert torch.count_nonzero(a_wy["attention"][:, :, 0]) == 0
        assert torch.isfinite(a_wy["attention"]).all()
        print(f"{label}: passed")

    # A common-mode score shift must not change safe softmax.
    scores = torch.randn(2, 3, L, L, dtype=dtype)
    shift = 1e4 * torch.randn(2, 3, L, 1, dtype=dtype)
    torch.testing.assert_close(
        safe_strict_causal_softmax(scores),
        safe_strict_causal_softmax(scores + shift),
        rtol=2e-12,
        atol=2e-12,
    )
    print("safe softmax common-mode test: passed")


if __name__ == "__main__":
    _self_test()