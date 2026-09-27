"""PyTorch wrapper for the strict-cache Delta-Cache WY reference.

The projection/kernel-facing layout is BNHC (batch, sequence, head, channel).
Only the triangular solve uses a BHNC view because PyTorch treats the leading
dimensions as solve batch dimensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.modules.conv import ShortConvolution
except ImportError:  # fla >= 0.3 moved the module
    try:
        from fla.modules.convolution import ShortConvolution
    except ImportError:
        ShortConvolution = None
try:
    from fla.modules.fused_norm_gate import FusedRMSNormGated
except ImportError:  # Lets the standalone WY function be tested without FLA.
    FusedRMSNormGated = None

try:
    from .delta_cache_reference import safe_strict_causal_softmax
except ImportError:
    from delta_cache_reference import safe_strict_causal_softmax

try:
    from .gdc_chunk import ChunkGatedDeltaCacheFn
except ImportError:
    try:
        from gdc_chunk import ChunkGatedDeltaCacheFn
    except ImportError:
        ChunkGatedDeltaCacheFn = None


def delta_cache_wy_bnhc(
    k: torch.Tensor,          # [B,N,H,Dk]
    v: torch.Tensor,          # [B,N,H,Dv]
    lam: torch.Tensor,        # [B,N,H]
    q: torch.Tensor,          # [B,N,H,Dk]
    gamma: torch.Tensor,      # [B,N,H]
    initial_state: Optional[torch.Tensor] = None,  # [B,H,Dv,Dk]
    *,
    sm_scale: float = 1.0,
    return_intermediates: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[
    torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]
]:
    """Strict-cache triangular WY reference with native BNHC inputs/output.

    The cached vector at m is the pre-update erase vector S^{m-1} k_m.
    Therefore a single strict-lower key Gram matrix is used by both the Delta
    term and the persistent cache term.

    Returns:
        final_state: [B,H,Dv,Dk]
        output:      [B,N,H,Dv] (post-update readout)
    """
    if k.ndim != 4:
        raise ValueError(f"k must be [B,N,H,Dk], got {tuple(k.shape)}")
    B, N, H, Dk = k.shape
    if q.shape != k.shape:
        raise ValueError("q and k must have identical [B,N,H,Dk] shapes")
    if v.ndim != 4 or v.shape[:3] != (B, N, H):
        raise ValueError("v must have shape [B,N,H,Dv]")
    Dv = v.shape[-1]
    if lam.shape != (B, N, H) or gamma.shape != (B, N, H):
        raise ValueError("lam and gamma must have shape [B,N,H]")
    if initial_state is not None and initial_state.shape != (B, H, Dv, Dk):
        raise ValueError("initial_state must have shape [B,H,Dv,Dk]")
    inputs = (k, v, q, lam, gamma) + (
        () if initial_state is None else (initial_state,)
    )
    if any(x.device != k.device for x in inputs):
        raise ValueError("all inputs must be on the same device")
    if any(x.dtype != k.dtype for x in inputs):
        raise ValueError("all inputs must have the same dtype")

    out_dtype = k.dtype
    work_dtype = (
        torch.float32
        if out_dtype in (torch.float16, torch.bfloat16)
        else out_dtype
    )
    k, v, q, lam, gamma = (x.to(work_dtype) for x in (k, v, q, lam, gamma))
    if initial_state is None:
        S0 = torch.zeros(B, H, Dv, Dk, dtype=work_dtype, device=k.device)
    else:
        S0 = initial_state.to(work_dtype)

    strict = torch.tril(
        torch.ones(N, N, dtype=torch.bool, device=k.device), diagonal=-1
    )
    inclusive = torch.tril(
        torch.ones(N, N, dtype=torch.bool, device=k.device), diagonal=0
    )

    # Both matrices have [target, source] time order.
    # A[n,l] = <k_n,k_l>; C[n,m] = <q_n,k_m>.
    A = torch.einsum("bnhd,blhd->bhnl", k, k)
    A_strict = A.masked_fill(~strict, 0)
    if sm_scale <= 0:
        raise ValueError("sm_scale must be positive")
    C = torch.einsum("bnhd,bmhd->bhnm", q, k)
    scores = C * sm_scale
    W = safe_strict_causal_softmax(scores)

    # D_n=S0 k_n and E_n=S0 q_n; solve tensors use [B,H,N,*].
    D = torch.einsum("bhvd,bnhd->bhnv", S0, k)
    E = torch.einsum("bhvd,bnhd->bhnv", S0, q)
    v_h = v.transpose(1, 2)
    lam_h = lam.transpose(1, 2)
    gamma_h = gamma.transpose(1, 2)

    WA = torch.matmul(W, A_strict)
    eye = torch.eye(N, dtype=work_dtype, device=k.device).view(1, 1, N, N)
    system = (
        eye
        + lam_h[..., :, None] * A_strict
        - gamma_h[..., :, None] * WA
    )
    rhs = (
        lam_h[..., None] * (v_h - D)
        + gamma_h[..., None] * torch.matmul(W, D)
    )
    U = torch.linalg.solve_triangular(
        system, rhs, upper=False, unitriangular=True
    )  # [B,H,N,Dv]

    final_state = S0 + torch.einsum("bhnv,bnhd->bhvd", U, k)
    readout_gram = C.masked_fill(~inclusive, 0)
    output_h = E + torch.matmul(readout_gram, U)
    output = output_h.transpose(1, 2)  # [B,N,H,Dv]

    if return_intermediates:
        aux = {
            "A_strict": A_strict.to(out_dtype),
            "C_qk": C.to(out_dtype),
            "attention_scores": scores.to(out_dtype),
            "attention": W.to(out_dtype),
            "W_A_strict": WA.to(out_dtype),
            "system": system.to(out_dtype),
            "rhs": rhs.to(out_dtype),
            "U": U.transpose(1, 2).to(out_dtype),  # expose BNHC
            "cache_values": (D + torch.matmul(A_strict, U))
            .transpose(1, 2)
            .to(out_dtype),
            "readout_gram": readout_gram.to(out_dtype),
        }
        return final_state.to(out_dtype), output.to(out_dtype), aux
    return final_state.to(out_dtype), output.to(out_dtype)


@dataclass
class DeltaCacheRecurrentState:
    """State required to continue the Delta-Cache recurrence across chunks."""

    matrix: torch.Tensor       # [B,H,Dv,Dk]
    keys: torch.Tensor         # [B,M,H,Dk]
    erase_values: torch.Tensor # [B,M,H,Dv], pre-update S^{m-1} k_m


@dataclass
class DeltaCacheNetState:
    """Complete layer state, including the three ShortConvolution caches."""

    delta: DeltaCacheRecurrentState
    q_conv: Optional[torch.Tensor] = None
    k_conv: Optional[torch.Tensor] = None
    v_conv: Optional[torch.Tensor] = None


def delta_cache_recurrent_step_bnhc(
    k_t: torch.Tensor,      # [B,H,Dk]
    v_t: torch.Tensor,      # [B,H,Dv]
    lam_t: torch.Tensor,    # [B,H]
    q_t: torch.Tensor,      # [B,H,Dk]
    gamma_t: torch.Tensor,  # [B,H]
    state: Optional[DeltaCacheRecurrentState] = None,
    *,
    sm_scale: float = 1.0,
) -> Tuple[torch.Tensor, DeltaCacheRecurrentState]:
    """Decode one token and append its key/pre-update erase value to cache.

    Returns the post-update readout [B,H,Dv] and the updated recurrent state.
    This is an exact O(M) attention step over all M cached positions.
    """
    if k_t.ndim != 3:
        raise ValueError("k_t must have shape [B,H,Dk]")
    B, H, Dk = k_t.shape
    if q_t.shape != k_t.shape:
        raise ValueError("q_t and k_t must have identical shapes")
    if v_t.ndim != 3 or v_t.shape[:2] != (B, H):
        raise ValueError("v_t must have shape [B,H,Dv]")
    Dv = v_t.shape[-1]
    if lam_t.shape != (B, H) or gamma_t.shape != (B, H):
        raise ValueError("lam_t and gamma_t must have shape [B,H]")
    if sm_scale <= 0:
        raise ValueError("sm_scale must be positive")

    if state is None:
        matrix = torch.zeros(B, H, Dv, Dk, dtype=k_t.dtype, device=k_t.device)
        keys = torch.empty(B, 0, H, Dk, dtype=k_t.dtype, device=k_t.device)
        erase_values = torch.empty(
            B, 0, H, Dv, dtype=k_t.dtype, device=k_t.device
        )
    else:
        matrix, keys, erase_values = state.matrix, state.keys, state.erase_values
        if matrix.shape != (B, H, Dv, Dk):
            raise ValueError("state.matrix has an incompatible shape")
        if keys.ndim != 4 or keys.shape[0] != B or keys.shape[2:] != (H, Dk):
            raise ValueError("state.keys has an incompatible shape")
        if erase_values.shape != (B, keys.shape[1], H, Dv):
            raise ValueError("state.erase_values has an incompatible shape")
        if any(x.device != k_t.device for x in (matrix, keys, erase_values)):
            raise ValueError("recurrent state and inputs must share a device")
        if any(x.dtype != k_t.dtype for x in (matrix, keys, erase_values)):
            raise ValueError("recurrent state and inputs must share a dtype")

    history_len = keys.shape[1]
    if history_len == 0:
        cache_read = torch.zeros(B, H, Dv, dtype=k_t.dtype, device=k_t.device)
    else:
        logits = torch.einsum("bhd,bmhd->bhm", q_t, keys) * sm_scale
        # PyTorch softmax is max-shifted; accumulate low-precision logits in
        # fp32. history_len>0 means there is no empty row in decoding.
        softmax_dtype = (
            torch.float32
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits.dtype
        )
        weights = torch.softmax(logits, dim=-1, dtype=softmax_dtype).to(logits.dtype)
        cache_read = torch.einsum("bhm,bmhv->bhv", weights, erase_values)

    erased = torch.einsum("bhvd,bhd->bhv", matrix, k_t)
    U_t = lam_t[..., None] * (v_t - erased) + gamma_t[..., None] * cache_read
    matrix_next = matrix + torch.einsum("bhv,bhd->bhvd", U_t, k_t)
    output_t = torch.einsum("bhvd,bhd->bhv", matrix_next, q_t)
    next_state = DeltaCacheRecurrentState(
        matrix=matrix_next,
        keys=torch.cat((keys, k_t[:, None]), dim=1),
        erase_values=torch.cat((erase_values, erased[:, None]), dim=1),
    )
    return output_t, next_state


def delta_cache_recurrent_bnhc(
    k: torch.Tensor,          # [B,N,H,Dk]
    v: torch.Tensor,          # [B,N,H,Dv]
    lam: torch.Tensor,        # [B,N,H]
    q: torch.Tensor,          # [B,N,H,Dk]
    gamma: torch.Tensor,      # [B,N,H]
    state: Optional[DeltaCacheRecurrentState] = None,
    *,
    sm_scale: float = 1.0,
) -> Tuple[torch.Tensor, DeltaCacheRecurrentState]:
    """Correctness recurrent path for prefill, chunk continuation, or decode."""
    if k.ndim != 4:
        raise ValueError("k must have shape [B,N,H,Dk]")
    B, N, H, Dk = k.shape
    if q.shape != k.shape or v.shape[:3] != (B, N, H):
        raise ValueError("q/k/v sequence shapes are incompatible")
    if lam.shape != (B, N, H) or gamma.shape != (B, N, H):
        raise ValueError("lam and gamma must have shape [B,N,H]")

    outputs = []
    next_state = state
    for n in range(N):
        output_t, next_state = delta_cache_recurrent_step_bnhc(
            k[:, n],
            v[:, n],
            lam[:, n],
            q[:, n],
            gamma[:, n],
            next_state,
            sm_scale=sm_scale,
        )
        outputs.append(output_t)
    if N == 0:
        raise ValueError("the recurrent path requires at least one token")
    return torch.stack(outputs, dim=1), next_state


class DeltaCacheNet(nn.Module):
    """Projection/convolution/norm wrapper around :func:`delta_cache_wy_bnhc`.

    Input and output use [B,N,C].  Internal projected tensors use [B,N,H,D]
    by default, avoiding the eager BHND rearranges in the supplied wrapper.

    `initial_state` is the recurrent matrix state only.  This full-sequence
    correctness wrapper does not expose ShortConvolution streaming caches.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 4,
        key_dim: int = 64,
        val_dim: int = 64,
        conv_kernel_size: int = 4,
        sm_scale: float = 1.0,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        if ShortConvolution is None or FusedRMSNormGated is None:
            raise ImportError(
                "DeltaCacheNet requires flash-linear-attention (the `fla` package)"
            )
        if hidden_size <= 0 or num_heads <= 0 or key_dim <= 0 or val_dim <= 0:
            raise ValueError("all dimensions must be positive")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.val_dim = val_dim
        if sm_scale <= 0:
            raise ValueError("sm_scale must be positive")
        self.sm_scale = float(sm_scale)

        qk_width = num_heads * key_dim
        v_width = num_heads * val_dim
        self.q_proj = nn.Linear(hidden_size, qk_width, bias=False)
        self.k_proj = nn.Linear(hidden_size, qk_width, bias=False)
        self.v_proj = nn.Linear(hidden_size, v_width, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=qk_width,
            kernel_size=conv_kernel_size,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=qk_width,
            kernel_size=conv_kernel_size,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=v_width,
            kernel_size=conv_kernel_size,
            activation="silu",
        )

        self.gamma_proj = nn.Linear(hidden_size, num_heads)
        self.lambda_proj = nn.Linear(hidden_size, num_heads)

        self.g_proj = nn.Linear(hidden_size, v_width, bias=False)
        self.o_norm = FusedRMSNormGated(val_dim, eps=norm_eps)
        self.o_proj = nn.Linear(v_width, hidden_size, bias=False)

    def _project(
        self,
        x: torch.Tensor,
        conv_state: Optional[Tuple[Optional[torch.Tensor], ...]] = None,
        *,
        output_final_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        q_state, k_state, v_state = (None, None, None) if conv_state is None else conv_state
        q, q_state = self.q_conv1d(
            x=self.q_proj(x), cache=q_state, output_final_state=output_final_state
        )
        k, k_state = self.k_conv1d(
            x=self.k_proj(x), cache=k_state, output_final_state=output_final_state
        )
        v, v_state = self.v_conv1d(
            x=self.v_proj(x), cache=v_state, output_final_state=output_final_state
        )
        return q, k, v, (q_state, k_state, v_state)

    def _finish_output(self, x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        g = self.g_proj(x).reshape(B, N, self.num_heads, self.val_dim)
        o = self.o_norm(o, g)
        return self.o_proj(o.reshape(B, N, self.num_heads * self.val_dim))

    def forward(
        self,
        x: torch.Tensor,                         # [B,N,C]
        initial_state: Optional[torch.Tensor] = None,
        *,
        return_final_state: bool = False,
        return_intermediates: bool = False,
    ):
        if x.ndim != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"x must be [B,N,{self.hidden_size}], got {tuple(x.shape)}"
            )
        B, N, _ = x.shape

        q, k, v, _ = self._project(x)

        # Projection output is contiguous in the combined (head,channel) axis,
        # so reshape directly into native BNHC without a transpose/rearrange.
        q = q.reshape(B, N, self.num_heads, self.key_dim)
        k = k.reshape(B, N, self.num_heads, self.key_dim)
        v = v.reshape(B, N, self.num_heads, self.val_dim)
        k = F.normalize(k, p=2, dim=-1)

        gamma = self.gamma_proj(x).sigmoid()       # [B,N,H]
        lam = self.lambda_proj(x).sigmoid() * 2.0  # follows supplied wrapper

        result = delta_cache_wy_bnhc(
            k,
            v,
            lam,
            q,
            gamma,
            initial_state,
            sm_scale=self.sm_scale,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            final_state, o, aux = result
        else:
            final_state, o = result

        o = self._finish_output(x, o)

        if return_intermediates and return_final_state:
            return o, final_state, aux
        if return_intermediates:
            return o, aux
        if return_final_state:
            return o, final_state
        return o

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: Optional[DeltaCacheNetState] = None,
    ) -> Tuple[torch.Tensor, DeltaCacheNetState]:
        """Run an arbitrary non-empty chunk while carrying all decode state.

        This path is the correctness implementation for recurrent prefill and
        chunked continuation.  It is O(N * history) and intentionally does not
        replace an optimized recurrent/cache kernel.
        """
        if x.ndim != 3 or x.shape[-1] != self.hidden_size or x.shape[1] == 0:
            raise ValueError(f"x must be non-empty [B,N,{self.hidden_size}]")
        B, N, _ = x.shape
        conv_state = None if state is None else (state.q_conv, state.k_conv, state.v_conv)
        q, k, v, next_conv = self._project(
            x, conv_state, output_final_state=True
        )
        q = q.reshape(B, N, self.num_heads, self.key_dim)
        k = F.normalize(k.reshape(B, N, self.num_heads, self.key_dim), p=2, dim=-1)
        v = v.reshape(B, N, self.num_heads, self.val_dim)
        gamma = self.gamma_proj(x).sigmoid()
        lam = self.lambda_proj(x).sigmoid() * 2.0

        delta_state = None if state is None else state.delta
        o, delta_state = delta_cache_recurrent_bnhc(
            k,
            v,
            lam,
            q,
            gamma,
            delta_state,
            sm_scale=self.sm_scale,
        )
        o = self._finish_output(x, o)
        next_state = DeltaCacheNetState(
            delta=delta_state,
            q_conv=next_conv[0],
            k_conv=next_conv[1],
            v_conv=next_conv[2],
        )
        return o, next_state

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[DeltaCacheNetState] = None,
    ) -> Tuple[torch.Tensor, DeltaCacheNetState]:
        """Decode one token. Accepts [B,C] or [B,1,C]."""
        squeeze_time = x_t.ndim == 2
        if squeeze_time:
            x_t = x_t[:, None, :]
        if x_t.ndim != 3 or x_t.shape[1] != 1:
            raise ValueError("x_t must have shape [B,C] or [B,1,C]")
        o, next_state = self.forward_recurrent(x_t, state)
        return (o[:, 0] if squeeze_time else o), next_state


def _self_test_core() -> None:
    """Check native BNHC core against the BHNC correctness reference."""
    try:
        from .delta_cache_reference import delta_cache_wy
    except ImportError:
        from delta_cache_reference import delta_cache_wy

    torch.manual_seed(11)
    B, N, H, Dk, Dv = 2, 6, 3, 5, 4
    dtype = torch.float64
    k = torch.randn(B, N, H, Dk, dtype=dtype) / Dk**0.5
    q = torch.randn(B, N, H, Dk, dtype=dtype) / Dk**0.5
    v = torch.randn(B, N, H, Dv, dtype=dtype)
    lam = torch.sigmoid(torch.randn(B, N, H, dtype=dtype))
    gamma = 0.2 * torch.tanh(torch.randn(B, N, H, dtype=dtype))
    sm_scale = 2.3

    for S0 in (None, torch.randn(B, H, Dv, Dk, dtype=dtype)):
        S_native, O_native = delta_cache_wy_bnhc(
            k, v, lam, q, gamma, S0, sm_scale=sm_scale
        )
        S_ref, O_ref = delta_cache_wy(
            k.transpose(1, 2),
            v.transpose(1, 2),
            q.transpose(1, 2),
            lam.transpose(1, 2),
            gamma.transpose(1, 2),
            S0,
            sm_scale=sm_scale,
        )
        torch.testing.assert_close(S_native, S_ref, rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(
            O_native, O_ref.transpose(1, 2), rtol=2e-11, atol=2e-11
        )

    # Full recurrent and split-chunk recurrent paths must match parallel WY.
    S_wy, O_wy = delta_cache_wy_bnhc(
        k, v, lam, q, gamma, sm_scale=sm_scale
    )
    O_rec, state_full = delta_cache_recurrent_bnhc(
        k, v, lam, q, gamma, sm_scale=sm_scale
    )
    split = 3
    O_a, state_a = delta_cache_recurrent_bnhc(
        k[:, :split], v[:, :split], lam[:, :split], q[:, :split],
        gamma[:, :split], sm_scale=sm_scale,
    )
    O_b, state_b = delta_cache_recurrent_bnhc(
        k[:, split:], v[:, split:], lam[:, split:], q[:, split:],
        gamma[:, split:], state_a, sm_scale=sm_scale,
    )
    torch.testing.assert_close(O_rec, O_wy, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(state_full.matrix, S_wy, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(
        torch.cat((O_a, O_b), dim=1), O_wy, rtol=2e-11, atol=2e-11
    )
    torch.testing.assert_close(state_b.matrix, S_wy, rtol=2e-11, atol=2e-11)
    print("BNHC parallel/recurrent/chunked decode: passed")


def gated_delta_cache_wy_bnhc(
    k: torch.Tensor,          # [B,N,H,Dk]
    v: torch.Tensor,          # [B,N,H,Dv]
    lam: torch.Tensor,        # [B,N,H]
    q: torch.Tensor,          # [B,N,H,Dk]
    gamma: torch.Tensor,      # [B,N,H]
    r: torch.Tensor,          # [B,N,H,Dk]
    log_decay: torch.Tensor,  # [B,N,H], log of the state decay gate, <= 0
    initial_state: Optional[torch.Tensor] = None,  # [B,H,Dv,Dk]
    *,
    sm_scale: float = 1.0,
    return_intermediates: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[
    torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]
]:
    """Parallel WY solve for the gated Delta-Cache variant (delta_cache.typ).

    Recurrence at position n (k is normalized to unit norm inside,
    g_n = exp(log_decay_n)):

      w[n,m] = softmax_m(<q_n,k_m>), m < n
      U_n = lam_n (v_n - g_n S^{n-1} k_n) + gamma_n sum_m w[n,m] (S^m r_m)
      S^n = g_n S^{n-1} + U_n k_n^T
      O_n = S^n q_n

    Gate factors are folded into the Gram matrices through the cumulative
    log-gates log G_n = sum_{t<=n} log g_t:

      A~[n,l] = (G_n/G_l) <k_n,k_l>   (l<n, strict)
      B~[m,l] = (G_m/G_l) <r_m,k_l>   (l<=m, inclusive)
      (I + Lam A~ - Gam W B~) U = Lam (V - G*D) + Gam W (G*F)
      O = G*E + (C * G_n/G_l) U       (l<=n, inclusive)

    The system matrix is unit lower triangular, so the solve needs no
    diagonal division.
    """
    if k.ndim != 4:
        raise ValueError(f"k must be [B,N,H,Dk], got {tuple(k.shape)}")
    B, N, H, Dk = k.shape
    if q.shape != k.shape:
        raise ValueError("q and k must have identical [B,N,H,Dk] shapes")
    if r.shape != k.shape:
        raise ValueError("r and k must have identical [B,N,H,Dk] shapes")
    if v.ndim != 4 or v.shape[:3] != (B, N, H):
        raise ValueError("v must have shape [B,N,H,Dv]")
    Dv = v.shape[-1]
    if lam.shape != (B, N, H) or gamma.shape != (B, N, H):
        raise ValueError("lam and gamma must have shape [B,N,H]")
    if log_decay.shape != (B, N, H):
        raise ValueError("log_decay must have shape [B,N,H]")
    if initial_state is not None and initial_state.shape != (B, H, Dv, Dk):
        raise ValueError("initial_state must have shape [B,H,Dv,Dk]")
    if sm_scale <= 0:
        raise ValueError("sm_scale must be positive")
    inputs = (k, v, q, lam, gamma, r, log_decay) + (
        () if initial_state is None else (initial_state,)
    )
    if any(x.device != k.device for x in inputs):
        raise ValueError("all inputs must be on the same device")
    if any(x.dtype != k.dtype for x in inputs):
        raise ValueError("all inputs must have the same dtype")

    out_dtype = k.dtype
    work_dtype = (
        torch.float32
        if out_dtype in (torch.float16, torch.bfloat16)
        else out_dtype
    )
    k, v, q, lam, gamma, r, log_decay = (
        x.to(work_dtype) for x in (k, v, q, lam, gamma, r, log_decay)
    )
    k = F.normalize(k, p=2, dim=-1)
    if initial_state is None:
        S0 = torch.zeros(B, H, Dv, Dk, dtype=work_dtype, device=k.device)
    else:
        S0 = initial_state.to(work_dtype)

    strict = torch.tril(
        torch.ones(N, N, dtype=torch.bool, device=k.device), diagonal=-1
    )
    inclusive = torch.tril(
        torch.ones(N, N, dtype=torch.bool, device=k.device), diagonal=0
    )

    # Cumulative log-gates; G_n = prod_{t<=n} g_t.
    log_G = log_decay.cumsum(dim=1).transpose(1, 2)  # [B,H,N]
    G = torch.exp(log_G)                             # [B,H,N]
    log_ratio = log_G[..., :, None] - log_G[..., None, :]              # [B,H,N,N]
    decay_strict = torch.exp(log_ratio.masked_fill(~strict, -torch.inf))
    decay_incl = torch.exp(log_ratio.masked_fill(~inclusive, -torch.inf))

    # Gram matrices, [target, source] time order.
    A = torch.einsum("bnhd,blhd->bhnl", k, k)          # A[n,l]=<k_n,k_l>
    Bm = torch.einsum("bmhd,blhd->bhml", r, k)         # B[m,l]=<r_m,k_l>
    C = torch.einsum("bnhd,bmhd->bhnm", q, k)          # C[n,m]=<q_n,k_m>
    A_tilde = A * decay_strict
    B_tilde = Bm * decay_incl
    W = safe_strict_causal_softmax(C * sm_scale)

    # Initial-state terms with gate prefactors.
    D_tilde = G[..., None] * torch.einsum("bhvd,bnhd->bhnv", S0, k)
    E_tilde = G[..., None] * torch.einsum("bhvd,bnhd->bhnv", S0, q)
    F_tilde = G[..., None] * torch.einsum("bhvd,bmhd->bhmv", S0, r)

    v_h = v.transpose(1, 2)
    lam_h = lam.transpose(1, 2)
    gamma_h = gamma.transpose(1, 2)

    WB = torch.matmul(W, B_tilde)
    eye = torch.eye(N, dtype=work_dtype, device=k.device).view(1, 1, N, N)
    system = (
        eye
        + lam_h[..., :, None] * A_tilde
        - gamma_h[..., :, None] * WB
    )
    rhs = (
        lam_h[..., None] * (v_h - D_tilde)
        + gamma_h[..., None] * torch.matmul(W, F_tilde)
    )
    U = torch.linalg.solve_triangular(
        system, rhs, upper=False, unitriangular=True
    )  # [B,H,N,Dv]

    # S^N = G_N S0 + sum_l (G_N/G_l) U_l k_l^T.
    tail = torch.exp(log_G[..., -1:] - log_G)                        # [B,H,N]
    final_state = (
        G[..., -1, None, None] * S0
        + torch.einsum("bhn,bhnv,bnhd->bhvd", tail, U, k)
    )

    readout_gram = C * decay_incl
    output_h = E_tilde + torch.matmul(readout_gram, U)
    output = output_h.transpose(1, 2)  # [B,N,H,Dv]

    if return_intermediates:
        aux = {
            "A_tilde": A_tilde.to(out_dtype),
            "B_tilde": B_tilde.to(out_dtype),
            "attention": W.to(out_dtype),
            "W_B_tilde": WB.to(out_dtype),
            "system": system.to(out_dtype),
            "rhs": rhs.to(out_dtype),
            "U": U.transpose(1, 2).to(out_dtype),  # expose BNHC
            # Cached value at m is the post-update projection S^{m} r_m.
            "cache_values": (F_tilde + torch.matmul(B_tilde, U))
            .transpose(1, 2)
            .to(out_dtype),
            "readout_gram": readout_gram.to(out_dtype),
            "cum_gate": G.to(out_dtype),
        }
        return final_state.to(out_dtype), output.to(out_dtype), aux
    return final_state.to(out_dtype), output.to(out_dtype)


@dataclass
class GatedDeltaCacheRecurrentState:
    """State required to continue the gated Delta-Cache recurrence."""

    matrix: torch.Tensor        # [B,H,Dv,Dk]
    keys: torch.Tensor          # [B,M,H,Dk]
    cache_values: torch.Tensor  # [B,M,H,Dv], post-update S^{m} r_m


@dataclass
class GatedDeltaCacheNetState:
    """Complete gated layer state, including the ShortConvolution caches."""

    delta: GatedDeltaCacheRecurrentState
    q_conv: Optional[torch.Tensor] = None
    k_conv: Optional[torch.Tensor] = None
    v_conv: Optional[torch.Tensor] = None
    r_conv: Optional[torch.Tensor] = None


def gated_delta_cache_recurrent_step_bnhc(
    k_t: torch.Tensor,      # [B,H,Dk]
    v_t: torch.Tensor,      # [B,H,Dv]
    lam_t: torch.Tensor,    # [B,H]
    q_t: torch.Tensor,      # [B,H,Dk]
    gamma_t: torch.Tensor,  # [B,H]
    r_t: torch.Tensor,      # [B,H,Dk]
    log_decay_t: torch.Tensor,  # [B,H], log of the decay gate, <= 0
    state: Optional[GatedDeltaCacheRecurrentState] = None,
    *,
    sm_scale: float = 1.0,
) -> Tuple[torch.Tensor, GatedDeltaCacheRecurrentState]:
    """Decode one gated token; ground-truth oracle for the parallel WY solve."""
    if k_t.ndim != 3:
        raise ValueError("k_t must have shape [B,H,Dk]")
    B, H, Dk = k_t.shape
    if q_t.shape != k_t.shape or r_t.shape != k_t.shape:
        raise ValueError("q_t/r_t and k_t must have identical shapes")
    if v_t.ndim != 3 or v_t.shape[:2] != (B, H):
        raise ValueError("v_t must have shape [B,H,Dv]")
    Dv = v_t.shape[-1]
    if lam_t.shape != (B, H) or gamma_t.shape != (B, H):
        raise ValueError("lam_t and gamma_t must have shape [B,H]")
    if log_decay_t.shape != (B, H):
        raise ValueError("log_decay_t must have shape [B,H]")
    if sm_scale <= 0:
        raise ValueError("sm_scale must be positive")
    k_t = F.normalize(k_t, p=2, dim=-1)
    decay_t = log_decay_t.exp()

    if state is None:
        matrix = torch.zeros(B, H, Dv, Dk, dtype=k_t.dtype, device=k_t.device)
        keys = torch.empty(B, 0, H, Dk, dtype=k_t.dtype, device=k_t.device)
        cache_values = torch.empty(
            B, 0, H, Dv, dtype=k_t.dtype, device=k_t.device
        )
    else:
        matrix, keys, cache_values = (
            state.matrix,
            state.keys,
            state.cache_values,
        )

    history_len = keys.shape[1]
    if history_len == 0:
        cache_read = torch.zeros(B, H, Dv, dtype=k_t.dtype, device=k_t.device)
    else:
        logits = torch.einsum("bhd,bmhd->bhm", q_t, keys) * sm_scale
        softmax_dtype = (
            torch.float32
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits.dtype
        )
        weights = torch.softmax(logits, dim=-1, dtype=softmax_dtype).to(logits.dtype)
        cache_read = torch.einsum("bhm,bmhv->bhv", weights, cache_values)

    erased = decay_t[..., None] * torch.einsum("bhvd,bhd->bhv", matrix, k_t)
    U_t = lam_t[..., None] * (v_t - erased) + gamma_t[..., None] * cache_read
    matrix_next = (
        decay_t[..., None, None] * matrix
        + torch.einsum("bhv,bhd->bhvd", U_t, k_t)
    )
    output_t = torch.einsum("bhvd,bhd->bhv", matrix_next, q_t)
    cache_value_t = torch.einsum("bhvd,bhd->bhv", matrix_next, r_t)
    next_state = GatedDeltaCacheRecurrentState(
        matrix=matrix_next,
        keys=torch.cat((keys, k_t[:, None]), dim=1),
        cache_values=torch.cat((cache_values, cache_value_t[:, None]), dim=1),
    )
    return output_t, next_state


def gated_delta_cache_recurrent_bnhc(
    k: torch.Tensor,          # [B,N,H,Dk]
    v: torch.Tensor,          # [B,N,H,Dv]
    lam: torch.Tensor,        # [B,N,H]
    q: torch.Tensor,          # [B,N,H,Dk]
    gamma: torch.Tensor,      # [B,N,H]
    r: torch.Tensor,          # [B,N,H,Dk]
    log_decay: torch.Tensor,  # [B,N,H], log of the decay gate, <= 0
    state: Optional[GatedDeltaCacheRecurrentState] = None,
    *,
    sm_scale: float = 1.0,
    return_intermediates: bool = False,
):
    """Sequential gated recurrence; correctness oracle for the WY solve."""
    if k.ndim != 4:
        raise ValueError("k must have shape [B,N,H,Dk]")
    B, N, H, Dk = k.shape
    if q.shape != k.shape or r.shape != k.shape or v.shape[:3] != (B, N, H):
        raise ValueError("q/k/v/r sequence shapes are incompatible")
    if lam.shape != (B, N, H) or gamma.shape != (B, N, H):
        raise ValueError("lam and gamma must have shape [B,N,H]")
    if log_decay.shape != (B, N, H):
        raise ValueError("log_decay must have shape [B,N,H]")
    if N == 0:
        raise ValueError("the recurrent path requires at least one token")

    outputs = []
    next_state = state
    for n in range(N):
        output_t, next_state = gated_delta_cache_recurrent_step_bnhc(
            k[:, n],
            v[:, n],
            lam[:, n],
            q[:, n],
            gamma[:, n],
            r[:, n],
            log_decay[:, n],
            next_state,
            sm_scale=sm_scale,
        )
        outputs.append(output_t)
    if return_intermediates:
        aux = {
            "keys": next_state.keys,
            "cache_values": next_state.cache_values,
        }
        return torch.stack(outputs, dim=1), next_state, aux
    return torch.stack(outputs, dim=1), next_state


class GatedDeltaCacheNet(nn.Module):
    """Gated Delta-Cache layer wrapping :func:`gated_delta_cache_wy_bnhc`.

    Input and output use [B,N,C].  Compared to DeltaCacheNet this adds an
    r projection (cache write projection) and a per-head decay gate g.
    Keys are normalized to unit norm before entering the recurrence.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 4,
        key_dim: int = 64,
        val_dim: int = 64,
        conv_kernel_size: int = 4,
        sm_scale: float = 1.0,
        norm_eps: float = 1e-5,
        use_chunk_kernel: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if ShortConvolution is None or FusedRMSNormGated is None:
            raise ImportError(
                "GatedDeltaCacheNet requires flash-linear-attention (the `fla` package)"
            )
        if hidden_size <= 0 or num_heads <= 0 or key_dim <= 0 or val_dim <= 0:
            raise ValueError("all dimensions must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.use_chunk_kernel = bool(use_chunk_kernel)
        if sm_scale <= 0:
            raise ValueError("sm_scale must be positive")
        self.sm_scale = float(sm_scale)

        qk_width = num_heads * key_dim
        v_width = num_heads * val_dim
        self.q_proj = nn.Linear(hidden_size, qk_width, bias=False)
        self.k_proj = nn.Linear(hidden_size, qk_width, bias=False)
        self.r_proj = nn.Linear(hidden_size, qk_width, bias=False)
        self.v_proj = nn.Linear(hidden_size, v_width, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=qk_width, kernel_size=conv_kernel_size, activation="silu"
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=qk_width, kernel_size=conv_kernel_size, activation="silu"
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=v_width, kernel_size=conv_kernel_size, activation="silu"
        )

        self.gamma_proj = nn.Linear(hidden_size, num_heads)
        self.lambda_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # Decay gate parameterization follows fla's GatedDeltaNet:
        #   g = -exp(A_log) * softplus(a_proj(x) + dt_bias), decay = exp(g).
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(
            torch.rand(num_heads) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.g_proj = nn.Linear(hidden_size, v_width, bias=False)
        self.o_norm = FusedRMSNormGated(val_dim, eps=norm_eps)
        self.o_proj = nn.Linear(v_width, hidden_size, bias=False)

    def _project(
        self,
        x: torch.Tensor,
        conv_state: Optional[Tuple[Optional[torch.Tensor], ...]] = None,
        *,
        output_final_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        q_state, k_state, v_state = (
            (None, None, None) if conv_state is None else conv_state
        )
        q, q_state = self.q_conv1d(
            x=self.q_proj(x), cache=q_state, output_final_state=output_final_state
        )
        k, k_state = self.k_conv1d(
            x=self.k_proj(x), cache=k_state, output_final_state=output_final_state
        )
        r = torch.nn.functional.silu(self.r_proj(x))
        v, v_state = self.v_conv1d(
            x=self.v_proj(x), cache=v_state, output_final_state=output_final_state
        )
        return q, k, r, v, (q_state, k_state, v_state)

    def _finish_output(self, x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        g = self.g_proj(x).reshape(B, N, self.num_heads, self.val_dim)
        o = self.o_norm(o, g)
        return self.o_proj(o.reshape(B, N, self.num_heads * self.val_dim))

    def _gates(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = self.gamma_proj(x).sigmoid()
        lam = self.lambda_proj(x).sigmoid() * 2.0
        # fla GatedDeltaNet log-decay parameterization, computed in fp32.
        log_decay = -self.A_log.float().exp() * F.softplus(
            self.a_proj(x).float() + self.dt_bias
        )
        return gamma, lam, log_decay.to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,                         # [B,N,C]
        initial_state: Optional[torch.Tensor] = None,
        *,
        return_final_state: bool = False,
        return_intermediates: bool = False,
    ):
        if x.ndim != 3 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"x must be [B,N,{self.hidden_size}], got {tuple(x.shape)}"
            )
        B, N, _ = x.shape

        q, k, r, v, _ = self._project(x)
        q = q.reshape(B, N, self.num_heads, self.key_dim)
        k = k.reshape(B, N, self.num_heads, self.key_dim)
        r = r.reshape(B, N, self.num_heads, self.key_dim)
        v = v.reshape(B, N, self.num_heads, self.val_dim)
        gamma, lam, log_decay = self._gates(x)

        use_chunk = (
            self.use_chunk_kernel
            and ChunkGatedDeltaCacheFn is not None
            and x.is_cuda
            and N % 64 == 0
            and not return_intermediates
        )
        if use_chunk:
            # The chunk kernels normalize k internally (and their backward
            # includes the l2-norm adjoint), so pass the raw k here.
            final_state, o = ChunkGatedDeltaCacheFn.apply(
                k, v, lam, q, gamma, r, log_decay, initial_state,
                self.sm_scale, 64,
            )
            o = self._finish_output(x, o)
            return (o, final_state) if return_final_state else o

        k = F.normalize(k, p=2, dim=-1)

        result = gated_delta_cache_wy_bnhc(
            k, v, lam, q, gamma, r, log_decay, initial_state,
            sm_scale=self.sm_scale,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            final_state, o, aux = result
        else:
            final_state, o = result

        o = self._finish_output(x, o)

        if return_intermediates and return_final_state:
            return o, final_state, aux
        if return_intermediates:
            return o, aux
        if return_final_state:
            return o, final_state
        return o

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: Optional[GatedDeltaCacheNetState] = None,
    ) -> Tuple[torch.Tensor, GatedDeltaCacheNetState]:
        """Run a non-empty chunk while carrying all decode state (oracle path)."""
        if x.ndim != 3 or x.shape[-1] != self.hidden_size or x.shape[1] == 0:
            raise ValueError(f"x must be non-empty [B,N,{self.hidden_size}]")
        B, N, _ = x.shape
        conv_state = (
            None
            if state is None
            else (state.q_conv, state.k_conv, state.v_conv, state.r_conv)
        )
        q, k, r, v, next_conv = self._project(
            x, conv_state, output_final_state=True
        )
        q = q.reshape(B, N, self.num_heads, self.key_dim)
        k = F.normalize(k.reshape(B, N, self.num_heads, self.key_dim), p=2, dim=-1)
        r = r.reshape(B, N, self.num_heads, self.key_dim)
        v = v.reshape(B, N, self.num_heads, self.val_dim)
        gamma, lam, log_decay = self._gates(x)

        delta_state = None if state is None else state.delta
        o, delta_state = gated_delta_cache_recurrent_bnhc(
            k, v, lam, q, gamma, r, log_decay, delta_state, sm_scale=self.sm_scale
        )
        o = self._finish_output(x, o)
        next_state = GatedDeltaCacheNetState(
            delta=delta_state,
            q_conv=next_conv[0],
            k_conv=next_conv[1],
            v_conv=next_conv[2],
            r_conv=next_conv[3],
        )
        return o, next_state

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[GatedDeltaCacheNetState] = None,
    ) -> Tuple[torch.Tensor, GatedDeltaCacheNetState]:
        """Decode one token. Accepts [B,C] or [B,1,C]."""
        squeeze_time = x_t.ndim == 2
        if squeeze_time:
            x_t = x_t[:, None, :]
        if x_t.ndim != 3 or x_t.shape[1] != 1:
            raise ValueError("x_t must have shape [B,C] or [B,1,C]")
        o, next_state = self.forward_recurrent(x_t, state)
        return (o[:, 0] if squeeze_time else o), next_state


def _self_test_gated() -> None:
    """Verify the gated parallel WY solve against the sequential oracle."""
    torch.manual_seed(13)
    B, N, H, Dk, Dv = 2, 7, 3, 5, 4
    dtype = torch.float64
    k = torch.randn(B, N, H, Dk, dtype=dtype) / Dk**0.5
    q = torch.randn(B, N, H, Dk, dtype=dtype) / Dk**0.5
    r = torch.randn(B, N, H, Dk, dtype=dtype) / Dk**0.5
    v = torch.randn(B, N, H, Dv, dtype=dtype)
    lam = torch.sigmoid(torch.randn(B, N, H, dtype=dtype))
    gamma = 0.2 * torch.tanh(torch.randn(B, N, H, dtype=dtype))
    log_decay = torch.sigmoid(torch.randn(B, N, H, dtype=dtype)).log()
    sm_scale = 1.7

    for S0 in (None, torch.randn(B, H, Dv, Dk, dtype=dtype)):
        S_wy, O_wy, aux = gated_delta_cache_wy_bnhc(
            k, v, lam, q, gamma, r, log_decay, S0,
            sm_scale=sm_scale, return_intermediates=True,
        )
        # Sequential oracle.
        state = None
        outputs, u_seq, cache_seq = [], [], []
        matrix = torch.zeros(B, H, Dv, Dk, dtype=dtype) if S0 is None else S0.clone()
        keys, cvals = [], []
        for n in range(N):
            k_n = F.normalize(k[:, n], p=2, dim=-1)  # [B,H,Dk]
            if keys:
                logits = torch.einsum(
                    "bhd,bmhd->bhm", q[:, n], torch.stack(keys, dim=1)
                ) * sm_scale
                weights = torch.softmax(logits, dim=-1)
                cache_read = torch.einsum(
                    "bhm,bmhv->bhv", weights, torch.stack(cvals, dim=1)
                )
            else:
                cache_read = torch.zeros(B, H, Dv, dtype=dtype)
            d_t = log_decay[:, n].exp()  # [B,H]
            erased = d_t[..., None] * torch.einsum("bhvd,bhd->bhv", matrix, k_n)
            U_n = (
                lam[:, n][..., None] * (v[:, n] - erased)
                + gamma[:, n][..., None] * cache_read
            )
            matrix = d_t[..., None, None] * matrix + torch.einsum(
                "bhv,bhd->bhvd", U_n, k_n
            )
            outputs.append(torch.einsum("bhvd,bhd->bhv", matrix, q[:, n]))
            u_seq.append(U_n)
            keys.append(k_n)
            cvals.append(torch.einsum("bhvd,bhd->bhv", matrix, r[:, n]))
        O_seq = torch.stack(outputs, dim=1)
        U_seq = torch.stack(u_seq, dim=1)
        C_seq = torch.stack(cvals, dim=1)

        torch.testing.assert_close(O_wy, O_seq, rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(S_wy, matrix, rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(aux["U"], U_seq, rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(aux["cache_values"], C_seq, rtol=2e-11, atol=2e-11)
        assert torch.count_nonzero(aux["attention"][:, :, 0]) == 0
        assert torch.isfinite(O_wy).all()
        print(f"gated WY vs oracle (S0={'0' if S0 is None else 'general'}): passed")

    # Recurrent step path must match parallel WY, including chunked splits.
    S_wy, O_wy = gated_delta_cache_wy_bnhc(
        k, v, lam, q, gamma, r, log_decay, sm_scale=sm_scale
    )
    O_rec, st_full = gated_delta_cache_recurrent_bnhc(
        k, v, lam, q, gamma, r, log_decay, sm_scale=sm_scale
    )
    split = 3
    O_a, st_a = gated_delta_cache_recurrent_bnhc(
        k[:, :split], v[:, :split], lam[:, :split], q[:, :split],
        gamma[:, :split], r[:, :split], log_decay[:, :split], sm_scale=sm_scale,
    )
    O_b, st_b = gated_delta_cache_recurrent_bnhc(
        k[:, split:], v[:, split:], lam[:, split:], q[:, split:],
        gamma[:, split:], r[:, split:], log_decay[:, split:], st_a, sm_scale=sm_scale,
    )
    torch.testing.assert_close(O_rec, O_wy, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(st_full.matrix, S_wy, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(
        torch.cat((O_a, O_b), dim=1), O_wy, rtol=2e-11, atol=2e-11
    )
    torch.testing.assert_close(st_b.matrix, S_wy, rtol=2e-11, atol=2e-11)

    # Unit-norm keys are enforced inside the kernels.
    k_bad = 10.0 * k
    S_a1, O_a1 = gated_delta_cache_wy_bnhc(
        k, v, lam, q, gamma, r, log_decay, sm_scale=sm_scale
    )
    S_a2, O_a2 = gated_delta_cache_wy_bnhc(
        k_bad, v, lam, q, gamma, r, log_decay, sm_scale=sm_scale
    )
    torch.testing.assert_close(S_a1, S_a2, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(O_a1, O_a2, rtol=2e-11, atol=2e-11)
    print("gated BNHC parallel/recurrent/chunked/normalization: passed")


if __name__ == "__main__":
    _self_test_core()
    _self_test_gated()
