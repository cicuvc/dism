"""Chunkwise parallel gated Delta-Cache (delta_cache.typ, "Chunkwise 形式").

Triton kernels are vendored from fla's chunk GatedDeltaRule forward path
(fla/ops/common/chunk_scaled_dot_kkt.py, fla/ops/utils/solve_tril.py) and
modified:

  - `chunk_decayed_gram_fwd_kernel` generalizes the KKT kernel to two distinct
    inputs (K@K^T for the Delta Gram, R@K^T for the cache Gram), an optional
    row scale (lam), and a selectable mask diagonal (strict for A, inclusive
    for B).  Autotune is stripped for fast dev iteration.
  - `solve_tril_64_kernel` is fla's merge_16x16_to_64x64 inverse kernel with
    autotune/TMA/varlen removed (BT=64 only).

Pipeline (per delta_cache.typ):

  Phase 1 (parallel over chunks):
      X  = Lam A^ - (gamma/L) W^loc B^
      T  = (I + X)^-1                       (vendored solve_tril)
      u1 = T (Lam V),  w = T Z,
      Z  = G (Lam K) - (gamma/L) W^loc (G R)
  Phase 2 (scan over chunks, cache dependency):
      U_k = u1_k - w_k S^T + T_k [ gamma_k (e^hist_k / L_k) C^hist ]
      O_k = G_k (Q_k S^T) + (C^raw_k * decay) U_k
      c^_k = G_k (R_k S^T) + B^_k U_k        (appended to cache)
      S   <- G_last S + U_k^T (G_last/G) K_k

Layout: BNHC, matching deltacache.py.  Dev constraints: fp32, N % 64 == 0.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Vendored + modified from fla/ops/common/chunk_scaled_dot_kkt.py
# A[i,j] = beta_i <a_i, b_j> exp(g_i - g_j), masked to j <= i + DIAG.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=['T'])
def chunk_decayed_gram_fwd_kernel(
    a,
    b,
    beta,
    g,
    A,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_BETA: tl.constexpr,
    DIAG: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_a = tl.make_block_ptr(a + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t*BT, i_k*BK), (BT, BK), (1, 0))
        p_b = tl.make_block_ptr(b + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t*BT, i_k*BK), (BT, BK), (1, 0))
        b_a = tl.load(p_a, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))
        b_A += tl.dot(b_a, tl.trans(b_b), input_precision="ieee")

    p_g = tl.make_block_ptr(g + bos*H + i_h, (T,), (H,), (i_t*BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_A *= tl.exp(b_g[:, None] - b_g[None, :])
    if USE_BETA:
        p_be = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t*BT,), (BT,), (0,))
        b_A *= tl.load(p_be, boundary_check=(0,))[:, None]

    m_A = (o_t[:, None] >= o_t[None, :] - DIAG) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t*BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_decayed_gram_fwd(
    a: torch.Tensor,               # [B,T,H,K]
    b: torch.Tensor,               # [B,T,H,K]
    g: torch.Tensor,               # [B,T,H] within-chunk cumulative log-gate
    beta: Optional[torch.Tensor],  # [B,T,H] or None
    diag: int,                     # -1 strict lower, 0 inclusive
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, K = a.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    A = torch.empty(B, T, H, BT, device=a.device, dtype=torch.float32)
    chunk_decayed_gram_fwd_kernel[(NT, B * H)](
        a=a,
        b=b,
        beta=beta if beta is not None else g,
        g=g,
        A=A,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=64,
        USE_BETA=beta is not None,
        DIAG=diag,
        num_warps=4,
        num_stages=2,
    )
    return A


# ---------------------------------------------------------------------------
# Vendored from fla/ops/utils/solve_tril.py (merge_16x16_to_64x64, no TMA).
# Computes Ai = (I + A)^-1 for strictly lower triangular A.  BT == 64 only.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=['T'])
def solve_tril_64_kernel(
    A,
    Ai,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    p_A_11 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_A_33 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_A_44 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision="ieee"), b_Ai_11, input_precision="ieee")
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision="ieee"), b_Ai_22, input_precision="ieee")
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision="ieee"), b_Ai_33, input_precision="ieee")
    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision="ieee") + tl.dot(b_A_32, b_Ai_21, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision="ieee") + tl.dot(b_A_43, b_Ai_32, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision="ieee") + tl.dot(b_A_42, b_Ai_21, input_precision="ieee") + tl.dot(b_A_43, b_Ai_31, input_precision="ieee"),
        input_precision="ieee",
    )

    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty), boundary_check=(0, 1))


def solve_tril_64(A: torch.Tensor) -> torch.Tensor:
    """(I + A)^-1 for strictly lower A of shape [B,T,H,64]."""
    B, T, H, BT = A.shape
    assert BT == 64
    NT = triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=torch.float32)
    solve_tril_64_kernel[(NT, B * H)](
        A=A,
        Ai=Ai,
        T=T,
        H=H,
        BT=BT,
        num_warps=4,
        num_stages=2,
    )
    return Ai


# ---------------------------------------------------------------------------
# Chunkwise parallel pipeline.
# ---------------------------------------------------------------------------
def _chunk_gdc_impl(
    k, v, lam, q, gamma, r, log_decay, initial_state, initial_cache,
    sm_scale, chunk_size, save_bwd,
):
    if k.ndim != 4:
        raise ValueError(f"k must be [B,N,H,Dk], got {tuple(k.shape)}")
    B, N, H, Dk = k.shape
    P = chunk_size
    if P != 64:
        raise ValueError("dev kernel requires chunk_size == 64")
    if N % P != 0:
        raise ValueError("dev kernel requires N % chunk_size == 0")
    if q.shape != k.shape or r.shape != k.shape:
        raise ValueError("q/r and k must have identical [B,N,H,Dk] shapes")
    if v.ndim != 4 or v.shape[:3] != (B, N, H):
        raise ValueError("v must have shape [B,N,H,Dv]")
    Dv = v.shape[-1]
    if lam.shape != (B, N, H) or gamma.shape != (B, N, H):
        raise ValueError("lam and gamma must have shape [B,N,H]")
    if log_decay.shape != (B, N, H):
        raise ValueError("log_decay must have shape [B,N,H]")
    if sm_scale <= 0:
        raise ValueError("sm_scale must be positive")
    NT = N // P

    out_dtype = k.dtype
    dev = k.device
    k, v, q, r = (x.float().contiguous() for x in (k, v, q, r))
    lam, gamma, log_decay = (x.float().contiguous() for x in (lam, gamma, log_decay))
    k_raw = k
    k = F.normalize(k, p=2, dim=-1)

    if initial_state is None:
        S = torch.zeros(B, H, Dv, Dk, dtype=torch.float32, device=dev)
    else:
        S = initial_state.float().contiguous()
    if initial_cache is None:
        cache_k = torch.empty(B, H, 0, Dk, dtype=torch.float32, device=dev)
        cache_v = torch.empty(B, H, 0, Dv, dtype=torch.float32, device=dev)
    else:
        cache_k = initial_cache[0].float().transpose(1, 2).contiguous()  # [B,H,M,Dk]
        cache_v = initial_cache[1].float().transpose(1, 2).contiguous()
    M0 = cache_k.shape[2]

    def ch(x: torch.Tensor) -> torch.Tensor:
        # [B,N,H,*] -> [B,H,NT,P,*]
        if x.ndim == 3:
            return x.view(B, NT, P, H).permute(0, 3, 1, 2)
        return x.view(B, NT, P, H, *x.shape[3:]).permute(0, 3, 1, 2, 4)

    # ---------------- Phase 1: parallel over chunks ----------------
    g_flat = log_decay.view(B, NT, P, H).cumsum(2).view(B, N, H)
    g_loc = ch(g_flat)                    # [B,H,NT,P] within-chunk cum log-gate
    G = g_loc.exp()
    q_c, k_c, r_c, v_c = ch(q), ch(k), ch(r), ch(v)
    lam_c, gam_c = ch(lam), ch(gamma)

    A_lam = ch(chunk_decayed_gram_fwd(k, k, g_flat, lam, diag=-1, chunk_size=P))
    B_hat = ch(chunk_decayed_gram_fwd(r, k, g_flat, None, diag=0, chunk_size=P))

    # Full raw QK gram (sequence keys + initial cache keys).
    k_all = torch.cat((cache_k, k.transpose(1, 2)), dim=2)  # [B,H,M0+N,Dk]
    C_raw = torch.einsum("bnhd,bhmd->bhnm", q, k_all)       # [B,H,N,M0+N]
    n_idx = torch.arange(N, device=dev)
    m_idx = torch.arange(M0 + N, device=dev)
    valid = m_idx[None, :] < (M0 + n_idx)[:, None]          # [N,M0+N]
    masked = (C_raw * sm_scale).masked_fill(~valid, -torch.inf)
    row_max = masked.amax(-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    e = torch.exp(masked - row_max)                         # zeros on invalid
    L = e.sum(-1).clamp_min(torch.finfo(torch.float32).tiny)  # [B,H,N]
    L_c = L.view(B, H, NT, P)
    # W_loc and the history weights are normalized by L, so gamma is used
    # directly (Gamma = diag(gamma/L) times unnormalized e^C == gamma * W).

    # Local (within-chunk) blocks of e and C_raw.
    e_c = e.view(B, H, NT, P, M0 + N)
    e_loc = torch.stack(
        [e_c[:, :, kc_, :, M0 + kc_ * P:M0 + (kc_ + 1) * P] for kc_ in range(NT)],
        dim=2,
    )  # [B,H,NT,P,P], already strict-lower masked
    C_loc = torch.stack(
        [C_raw.view(B, H, NT, P, M0 + N)[:, :, kc_, :, M0 + kc_ * P:M0 + (kc_ + 1) * P] for kc_ in range(NT)],
        dim=2,
    )
    W_loc = e_loc / L_c[..., None]

    X = A_lam - gam_c[..., None] * torch.matmul(W_loc, B_hat)
    X_flat = X.permute(0, 2, 3, 1, 4).reshape(B, N, H, P).contiguous()
    T_inv = ch(solve_tril_64(X_flat))                       # [B,H,NT,P,P]

    u1 = torch.matmul(T_inv, lam_c[..., None] * v_c)        # [B,H,NT,P,Dv]
    GR = G[..., None] * r_c
    Z = G[..., None] * (lam_c[..., None] * k_c) - gam_c[..., None] * torch.matmul(W_loc, GR)
    w = torch.matmul(T_inv, Z)                              # [B,H,NT,P,Dk]

    incl = torch.tril(torch.ones(P, P, dtype=torch.bool, device=dev))
    g_diff = g_loc[..., :, None] - g_loc[..., None, :]
    decay_incl = torch.exp(g_diff.masked_fill(~incl, -torch.inf))
    readout_gram = C_loc * decay_incl                       # j<=i, raw C

    if save_bwd:
        A_hat = ch(chunk_decayed_gram_fwd(k, k, g_flat, None, diag=-1, chunk_size=P))
        strict = torch.tril(torch.ones(P, P, dtype=torch.bool, device=dev), -1)
        decay_strict = torch.exp(g_diff.masked_fill(~strict, -torch.inf))
        W_full = e / L[..., None]                           # [B,H,N,M0+N]
    else:
        A_hat = decay_strict = W_full = None

    # ---------------- Phase 2: scan over chunks ----------------
    O = torch.empty(B, H, NT, P, Dv, dtype=torch.float32, device=dev)
    U_all = torch.empty(B, H, NT, P, Dv, dtype=torch.float32, device=dev)
    if save_bwd:
        S_list = torch.empty(B, H, NT, Dv, Dk, dtype=torch.float32, device=dev)
        Omega_all = torch.empty(B, H, NT, P, Dv, dtype=torch.float32, device=dev)
    else:
        S_list = Omega_all = None
    cache_out_v = [cache_v] if M0 > 0 else []
    cache_out_k = [cache_k] if M0 > 0 else []
    for kc_ in range(NT):
        e_h = e[:, :, kc_ * P:(kc_ + 1) * P, :M0 + kc_ * P]   # [B,H,P,M]
        hist = torch.matmul(e_h, cache_v) / L_c[:, :, kc_][..., None]
        if save_bwd:
            Omega_all[:, :, kc_] = hist
            S_list[:, :, kc_] = S
        u2 = torch.matmul(T_inv[:, :, kc_], gam_c[:, :, kc_][..., None] * hist)
        S_t = S.transpose(-1, -2)                             # [B,H,Dk,Dv]
        U = u1[:, :, kc_] - torch.matmul(w[:, :, kc_], S_t) + u2
        U_all[:, :, kc_] = U

        O[:, :, kc_] = (
            G[:, :, kc_][..., None] * torch.matmul(q_c[:, :, kc_], S_t)
            + torch.matmul(readout_gram[:, :, kc_], U)
        )
        c_new = (
            G[:, :, kc_][..., None] * torch.matmul(r_c[:, :, kc_], S_t)
            + torch.matmul(B_hat[:, :, kc_], U)
        )
        cache_out_v.append(c_new)
        cache_out_k.append(k_c[:, :, kc_])
        cache_v = torch.cat((cache_v, c_new), dim=2)

        tail = torch.exp(g_loc[:, :, kc_, -1:] - g_loc[:, :, kc_])  # [B,H,P]
        S = G[:, :, kc_, -1][..., None, None] * S + torch.einsum(
            "bhpv,bhpd->bhvd", U, tail[..., None] * k_c[:, :, kc_]
        )

    cache_v = torch.cat(cache_out_v, dim=2) if cache_out_v else cache_v
    cache_k = torch.cat(cache_out_k, dim=2) if cache_out_k else cache_k

    return {
        "S": S, "O": O, "cache_k": cache_k, "cache_v": cache_v,
        "meta": (B, N, H, NT, P, Dk, Dv, M0, out_dtype),
        "q": q, "k_raw": k_raw, "k_all": k_all,
        "g_loc": g_loc, "G": G, "q_c": q_c, "k_c": k_c, "r_c": r_c, "v_c": v_c,
        "lam_c": lam_c, "gam_c": gam_c,
        "A_lam": A_lam, "B_hat": B_hat, "W_loc": W_loc, "T_inv": T_inv,
        "w": w, "Z": Z, "readout_gram": readout_gram, "decay_incl": decay_incl,
        "U_all": U_all, "X": X, "L": L,
        "A_hat": A_hat, "decay_strict": decay_strict, "W_full": W_full,
        "S_list": S_list, "Omega_all": Omega_all,
    }


def chunk_gated_delta_cache(
    k: torch.Tensor,          # [B,N,H,Dk]
    v: torch.Tensor,          # [B,N,H,Dv]
    lam: torch.Tensor,        # [B,N,H]
    q: torch.Tensor,          # [B,N,H,Dk]
    gamma: torch.Tensor,      # [B,N,H]
    r: torch.Tensor,          # [B,N,H,Dk]
    log_decay: torch.Tensor,  # [B,N,H], log of the decay gate, <= 0
    initial_state: Optional[torch.Tensor] = None,   # [B,H,Dv,Dk]
    initial_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    #  ([B,M,H,Dk] keys, [B,M,H,Dv] values)
    *,
    sm_scale: float = 1.0,
    chunk_size: int = 64,
    return_intermediates: bool = False,
):
    fw = _chunk_gdc_impl(
        k, v, lam, q, gamma, r, log_decay, initial_state, initial_cache,
        sm_scale=sm_scale, chunk_size=chunk_size, save_bwd=False,
    )
    B, N, H, NT, P, Dk, Dv, M0, out_dtype = fw["meta"]
    O_out = fw["O"].permute(0, 2, 3, 1, 4).reshape(B, N, H, Dv).to(out_dtype)
    S_out = fw["S"].to(out_dtype)
    cache_out = (
        fw["cache_k"].transpose(1, 2).to(out_dtype),
        fw["cache_v"].transpose(1, 2).to(out_dtype),
    )
    if return_intermediates:
        aux = {
            "U": fw["U_all"].permute(0, 2, 3, 1, 4).reshape(B, N, H, Dv).to(out_dtype),
            "A_lam": fw["A_lam"].to(out_dtype),
            "B_hat": fw["B_hat"].to(out_dtype),
            "X": fw["X"].to(out_dtype),
            "T_inv": fw["T_inv"].to(out_dtype),
            "W_loc": fw["W_loc"].to(out_dtype),
            "L": fw["L"].to(out_dtype),
            "readout_gram": fw["readout_gram"].to(out_dtype),
        }
        return S_out, O_out, cache_out, aux
    return S_out, O_out, cache_out


# ---------------------------------------------------------------------------
# Backward. Stages follow the "Chunkwise 反向" section of delta_cache.typ and
# map to fla's GDN backward kernels:
#   stage 1 (dU readout part)      gdc_bwd_dv_local_kernel    ~ chunk_bwd_dv_local
#   stage 2 (reverse chunk scan)   gdc_bwd_scan_*             ~ chunk_bwd_dhu
#   stage 3 (WY adjoint)           gdc_bwd_wy_*               ~ prepare_wy_repr_bwd
#   stage 4 (softmax + dQ/dK)      gdc_bwd_{rowsum,dq,dk}     ~ chunk_bwd_dqkwg
#   stage 5 (gate + reverse cumsum)gdc_bwd_gate_cumsum_kernel
# Dev kernels: P=64, Dk/Dv<=BDk/BDv<=64, fp32, no autotune. This GPU has ~99KB
# smem per block and this triton stages every distinct dot-operand tensor
# through smem, so the scan runs as a host loop of per-chunk micro-kernels and
# the WY adjoint is split into small kernels (<=5 dot operands each).
# Torch stage fallbacks below are the autograd-verified golden reference;
# stages can be mixed via the stage_impl argument of chunk_gated_delta_cache_bwd.
# ---------------------------------------------------------------------------
@triton.jit
def gdc_bwd_dv_local_kernel(
    RG, DO, U, DU1, DAPR,
    NT, Dv,
    BDv: tl.constexpr,
):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_v = tl.arange(0, BDv)
    m_v = o_v < Dv
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    b_rg = tl.load(RG + base * 4096 + o_pp)
    b_do = tl.load(DO + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_u = tl.load(U + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_du1 = tl.dot(tl.trans(b_rg), b_do, input_precision="ieee")
    b_dapr = tl.dot(b_do, tl.trans(b_u), input_precision="ieee")
    tl.store(DU1 + base * 64 * Dv + o_pv, b_du1, mask=m_v[None, :])
    tl.store(DAPR + base * 4096 + o_pp, b_dapr)


# ---- Stage 2: reverse scan as a host loop of per-chunk micro-kernels ----
@triton.jit
def gdc_bwd_scan_du_kernel(
    GLOC, K, DU1, B_HAT, DCACHE_V, DS_STATE,
    DU_ALL, DS_LIST_BWD,
    i_c, NT, Dk, Dv, Mtot, M0,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # dU = dU1 + B^T dc + (tail*K) dS^T; also checkpoints dS^T for this chunk.
    i_bh = tl.program_id(0)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    o_vkt = o_k[:, None] + o_v[None, :] * Dk
    b_g = tl.load(GLOC + base * 64 + o_p)
    g_last = tl.load(GLOC + base * 64 + 63)
    b_tail = tl.exp(g_last - b_g)
    b_du1 = tl.load(DU1 + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_Bt = tl.load(B_HAT + base * 4096 + o_ppt)
    M_k = M0 + i_c * 64
    b_dc = tl.load(DCACHE_V + (i_bh * Mtot + M_k) * Dv + o_pv,
                   mask=m_v[None, :], other=0.0)
    b_k = tl.load(K + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_dSt = tl.load(DS_STATE + i_bh * Dv * Dk + o_vkt,
                    mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_dU = (b_du1
            + tl.dot(b_Bt, b_dc, input_precision="ieee")
            + tl.dot(b_tail[:, None] * b_k, b_dSt, input_precision="ieee"))
    tl.store(DU_ALL + base * 64 * Dv + o_pv, b_dU, mask=m_v[None, :])
    tl.store(DS_LIST_BWD + base * Dv * Dk + o_vkt, b_dSt,
             mask=m_k[:, None] & m_v[None, :])


@triton.jit
def gdc_bwd_scan_dom_kernel(
    T, GAM, OMEGA, DU_ALL, DOM, DGAM,
    i_c, NT, Dv,
    BDv: tl.constexpr,
):
    # dOmega = gamma * T^T dU; dgam contribution from Omega.
    i_bh = tl.program_id(0)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_v = tl.arange(0, BDv)
    m_v = o_v < Dv
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    b_Tt = tl.load(T + base * 4096 + o_ppt)
    b_dU = tl.load(DU_ALL + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_Om = tl.load(OMEGA + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_gam = tl.load(GAM + base * 64 + o_p)
    b_TtdU = tl.dot(b_Tt, b_dU, input_precision="ieee")
    tl.store(DOM + base * 64 * Dv + o_pv, b_gam[:, None] * b_TtdU, mask=m_v[None, :])
    tl.store(DGAM + base * 64 + o_p, tl.sum(b_TtdU * b_Om, 1))


@triton.jit
def gdc_bwd_scan_hist_kernel(
    DOM, W_FULL, CACHE_V, DW_FULL, DCACHE_V,
    i_c, NT, N, Dv, Mtot, M0,
    BDv: tl.constexpr,
):
    # dW^hist block + accumulate dC^hist = W^T dOmega into the cache grads.
    i_bh = tl.program_id(0)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_v = tl.arange(0, BDv)
    m_v = o_v < Dv
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_pvt = o_v[:, None] + o_p[None, :] * Dv
    b_dOm = tl.load(DOM + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    M_k = M0 + i_c * 64
    for m0 in range(0, M_k, 64):
        m_m = (m0 + o_p) < M_k
        b_cvt = tl.load(CACHE_V + (i_bh * Mtot + m0) * Dv + o_pvt,
                        mask=m_v[:, None] & m_m[None, :], other=0.0)
        p_dwh = (DW_FULL + (i_bh * N + i_c * 64) * Mtot
                 + o_p[:, None] * Mtot + m0 + o_p[None, :])
        tl.store(p_dwh, tl.dot(b_dOm, b_cvt, input_precision="ieee"),
                 mask=m_m[None, :])
        p_wht = (W_FULL + (i_bh * N + i_c * 64) * Mtot
                 + o_p[None, :] * Mtot + m0 + o_p[:, None])
        b_wht = tl.load(p_wht, mask=m_m[:, None], other=0.0)
        p_dcv = DCACHE_V + (i_bh * Mtot + m0) * Dv + o_pv
        prev = tl.load(p_dcv, mask=m_m[:, None] & m_v[None, :], other=0.0)
        tl.store(p_dcv, prev + tl.dot(b_wht, b_dOm, input_precision="ieee"),
                 mask=m_m[:, None] & m_v[None, :])


@triton.jit
def gdc_bwd_scan_ds1_kernel(
    GLOC, Q, R, DO, DCACHE_V, DS_STATE,
    i_c, NT, Dk, Dv, Mtot, M0,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # dS^T <- G_last dS^T + Q^T (G dO) + R^T (G dc).
    i_bh = tl.program_id(0)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_pkt = o_k[:, None] + o_p[None, :] * Dk
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_vkt = o_k[:, None] + o_v[None, :] * Dk
    b_g = tl.load(GLOC + base * 64 + o_p)
    b_G = tl.exp(b_g)
    g_last = tl.load(GLOC + base * 64 + 63)
    b_qt = tl.load(Q + base * 64 * Dk + o_pkt, mask=m_k[:, None], other=0.0)
    b_rt = tl.load(R + base * 64 * Dk + o_pkt, mask=m_k[:, None], other=0.0)
    b_do = tl.load(DO + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    M_k = M0 + i_c * 64
    b_dc = tl.load(DCACHE_V + (i_bh * Mtot + M_k) * Dv + o_pv,
                   mask=m_v[None, :], other=0.0)
    b_dSt = tl.load(DS_STATE + i_bh * Dv * Dk + o_vkt,
                    mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_dSt = (tl.exp(g_last) * b_dSt
             + tl.dot(b_qt, b_G[:, None] * b_do, input_precision="ieee")
             + tl.dot(b_rt, b_G[:, None] * b_dc, input_precision="ieee"))
    tl.store(DS_STATE + i_bh * Dv * Dk + o_vkt, b_dSt,
             mask=m_k[:, None] & m_v[None, :])


@triton.jit
def gdc_bwd_scan_ds2_kernel(
    W_MAT, DU_ALL, DS_STATE,
    i_c, NT, Dk, Dv,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # dS^T -= w^T dU.
    i_bh = tl.program_id(0)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_pkt = o_k[:, None] + o_p[None, :] * Dk
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_vkt = o_k[:, None] + o_v[None, :] * Dk
    b_wt = tl.load(W_MAT + base * 64 * Dk + o_pkt, mask=m_k[:, None], other=0.0)
    b_dU = tl.load(DU_ALL + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_dSt = tl.load(DS_STATE + i_bh * Dv * Dk + o_vkt,
                    mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_dSt -= tl.dot(b_wt, b_dU, input_precision="ieee")
    tl.store(DS_STATE + i_bh * Dv * Dk + o_vkt, b_dSt,
             mask=m_k[:, None] & m_v[None, :])


# ---- Stage 2 post-pass (parallel over chunks; dc is final here) ----
@triton.jit
def gdc_bwd_scan_dbh_dt_kernel(
    DCACHE_V, U, K, S_LIST, DS_LIST_BWD,
    DB_HAT, DT_ALL, DG_ACC,
    NT, Dk, Dv, Mtot, M0,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # dB_hat = dc U^T; dt_j = U_j^T dS' K_j; dG_acc[P-1] = <dS', S_hat>.
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_pvt = o_v[:, None] + o_p[None, :] * Dv
    o_vkt = o_k[:, None] + o_v[None, :] * Dk
    M_k = M0 + i_c * 64
    b_dc = tl.load(DCACHE_V + (i_bh * Mtot + M_k) * Dv + o_pv,
                   mask=m_v[None, :], other=0.0)
    b_ut = tl.load(U + base * 64 * Dv + o_pvt, mask=m_v[:, None], other=0.0)
    b_k = tl.load(K + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_dSt = tl.load(DS_LIST_BWD + base * Dv * Dk + o_vkt,
                    mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_Skt = tl.load(S_LIST + base * Dv * Dk + o_vkt,
                    mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_u = tl.load(U + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    tl.store(DB_HAT + base * 4096 + o_pp,
             tl.dot(b_dc, b_ut, input_precision="ieee"))
    tl.store(DT_ALL + base * 64 + o_p,
             tl.sum(b_u * tl.dot(b_k, b_dSt, input_precision="ieee"), 1))
    dgl = tl.sum(b_dSt * b_Skt)
    tl.store(DG_ACC + base * 64 + o_p, tl.where(o_p == 63, dgl, 0.0))


@triton.jit
def gdc_bwd_scan_dgacc_kernel(
    S_LIST, Q, R, DO, DCACHE_V, DG_ACC,
    NT, Dk, Dv, Mtot, M0,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # dG_acc += rowsum(dO * Q S^T) + rowsum(dc * R S^T).
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_vkt = o_k[:, None] + o_v[None, :] * Dk
    b_Skt = tl.load(S_LIST + base * Dv * Dk + o_vkt,
                    mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_q = tl.load(Q + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_r = tl.load(R + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_do = tl.load(DO + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    M_k = M0 + i_c * 64
    b_dc = tl.load(DCACHE_V + (i_bh * Mtot + M_k) * Dv + o_pv,
                   mask=m_v[None, :], other=0.0)
    b_dga = (tl.sum(b_do * tl.dot(b_q, b_Skt, input_precision="ieee"), 1)
             + tl.sum(b_dc * tl.dot(b_r, b_Skt, input_precision="ieee"), 1))
    p_dga = DG_ACC + base * 64 + o_p
    tl.store(p_dga, tl.load(p_dga) + b_dga)


@triton.jit
def gdc_bwd_scan_dqdr_kernel(
    DS_LIST_BWD, U, DO, DCACHE_V, S_LIST, GLOC,
    DK, DQ, DR,
    NT, Dk, Dv, Mtot, M0,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # dK += tail * (U dS') (stored transposed); dQ += G (dO S); dR += G (dc S).
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    o_pkt = o_k[:, None] + o_p[None, :] * Dk
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_pvt = o_v[:, None] + o_p[None, :] * Dv
    o_vk = o_v[:, None] * Dk + o_k[None, :]
    o_vkt = o_k[:, None] + o_v[None, :] * Dk
    b_g = tl.load(GLOC + base * 64 + o_p)
    b_G = tl.exp(b_g)
    g_last = tl.load(GLOC + base * 64 + 63)
    b_tail = tl.exp(g_last - b_g)
    b_dSt = tl.load(DS_LIST_BWD + base * Dv * Dk + o_vkt,
                    mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_ut = tl.load(U + base * 64 * Dv + o_pvt, mask=m_v[:, None], other=0.0)
    b_do = tl.load(DO + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    M_k = M0 + i_c * 64
    b_dc = tl.load(DCACHE_V + (i_bh * Mtot + M_k) * Dv + o_pv,
                   mask=m_v[None, :], other=0.0)
    b_Sk = tl.load(S_LIST + base * Dv * Dk + o_vk,
                   mask=m_v[:, None] & m_k[None, :], other=0.0)
    b_dKt = b_tail[None, :] * tl.dot(b_dSt, b_ut, input_precision="ieee")
    p_dk = DK + base * 64 * Dk + o_pkt
    tl.store(p_dk, tl.load(p_dk, mask=m_k[:, None], other=0.0) + b_dKt,
             mask=m_k[:, None])
    p_dq = DQ + base * 64 * Dk + o_pk
    tl.store(p_dq, tl.load(p_dq, mask=m_k[None, :], other=0.0)
             + b_G[:, None] * tl.dot(b_do, b_Sk, input_precision="ieee"),
             mask=m_k[None, :])
    p_dr = DR + base * 64 * Dk + o_pk
    tl.store(p_dr, tl.load(p_dr, mask=m_k[None, :], other=0.0)
             + b_G[:, None] * tl.dot(b_dc, b_Sk, input_precision="ieee"),
             mask=m_k[None, :])


# ---- Stage 3: WY adjoint micro-kernels (parallel over chunks) ----
@triton.jit
def gdc_bwd_wy_3a_kernel(
    T, DU, LAM, V, S_LIST, DUS, DV, DLAM,
    NT, Dk, Dv,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # TtdU = T^T dU; dV = lam*TtdU; dlam = rowsum(TtdU*V); dUS = dU S.
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_vk = o_v[:, None] * Dk + o_k[None, :]
    b_Tt = tl.load(T + base * 4096 + o_ppt)
    b_dU = tl.load(DU + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_lam = tl.load(LAM + base * 64 + o_p)
    b_v = tl.load(V + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_S = tl.load(S_LIST + base * Dv * Dk + o_vk,
                  mask=m_v[:, None] & m_k[None, :], other=0.0)
    b_TtdU = tl.dot(b_Tt, b_dU, input_precision="ieee")
    tl.store(DV + base * 64 * Dv + o_pv, b_lam[:, None] * b_TtdU, mask=m_v[None, :])
    tl.store(DLAM + base * 64 + o_p, tl.sum(b_TtdU * b_v, 1))
    tl.store(DUS + base * 64 * Dk + o_pk,
             tl.dot(b_dU, b_S, input_precision="ieee"), mask=m_k[None, :])


@triton.jit
def gdc_bwd_wy_3b_kernel(
    DU, LAM, V, GAM, OMEGA, DUS, Z, DT,
    NT, Dk, Dv,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # dT = dU (lam V)^T + dU (gam Omega)^T - dUS Z^T.
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    o_pkt = o_k[:, None] + o_p[None, :] * Dk
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_pvt = o_v[:, None] + o_p[None, :] * Dv
    b_dU = tl.load(DU + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_lam = tl.load(LAM + base * 64 + o_p)
    b_gam = tl.load(GAM + base * 64 + o_p)
    b_vt = tl.load(V + base * 64 * Dv + o_pvt, mask=m_v[:, None], other=0.0)
    b_Omt = tl.load(OMEGA + base * 64 * Dv + o_pvt, mask=m_v[:, None], other=0.0)
    b_dUS = tl.load(DUS + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_Zt = tl.load(Z + base * 64 * Dk + o_pkt, mask=m_k[:, None], other=0.0)
    b_dT = (tl.dot(b_dU, b_lam[None, :] * b_vt, input_precision="ieee")
            + tl.dot(b_dU, b_gam[None, :] * b_Omt, input_precision="ieee")
            - tl.dot(b_dUS, b_Zt, input_precision="ieee"))
    tl.store(DT + base * 4096 + o_pp, b_dT)


@triton.jit
def gdc_bwd_wy_3c_kernel(
    T, DUS, K, LAM, GLOC, DZ, DLAM, DK, DG_ACC,
    NT, Dk,
    BDk: tl.constexpr,
):
    # dZ = -T^T dUS; dK += (lam*G) dZ; dlam += G e; dG_acc += lam e.
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    m_k = o_k < Dk
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    b_Tt = tl.load(T + base * 4096 + o_ppt)
    b_dUS = tl.load(DUS + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_k = tl.load(K + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_lam = tl.load(LAM + base * 64 + o_p)
    b_G = tl.exp(tl.load(GLOC + base * 64 + o_p))
    b_dZ = -tl.dot(b_Tt, b_dUS, input_precision="ieee")
    tl.store(DZ + base * 64 * Dk + o_pk, b_dZ, mask=m_k[None, :])
    e = tl.sum(b_dZ * b_k, 1)
    p_dk = DK + base * 64 * Dk + o_pk
    tl.store(p_dk, tl.load(p_dk, mask=m_k[None, :], other=0.0)
             + (b_lam * b_G)[:, None] * b_dZ, mask=m_k[None, :])
    p_dlam = DLAM + base * 64 + o_p
    tl.store(p_dlam, tl.load(p_dlam) + b_G * e)
    p_dga = DG_ACC + base * 64 + o_p
    tl.store(p_dga, tl.load(p_dga) + b_lam * e)


@triton.jit
def gdc_bwd_wy_3d_kernel(
    W_LOC, R, GLOC, DZ, GAM, DY, DGAM,
    NT, Dk,
    BDk: tl.constexpr,
):
    # Y = W^loc (G R); dY = -gam dZ; dgam -= rowsum(dZ * Y).
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    m_k = o_k < Dk
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    b_Wl = tl.load(W_LOC + base * 4096 + o_pp)
    b_r = tl.load(R + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_G = tl.exp(tl.load(GLOC + base * 64 + o_p))
    b_dZ = tl.load(DZ + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_gam = tl.load(GAM + base * 64 + o_p)
    b_Y = tl.dot(b_Wl, b_G[:, None] * b_r, input_precision="ieee")
    tl.store(DY + base * 64 * Dk + o_pk, -b_gam[:, None] * b_dZ, mask=m_k[None, :])
    p_dgam = DGAM + base * 64 + o_p
    tl.store(p_dgam, tl.load(p_dgam) - tl.sum(b_dZ * b_Y, 1))


@triton.jit
def gdc_bwd_wy_3e_kernel(
    DY, R, GLOC, W_LOC, DW_FULL, DR, DG_ACC,
    NT, N, Dk, Mtot, M0,
    BDk: tl.constexpr,
):
    # dW^loc(part 1) = dY (G R)^T -> local block; dR += G (W^T dY); dG_acc.
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    m_k = o_k < Dk
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    o_pkt = o_k[:, None] + o_p[None, :] * Dk
    b_dY = tl.load(DY + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_r = tl.load(R + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_rt = tl.load(R + base * 64 * Dk + o_pkt, mask=m_k[:, None], other=0.0)
    b_G = tl.exp(tl.load(GLOC + base * 64 + o_p))
    b_Wlt = tl.load(W_LOC + base * 4096 + o_ppt)
    b_dWl = tl.dot(b_dY, b_G[None, :] * b_rt, input_precision="ieee")
    p_dwl = (DW_FULL + (i_bh * N + i_c * 64) * Mtot + o_p[:, None] * Mtot
             + (M0 + i_c * 64) + o_p[None, :])
    tl.store(p_dwl, b_dWl)
    b_WldY = tl.dot(b_Wlt, b_dY, input_precision="ieee")
    p_dr = DR + base * 64 * Dk + o_pk
    tl.store(p_dr, tl.load(p_dr, mask=m_k[None, :], other=0.0)
             + b_G[:, None] * b_WldY, mask=m_k[None, :])
    p_dga = DG_ACC + base * 64 + o_p
    tl.store(p_dga, tl.load(p_dga) + tl.sum(b_r * b_WldY, 1))


@triton.jit
def gdc_bwd_wy_3f_kernel(
    T, DT, DX, DXT,
    NT,
):
    # dX = -T^T dT T^T and its transpose.
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    b_T = tl.load(T + base * 4096 + o_pp)
    b_Tt = tl.load(T + base * 4096 + o_ppt)
    b_dT = tl.load(DT + base * 4096 + o_pp)
    b_dTt = tl.load(DT + base * 4096 + o_ppt)
    b_dX = -tl.dot(b_Tt, tl.dot(b_dT, b_Tt, input_precision="ieee"),
                   input_precision="ieee")
    b_dXt = -tl.dot(b_T, tl.dot(b_dTt, b_T, input_precision="ieee"),
                    input_precision="ieee")
    tl.store(DX + base * 4096 + o_pp, b_dX)
    tl.store(DXT + base * 4096 + o_pp, b_dXt)


@triton.jit
def gdc_bwd_wy_3g_kernel(
    DX, A_HAT, LAM, W_LOC, B_HAT, DLAM, DGAM,
    NT,
):
    # dlam += rowsum(dX * A_hat); dgam -= rowsum(W^loc * (dX B^T)).
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    b_dX = tl.load(DX + base * 4096 + o_pp)
    b_Ah = tl.load(A_HAT + base * 4096 + o_pp)
    b_Wl = tl.load(W_LOC + base * 4096 + o_pp)
    b_Bt = tl.load(B_HAT + base * 4096 + o_ppt)
    p_dlam = DLAM + base * 64 + o_p
    tl.store(p_dlam, tl.load(p_dlam) + tl.sum(b_dX * b_Ah, 1))
    b_t = tl.dot(b_dX, b_Bt, input_precision="ieee")
    p_dgam = DGAM + base * 64 + o_p
    tl.store(p_dgam, tl.load(p_dgam) - tl.sum(b_Wl * b_t, 1))


@triton.jit
def gdc_bwd_wy_3h_kernel(
    DX, GAM, B_HAT, DW_FULL,
    NT, N, Mtot, M0,
):
    # dW^loc(part 2) = (-gam dX) B^T, accumulated into the local block.
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    b_dX = tl.load(DX + base * 4096 + o_pp)
    b_gam = tl.load(GAM + base * 64 + o_p)
    b_Bt = tl.load(B_HAT + base * 4096 + o_ppt)
    b_dWl2 = tl.dot(-b_gam[:, None] * b_dX, b_Bt, input_precision="ieee")
    p_dwl = (DW_FULL + (i_bh * N + i_c * 64) * Mtot + o_p[:, None] * Mtot
             + (M0 + i_c * 64) + o_p[None, :])
    tl.store(p_dwl, tl.load(p_dwl) + b_dWl2)


@triton.jit
def gdc_bwd_wy_3i_kernel(
    DX, DXT, GAM, W_LOC, DB_IN, DBT, DBTT,
    NT,
):
    # dB_total = dB_in + W^T dH and its transpose, dH = -gam dX.
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    b_dX = tl.load(DX + base * 4096 + o_pp)
    b_dXt = tl.load(DXT + base * 4096 + o_pp)
    b_gam = tl.load(GAM + base * 64 + o_p)
    b_Wl = tl.load(W_LOC + base * 4096 + o_pp)
    b_Wlt = tl.load(W_LOC + base * 4096 + o_ppt)
    b_dB = tl.load(DB_IN + base * 4096 + o_pp)
    b_dBt_in = tl.load(DB_IN + base * 4096 + o_ppt)
    b_dH = -b_gam[:, None] * b_dX
    b_dHt = -b_gam[None, :] * b_dXt
    tl.store(DBT + base * 4096 + o_pp,
             b_dB + tl.dot(b_Wlt, b_dH, input_precision="ieee"))
    tl.store(DBTT + base * 4096 + o_pp,
             b_dBt_in + tl.dot(b_dHt, b_Wl, input_precision="ieee"))


@triton.jit
def gdc_bwd_wy_3j1_kernel(
    DX, DXT, LAM, DEC_S, K, DK,
    NT, Dk,
    BDk: tl.constexpr,
):
    # dK += (decay*lam*dX) K + (decay*lam*dX)^T K  (A-gram adjoint).
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    m_k = o_k < Dk
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    b_dX = tl.load(DX + base * 4096 + o_pp)
    b_dXt = tl.load(DXT + base * 4096 + o_pp)
    b_lam = tl.load(LAM + base * 64 + o_p)
    b_ds = tl.load(DEC_S + base * 4096 + o_pp)
    b_dst = tl.load(DEC_S + base * 4096 + o_ppt)
    b_k = tl.load(K + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_acc = (tl.dot(b_ds * (b_lam[:, None] * b_dX), b_k, input_precision="ieee")
             + tl.dot(b_dst * (b_lam[None, :] * b_dXt), b_k, input_precision="ieee"))
    p_dk = DK + base * 64 * Dk + o_pk
    tl.store(p_dk, tl.load(p_dk, mask=m_k[None, :], other=0.0) + b_acc,
             mask=m_k[None, :])


@triton.jit
def gdc_bwd_wy_3j2_kernel(
    DEC_I, DBT, DBTT, K, R, DK, DR,
    NT, Dk,
    BDk: tl.constexpr,
):
    # dR += (decay*dBt) K; dK += (decay*dBt)^T R  (B-gram adjoint).
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    m_k = o_k < Dk
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    b_di = tl.load(DEC_I + base * 4096 + o_pp)
    b_dit = tl.load(DEC_I + base * 4096 + o_ppt)
    b_dBt = tl.load(DBT + base * 4096 + o_pp)
    b_dBtt = tl.load(DBTT + base * 4096 + o_pp)
    b_k = tl.load(K + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_r = tl.load(R + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    p_dr = DR + base * 64 * Dk + o_pk
    tl.store(p_dr, tl.load(p_dr, mask=m_k[None, :], other=0.0)
             + tl.dot(b_di * b_dBt, b_k, input_precision="ieee"),
             mask=m_k[None, :])
    p_dk = DK + base * 64 * Dk + o_pk
    tl.store(p_dk, tl.load(p_dk, mask=m_k[None, :], other=0.0)
             + tl.dot(b_dit * b_dBtt, b_r, input_precision="ieee"),
             mask=m_k[None, :])


@triton.jit
def gdc_bwd_wy_3k_kernel(
    A_HAT, DX, DXT, LAM, B_HAT, DBT, DBTT, DG_LOC,
    NT,
):
    # dg_loc = rowsum(E_a) - colsum(E_a) + rowsum(E_b) - colsum(E_b),
    # elementwise only (column sums read off the transposed loads).
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    b_Ah = tl.load(A_HAT + base * 4096 + o_pp)
    b_Aht = tl.load(A_HAT + base * 4096 + o_ppt)
    b_dX = tl.load(DX + base * 4096 + o_pp)
    b_dXt = tl.load(DXT + base * 4096 + o_pp)
    b_lam = tl.load(LAM + base * 64 + o_p)
    b_B = tl.load(B_HAT + base * 4096 + o_pp)
    b_Bt = tl.load(B_HAT + base * 4096 + o_ppt)
    b_dBt = tl.load(DBT + base * 4096 + o_pp)
    b_dBtt = tl.load(DBTT + base * 4096 + o_pp)
    e_a_r = tl.sum(b_Ah * (b_lam[:, None] * b_dX), 1)
    e_a_c = tl.sum(b_Aht * (b_lam[None, :] * b_dXt), 1)
    e_b_r = tl.sum(b_B * b_dBt, 1)
    e_b_c = tl.sum(b_Bt * b_dBtt, 1)
    tl.store(DG_LOC + base * 64 + o_p, e_a_r - e_a_c + e_b_r - e_b_c)


# ---------------------------------------------------------------------------
# Datacenter variants (A100/H100: ~160KB+ smem per block). Same math as the
# micro-kernels above, fused to cut kernel launches and global round-trips.
# Selected at launch time when the device's shared memory budget allows.
# ---------------------------------------------------------------------------
@triton.jit
def gdc_bwd_scan_du_dom_kernel(
    GLOC, K, DU1, B_HAT, DCACHE_V, DS_STATE, T, GAM, OMEGA,
    DU_ALL, DS_LIST_BWD, DOM, DGAM,
    i_c, NT, Dk, Dv, Mtot, M0,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # scan_du + scan_dom fused: dU + dS^T checkpoint, then dOmega/dgam.
    i_bh = tl.program_id(0)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    o_vkt = o_k[:, None] + o_v[None, :] * Dk
    b_g = tl.load(GLOC + base * 64 + o_p)
    g_last = tl.load(GLOC + base * 64 + 63)
    b_tail = tl.exp(g_last - b_g)
    b_du1 = tl.load(DU1 + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_Bt = tl.load(B_HAT + base * 4096 + o_ppt)
    M_k = M0 + i_c * 64
    b_dc = tl.load(DCACHE_V + (i_bh * Mtot + M_k) * Dv + o_pv,
                   mask=m_v[None, :], other=0.0)
    b_k = tl.load(K + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_dSt = tl.load(DS_STATE + i_bh * Dv * Dk + o_vkt,
                    mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_dU = (b_du1
            + tl.dot(b_Bt, b_dc, input_precision="ieee")
            + tl.dot(b_tail[:, None] * b_k, b_dSt, input_precision="ieee"))
    tl.store(DU_ALL + base * 64 * Dv + o_pv, b_dU, mask=m_v[None, :])
    tl.store(DS_LIST_BWD + base * Dv * Dk + o_vkt, b_dSt,
             mask=m_k[:, None] & m_v[None, :])
    b_Tt = tl.load(T + base * 4096 + o_ppt)
    b_Om = tl.load(OMEGA + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_gam = tl.load(GAM + base * 64 + o_p)
    b_TtdU = tl.dot(b_Tt, b_dU, input_precision="ieee")
    tl.store(DOM + base * 64 * Dv + o_pv, b_gam[:, None] * b_TtdU, mask=m_v[None, :])
    tl.store(DGAM + base * 64 + o_p, tl.sum(b_TtdU * b_Om, 1))


@triton.jit
def gdc_bwd_scan_ds12_kernel(
    GLOC, Q, R, DO, DCACHE_V, W_MAT, DU_ALL, DS_STATE,
    i_c, NT, Dk, Dv, Mtot, M0,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # scan_ds1 + scan_ds2 fused: one dS^T state update per chunk.
    i_bh = tl.program_id(0)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_pkt = o_k[:, None] + o_p[None, :] * Dk
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_vkt = o_k[:, None] + o_v[None, :] * Dk
    b_g = tl.load(GLOC + base * 64 + o_p)
    b_G = tl.exp(b_g)
    g_last = tl.load(GLOC + base * 64 + 63)
    b_qt = tl.load(Q + base * 64 * Dk + o_pkt, mask=m_k[:, None], other=0.0)
    b_rt = tl.load(R + base * 64 * Dk + o_pkt, mask=m_k[:, None], other=0.0)
    b_do = tl.load(DO + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    M_k = M0 + i_c * 64
    b_dc = tl.load(DCACHE_V + (i_bh * Mtot + M_k) * Dv + o_pv,
                   mask=m_v[None, :], other=0.0)
    b_wt = tl.load(W_MAT + base * 64 * Dk + o_pkt, mask=m_k[:, None], other=0.0)
    b_dU = tl.load(DU_ALL + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_dSt = tl.load(DS_STATE + i_bh * Dv * Dk + o_vkt,
                    mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_dSt = (tl.exp(g_last) * b_dSt
             + tl.dot(b_qt, b_G[:, None] * b_do, input_precision="ieee")
             + tl.dot(b_rt, b_G[:, None] * b_dc, input_precision="ieee")
             - tl.dot(b_wt, b_dU, input_precision="ieee"))
    tl.store(DS_STATE + i_bh * Dv * Dk + o_vkt, b_dSt,
             mask=m_k[:, None] & m_v[None, :])


@triton.jit
def gdc_bwd_scan_dga_dqdr_kernel(
    S_LIST, Q, R, DO, DCACHE_V, DS_LIST_BWD, U, GLOC,
    DG_ACC, DK, DQ, DR,
    NT, Dk, Dv, Mtot, M0,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # scan_dgacc + scan_dqdr fused.
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    o_pkt = o_k[:, None] + o_p[None, :] * Dk
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_pvt = o_v[:, None] + o_p[None, :] * Dv
    o_vk = o_v[:, None] * Dk + o_k[None, :]
    o_vkt = o_k[:, None] + o_v[None, :] * Dk
    b_g = tl.load(GLOC + base * 64 + o_p)
    b_G = tl.exp(b_g)
    g_last = tl.load(GLOC + base * 64 + 63)
    b_tail = tl.exp(g_last - b_g)
    b_Skt = tl.load(S_LIST + base * Dv * Dk + o_vkt,
                    mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_Sk = tl.load(S_LIST + base * Dv * Dk + o_vk,
                   mask=m_v[:, None] & m_k[None, :], other=0.0)
    b_q = tl.load(Q + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_r = tl.load(R + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_do = tl.load(DO + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    M_k = M0 + i_c * 64
    b_dc = tl.load(DCACHE_V + (i_bh * Mtot + M_k) * Dv + o_pv,
                   mask=m_v[None, :], other=0.0)
    b_dga = (tl.sum(b_do * tl.dot(b_q, b_Skt, input_precision="ieee"), 1)
             + tl.sum(b_dc * tl.dot(b_r, b_Skt, input_precision="ieee"), 1))
    p_dga = DG_ACC + base * 64 + o_p
    tl.store(p_dga, tl.load(p_dga) + b_dga)
    b_dSt = tl.load(DS_LIST_BWD + base * Dv * Dk + o_vkt,
                    mask=m_k[:, None] & m_v[None, :], other=0.0)
    b_ut = tl.load(U + base * 64 * Dv + o_pvt, mask=m_v[:, None], other=0.0)
    b_dKt = b_tail[None, :] * tl.dot(b_dSt, b_ut, input_precision="ieee")
    p_dk = DK + base * 64 * Dk + o_pkt
    tl.store(p_dk, tl.load(p_dk, mask=m_k[:, None], other=0.0) + b_dKt,
             mask=m_k[:, None])
    p_dq = DQ + base * 64 * Dk + o_pk
    tl.store(p_dq, tl.load(p_dq, mask=m_k[None, :], other=0.0)
             + b_G[:, None] * tl.dot(b_do, b_Sk, input_precision="ieee"),
             mask=m_k[None, :])
    p_dr = DR + base * 64 * Dk + o_pk
    tl.store(p_dr, tl.load(p_dr, mask=m_k[None, :], other=0.0)
             + b_G[:, None] * tl.dot(b_dc, b_Sk, input_precision="ieee"),
             mask=m_k[None, :])


@triton.jit
def gdc_bwd_wy_a_fused_kernel(
    T, DU, LAM, V, GAM, OMEGA, S_LIST, Z, GLOC, K, R, W_LOC,
    DT, DV, DLAM, DGAM, DK, DR, DG_ACC, DW_FULL,
    NT, N, Dk, Dv, Mtot, M0,
    BDk: tl.constexpr, BDv: tl.constexpr,
):
    # WY adjoint part A (micro-kernels 3a..3e fused).
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    o_v = tl.arange(0, BDv)
    m_k = o_k < Dk
    m_v = o_v < Dv
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    o_pkt = o_k[:, None] + o_p[None, :] * Dk
    o_pv = o_p[:, None] * Dv + o_v[None, :]
    o_pvt = o_v[:, None] + o_p[None, :] * Dv
    o_vk = o_v[:, None] * Dk + o_k[None, :]
    b_Tt = tl.load(T + base * 4096 + o_ppt)
    b_dU = tl.load(DU + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_lam = tl.load(LAM + base * 64 + o_p)
    b_gam = tl.load(GAM + base * 64 + o_p)
    b_v = tl.load(V + base * 64 * Dv + o_pv, mask=m_v[None, :], other=0.0)
    b_vt = tl.load(V + base * 64 * Dv + o_pvt, mask=m_v[:, None], other=0.0)
    b_Omt = tl.load(OMEGA + base * 64 * Dv + o_pvt, mask=m_v[:, None], other=0.0)
    b_S = tl.load(S_LIST + base * Dv * Dk + o_vk,
                  mask=m_v[:, None] & m_k[None, :], other=0.0)
    b_Zt = tl.load(Z + base * 64 * Dk + o_pkt, mask=m_k[:, None], other=0.0)
    b_G = tl.exp(tl.load(GLOC + base * 64 + o_p))
    b_k = tl.load(K + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_r = tl.load(R + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_rt = tl.load(R + base * 64 * Dk + o_pkt, mask=m_k[:, None], other=0.0)
    b_Wl = tl.load(W_LOC + base * 4096 + o_pp)
    b_Wlt = tl.load(W_LOC + base * 4096 + o_ppt)
    # 3a
    b_TtdU = tl.dot(b_Tt, b_dU, input_precision="ieee")
    tl.store(DV + base * 64 * Dv + o_pv, b_lam[:, None] * b_TtdU, mask=m_v[None, :])
    b_dlam = tl.sum(b_TtdU * b_v, 1)
    b_dUS = tl.dot(b_dU, b_S, input_precision="ieee")
    # 3b
    b_dT = (tl.dot(b_dU, b_lam[None, :] * b_vt, input_precision="ieee")
            + tl.dot(b_dU, b_gam[None, :] * b_Omt, input_precision="ieee")
            - tl.dot(b_dUS, b_Zt, input_precision="ieee"))
    tl.store(DT + base * 4096 + o_pp, b_dT)
    # 3c
    b_dZ = -tl.dot(b_Tt, b_dUS, input_precision="ieee")
    e = tl.sum(b_dZ * b_k, 1)
    p_dk = DK + base * 64 * Dk + o_pk
    tl.store(p_dk, tl.load(p_dk, mask=m_k[None, :], other=0.0)
             + (b_lam * b_G)[:, None] * b_dZ, mask=m_k[None, :])
    b_dlam += b_G * e
    b_dga = b_lam * e
    # 3d
    b_Y = tl.dot(b_Wl, b_G[:, None] * b_r, input_precision="ieee")
    b_dY = -b_gam[:, None] * b_dZ
    p_dgam = DGAM + base * 64 + o_p
    tl.store(p_dgam, tl.load(p_dgam) - tl.sum(b_dZ * b_Y, 1))
    # 3e
    p_dwl = (DW_FULL + (i_bh * N + i_c * 64) * Mtot + o_p[:, None] * Mtot
             + (M0 + i_c * 64) + o_p[None, :])
    tl.store(p_dwl, tl.dot(b_dY, b_G[None, :] * b_rt, input_precision="ieee"))
    b_WldY = tl.dot(b_Wlt, b_dY, input_precision="ieee")
    p_dr = DR + base * 64 * Dk + o_pk
    tl.store(p_dr, tl.load(p_dr, mask=m_k[None, :], other=0.0)
             + b_G[:, None] * b_WldY, mask=m_k[None, :])
    b_dga += tl.sum(b_r * b_WldY, 1)
    p_dga = DG_ACC + base * 64 + o_p
    tl.store(p_dga, tl.load(p_dga) + b_dga)
    tl.store(DLAM + base * 64 + o_p, b_dlam)


@triton.jit
def gdc_bwd_wy_b1_kernel(
    T, DT, A_HAT, LAM, GAM, W_LOC, B_HAT,
    DX, DXT, DLAM, DGAM, DW_FULL,
    NT, N, Mtot, M0,
):
    # WY adjoint part B1 (3f+3g+3h): dX/dX^T and their direct contributions.
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    b_T = tl.load(T + base * 4096 + o_pp)
    b_Tt = tl.load(T + base * 4096 + o_ppt)
    b_dT = tl.load(DT + base * 4096 + o_pp)
    b_dTt = tl.load(DT + base * 4096 + o_ppt)
    b_dX = -tl.dot(b_Tt, tl.dot(b_dT, b_Tt, input_precision="ieee"),
                   input_precision="ieee")
    b_dXt = -tl.dot(b_T, tl.dot(b_dTt, b_T, input_precision="ieee"),
                    input_precision="ieee")
    tl.store(DX + base * 4096 + o_pp, b_dX)
    tl.store(DXT + base * 4096 + o_pp, b_dXt)
    b_Ah = tl.load(A_HAT + base * 4096 + o_pp)
    p_dlam = DLAM + base * 64 + o_p
    tl.store(p_dlam, tl.load(p_dlam) + tl.sum(b_dX * b_Ah, 1))
    b_Wl = tl.load(W_LOC + base * 4096 + o_pp)
    b_Bt = tl.load(B_HAT + base * 4096 + o_ppt)
    p_dgam = DGAM + base * 64 + o_p
    tl.store(p_dgam, tl.load(p_dgam)
             - tl.sum(b_Wl * tl.dot(b_dX, b_Bt, input_precision="ieee"), 1))
    b_gam = tl.load(GAM + base * 64 + o_p)
    p_dwl = (DW_FULL + (i_bh * N + i_c * 64) * Mtot + o_p[:, None] * Mtot
             + (M0 + i_c * 64) + o_p[None, :])
    tl.store(p_dwl, tl.load(p_dwl)
             + tl.dot(-b_gam[:, None] * b_dX, b_Bt, input_precision="ieee"))


@triton.jit
def gdc_bwd_wy_b2_kernel(
    DX, DXT, GAM, W_LOC, DB_IN, LAM, DEC_S, DEC_I, K, R, A_HAT, B_HAT,
    DK, DR, DG_LOC,
    NT, Dk,
    BDk: tl.constexpr,
):
    # WY adjoint part B2 (3i+3j1+3j2+3k): dB_total, gram adjoints, dg_loc.
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    m_k = o_k < Dk
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    o_ppt = o_p[:, None] + o_p[None, :] * 64
    o_pk = o_p[:, None] * Dk + o_k[None, :]
    b_dX = tl.load(DX + base * 4096 + o_pp)
    b_dXt = tl.load(DXT + base * 4096 + o_pp)
    b_gam = tl.load(GAM + base * 64 + o_p)
    b_Wl = tl.load(W_LOC + base * 4096 + o_pp)
    b_Wlt = tl.load(W_LOC + base * 4096 + o_ppt)
    b_dB = tl.load(DB_IN + base * 4096 + o_pp)
    b_dBt_in = tl.load(DB_IN + base * 4096 + o_ppt)
    b_dH = -b_gam[:, None] * b_dX
    b_dHt = -b_gam[None, :] * b_dXt
    b_dBt = b_dB + tl.dot(b_Wlt, b_dH, input_precision="ieee")
    b_dBtt = b_dBt_in + tl.dot(b_dHt, b_Wl, input_precision="ieee")
    b_lam = tl.load(LAM + base * 64 + o_p)
    b_ds = tl.load(DEC_S + base * 4096 + o_pp)
    b_dst = tl.load(DEC_S + base * 4096 + o_ppt)
    b_k = tl.load(K + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    b_r = tl.load(R + base * 64 * Dk + o_pk, mask=m_k[None, :], other=0.0)
    p_dk = DK + base * 64 * Dk + o_pk
    b_dk = tl.load(p_dk, mask=m_k[None, :], other=0.0)
    b_dk += (tl.dot(b_ds * (b_lam[:, None] * b_dX), b_k, input_precision="ieee")
             + tl.dot(b_dst * (b_lam[None, :] * b_dXt), b_k, input_precision="ieee"))
    b_di = tl.load(DEC_I + base * 4096 + o_pp)
    b_dit = tl.load(DEC_I + base * 4096 + o_ppt)
    p_dr = DR + base * 64 * Dk + o_pk
    tl.store(p_dr, tl.load(p_dr, mask=m_k[None, :], other=0.0)
             + tl.dot(b_di * b_dBt, b_k, input_precision="ieee"),
             mask=m_k[None, :])
    b_dk += tl.dot(b_dit * b_dBtt, b_r, input_precision="ieee")
    tl.store(p_dk, b_dk, mask=m_k[None, :])
    b_Ah = tl.load(A_HAT + base * 4096 + o_pp)
    b_Aht = tl.load(A_HAT + base * 4096 + o_ppt)
    b_B = tl.load(B_HAT + base * 4096 + o_pp)
    b_Bt = tl.load(B_HAT + base * 4096 + o_ppt)
    e_a_r = tl.sum(b_Ah * (b_lam[:, None] * b_dX), 1)
    e_a_c = tl.sum(b_Aht * (b_lam[None, :] * b_dXt), 1)
    e_b_r = tl.sum(b_B * b_dBt, 1)
    e_b_c = tl.sum(b_Bt * b_dBtt, 1)
    tl.store(DG_LOC + base * 64 + o_p, e_a_r - e_a_c + e_b_r - e_b_c)


@triton.jit
def gdc_bwd_rowsum_kernel(
    W, DW, RS,
    NT, N, Mtot,
):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    o_p = tl.arange(0, 64)
    row0 = i_c * 64
    acc = tl.zeros([64], dtype=tl.float32)
    for m0 in range(0, Mtot, 64):
        m_m = (m0 + o_p) < Mtot
        p = (i_bh * N + row0) * Mtot + o_p[:, None] * Mtot + m0 + o_p[None, :]
        b_w = tl.load(W + p, mask=m_m[None, :], other=0.0)
        b_dw = tl.load(DW + p, mask=m_m[None, :], other=0.0)
        acc += tl.sum(b_w * b_dw, 1)
    tl.store(RS + i_bh * N + row0 + o_p, acc)


@triton.jit
def gdc_bwd_dq_kernel(
    W, DW, RS, K_ALL, DEC_I, DAPR, RG,
    DQ, DG_LOC,
    NT, N, Dk, Mtot, M0, sm_scale,
    BDk: tl.constexpr,
):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    m_k = o_k < Dk
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    row0 = i_c * 64
    b_rs = tl.load(RS + i_bh * N + row0 + o_p)
    b_dec = tl.load(DEC_I + base * 4096 + o_pp)
    b_apr = tl.load(DAPR + base * 4096 + o_pp)
    acc = tl.zeros([64, BDk], dtype=tl.float32)
    for m0 in range(0, Mtot, 64):
        m_m = (m0 + o_p) < Mtot
        p = (i_bh * N + row0) * Mtot + o_p[:, None] * Mtot + m0 + o_p[None, :]
        b_w = tl.load(W + p, mask=m_m[None, :], other=0.0)
        b_dw = tl.load(DW + p, mask=m_m[None, :], other=0.0)
        b_dC = sm_scale * b_w * (b_dw - b_rs[:, None])
        is_loc = (m0 == M0 + row0).to(tl.float32)
        b_dC += is_loc * b_dec * b_apr
        p_ka = (i_bh * Mtot + m0) * Dk + o_p[:, None] * Dk + o_k[None, :]
        b_ka = tl.load(K_ALL + p_ka, mask=m_m[:, None] & m_k[None, :], other=0.0)
        acc += tl.dot(b_dC, b_ka, input_precision="ieee")
    p_dq = DQ + base * 64 * Dk + o_p[:, None] * Dk + o_k[None, :]
    tl.store(p_dq, tl.load(p_dq, mask=m_k[None, :], other=0.0) + acc,
             mask=m_k[None, :])
    b_rg = tl.load(RG + base * 4096 + o_pp)
    e = b_rg * b_apr
    p_dg = DG_LOC + base * 64 + o_p
    tl.store(p_dg, tl.load(p_dg) + tl.sum(e, 1) - tl.sum(e, 0))


@triton.jit
def gdc_bwd_dk_kernel(
    W, DW, RS, Q, DEC_I, DAPR,
    DK,
    NT, N, Dk, Mtot, M0, sm_scale,
    BDk: tl.constexpr,
):
    i_j = tl.program_id(0)
    i_bh = tl.program_id(1)
    o_p = tl.arange(0, 64)
    o_k = tl.arange(0, BDk)
    m_k = o_k < Dk
    o_pp = o_p[:, None] * 64 + o_p[None, :]
    m0 = M0 + i_j * 64
    acc = tl.zeros([64, BDk], dtype=tl.float32)
    for i_c in range(i_j, NT):
        base = i_bh * NT + i_c
        row0 = i_c * 64
        b_rs = tl.load(RS + i_bh * N + row0 + o_p)
        p = (i_bh * N + row0) * Mtot + o_p[:, None] * Mtot + m0 + o_p[None, :]
        b_w = tl.load(W + p)
        b_dw = tl.load(DW + p)
        b_dC = sm_scale * b_w * (b_dw - b_rs[:, None])
        is_loc = (i_c == i_j).to(tl.float32)
        b_dec = tl.load(DEC_I + base * 4096 + o_pp)
        b_apr = tl.load(DAPR + base * 4096 + o_pp)
        b_dC += is_loc * b_dec * b_apr
        b_q = tl.load(Q + (i_bh * N + row0) * Dk + o_p[:, None] * Dk + o_k[None, :],
                      mask=m_k[None, :], other=0.0)
        acc += tl.dot(tl.trans(b_dC), b_q, input_precision="ieee")
    p_dk = DK + (i_bh * N + i_j * 64) * Dk + o_p[:, None] * Dk + o_k[None, :]
    tl.store(p_dk, tl.load(p_dk, mask=m_k[None, :], other=0.0) + acc,
             mask=m_k[None, :])


@triton.jit
def gdc_bwd_gate_cumsum_kernel(
    DG, DG_ACC, DT_ALL, GLOC,
    NT,
):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    base = i_bh * NT + i_c
    o_p = tl.arange(0, 64)
    b_g = tl.load(GLOC + base * 64 + o_p)
    b_dg = tl.load(DG + base * 64 + o_p) + tl.exp(b_g) * tl.load(DG_ACC + base * 64 + o_p)
    b_dt = tl.load(DT_ALL + base * 64 + o_p)
    g_last = tl.load(GLOC + base * 64 + 63)
    contrib = tl.exp(g_last - b_g) * b_dt
    b_dg += tl.where(o_p == 63, tl.sum(contrib), 0.0) - contrib
    tl.store(DG + base * 64 + o_p, tl.cumsum(b_dg, 0, reverse=True))


# ---- Torch stage fallbacks (autograd-verified golden reference) ----
def _bwd_s1_torch(fw, bufs, dO_c):
    bufs["dU1"] = torch.matmul(fw["readout_gram"].transpose(-1, -2), dO_c)
    bufs["dApr"] = torch.matmul(dO_c, fw["U_all"].transpose(-1, -2))


def _bwd_s2_torch(fw, bufs, dO_c, dht):
    B, N, H, NT, P, Dk, Dv, M0, _ = fw["meta"]
    g_loc, G = fw["g_loc"], fw["G"]
    q_c, k_c, r_c = fw["q_c"], fw["k_c"], fw["r_c"]
    gam_c = fw["gam_c"]
    B_hat, T_inv, w = fw["B_hat"], fw["T_inv"], fw["w"]
    U_all, S_list, Omega_all = fw["U_all"], fw["S_list"], fw["Omega_all"]
    W_full, cache_v = fw["W_full"], fw["cache_v"]
    if dht is not None:
        dS_next = dht
    else:
        dS_next = torch.zeros(B, H, Dv, Dk, dtype=torch.float32, device=dO_c.device)
    for kc_ in reversed(range(NT)):
        S_k = S_list[:, :, kc_]
        U_k = U_all[:, :, kc_]
        G_k = G[:, :, kc_]
        M_k = M0 + kc_ * P
        tail = torch.exp(g_loc[:, :, kc_, -1:] - g_loc[:, :, kc_])
        dc = bufs["dcache_v"][:, :, M_k:M_k + P]

        dU = (
            bufs["dU1"][:, :, kc_]
            + torch.matmul(B_hat[:, :, kc_].transpose(-1, -2), dc)
            + torch.matmul(tail[..., None] * k_c[:, :, kc_], dS_next.transpose(-1, -2))
        )
        bufs["dB_hat"][:, :, kc_] = torch.matmul(dc, U_k.transpose(-1, -2))

        TtdU = torch.matmul(T_inv[:, :, kc_].transpose(-1, -2), dU)
        dOmega = gam_c[:, :, kc_][..., None] * TtdU
        bufs["dgam"][:, :, kc_] += (TtdU * Omega_all[:, :, kc_]).sum(-1)
        if M_k > 0:
            cv_hist = cache_v[:, :, :M_k]
            bufs["dW_full"][:, :, kc_ * P:(kc_ + 1) * P, :M_k] = torch.matmul(
                dOmega, cv_hist.transpose(-1, -2))
            bufs["dcache_v"][:, :, :M_k] += torch.matmul(
                W_full[:, :, kc_ * P:(kc_ + 1) * P, :M_k].transpose(-1, -2), dOmega)

        dS = (
            G_k[..., -1][..., None, None] * dS_next
            + torch.matmul((G_k[..., None] * dO_c[:, :, kc_]).transpose(-1, -2), q_c[:, :, kc_])
            + torch.matmul((G_k[..., None] * dc).transpose(-1, -2), r_c[:, :, kc_])
            - torch.matmul(dU.transpose(-1, -2), w[:, :, kc_])
        )
        bufs["dK"][:, :, kc_] += tail[..., None] * torch.matmul(U_k, dS_next)
        bufs["dQ"][:, :, kc_] += G_k[..., None] * torch.matmul(dO_c[:, :, kc_], S_k)
        bufs["dR"][:, :, kc_] += G_k[..., None] * torch.matmul(dc, S_k)
        bufs["dt_all"][:, :, kc_] = torch.einsum(
            "bhpv,bhvd,bhpd->bhp", U_k, dS_next, k_c[:, :, kc_])
        bufs["dG_acc"][:, :, kc_, -1] += (dS_next * S_k).sum((-1, -2))
        bufs["dG_acc"][:, :, kc_] += (
            (dO_c[:, :, kc_] * torch.matmul(q_c[:, :, kc_], S_k.transpose(-1, -2))).sum(-1)
            + (dc * torch.matmul(r_c[:, :, kc_], S_k.transpose(-1, -2))).sum(-1)
        )
        bufs["dU_all"][:, :, kc_] = dU
        dS_next = dS
    bufs["dS0"] = dS_next


def _bwd_s3_torch(fw, bufs):
    G = fw["G"]
    k_c, r_c, v_c = fw["k_c"], fw["r_c"], fw["v_c"]
    lam_c, gam_c = fw["lam_c"], fw["gam_c"]
    A_hat, B_hat, W_loc, T_inv, Z = (
        fw["A_hat"], fw["B_hat"], fw["W_loc"], fw["T_inv"], fw["Z"])
    decay_incl, decay_strict = fw["decay_incl"], fw["decay_strict"]
    S_list, Omega_all = fw["S_list"], fw["Omega_all"]
    dU_all = bufs["dU_all"]

    TtdU_all = torch.matmul(T_inv.transpose(-1, -2), dU_all)
    bufs["dV"] += lam_c[..., None] * TtdU_all
    bufs["dlam"] += (TtdU_all * v_c).sum(-1)
    dUS = torch.matmul(dU_all, S_list)
    dT = (
        torch.matmul(dU_all, (lam_c[..., None] * v_c).transpose(-1, -2))
        + torch.matmul(dU_all, (gam_c[..., None] * Omega_all).transpose(-1, -2))
        - torch.matmul(dUS, Z.transpose(-1, -2))
    )
    dZ = -torch.matmul(T_inv.transpose(-1, -2), dUS)
    bufs["dK"] += (lam_c * G)[..., None] * dZ
    bufs["dlam"] += G * (dZ * k_c).sum(-1)
    bufs["dG_acc"] += lam_c * (dZ * k_c).sum(-1)
    GR = G[..., None] * r_c
    Y = torch.matmul(W_loc, GR)
    dY = -gam_c[..., None] * dZ
    bufs["dgam"] += -(dZ * Y).sum(-1)
    dW_loc = torch.matmul(dY, GR.transpose(-1, -2))
    bufs["dR"] += G[..., None] * torch.matmul(W_loc.transpose(-1, -2), dY)
    bufs["dG_acc"] += (r_c * torch.matmul(W_loc.transpose(-1, -2), dY)).sum(-1)
    dX = -torch.matmul(T_inv.transpose(-1, -2),
                       torch.matmul(dT, T_inv.transpose(-1, -2)))
    dA_hat = lam_c[..., None] * dX
    bufs["dlam"] += (dX * A_hat).sum(-1)
    H_mat = torch.matmul(W_loc, B_hat)
    dH = -gam_c[..., None] * dX
    bufs["dgam"] += -(dX * H_mat).sum(-1)
    dW_loc += torch.matmul(dH, B_hat.transpose(-1, -2))
    dB_hat = bufs["dB_hat"] + torch.matmul(W_loc.transpose(-1, -2), dH)
    dA_raw = decay_strict * dA_hat
    bufs["dK"] += torch.matmul(dA_raw + dA_raw.transpose(-1, -2), k_c)
    dB_raw = decay_incl * dB_hat
    bufs["dR"] += torch.matmul(dB_raw, k_c)
    bufs["dK"] += torch.matmul(dB_raw.transpose(-1, -2), r_c)
    E_a = A_hat * dA_hat
    bufs["dg_loc"] += E_a.sum(-1) - E_a.sum(-2)
    E_b = B_hat * dB_hat
    bufs["dg_loc"] += E_b.sum(-1) - E_b.sum(-2)
    bufs["dW_loc"] = dW_loc


def _bwd_s4_torch(fw, bufs, sm_scale, fill_local):
    B, N, H, NT, P, Dk, Dv, M0, _ = fw["meta"]
    W_full, k_all, q_h = fw["W_full"], fw["k_all"], fw["q_h"]
    dW_full = bufs["dW_full"]
    if fill_local:
        dW_loc = bufs["dW_loc"]
        for kc_ in range(NT):
            dW_full[:, :, kc_ * P:(kc_ + 1) * P, M0 + kc_ * P:M0 + (kc_ + 1) * P] = dW_loc[:, :, kc_]
    ds = W_full * (dW_full - (W_full * dW_full).sum(-1, keepdim=True))
    dC = sm_scale * ds
    decay_incl, dApr = fw["decay_incl"], bufs["dApr"]
    for kc_ in range(NT):
        dC[:, :, kc_ * P:(kc_ + 1) * P, M0 + kc_ * P:M0 + (kc_ + 1) * P] += (
            decay_incl[:, :, kc_] * dApr[:, :, kc_])
    bufs["dQ"] += torch.matmul(dC, k_all).reshape(B, H, NT, P, Dk)
    bufs["dK"] += torch.matmul(dC.transpose(-1, -2), q_h)[:, :, M0:].reshape(B, H, NT, P, Dk)
    E_r = fw["readout_gram"] * dApr
    bufs["dg_loc"] += E_r.sum(-1) - E_r.sum(-2)


def _bwd_s5_torch(fw, bufs):
    g_loc, G = fw["g_loc"], fw["G"]
    dg_loc = bufs["dg_loc"] + G * bufs["dG_acc"]
    tail_all = torch.exp(g_loc[..., -1:] - g_loc)
    dt_all = bufs["dt_all"]
    dg_loc[..., -1] += (tail_all * dt_all).sum(-1)
    dg_loc -= tail_all * dt_all
    bufs["dg_loc"] = dg_loc.flip(-1).cumsum(-1).flip(-1)


# ---- Kernel launchers ----
def _bd(x):
    return max(16, triton.next_power_of_2(x))


def _max_smem() -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(
            torch.cuda.current_device())["max_shared_mem"]
    except Exception:
        return 0


# Fused (datacenter) bwd kernels fit A100/H100 smem but measure ~0.91x vs the
# micro-kernel path on A100 (latency-bound at small grids: fusion raises
# register pressure more than it saves in launches). Opt in by setting this to
# 150_000.
_FUSED_SMEM_MIN = float("inf")


def _bwd_s1_triton(fw, bufs, dO_c):
    B, N, H, NT, P, Dk, Dv, M0, _ = fw["meta"]
    gdc_bwd_dv_local_kernel[(NT, B * H)](
        fw["readout_gram"], dO_c, fw["U_all"], bufs["dU1"], bufs["dApr"],
        NT, Dv, BDv=_bd(Dv), num_warps=4)


def _bwd_s2_triton(fw, bufs, dO_c, dht):
    B, N, H, NT, P, Dk, Dv, M0, _ = fw["meta"]
    Mtot = M0 + N
    BDk, BDv = _bd(Dk), _bd(Dv)
    dS_state = bufs["dS_state"]
    if dht is not None:
        dS_state.copy_(dht)
    if _max_smem() >= _FUSED_SMEM_MIN:
        for kc_ in reversed(range(NT)):
            gdc_bwd_scan_du_dom_kernel[(B * H,)](
                fw["g_loc"], fw["k_c"], bufs["dU1"], fw["B_hat"], bufs["dcache_v"],
                dS_state, fw["T_inv"], fw["gam_c"], fw["Omega_all"],
                bufs["dU_all"], bufs["dS_list_bwd"], bufs["dOm"], bufs["dgam"],
                kc_, NT, Dk, Dv, Mtot, M0, BDk=BDk, BDv=BDv, num_warps=8)
            gdc_bwd_scan_hist_kernel[(B * H,)](
                bufs["dOm"], fw["W_full"], fw["cache_v"], bufs["dW_full"],
                bufs["dcache_v"],
                kc_, NT, N, Dv, Mtot, M0, BDv=BDv, num_warps=8, num_stages=1)
            gdc_bwd_scan_ds12_kernel[(B * H,)](
                fw["g_loc"], fw["q_c"], fw["r_c"], dO_c, bufs["dcache_v"],
                fw["w"], bufs["dU_all"], dS_state,
                kc_, NT, Dk, Dv, Mtot, M0, BDk=BDk, BDv=BDv, num_warps=8)
        bufs["dS0"] = dS_state
        gdc_bwd_scan_dbh_dt_kernel[(NT, B * H)](
            bufs["dcache_v"], fw["U_all"], fw["k_c"], fw["S_list"],
            bufs["dS_list_bwd"], bufs["dB_hat"], bufs["dt_all"], bufs["dG_acc"],
            NT, Dk, Dv, Mtot, M0, BDk=BDk, BDv=BDv, num_warps=8)
        gdc_bwd_scan_dga_dqdr_kernel[(NT, B * H)](
            fw["S_list"], fw["q_c"], fw["r_c"], dO_c, bufs["dcache_v"],
            bufs["dS_list_bwd"], fw["U_all"], fw["g_loc"],
            bufs["dG_acc"], bufs["dK"], bufs["dQ"], bufs["dR"],
            NT, Dk, Dv, Mtot, M0, BDk=BDk, BDv=BDv, num_warps=8)
        return
    for kc_ in reversed(range(NT)):
        gdc_bwd_scan_du_kernel[(B * H,)](
            fw["g_loc"], fw["k_c"], bufs["dU1"], fw["B_hat"], bufs["dcache_v"],
            dS_state, bufs["dU_all"], bufs["dS_list_bwd"],
            kc_, NT, Dk, Dv, Mtot, M0, BDk=BDk, BDv=BDv, num_warps=8)
        gdc_bwd_scan_dom_kernel[(B * H,)](
            fw["T_inv"], fw["gam_c"], fw["Omega_all"], bufs["dU_all"],
            bufs["dOm"], bufs["dgam"],
            kc_, NT, Dv, BDv=BDv, num_warps=8)
        gdc_bwd_scan_hist_kernel[(B * H,)](
            bufs["dOm"], fw["W_full"], fw["cache_v"], bufs["dW_full"],
            bufs["dcache_v"],
            kc_, NT, N, Dv, Mtot, M0, BDv=BDv, num_warps=8, num_stages=1)
        gdc_bwd_scan_ds1_kernel[(B * H,)](
            fw["g_loc"], fw["q_c"], fw["r_c"], dO_c, bufs["dcache_v"], dS_state,
            kc_, NT, Dk, Dv, Mtot, M0, BDk=BDk, BDv=BDv, num_warps=8)
        gdc_bwd_scan_ds2_kernel[(B * H,)](
            fw["w"], bufs["dU_all"], dS_state,
            kc_, NT, Dk, Dv, BDk=BDk, BDv=BDv, num_warps=8)
    bufs["dS0"] = dS_state
    gdc_bwd_scan_dbh_dt_kernel[(NT, B * H)](
        bufs["dcache_v"], fw["U_all"], fw["k_c"], fw["S_list"], bufs["dS_list_bwd"],
        bufs["dB_hat"], bufs["dt_all"], bufs["dG_acc"],
        NT, Dk, Dv, Mtot, M0, BDk=BDk, BDv=BDv, num_warps=8)
    gdc_bwd_scan_dgacc_kernel[(NT, B * H)](
        fw["S_list"], fw["q_c"], fw["r_c"], dO_c, bufs["dcache_v"], bufs["dG_acc"],
        NT, Dk, Dv, Mtot, M0, BDk=BDk, BDv=BDv, num_warps=8)
    gdc_bwd_scan_dqdr_kernel[(NT, B * H)](
        bufs["dS_list_bwd"], fw["U_all"], dO_c, bufs["dcache_v"], fw["S_list"],
        fw["g_loc"], bufs["dK"], bufs["dQ"], bufs["dR"],
        NT, Dk, Dv, Mtot, M0, BDk=BDk, BDv=BDv, num_warps=8)


def _bwd_s3_triton(fw, bufs):
    B, N, H, NT, P, Dk, Dv, M0, _ = fw["meta"]
    Mtot = M0 + N
    BDk, BDv = _bd(Dk), _bd(Dv)
    if _max_smem() >= _FUSED_SMEM_MIN:
        gdc_bwd_wy_a_fused_kernel[(NT, B * H)](
            fw["T_inv"], bufs["dU_all"], fw["lam_c"], fw["v_c"], fw["gam_c"],
            fw["Omega_all"], fw["S_list"], fw["Z"], fw["g_loc"], fw["k_c"],
            fw["r_c"], fw["W_loc"],
            bufs["dT"], bufs["dV"], bufs["dlam"], bufs["dgam"], bufs["dK"],
            bufs["dR"], bufs["dG_acc"], bufs["dW_full"],
            NT, N, Dk, Dv, Mtot, M0, BDk=BDk, BDv=BDv, num_warps=8)
        gdc_bwd_wy_b1_kernel[(NT, B * H)](
            fw["T_inv"], bufs["dT"], fw["A_hat"], fw["lam_c"], fw["gam_c"],
            fw["W_loc"], fw["B_hat"],
            bufs["dX"], bufs["dXt"], bufs["dlam"], bufs["dgam"], bufs["dW_full"],
            NT, N, Mtot, M0, num_warps=8)
        gdc_bwd_wy_b2_kernel[(NT, B * H)](
            bufs["dX"], bufs["dXt"], fw["gam_c"], fw["W_loc"], bufs["dB_hat"],
            fw["lam_c"], fw["decay_strict"], fw["decay_incl"], fw["k_c"],
            fw["r_c"], fw["A_hat"], fw["B_hat"],
            bufs["dK"], bufs["dR"], bufs["dg_loc"],
            NT, Dk, BDk=BDk, num_warps=8)
        return
    gdc_bwd_wy_3a_kernel[(NT, B * H)](
        fw["T_inv"], bufs["dU_all"], fw["lam_c"], fw["v_c"], fw["S_list"],
        bufs["dUS"], bufs["dV"], bufs["dlam"],
        NT, Dk, Dv, BDk=BDk, BDv=BDv, num_warps=8)
    gdc_bwd_wy_3b_kernel[(NT, B * H)](
        bufs["dU_all"], fw["lam_c"], fw["v_c"], fw["gam_c"], fw["Omega_all"],
        bufs["dUS"], fw["Z"], bufs["dT"],
        NT, Dk, Dv, BDk=BDk, BDv=BDv, num_warps=8)
    gdc_bwd_wy_3c_kernel[(NT, B * H)](
        fw["T_inv"], bufs["dUS"], fw["k_c"], fw["lam_c"], fw["g_loc"],
        bufs["dZ"], bufs["dlam"], bufs["dK"], bufs["dG_acc"],
        NT, Dk, BDk=BDk, num_warps=8)
    gdc_bwd_wy_3d_kernel[(NT, B * H)](
        fw["W_loc"], fw["r_c"], fw["g_loc"], bufs["dZ"], fw["gam_c"],
        bufs["dY"], bufs["dgam"],
        NT, Dk, BDk=BDk, num_warps=8)
    gdc_bwd_wy_3e_kernel[(NT, B * H)](
        bufs["dY"], fw["r_c"], fw["g_loc"], fw["W_loc"],
        bufs["dW_full"], bufs["dR"], bufs["dG_acc"],
        NT, N, Dk, Mtot, M0, BDk=BDk, num_warps=8)
    gdc_bwd_wy_3f_kernel[(NT, B * H)](
        fw["T_inv"], bufs["dT"], bufs["dX"], bufs["dXt"],
        NT, num_warps=8)
    gdc_bwd_wy_3g_kernel[(NT, B * H)](
        bufs["dX"], fw["A_hat"], fw["lam_c"], fw["W_loc"], fw["B_hat"],
        bufs["dlam"], bufs["dgam"],
        NT, num_warps=8)
    gdc_bwd_wy_3h_kernel[(NT, B * H)](
        bufs["dX"], fw["gam_c"], fw["B_hat"], bufs["dW_full"],
        NT, N, Mtot, M0, num_warps=8)
    gdc_bwd_wy_3i_kernel[(NT, B * H)](
        bufs["dX"], bufs["dXt"], fw["gam_c"], fw["W_loc"], bufs["dB_hat"],
        bufs["dBt"], bufs["dBtt"],
        NT, num_warps=8)
    gdc_bwd_wy_3j1_kernel[(NT, B * H)](
        bufs["dX"], bufs["dXt"], fw["lam_c"], fw["decay_strict"], fw["k_c"],
        bufs["dK"],
        NT, Dk, BDk=BDk, num_warps=8)
    gdc_bwd_wy_3j2_kernel[(NT, B * H)](
        fw["decay_incl"], bufs["dBt"], bufs["dBtt"], fw["k_c"], fw["r_c"],
        bufs["dK"], bufs["dR"],
        NT, Dk, BDk=BDk, num_warps=8)
    gdc_bwd_wy_3k_kernel[(NT, B * H)](
        fw["A_hat"], bufs["dX"], bufs["dXt"], fw["lam_c"], fw["B_hat"],
        bufs["dBt"], bufs["dBtt"], bufs["dg_loc"],
        NT, num_warps=4)


def _bwd_s4_triton(fw, bufs, sm_scale, fill_local=False):
    B, N, H, NT, P, Dk, Dv, M0, _ = fw["meta"]
    Mtot = M0 + N
    if fill_local:
        # torch s3 left dW^loc in bufs["dW_loc"]; move it into dW_full.
        dW_loc, dW_full = bufs["dW_loc"], bufs["dW_full"]
        for kc_ in range(NT):
            dW_full[:, :, kc_ * P:(kc_ + 1) * P, M0 + kc_ * P:M0 + (kc_ + 1) * P] = dW_loc[:, :, kc_]
    gdc_bwd_rowsum_kernel[(NT, B * H)](
        fw["W_full"], bufs["dW_full"], bufs["rowsum"], NT, N, Mtot,
        num_warps=4, num_stages=1)
    gdc_bwd_dq_kernel[(NT, B * H)](
        fw["W_full"], bufs["dW_full"], bufs["rowsum"], fw["k_all"],
        fw["decay_incl"], bufs["dApr"], fw["readout_gram"],
        bufs["dQ"], bufs["dg_loc"],
        NT, N, Dk, Mtot, M0, sm_scale, BDk=_bd(Dk), num_warps=4, num_stages=1)
    gdc_bwd_dk_kernel[(NT, B * H)](
        fw["W_full"], bufs["dW_full"], bufs["rowsum"], fw["q_h"],
        fw["decay_incl"], bufs["dApr"], bufs["dK"],
        NT, N, Dk, Mtot, M0, sm_scale, BDk=_bd(Dk), num_warps=4, num_stages=1)


def _bwd_s5_triton(fw, bufs):
    B, N, H, NT, P, Dk, Dv, M0, _ = fw["meta"]
    gdc_bwd_gate_cumsum_kernel[(NT, B * H)](
        bufs["dg_loc"], bufs["dG_acc"], bufs["dt_all"], fw["g_loc"],
        NT, num_warps=1)


# ---------------------------------------------------------------------------
# Backward entry point (dispatches each stage to torch or triton).
# ---------------------------------------------------------------------------
def chunk_gated_delta_cache_bwd(
    k: torch.Tensor,          # [B,N,H,Dk]
    v: torch.Tensor,          # [B,N,H,Dv]
    lam: torch.Tensor,        # [B,N,H]
    q: torch.Tensor,          # [B,N,H,Dk]
    gamma: torch.Tensor,      # [B,N,H]
    r: torch.Tensor,          # [B,N,H,Dk]
    log_decay: torch.Tensor,  # [B,N,H]
    dO: torch.Tensor,         # [B,N,H,Dv]
    dht: Optional[torch.Tensor] = None,   # [B,H,Dv,Dk] grad on final state
    initial_state: Optional[torch.Tensor] = None,
    initial_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *,
    sm_scale: float = 1.0,
    chunk_size: int = 64,
    stage_impl: Optional[dict] = None,
):
    impl = {"s1": "triton", "s2": "triton", "s3": "triton",
            "s4": "triton", "s5": "triton"}
    if stage_impl:
        impl.update(stage_impl)
    fw = _chunk_gdc_impl(
        k, v, lam, q, gamma, r, log_decay, initial_state, initial_cache,
        sm_scale=sm_scale, chunk_size=chunk_size, save_bwd=True,
    )
    B, N, H, NT, P, Dk, Dv, M0, out_dtype = fw["meta"]
    dev = k.device
    Mtot = M0 + N
    if "triton" in impl.values():
        if M0 % P != 0:
            raise ValueError("triton bwd stages require len(initial_cache) % chunk_size == 0")
        if k.dtype != torch.float32:
            raise ValueError("triton bwd dev kernels require fp32 inputs")

    # Kernel indexing assumes contiguous per-chunk layouts.
    for name in ("g_loc", "q_c", "k_c", "r_c", "v_c", "lam_c", "gam_c", "A_hat",
                 "B_hat", "W_loc", "T_inv", "w", "Z", "readout_gram", "decay_incl",
                 "decay_strict", "U_all", "S_list", "Omega_all", "W_full", "k_all",
                 "cache_v"):
        fw[name] = fw[name].contiguous()
    fw["G"] = fw["g_loc"].exp()
    fw["q_h"] = fw["q"].transpose(1, 2).contiguous()       # [B,H,N,Dk]

    dO_c = dO.float().view(B, NT, P, H, Dv).permute(0, 3, 1, 2, 4).contiguous()
    dht_f = dht.float().contiguous() if dht is not None else None

    def z(*shape):
        return torch.zeros(*shape, dtype=torch.float32, device=dev)

    bufs = {
        "dU1": z(B, H, NT, P, Dv), "dApr": z(B, H, NT, P, P),
        "dU_all": z(B, H, NT, P, Dv), "dB_hat": z(B, H, NT, P, P),
        "dW_full": z(B, H, N, Mtot), "dcache_v": z(B, H, Mtot, Dv),
        "dK": z(B, H, NT, P, Dk), "dQ": z(B, H, NT, P, Dk),
        "dR": z(B, H, NT, P, Dk), "dV": z(B, H, NT, P, Dv),
        "dlam": z(B, H, NT, P), "dgam": z(B, H, NT, P),
        "dG_acc": z(B, H, NT, P), "dt_all": z(B, H, NT, P),
        "dg_loc": z(B, H, NT, P), "rowsum": z(B, H, N),
        "dT": z(B, H, NT, P, P), "dY": z(B, H, NT, P, Dk),
        "dUS": z(B, H, NT, P, Dk), "dZ": z(B, H, NT, P, Dk),
        "dX": z(B, H, NT, P, P), "dXt": z(B, H, NT, P, P),
        "dBt": z(B, H, NT, P, P), "dBtt": z(B, H, NT, P, P),
        "dOm": z(B, H, NT, P, Dv),
        "dS_state": z(B, H, Dv, Dk), "dS_list_bwd": z(B, H, NT, Dv, Dk),
        "dS0": torch.empty(B, H, Dv, Dk, dtype=torch.float32, device=dev),
    }

    if impl["s1"] == "torch":
        _bwd_s1_torch(fw, bufs, dO_c)
    else:
        _bwd_s1_triton(fw, bufs, dO_c)
    if impl["s2"] == "torch":
        _bwd_s2_torch(fw, bufs, dO_c, dht_f)
    else:
        _bwd_s2_triton(fw, bufs, dO_c, dht_f)
    if impl["s3"] == "torch":
        _bwd_s3_torch(fw, bufs)
    else:
        _bwd_s3_triton(fw, bufs)
    if impl["s4"] == "torch":
        _bwd_s4_torch(fw, bufs, sm_scale, fill_local=impl["s3"] == "torch")
    else:
        _bwd_s4_triton(fw, bufs, sm_scale, fill_local=impl["s3"] == "torch")
    if impl["s5"] == "torch":
        _bwd_s5_torch(fw, bufs)
    else:
        _bwd_s5_triton(fw, bufs)

    # ---- l2-normalization adjoint for K + output layout ----
    def out(x: torch.Tensor) -> torch.Tensor:
        # [B,H,NT,P,*] -> [B,N,H,*]
        return x.reshape(B, H, N, *x.shape[4:]).transpose(1, 2).to(out_dtype)

    dK_n = out(bufs["dK"]).float()                         # grad wrt normalized k
    k_raw = fw["k_raw"]
    rstd = k_raw.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    kn = k_raw / rstd
    dK_out = ((dK_n - kn * (dK_n * kn).sum(-1, keepdim=True)) / rstd).to(out_dtype)

    return (
        dK_out,
        out(bufs["dV"]),
        out(bufs["dlam"]),
        out(bufs["dQ"]),
        out(bufs["dgam"]),
        out(bufs["dR"]),
        out(bufs["dg_loc"]),
        bufs["dS0"].to(out_dtype),
    )


class ChunkGatedDeltaCacheFn(torch.autograd.Function):
    """Autograd bridge: chunkwise triton forward + backward (fp32 internally).

    Inputs may be any dtype; they are cast to fp32 for the kernels and grads
    are cast back to the input dtypes. Requires N % chunk_size == 0 and CUDA.
    """

    @staticmethod
    def forward(ctx, k, v, lam, q, gamma, r, log_decay, initial_state,
                sm_scale, chunk_size):
        in_dtypes = tuple(
            None if x is None else x.dtype
            for x in (k, v, lam, q, gamma, r, log_decay, initial_state)
        )
        k, v, lam, q, gamma, r, log_decay = (
            x.float() for x in (k, v, lam, q, gamma, r, log_decay)
        )
        initial_state = None if initial_state is None else initial_state.float()
        S, O, _ = chunk_gated_delta_cache(
            k, v, lam, q, gamma, r, log_decay, initial_state, None,
            sm_scale=sm_scale, chunk_size=chunk_size,
        )
        ctx.save_for_backward(k, v, lam, q, gamma, r, log_decay, initial_state)
        ctx.sm_scale = sm_scale
        ctx.chunk_size = chunk_size
        ctx.in_dtypes = in_dtypes
        return S.to(in_dtypes[0]), O.to(in_dtypes[0])

    @staticmethod
    def backward(ctx, dS, dO):
        k, v, lam, q, gamma, r, log_decay, initial_state = ctx.saved_tensors
        grads = chunk_gated_delta_cache_bwd(
            k, v, lam, q, gamma, r, log_decay,
            dO.contiguous(), dS.contiguous(), initial_state, None,
            sm_scale=ctx.sm_scale, chunk_size=ctx.chunk_size,
        )
        out = []
        for i, g in enumerate(grads):
            if ctx.in_dtypes[i] is None:
                out.append(None)
            else:
                # contiguous(): fla's ShortConvolution bwd reads dy with
                # hardcoded contiguous strides and silently corrupts grads
                # on non-contiguous input (see repro in fla conv triton ops).
                out.append(g.contiguous())
        return (*out, None, None)


def _self_test() -> None:
    try:
        from .deltacache import gated_delta_cache_wy_bnhc
    except ImportError:
        from deltacache import gated_delta_cache_wy_bnhc

    torch.manual_seed(17)
    dev = "cuda"
    B, N, H, Dk, Dv = 2, 192, 3, 64, 64
    dtype = torch.float32
    k = torch.randn(B, N, H, Dk, dtype=dtype, device=dev) / Dk**0.5
    q = torch.randn(B, N, H, Dk, dtype=dtype, device=dev) / Dk**0.5
    r = torch.randn(B, N, H, Dk, dtype=dtype, device=dev) / Dk**0.5
    v = torch.randn(B, N, H, Dv, dtype=dtype, device=dev)
    lam = torch.sigmoid(torch.randn(B, N, H, dtype=dtype, device=dev))
    gamma = 0.2 * torch.tanh(torch.randn(B, N, H, dtype=dtype, device=dev))
    log_decay = torch.sigmoid(torch.randn(B, N, H, dtype=dtype, device=dev)).log()
    sm_scale = 1.7

    for S0 in (None, torch.randn(B, H, Dv, Dk, dtype=dtype, device=dev)):
        S_wy, O_wy, aux = gated_delta_cache_wy_bnhc(
            k, v, lam, q, gamma, r, log_decay, S0,
            sm_scale=sm_scale, return_intermediates=True,
        )
        S_ck, O_ck, cache_ck, aux_ck = chunk_gated_delta_cache(
            k, v, lam, q, gamma, r, log_decay, S0,
            sm_scale=sm_scale, return_intermediates=True,
        )
        torch.testing.assert_close(O_ck, O_wy, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(S_ck, S_wy, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(aux_ck["U"], aux["U"], rtol=1e-4, atol=1e-5)
        # Cache values appended by chunk k are the post-update S^m r_m.
        torch.testing.assert_close(
            cache_ck[1], aux["cache_values"], rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(
            cache_ck[0], F.normalize(k, p=2, dim=-1), rtol=1e-5, atol=1e-6
        )
        print(f"chunk vs WY (S0={'0' if S0 is None else 'general'}): passed")

    # Continuation: second call carries state + cache, must equal full run.
    split = 128
    S_full, O_full, cache_full = chunk_gated_delta_cache(
        k, v, lam, q, gamma, r, log_decay, sm_scale=sm_scale
    )
    S_a, O_a, cache_a = chunk_gated_delta_cache(
        k[:, :split], v[:, :split], lam[:, :split], q[:, :split],
        gamma[:, :split], r[:, :split], log_decay[:, :split], sm_scale=sm_scale,
    )
    S_b, O_b, cache_b = chunk_gated_delta_cache(
        k[:, split:], v[:, split:], lam[:, split:], q[:, split:],
        gamma[:, split:], r[:, split:], log_decay[:, split:],
        S_a, cache_a, sm_scale=sm_scale,
    )
    torch.testing.assert_close(
        torch.cat((O_a, O_b), dim=1), O_full, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(S_b, S_full, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(cache_b[0], cache_full[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cache_b[1], cache_full[1], rtol=1e-4, atol=1e-5)
    print("chunk continuation with carried cache: passed")


def _self_test_bwd() -> None:
    try:
        from .deltacache import gated_delta_cache_wy_bnhc
    except ImportError:
        from deltacache import gated_delta_cache_wy_bnhc

    torch.manual_seed(23)
    dev = "cuda"
    sm_scale = 1.7

    for (Dk, Dv) in ((64, 64), (32, 64)):
      B, N, H = 2, 192, 3
      dtype = torch.float32
      k = torch.randn(B, N, H, Dk, dtype=dtype, device=dev) / Dk**0.5
      q = torch.randn(B, N, H, Dk, dtype=dtype, device=dev) / Dk**0.5
      r = torch.randn(B, N, H, Dk, dtype=dtype, device=dev) / Dk**0.5
      v = torch.randn(B, N, H, Dv, dtype=dtype, device=dev)
      lam = torch.sigmoid(torch.randn(B, N, H, dtype=dtype, device=dev))
      gamma = 0.2 * torch.tanh(torch.randn(B, N, H, dtype=dtype, device=dev))
      log_decay = torch.sigmoid(torch.randn(B, N, H, dtype=dtype, device=dev)).log()

      for use_ht in (False, True):
        dO = torch.randn(B, N, H, Dv, dtype=dtype, device=dev)
        dht = torch.randn(B, H, Dv, Dk, dtype=dtype, device=dev) if use_ht else None
        S0 = torch.randn(B, H, Dv, Dk, dtype=dtype, device=dev)

        refs = [x.clone().requires_grad_(True)
                for x in (k, v, lam, q, gamma, r, log_decay, S0)]
        S_ref, O_ref = gated_delta_cache_wy_bnhc(
            *refs[:7], refs[7], sm_scale=sm_scale)
        loss = (O_ref * dO).sum()
        if use_ht:
            loss = loss + (S_ref * dht).sum()
        grads_ref = torch.autograd.grad(loss, refs)

        grads_ck = chunk_gated_delta_cache_bwd(
            k, v, lam, q, gamma, r, log_decay, dO, dht, S0, sm_scale=sm_scale)

        names = ["dK", "dV", "dlam", "dQ", "dgamma", "dR", "dg", "dS0"]
        ok = True
        for name, g_ck, g_ref in zip(names, grads_ck, grads_ref):
            try:
                torch.testing.assert_close(g_ck, g_ref, rtol=2e-3, atol=1e-4)
                print(f"  {name}: passed")
            except AssertionError:
                diff = (g_ck - g_ref).abs()
                denom = g_ref.abs().clamp_min(1e-6)
                bad = (diff > 1e-4 + 2e-3 * g_ref.abs())
                idx = diff.flatten().argmax()
                print(
                    f"  {name}: FAILED maxabs={diff.max().item():.4g} "
                    f"nbad={bad.sum().item()}/{bad.numel()} "
                    f"worst at flat {idx.item()} ck={g_ck.flatten()[idx].item():.6g} "
                    f"ref={g_ref.flatten()[idx].item():.6g} "
                    f"rel_at_worst={(diff/denom).flatten()[idx].item():.4g}"
                )
                ok = False
        if ok:
            print(f"bwd vs WY autograd (Dk={Dk},Dv={Dv},dht={'yes' if use_ht else 'no'}): passed")


if __name__ == "__main__":
    _self_test()
    _self_test_bwd()
