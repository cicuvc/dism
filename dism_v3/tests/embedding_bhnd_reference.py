"""Frozen BHND interpolation oracle for BNHD migration regression tests only.

Extracted from main:dism_v2/emb_kernel.py, excluding its standalone benchmark.
Do not use this layout in production v3 models.
"""
import triton
import triton.language as tl
import torch


@triton.jit
def emb_fwd(
    q, k, emb_q, emb_k, out_q, out_k, olse_q, olse_k, top1_q, top1_k, idx_q, idx_k, BATCH, HEAD, N_CTX, sm_scale,
    q_s0, q_s1, q_s2,
    k_s0, k_s1, k_s2,
    eq_s0, eq_s1, ek_s0, ek_s1,
    o_s0, o_s1, o_s2,
    lse_s0, lse_s1,
    N_HEADDIM: tl.constexpr, N_VOCAB: tl.constexpr,
    N_RSIZE: tl.constexpr, N_CSIZE: tl.constexpr
    ):
    """
    q, k: [BATCH, HEAD, N_CTX, N_HEADDIM]
    emb_q, emb_k: [HEAD, N_VOCAB, N_HEADDIM]
    out_q, out_k: [BATCH, HEAD, N_CTX, N_HEADDIM]
    lse_q, lse_k, top1_q, top1_k, idx_q, idx_k: [BATCH, HEAD, N_CTX]
    """

    pid_bh, pid_n = tl.program_id(0), tl.program_id(1)
    batch, head = pid_bh // HEAD, pid_bh % HEAD

    SM_SCALE: tl.constexpr = 1.4426950408889634
    RCP_SCALE: tl.constexpr = 0.6931471805599453
    scale = sm_scale * SM_SCALE

    q_block = tl.load(tl.make_block_ptr(
        q + q_s0 * batch + q_s1 * head,
        shape=(N_CTX, N_HEADDIM),
        strides=(q_s2, 1),
        offsets=(pid_n * N_RSIZE, 0),
        block_shape=(N_RSIZE, N_HEADDIM),
        order=(1,0)
    ), boundary_check=(0,))

    k_block = tl.load(tl.make_block_ptr(
        k + k_s0 * batch + k_s1 * head,
        shape=(N_CTX, N_HEADDIM),
        strides=(k_s2, 1),
        offsets=(pid_n * N_RSIZE, 0),
        block_shape=(N_RSIZE, N_HEADDIM),
        order=(1,0)
    ), boundary_check=(0,))

    q_emb_block = tl.make_block_ptr(
        emb_q + head * eq_s0,
        shape = (N_VOCAB, N_HEADDIM),
        strides=(eq_s1, 1),
        offsets=(0, 0),
        block_shape=(N_CSIZE, N_HEADDIM),
        order=(1,0)
    )

    k_emb_block = tl.make_block_ptr(
        emb_k + head * ek_s0,
        shape = (N_VOCAB, N_HEADDIM),
        strides=(ek_s1, 1),
        offsets=(0, 0),
        block_shape=(N_CSIZE, N_HEADDIM),
        order=(1,0)
    )

    acc_qo = tl.zeros((N_RSIZE, N_HEADDIM), dtype=tl.float32)
    acc_ko = tl.zeros((N_RSIZE, N_HEADDIM), dtype=tl.float32)
    m_q = tl.full((N_RSIZE, 1), float('-inf'), dtype=tl.float32)
    m_k = tl.full((N_RSIZE, 1), float('-inf'), dtype=tl.float32)
    lse_q = tl.full((N_RSIZE, 1), 1.0, dtype = tl.float32)
    lse_k = tl.full((N_RSIZE, 1), 1.0, dtype = tl.float32)
    mxidx_q = tl.zeros((N_RSIZE, 1), dtype = tl.int32)
    mxidx_k = tl.zeros((N_RSIZE, 1), dtype = tl.int32)



    for i in tl.range(0, N_VOCAB, N_CSIZE):
        col_idx = tl.arange(0, N_CSIZE)[None, :] + i
        col_mask = col_idx < N_VOCAB

        q_embs = tl.load(q_emb_block, boundary_check=(0,))
        k_embs = tl.load(k_emb_block, boundary_check=(0,))

        p_qs = tl.dot(q_block, tl.trans(q_embs)) * scale
        p_ks = tl.dot(k_block, tl.trans(k_embs)) * scale

        p_qs = tl.where(col_mask, p_qs, -1e5)
        p_ks = tl.where(col_mask, p_ks, -1e5)

        mx_qs, ix_qs = tl.max(p_qs, -1, return_indices=True, keep_dims=True)
        mx_ks, ix_ks = tl.max(p_ks, -1, return_indices=True, keep_dims=True)

        new_m_q = tl.maximum(mx_qs, m_q)
        new_m_k = tl.maximum(mx_ks, m_k)

        mxidx_q = tl.where(mx_qs > m_q, ix_qs + i, mxidx_q)
        mxidx_k = tl.where(mx_ks > m_k, ix_ks + i, mxidx_k)

        s_qs = tl.exp2(p_qs - new_m_q)
        rs_qs = tl.exp2(m_q - new_m_q)
        s_ks = tl.exp2(p_ks - new_m_k)
        rs_ks = tl.exp2(m_k - new_m_k)

        acc_ko = acc_ko * rs_qs + tl.dot(s_qs.to(k_embs.dtype), k_embs)
        acc_qo = acc_qo * rs_ks + tl.dot(s_ks.to(q_embs.dtype), q_embs)

        lse_k = lse_k * rs_qs + tl.sum(s_qs, -1, keep_dims=True)
        lse_q = lse_q * rs_ks + tl.sum(s_ks, -1, keep_dims=True)

        q_emb_block = tl.advance(q_emb_block, (N_CSIZE, 0))
        k_emb_block = tl.advance(k_emb_block, (N_CSIZE, 0))
        m_q = new_m_q
        m_k = new_m_k

    tl.store(
        tl.make_block_ptr(
            out_q + batch * o_s0 + head * o_s1,
            shape=(N_CTX, N_HEADDIM),
            strides=(o_s2, 1),
            offsets=(N_RSIZE * pid_n, 0),
            block_shape=(N_RSIZE, N_HEADDIM),
            order=(1, 0)
        ), (acc_qo / lse_q).to(out_q.dtype.element_ty), boundary_check=(0,)
    )

    tl.store(
        tl.make_block_ptr(
            out_k + batch * o_s0 + head * o_s1,
            shape=(N_CTX, N_HEADDIM),
            strides=(o_s2, 1),
            offsets=(N_RSIZE * pid_n, 0),
            block_shape=(N_RSIZE, N_HEADDIM),
            order=(1, 0)
        ), (acc_ko / lse_k).to(out_k.dtype.element_ty), boundary_check=(0,)
    )

    idx = tl.arange(0, N_RSIZE) + pid_n * N_RSIZE
    lse_mask = idx[:, None] < N_CTX
    tl.store(olse_q + batch * lse_s0 + head * lse_s1 + idx[:, None], (tl.log2(lse_q) + m_k) * RCP_SCALE, mask=lse_mask)
    tl.store(olse_k + batch * lse_s0 + head * lse_s1 + idx[:, None], (tl.log2(lse_k) + m_q) * RCP_SCALE, mask=lse_mask)
    tl.store(top1_q + batch * lse_s0 + head * lse_s1 + idx[:, None], (1.0 / lse_q).to(top1_q.dtype.element_ty), mask=lse_mask)
    tl.store(top1_k + batch * lse_s0 + head * lse_s1 + idx[:, None], (1.0 / lse_k).to(top1_k.dtype.element_ty), mask=lse_mask)
    tl.store(idx_q + batch * lse_s0 + head * lse_s1 + idx[:, None], mxidx_k, mask=lse_mask)
    tl.store(idx_k + batch * lse_s0 + head * lse_s1 + idx[:, None], mxidx_q, mask=lse_mask)

def _check_emb_layout(q: torch.Tensor, k: torch.Tensor, q_emb: torch.Tensor, k_emb: torch.Tensor):
    assert len(q.shape) == 4 and len(q_emb.shape) == 3, \
        f"expected q/k [B,H,N,D] and q_emb/k_emb [H,V,D], got {tuple(q.shape)} / {tuple(q_emb.shape)}"
    assert q.shape == k.shape, f"q/k shape mismatch: {tuple(q.shape)} vs {tuple(k.shape)}"
    assert q_emb.shape == k_emb.shape, f"q_emb/k_emb shape mismatch: {tuple(q_emb.shape)} vs {tuple(k_emb.shape)}"
    assert q.shape[1] == q_emb.shape[0], f"H mismatch: q {q.shape[1]} vs q_emb {q_emb.shape[0]}"
    assert q.shape[3] == q_emb.shape[2], f"D mismatch: q {q.shape[3]} vs q_emb {q_emb.shape[2]}"
    for t, nm in ((q, "q"), (k, "k"), (q_emb, "q_emb"), (k_emb, "k_emb")):
        assert t.stride(-1) == 1, f"{nm} last dim must be contiguous (stride(-1)==1), got strides {t.stride()}"


def emb_fwd_wrapper(q: torch.Tensor, k: torch.Tensor, emb_q: torch.Tensor, emb_k: torch.Tensor, sm_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_emb_layout(q, k, emb_q, emb_k)
    B, H, N, D = q.shape
    VOC = emb_q.shape[1]

    N_RSIZE = 64
    N_CSIZE = 64

    out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    out_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    olse_q = torch.empty((B, H, N), dtype=torch.float32, device=q.device)
    olse_k = torch.empty((B, H, N), dtype=torch.float32, device=k.device)
    top1_q = torch.empty((B, H, N), dtype=torch.float32, device=q.device)
    top1_k = torch.empty((B, H, N), dtype=torch.float32, device=k.device)

    idx_q = torch.empty((B, H, N), dtype=torch.int32, device=q.device)
    idx_k = torch.empty((B, H, N), dtype=torch.int32, device=k.device)

    emb_fwd[(B * H, triton.cdiv(N, N_RSIZE))](
        q, k, emb_q, emb_k, out_q, out_k, olse_q, olse_k, top1_q, top1_k, idx_q, idx_k,
        B, H, N, float(sm_scale),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        emb_q.stride(0), emb_q.stride(1), emb_k.stride(0), emb_k.stride(1),
        out_q.stride(0), out_q.stride(1), out_q.stride(2),
        olse_q.stride(0), olse_q.stride(1),
        D, VOC, N_RSIZE, N_CSIZE,
        # D128 with multiple vocab blocks exceeds sm120 shared memory at default stages.
        num_stages=1 if D == 128 else 3
    )

    return out_q, out_k, olse_q, olse_k, top1_q, top1_k, idx_q, idx_k


@triton.jit
def _interp_bwd_preprocess(
    out_q, out_k, doq, dok, delta_q, delta_k,
    BATCH, HEAD, N_CTX,
    o_s0, o_s1, o_s2,
    doq_s0, doq_s1, doq_s2, dok_s0, dok_s1, dok_s2,
    lse_s0, lse_s1,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr
    ):
    pid_bh = tl.program_id(1)
    batch, head = pid_bh // HEAD, pid_bh % HEAD

    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, HEAD_DIM)
    rm = off_m < N_CTX

    base = batch * o_s0 + head * o_s1
    oq = tl.load(out_q + base + off_m[:, None] * o_s2 + off_d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    ok = tl.load(out_k + base + off_m[:, None] * o_s2 + off_d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    dq = tl.load(doq + batch * doq_s0 + head * doq_s1 + off_m[:, None] * doq_s2 + off_d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    dk = tl.load(dok + batch * dok_s0 + head * dok_s1 + off_m[:, None] * dok_s2 + off_d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)

    dq = tl.sum(oq * dq, 1)
    dk = tl.sum(ok * dk, 1)

    lse_base = batch * lse_s0 + head * lse_s1
    tl.store(delta_q + lse_base + off_m, dq, mask=rm)
    tl.store(delta_k + lse_base + off_m, dk, mask=rm)


@triton.jit
def _interp_bwd(
    q, k, q_emb, k_emb,
    doq, dok, dlq, dlk,
    olse_q, olse_k,
    delta_q, delta_k,
    dq, dk, dq_emb, dk_emb,
    BATCH, HEAD, N_CTX, N_VOCAB, sm_scale,
    q_s0, q_s1, q_s2,
    k_s0, k_s1, k_s2,
    eq_s0, eq_s1, ek_s0, ek_s1,
    doq_s0, doq_s1, doq_s2, dok_s0, dok_s1, dok_s2,
    dlq_s0, dlq_s1, dlq_s2, dlk_s0, dlk_s1, dlk_s2,
    lse_s0, lse_s1,
    dq_s0, dq_s1, dq_s2,
    deq_s0, deq_s1, dek_s0, dek_s1,
    HEAD_DIM: tl.constexpr,
    BLOCK_V: tl.constexpr, BLOCK_R1: tl.constexpr,
    BLOCK_R2: tl.constexpr, BLOCK_V2: tl.constexpr
    ):
    LOG2E: tl.constexpr = 1.4426950408889634
    scale = sm_scale * LOG2E

    pid_v, pid_r, pid_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = pid_bh // HEAD, pid_bh % HEAD

    q_off = batch * q_s0 + head * q_s1
    k_off = batch * k_s0 + head * k_s1
    lse_off = batch * lse_s0 + head * lse_s1

    off_d = tl.arange(0, HEAD_DIM)
    d_ty = q_emb.dtype.element_ty

    # Phase A: dq_emb / dk_emb. Each program owns a vocab column block (pid_v);
    # q_emb/k_emb are per-head and shared across batches, so a single program per
    # (head, vocab block) reduces over all batches and rows in-kernel (no atomics).
    if (pid_r == 0) & (pid_bh < HEAD):
        a_head = pid_bh

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        col_mask = offs_v < N_VOCAB

        q_emb_b = tl.load(
            q_emb + a_head * eq_s0 + offs_v[:, None] * eq_s1 + off_d[None, :],
            mask=col_mask[:, None], other=0.0)
        k_emb_b = tl.load(
            k_emb + a_head * ek_s0 + offs_v[:, None] * ek_s1 + off_d[None, :],
            mask=col_mask[:, None], other=0.0)

        dq_emb_acc = tl.zeros((BLOCK_V, HEAD_DIM), dtype=tl.float32)
        dk_emb_acc = tl.zeros((BLOCK_V, HEAD_DIM), dtype=tl.float32)

        for a_b in range(BATCH):
            a_qoff = a_b * q_s0 + a_head * q_s1
            a_koff = a_b * k_s0 + a_head * k_s1
            a_dqoff = a_b * doq_s0 + a_head * doq_s1
            a_dkoff = a_b * dok_s0 + a_head * dok_s1
            a_dlq = a_b * dlq_s0 + a_head * dlq_s1
            a_dlk = a_b * dlk_s0 + a_head * dlk_s1
            a_lse = a_b * lse_s0 + a_head * lse_s1

            for start_r in range(0, N_CTX, BLOCK_R1):
                offs_r = start_r + tl.arange(0, BLOCK_R1)
                rm = offs_r < N_CTX

                q_b = tl.load(q + a_qoff + offs_r[:, None] * q_s2 + off_d[None, :], mask=rm[:, None], other=0.0)
                k_b = tl.load(k + a_koff + offs_r[:, None] * k_s2 + off_d[None, :], mask=rm[:, None], other=0.0)
                doq_b = tl.load(doq + a_dqoff + offs_r[:, None] * doq_s2 + off_d[None, :], mask=rm[:, None], other=0.0).to(d_ty)
                dok_b = tl.load(dok + a_dkoff + offs_r[:, None] * dok_s2 + off_d[None, :], mask=rm[:, None], other=0.0).to(d_ty)

                lq = tl.load(olse_q + a_lse + offs_r, mask=rm, other=0.0)[:, None] * LOG2E
                lk = tl.load(olse_k + a_lse + offs_r, mask=rm, other=0.0)[:, None] * LOG2E
                dq_l = tl.load(dlq + a_dlq + offs_r * dlq_s2, mask=rm, other=0.0)[:, None]
                dk_l = tl.load(dlk + a_dlk + offs_r * dlk_s2, mask=rm, other=0.0)[:, None]
                dq_d = tl.load(delta_q + a_lse + offs_r, mask=rm, other=0.0)[:, None]
                dk_d = tl.load(delta_k + a_lse + offs_r, mask=rm, other=0.0)[:, None]

                mask = rm[:, None] & col_mask[None, :]

                S1 = tl.where(mask, tl.dot(k_b, tl.trans(k_emb_b)), float('-inf'))
                S2 = tl.where(mask, tl.dot(q_b, tl.trans(q_emb_b)), float('-inf'))

                P1 = tl.exp2(S1 * scale - lq)
                P2 = tl.exp2(S2 * scale - lk)

                dP1 = tl.dot(doq_b, tl.trans(q_emb_b))
                dP2 = tl.dot(dok_b, tl.trans(k_emb_b))

                dS1 = P1 * (dP1 - dq_d) + P1 * dq_l
                dS2 = P2 * (dP2 - dk_d) + P2 * dk_l

                dk_emb_acc += tl.dot(tl.trans(dS1).to(d_ty), k_b) * sm_scale
                dq_emb_acc += tl.dot(tl.trans(dS2).to(d_ty), q_b) * sm_scale
                dq_emb_acc += tl.dot(tl.trans(P1).to(d_ty), doq_b)
                dk_emb_acc += tl.dot(tl.trans(P2).to(d_ty), dok_b)

        tl.store(
            dq_emb + a_head * deq_s0 + offs_v[:, None] * deq_s1 + off_d[None, :],
            dq_emb_acc, mask=col_mask[:, None])
        tl.store(
            dk_emb + a_head * dek_s0 + offs_v[:, None] * dek_s1 + off_d[None, :],
            dk_emb_acc, mask=col_mask[:, None])

    # Phase B owns rows only; other pid_v programs must not duplicate its stores.
    if pid_v == 0:
        # Phase B: dq / dk, each program owns a row block (pid_r)
        offs_r = pid_r * BLOCK_R2 + tl.arange(0, BLOCK_R2)
        rm = offs_r < N_CTX

        q_b = tl.load(q + q_off + offs_r[:, None] * q_s2 + off_d[None, :], mask=rm[:, None], other=0.0)
        k_b = tl.load(k + k_off + offs_r[:, None] * k_s2 + off_d[None, :], mask=rm[:, None], other=0.0)
        doq_b = tl.load(doq + batch * doq_s0 + head * doq_s1 + offs_r[:, None] * doq_s2 + off_d[None, :], mask=rm[:, None], other=0.0).to(d_ty)
        dok_b = tl.load(dok + batch * dok_s0 + head * dok_s1 + offs_r[:, None] * dok_s2 + off_d[None, :], mask=rm[:, None], other=0.0).to(d_ty)

        lq = tl.load(olse_q + lse_off + offs_r, mask=rm, other=0.0)[:, None] * LOG2E
        lk = tl.load(olse_k + lse_off + offs_r, mask=rm, other=0.0)[:, None] * LOG2E
        dq_l = tl.load(dlq + batch * dlq_s0 + head * dlq_s1 + offs_r * dlq_s2, mask=rm, other=0.0)[:, None]
        dk_l = tl.load(dlk + batch * dlk_s0 + head * dlk_s1 + offs_r * dlk_s2, mask=rm, other=0.0)[:, None]
        dq_d = tl.load(delta_q + lse_off + offs_r, mask=rm, other=0.0)[:, None]
        dk_d = tl.load(delta_k + lse_off + offs_r, mask=rm, other=0.0)[:, None]

        dq_acc = tl.zeros((BLOCK_R2, HEAD_DIM), dtype=tl.float32)
        dk_acc = tl.zeros((BLOCK_R2, HEAD_DIM), dtype=tl.float32)

        for start_v in range(0, N_VOCAB, BLOCK_V2):
            offs_v2 = start_v + tl.arange(0, BLOCK_V2)
            col_mask2 = offs_v2 < N_VOCAB

            q_emb_b2 = tl.load(
                q_emb + head * eq_s0 + offs_v2[:, None] * eq_s1 + off_d[None, :],
                mask=col_mask2[:, None], other=0.0)
            k_emb_b2 = tl.load(
                k_emb + head * ek_s0 + offs_v2[:, None] * ek_s1 + off_d[None, :],
                mask=col_mask2[:, None], other=0.0)

            mask = rm[:, None] & col_mask2[None, :]

            S1 = tl.where(mask, tl.dot(k_b, tl.trans(k_emb_b2)), float('-inf'))
            S2 = tl.where(mask, tl.dot(q_b, tl.trans(q_emb_b2)), float('-inf'))

            P1 = tl.exp2(S1 * scale - lq)
            P2 = tl.exp2(S2 * scale - lk)

            dP1 = tl.dot(doq_b, tl.trans(q_emb_b2))
            dP2 = tl.dot(dok_b, tl.trans(k_emb_b2))

            dS1 = P1 * (dP1 - dq_d) + P1 * dq_l
            dS2 = P2 * (dP2 - dk_d) + P2 * dk_l

            dk_acc += tl.dot(dS1.to(d_ty), k_emb_b2) * sm_scale
            dq_acc += tl.dot(dS2.to(d_ty), q_emb_b2) * sm_scale

        tl.store(
            dq + q_off + offs_r[:, None] * dq_s2 + off_d[None, :],
            dq_acc, mask=rm[:, None])
        tl.store(
            dk + k_off + offs_r[:, None] * dq_s2 + off_d[None, :],
            dk_acc, mask=rm[:, None])


def emb_bwd_wrapper(q: torch.Tensor, k: torch.Tensor, q_emb: torch.Tensor, k_emb: torch.Tensor,
                    out_q: torch.Tensor, out_k: torch.Tensor, olse_q: torch.Tensor, olse_k: torch.Tensor,
                    doq: torch.Tensor, dok: torch.Tensor, dlq: torch.Tensor | None, dlk: torch.Tensor | None,
                    sm_scale: float = 1.0):
    _check_emb_layout(q, k, q_emb, k_emb)
    B, H, N, D = q.shape
    VOC = q_emb.shape[1]

    if dlq is None:
        dlq = torch.zeros((B, H, N), dtype=torch.float32, device=q.device)
    if dlk is None:
        dlk = torch.zeros((B, H, N), dtype=torch.float32, device=k.device)
    assert doq.shape == out_q.shape and dok.shape == out_k.shape, \
        f"doq/dok shape mismatch: {tuple(doq.shape)} / {tuple(dok.shape)} vs {tuple(out_q.shape)}"
    assert dlq.shape == olse_q.shape and dlk.shape == olse_k.shape, \
        f"dlq/dlk shape mismatch: {tuple(dlq.shape)} / {tuple(dlk.shape)} vs {tuple(olse_q.shape)}"
    #for t, nm in ((doq, "doq"), (dok, "dok"), (dlq, "dlq"), (dlk, "dlk")):
    #    assert t.stride(-1) == 1, f"{nm} last dim must be contiguous (stride(-1)==1), got strides {t.stride()}"

    doq = doq.contiguous()
    dok = dok.contiguous()
    dlq = dlq.contiguous()
    dlk = dlk.contiguous()

    dq = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    dk = torch.empty(k.shape, dtype=torch.float32, device=k.device)
    dq_emb = torch.empty(q_emb.shape, dtype=torch.float32, device=q_emb.device)
    dk_emb = torch.empty(k_emb.shape, dtype=torch.float32, device=k_emb.device)

    delta_q = torch.empty((B, H, N), dtype=torch.float32, device=q.device)
    delta_k = torch.empty((B, H, N), dtype=torch.float32, device=k.device)

    PRE_BLOCK = 64
    _interp_bwd_preprocess[(triton.cdiv(N, PRE_BLOCK), B * H)](
        out_q, out_k, doq, dok, delta_q, delta_k,
        B, H, N,
        out_q.stride(0), out_q.stride(1), out_q.stride(2),
        doq.stride(0), doq.stride(1), doq.stride(2),
        dok.stride(0), dok.stride(1), dok.stride(2),
        olse_q.stride(0), olse_q.stride(1),
        D, PRE_BLOCK
    )

    BLOCK_V, BLOCK_R1, BLOCK_R2, BLOCK_V2 = 64, 32, 64, 32

    _interp_bwd[(triton.cdiv(VOC, BLOCK_V), triton.cdiv(N, BLOCK_R2), B * H)](
        q, k, q_emb, k_emb,
        doq, dok, dlq, dlk,
        olse_q, olse_k,
        delta_q, delta_k,
        dq, dk, dq_emb, dk_emb,
        B, H, N, VOC, float(sm_scale),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        q_emb.stride(0), q_emb.stride(1), k_emb.stride(0), k_emb.stride(1),
        doq.stride(0), doq.stride(1), doq.stride(2),
        dok.stride(0), dok.stride(1), dok.stride(2),
        dlq.stride(0), dlq.stride(1), dlq.stride(2),
        dlk.stride(0), dlk.stride(1), dlk.stride(2),
        olse_q.stride(0), olse_q.stride(1),
        dq.stride(0), dq.stride(1), dq.stride(2),
        dq_emb.stride(0), dq_emb.stride(1), dk_emb.stride(0), dk_emb.stride(1),
        D,
        BLOCK_V=BLOCK_V, BLOCK_R1=BLOCK_R1, BLOCK_R2=BLOCK_R2, BLOCK_V2=BLOCK_V2,
        num_stages=1 if D == 128 else 3
    )

    return dq, dk, dq_emb, dk_emb


def _voc_interp(x: torch.Tensor, key_voc: torch.Tensor, val_voc: torch.Tensor, sm_scale: float = 1.0):
    """
        x: [B, H, N, D] (queries)
        key_voc, val_voc: [H, V, D]
        returns out [B, H, N, D], lse [B, H, N]
    """
    qq = torch.einsum('bhnd,hvd->bhnv', x.float(), key_voc.float()) * sm_scale
    out = torch.einsum('bhnv,hvd->bhnd', torch.softmax(qq, -1), val_voc.float())
    g_lse = torch.logsumexp(qq, -1)
    idx = torch.argmax(qq, -1)
    return out, g_lse, idx

class EmbInterpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.Function, q, k, q_emb, k_emb, sm_scale: float = 1.0):
        out_q, out_k, olse_q, olse_k, top1_q, top1_k, idx_q, idx_k = emb_fwd_wrapper(q, k, q_emb, k_emb, sm_scale)
        ctx.save_for_backward(q, k, q_emb, k_emb, out_q, out_k, olse_q, olse_k)
        ctx.sm_scale = float(sm_scale)
        return out_q, out_k, olse_q, olse_k, top1_q, top1_k, idx_q, idx_k

    @staticmethod
    def backward(ctx: torch.autograd.Function, doq, dok, dlq, dlk, *args): # grads from top1/idx are ignored
        q, k, q_emb, k_emb, out_q, out_k, olse_q, olse_k = ctx.saved_tensors
        dq, dk, dq_emb, dk_emb = emb_bwd_wrapper(q, k, q_emb, k_emb, out_q, out_k, olse_q, olse_k, doq, dok, dlq, dlk, ctx.sm_scale)
        return dq, dk, dq_emb, dk_emb, None
