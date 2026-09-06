import math
import os
from typing import Tuple

import tqdm
os.environ['TRITON_INTERPRET'] = '0'


import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



def lse0(x: torch.Tensor): # logsumexp(x, y)
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.clamp_min(x, 0)

def std_rosa_batched(logq: torch.Tensor, logk: torch.Tensor, values: torch. Tensor, betas: torch.Tensor, fallback_o: torch.Tensor, sel_tau: torch.Tensor, EPS: float = 1e-3):
    B, H, N, C = logq.shape

    q = torch.softmax(logq, -1)
    k = torch.softmax(logk, -1)
    
    QK_BIAS = math.log(1 - EPS * C)
    QK_LOG_OFFSET = EPS / (1 - EPS * C)

    logM = sel_tau[None,:,None,None] + QK_BIAS + torch.log(QK_LOG_OFFSET + torch.matmul(q, k.transpose(-2, -1)))  # [B, H, N, N]

    logM.retain_grad()
    globals()['logM'] = logM
    
    weights = [logM[:,:,0,None,:]]
    mask = torch.arange(0, N) == 0
    umask = torch.tril(torch.ones((B, H, N, N), dtype = torch.bool))

    for i in range(1, N):
        last_row = torch.roll(weights[-1],1)
        last_row = torch.where(mask, -1e5, last_row)
        weights.append(lse0(last_row) + logM[:,:,i,None,:])

    attn = torch.where(umask, torch.cat(weights, dim = -2), -1e5)
    globals()['attn_scores'] = attn
    Zmax = torch.clamp_min(attn.max(-1, keepdim=True).values, betas[...,None])
    attn = torch.exp(attn - Zmax)
    out = torch.matmul(attn, values) + fallback_o * torch.exp(betas[...,None] - Zmax)
    globals()['attn'] = attn

    return torch.nn.functional.rms_norm(out, (values.shape[-1],), eps=1e-7) 


def std_rosa_batched_hard(idx_q: torch.Tensor, idx_k: torch.Tensor, values: torch. Tensor, betas: torch.Tensor, fallback_o: torch.Tensor, sel_tau: torch.Tensor, EPS: float = 1e-3):
    B, H, N  = idx_q.shape
    

    logM = sel_tau[None,:,None,None] + torch.where(idx_q[:,:,:,None] == idx_k[:,:,None,:], 0, -1e5)  # [B, H, N, N]

    weights = [logM[:,:,0,None,:]]
    mask = torch.arange(0, N) == 0
    umask = torch.tril(torch.ones((B, H, N, N), dtype = torch.bool))

    for i in range(1, N):
        last_row = torch.roll(weights[-1],1)
        last_row = torch.where(mask, -1e5, last_row)
        weights.append(lse0(last_row) + logM[:,:,i,None,:])

    attn = torch.where(umask, torch.cat(weights, dim = -2), -1e5)
    Zmax = torch.clamp_min(attn.max(-1, keepdim=True).values, betas[...,None])
    attn = torch.exp(attn - Zmax)
    out = torch.matmul(attn, values) + fallback_o * torch.exp(betas[...,None] - Zmax)

    return torch.nn.functional.rms_norm(out, (values.shape[-1],), eps=1e-7)


@triton.jit
def add_mul_scan(la: tl.tensor, lb: tl.tensor, ra: tl.tensor, rb: tl.tensor):
    return la * ra, libdevice.fma(lb, ra, rb)


@triton.jit
def compute_qd_fused_partial_log_gemm(q_soft: tl.tensor, logk: tl.tensor, rtau: tl.tensor, eps: tl.constexpr = 1e-3):
    dim = q_soft.shape[-1]
    k_soft = logk
    m_qk = rtau + (tl.log2(1-dim*eps)) + libdevice.fast_log2f((eps / (1-dim*eps)) + tl.dot(q_soft, tl.trans(k_soft))) # [QCS, DCS]
    return m_qk, k_soft


@triton.jit
def compute_qd_fused_partial_hard_gemm(idx_q: tl.tensor, idx_k: tl.tensor, rtau: tl.tensor):
    m_qk = rtau + tl.where(idx_q[:, None] == idx_k[None, :], 0.0, -1e5) # [QCS, DCS]
    return m_qk


@triton.jit
def tril(N: tl.constexpr, dtype: tl.dtype) -> tl.tensor:
    idx = tl.arange(0, N)
    return (idx[:, None] >= idx[None, :]).to(dtype)

@triton.jit
def triu(N: tl.constexpr, dtype: tl.dtype) -> tl.tensor:
    idx = tl.arange(0, N)
    return (idx[:, None] <= idx[None, :]).to(dtype)

@triton.jit
def lse_scan(lc: tl.tensor, rc: tl.tensor):
    mx = tl.maximum(lc, rc)
    nabs = -tl.abs(lc - rc)
    p = tl.maximum(nabs, -4.3125)
    logs = libdevice.fma_rn(p, libdevice.fma_rn(p, libdevice.fma_rn(p, -4.553042169205498771e-03, -5.756650101889571047e-02), -2.793604998770852221e-01), nabs)
    #logs = nabs + p * (-2.793604998770852221e-01 + p * (-5.756650101889571047e-02 + p * (-4.553042169205498771e-03)))
    
    return mx + tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;",'=f,f', (logs,), (tl.float32), is_pure=True,pack=1)

@triton.jit
def neg_lse_scan(lc: tl.tensor, rc: tl.tensor):
    mi = tl.minimum(lc, rc)
    nabs = -tl.abs(lc - rc)
    p = tl.maximum(nabs, -4.3125)
    logs = libdevice.fma_rn(p, libdevice.fma_rn(p, libdevice.fma_rn(p, -4.553042169205498771e-03, -5.756650101889571047e-02), -2.793604998770852221e-01), nabs)
    #logs = nabs + p * (-2.793604998770852221e-01 + p * (-5.756650101889571047e-02 + p * (-4.553042169205498771e-03)))
    
    return mi - tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;",'=f,f', (logs,), (tl.float32), is_pure=True,pack=1)

@triton.jit
def lse_scan_comp(la: tl.tensor, lb: tl.tensor, ra: tl.tensor, rb: tl.tensor):
    return la + ra, lse_scan(lb + ra, rb)


@triton.jit
def recompute_qd_block(idx_q: tl.tensor, idx_d: tl.tensor, logM_prev: tl.tensor, logM_succ:tl.tensor, init_Vs: tl.tensor, Q_CHUNK_SIZE: tl.constexpr, D_CHUNK_SIZE: tl.constexpr, RET_PREV: tl.constexpr = True):
    col_idx = tl.arange(0, Q_CHUNK_SIZE)[:, None] + tl.arange(0, D_CHUNK_SIZE)[None, :] + 1
    col0_idx = tl.minimum(col_idx, Q_CHUNK_SIZE - 1)
    col1_idx = tl.maximum(col_idx - Q_CHUNK_SIZE, 0)


    NL = tl.where(col_idx >= Q_CHUNK_SIZE, tl.gather(logM_succ, col1_idx, 1), tl.gather(logM_prev, col0_idx, 1)) # [QCS, DCS]
    NL = tl.flip(NL, dim = 1)
    q_indices = (tl.arange(0, Q_CHUNK_SIZE) + idx_q * Q_CHUNK_SIZE)[:,None]
    d_indices = (tl.arange(0, D_CHUNK_SIZE) + idx_d * D_CHUNK_SIZE)[None,:]
    avail_mask = q_indices >= d_indices

    NL_C = tl.where(avail_mask, NL, 0)
    NL_D = tl.where(avail_mask, NL, -1e5)

    H, V = tl.associative_scan((NL_C, NL_D), axis = 0, combine_fn=lse_scan_comp)
    V = lse_scan(init_Vs[None, :] + H, V)

    return V, tl.where(RET_PREV, logM_prev, logM_succ)



@triton.jit
def recompute_qd_block_lastrow(idx_q: tl.tensor, idx_d: tl.tensor, logM_prev: tl.tensor, logM_succ:tl.tensor, Q_CHUNK_SIZE: tl.constexpr, D_CHUNK_SIZE: tl.constexpr, RET_PREV: tl.constexpr = True):
    col_idx = tl.arange(0, Q_CHUNK_SIZE)[:, None] + tl.arange(0, D_CHUNK_SIZE)[None, :] + 1
    col0_idx = tl.minimum(col_idx, Q_CHUNK_SIZE - 1)
    col1_idx = tl.maximum(col_idx - Q_CHUNK_SIZE, 0)


    NL = tl.where(col_idx >= Q_CHUNK_SIZE, tl.gather(logM_succ, col1_idx, 1), tl.gather(logM_prev, col0_idx, 1)) # [QCS, DCS]
    NL = tl.flip(NL, dim = 1)
    q_indices = (tl.arange(0, Q_CHUNK_SIZE) + idx_q * Q_CHUNK_SIZE)[:,None]
    d_indices = (tl.arange(0, D_CHUNK_SIZE) + idx_d * D_CHUNK_SIZE)[None,:]
    avail_mask = q_indices >= d_indices

    NL_C = tl.where(avail_mask, NL, 0)
    NL_D = tl.where(avail_mask, (tl.cumsum(NL_C, axis = 0) - NL_C), 1e5)
    V = tl.reduce(NL_D, axis = 0, combine_fn=neg_lse_scan, keep_dims=True)
    C = tl.sum(NL_C, axis = 0, keep_dims=True)

    return C - V, C


@triton.jit
def attn_bwd_kernel_hh(
    logq, logk, values, rmax, do, dV, dlm, # [B, H, N, C]
    V_ptr, H_ptr, J_ptr, # [B, H, Q_CHUNKS, N]
    tau_ptr, # [H]
    qk_stride0, qk_stride1, qk_stride2, qk_stride3,
    vo_stride0, vo_stride1, vo_stride2, vo_stride3,
    uvh_stride0, uvh_stride1, uvh_stride2, uvh_stride3,
    lm_stride0, lm_stride1, lm_stride2, lm_stride3,
    N_BATCH, N_HEADS, N_CTX, N_VOCAB: tl.constexpr,
    Q_CHUNK_SIZE: tl.constexpr, N_HEADDIM: tl.constexpr, SEPS: tl.constexpr
):
    pid_bh = tl.program_id(0)
    idx_q = tl.program_id(1)

    logq = logq + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    logk = logk + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    do = do + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)
    values = values + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)
    dV = dV + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)
    dlm = dlm + lm_stride0 * (pid_bh // N_HEADS) + lm_stride1 * (pid_bh % N_HEADS)


    V_ptr = V_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q - 1) + uvh_stride3 * tl.arange(0, Q_CHUNK_SIZE)
    H_ptr = H_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q) + uvh_stride3 * tl.arange(0, Q_CHUNK_SIZE)
    J_ptr = J_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q) + uvh_stride3 * tl.arange(0, Q_CHUNK_SIZE)
    rmax_ptr = rmax + (N_HEADS * N_CTX) * (pid_bh // N_HEADS) + N_CTX * (pid_bh % N_HEADS) + idx_q * Q_CHUNK_SIZE + tl.arange(0, Q_CHUNK_SIZE)
    
    dV_desc = tl.make_tensor_descriptor(dV, (N_CTX, N_HEADDIM), (vo_stride2, vo_stride3), (Q_CHUNK_SIZE, N_HEADDIM))

    rtau = tl.load(tau_ptr + (pid_bh % N_HEADS))
    RCP_LN2: tl.constexpr = 1.4426950216
    rtau = RCP_LN2 * rtau

    dlm_ptrs = tl.make_block_ptr(
        dlm,
        shape = (N_CTX, N_CTX),
        strides = (lm_stride2, lm_stride3),
        offsets=(idx_q * Q_CHUNK_SIZE, idx_q * Q_CHUNK_SIZE),
        block_shape=(Q_CHUNK_SIZE, Q_CHUNK_SIZE),
        order = (1, 0)
    )

    logq_ptrs = tl.make_block_ptr(
        logq,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    )
    soft_q = tl.load(logq_ptrs) # [Q_CHUNK_SIZE, N_VOCAB]
    
    logk_ptrs = tl.make_block_ptr(
        logk,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    ) 

    v_block_ptr = tl.make_block_ptr(
        values,
        shape = (N_CTX, N_HEADDIM),
        strides = (vo_stride2, vo_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_HEADDIM),
        order = (1, 0)
    )
    do_block_ptr = tl.make_block_ptr(
        do,
        shape = (Q_CHUNK_SIZE, N_HEADDIM),
        strides = (vo_stride2, vo_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape=(Q_CHUNK_SIZE, N_HEADDIM),
        order = (1, 0)
    )

    dAV = tl.load(do_block_ptr) # [QCS, D]
    Zf = tl.load(rmax_ptr) # [QCS]


    logM_succ, succ_soft_k = compute_qd_fused_partial_log_gemm(soft_q, tl.load(logk_ptrs), rtau, SEPS)

    prev_attn_Vs = tl.full((Q_CHUNK_SIZE, Q_CHUNK_SIZE), float("-inf"), dtype = tl.float32)

    v_succs = tl.load(v_block_ptr) # [Q_CHUNK_SIZE, N_HEADDIM]
    dQK_prev = tl.flip(tl.dot(dAV, tl.trans(v_succs)), 1)


    col_idx = tl.arange(0, Q_CHUNK_SIZE)[:, None] + tl.arange(0, Q_CHUNK_SIZE)[None, :] + 1
    col0_idx = tl.minimum(col_idx, Q_CHUNK_SIZE - 1)
    col1_idx = tl.maximum(col_idx - Q_CHUNK_SIZE, 0)

    inv_col_idx = - tl.arange(0, Q_CHUNK_SIZE)[:, None] + tl.arange(0, Q_CHUNK_SIZE)[None, :] - 1 + Q_CHUNK_SIZE
    inv_col0_idx = tl.minimum(inv_col_idx, Q_CHUNK_SIZE - 1)
    inv_col1_idx = tl.maximum(inv_col_idx - Q_CHUNK_SIZE, 0)

    v_idx = tl.full((1, Q_CHUNK_SIZE), 0, dtype = tl.int32)

    for idx_d in tl.range(0, idx_q + 1):
        init_Vs = tl.load(V_ptr, mask=idx_q>0, other=float("-inf"))
        logk_ptrs = tl.advance(logk_ptrs, (-Q_CHUNK_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (-Q_CHUNK_SIZE, 0))

        logk_vals = tl.load(logk_ptrs, boundary_check=(0,), padding_option='zero')
        logM_prev, prev_soft_k = compute_qd_fused_partial_log_gemm(soft_q, logk_vals, rtau, SEPS)
        V_n, logM_succ = recompute_qd_block(idx_q, idx_d, logM_prev, logM_succ, init_Vs, Q_CHUNK_SIZE, Q_CHUNK_SIZE, RET_PREV=True)

        attns_Vs = tl.flip(tl.where(col_idx >= Q_CHUNK_SIZE, tl.gather(V_n, col1_idx, 1), tl.gather(prev_attn_Vs, col0_idx, 1)), 1)
        
        tl.store(dlm_ptrs, attns_Vs)
        dlm_ptrs = tl.advance(dlm_ptrs, (0, -Q_CHUNK_SIZE))
        
        attns = tl.exp2(attns_Vs - Zf[:,None]).to(tl.bfloat16)
        dV_out = tl.dot(tl.trans(attns), dAV)
        dV_desc.atomic_add(((idx_q - idx_d) * Q_CHUNK_SIZE, 0), dV_out)

        v_prev = tl.load(v_block_ptr, boundary_check=(0,), padding_option='zero') # [Q_CHUNK_SIZE, N_HEADDIM]
        dQK_succ = tl.flip(tl.dot(dAV, tl.trans(v_prev)), 1)

        dQK = tl.where(inv_col_idx >= Q_CHUNK_SIZE, tl.gather(dQK_succ, inv_col1_idx, 1), tl.gather(dQK_prev, inv_col0_idx, 1))
        beta = dQK * tl.exp2(V_n - Zf[:, None])
        alpha = tl.fdiv(1., 1. + tl.exp2(-V_n))

        H_n, J_n = tl.associative_scan((alpha, beta), axis = 0, reverse=True, combine_fn=add_mul_scan) # [QCS, QCS]

        tl.store(H_ptr, tl.ravel(tl.gather(H_n, v_idx, axis = 0)))
        tl.store(J_ptr, tl.ravel(tl.gather(J_n, v_idx, axis = 0)))

        dQK_prev = dQK_succ
        prev_attn_Vs = V_n
        V_ptr += uvh_stride3 * Q_CHUNK_SIZE
        H_ptr += uvh_stride3 * Q_CHUNK_SIZE
        J_ptr += uvh_stride3 * Q_CHUNK_SIZE

        
@triton.jit
def chunk_passing_kernel_bwd(
    H_ptr, J_ptr, # [B, H, Q_CHUNKS, N]
    uvh_stride0, uvh_stride1, uvh_stride2, uvh_stride3,
    N_BATCH, N_HEADS, N_CTX, Q_CHUNKS, D_CHUNK_SIZE:tl.constexpr
):
    
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    J_ptr = J_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS)
    H_ptr = H_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS)
    
    J_block_ptr = tl.make_block_ptr(
        J_ptr,
        shape = (Q_CHUNKS, N_CTX),
        strides = (uvh_stride2, uvh_stride3),
        offsets=((Q_CHUNKS - 1), D_CHUNK_SIZE * pid_d),
        block_shape=(1, D_CHUNK_SIZE),
        order=(1,0)
    )
    H_block_ptr = tl.make_block_ptr(
        H_ptr,
        shape = (Q_CHUNKS, N_CTX),
        strides = (uvh_stride2, uvh_stride3),
        offsets=((Q_CHUNKS - 1), D_CHUNK_SIZE * pid_d),
        block_shape=(1, D_CHUNK_SIZE),
        order=(1,0)
    )

    J_last = tl.full((1, D_CHUNK_SIZE), 0, dtype = tl.float32)

    for i in tl.range(Q_CHUNKS - 1, -1, -1):
        Js = tl.load(J_block_ptr)
        Hs = tl.load(H_block_ptr)

        Jnew = Js + J_last * Hs 

        tl.store(J_block_ptr, Jnew)
        
        J_last = Jnew
        J_block_ptr = tl.advance(J_block_ptr, (-1, 0))
        H_block_ptr = tl.advance(H_block_ptr, (-1, 0))

@triton.jit
def attn_bwd_kernel_post_hh(
    logq, logk, values, rmax, do, dq, dk, dlm, # [B, H, N, C]
    V_ptr, J_ptr, # [B, H, Q_CHUNKS, N]
    tau_ptr, dtau_ptr, # [H]
    qk_stride0, qk_stride1, qk_stride2, qk_stride3,
    vo_stride0, vo_stride1, vo_stride2, vo_stride3,
    uvh_stride0, uvh_stride1, uvh_stride2, uvh_stride3,
    lm_stride0, lm_stride1, lm_stride2, lm_stride3,
    N_BATCH, N_HEADS, N_CTX, Q_CHUNKS, N_VOCAB: tl.constexpr,
    Q_CHUNK_SIZE: tl.constexpr, N_HEADDIM: tl.constexpr, SEPS: tl.constexpr
):
    pid_bh = tl.program_id(0)
    idx_q = tl.program_id(1)

    logq = logq + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    logk = logk + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    dk = dk + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    dq = dq + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    dlm = dlm + lm_stride0 * (pid_bh // N_HEADS) + lm_stride1 * (pid_bh % N_HEADS)

    values = values + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)
    do = do + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)

    V_ptr = V_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q - 1) + uvh_stride3 * tl.arange(0, Q_CHUNK_SIZE)
    J_ptr = J_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q + 1) + uvh_stride3 * tl.arange(0, Q_CHUNK_SIZE)
    rmax_ptr = rmax + (N_HEADS * N_CTX) * (pid_bh // N_HEADS) + N_CTX * (pid_bh % N_HEADS) + idx_q * Q_CHUNK_SIZE + tl.arange(0, Q_CHUNK_SIZE)
    
    dk_desc = tl.make_tensor_descriptor(dk, (N_CTX, N_VOCAB), (qk_stride2, qk_stride3), (Q_CHUNK_SIZE, N_VOCAB))

    rtau = tl.load(tau_ptr + (pid_bh % N_HEADS))
    RCP_LN2: tl.constexpr = 1.4426950216
    rtau = RCP_LN2 * rtau

    logq_ptrs = tl.make_block_ptr(
        logq,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    )
    soft_q = tl.load(logq_ptrs) # [Q_CHUNK_SIZE, N_VOCAB]
    
    logk_ptrs = tl.make_block_ptr(
        logk,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    ) 

    v_block_ptr = tl.make_block_ptr(
        values,
        shape = (N_CTX, N_HEADDIM),
        strides = (vo_stride2, vo_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_HEADDIM),
        order = (1, 0)
    )

    do_block_ptr = tl.make_block_ptr(
        do,
        shape = (N_CTX, N_HEADDIM),
        strides = (vo_stride2, vo_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape=(Q_CHUNK_SIZE, N_HEADDIM),
        order = (1, 0)
    )

    dlm_ptrs = tl.make_block_ptr(
        dlm,
        shape = (N_CTX, N_CTX),
        strides = (lm_stride2, lm_stride3),
        offsets=(idx_q * Q_CHUNK_SIZE, idx_q * Q_CHUNK_SIZE),
        block_shape=(Q_CHUNK_SIZE, Q_CHUNK_SIZE),
        order = (1, 0)
    )

    dAV = tl.load(do_block_ptr) # [QCS, D]
    Zf = tl.load(rmax_ptr) # [QCS]


    logM_succ, succ_soft_k = compute_qd_fused_partial_log_gemm(soft_q, tl.load(logk_ptrs), rtau, SEPS)

    v_succs = tl.load(v_block_ptr) # [Q_CHUNK_SIZE, N_HEADDIM]
    dQK_prev = tl.flip(tl.dot(dAV, tl.trans(v_succs)), 1)

    dQ_accum = tl.zeros(soft_q.shape, dtype = tl.float32)
    dlogM_qd_prev = tl.zeros((Q_CHUNK_SIZE, Q_CHUNK_SIZE), dtype = tl.float32)
    dtaus = tl.zeros((1,), dtype = tl.float32)

    col_idx = tl.arange(0, Q_CHUNK_SIZE)[:, None] + tl.arange(0, Q_CHUNK_SIZE)[None, :] + 1
    col0_idx = tl.minimum(col_idx, Q_CHUNK_SIZE - 1)
    col1_idx = tl.maximum(col_idx - Q_CHUNK_SIZE, 0)

    inv_col_idx = - tl.arange(0, Q_CHUNK_SIZE)[:, None] + tl.arange(0, Q_CHUNK_SIZE)[None, :] - 1 + Q_CHUNK_SIZE
    inv_col0_idx = tl.minimum(inv_col_idx, Q_CHUNK_SIZE - 1)
    inv_col1_idx = tl.maximum(inv_col_idx - Q_CHUNK_SIZE, 0)

    soft_q_fp32 = soft_q.to(tl.float32)

    for idx_d in tl.range(0, idx_q + 1):
        init_Vs = tl.load(V_ptr, mask=idx_q>0, other=float("-inf"))
        init_Js = tl.load(J_ptr, mask=idx_q<Q_CHUNKS-1, other=0)

        logk_ptrs = tl.advance(logk_ptrs, (-Q_CHUNK_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (-Q_CHUNK_SIZE, 0))

        logk_vals = tl.load(logk_ptrs, boundary_check=(0,), padding_option='zero')
        logM_prev, prev_soft_k = compute_qd_fused_partial_log_gemm(soft_q, logk_vals, rtau, SEPS)
        V_n, logM_succ_next = recompute_qd_block(idx_q, idx_d, logM_prev, logM_succ, init_Vs, Q_CHUNK_SIZE, Q_CHUNK_SIZE, RET_PREV=True)

        v_prev = tl.load(v_block_ptr, boundary_check=(0,), padding_option='zero') # [Q_CHUNK_SIZE, N_HEADDIM]
        dQK_succ = tl.flip(tl.dot(dAV, tl.trans(v_prev)), 1)

        dQK = tl.where(inv_col_idx >= Q_CHUNK_SIZE, tl.gather(dQK_succ, inv_col1_idx, 1), tl.gather(dQK_prev, inv_col0_idx, 1))
        
        beta = dQK * tl.exp2(V_n - Zf[:, None])
        alpha = tl.fdiv(1., 1. + tl.exp2(-V_n))

        H_n, J_n = tl.associative_scan((alpha, beta), axis = 0, reverse=True, combine_fn=add_mul_scan) # [QCS, QCS]
        dlogM_qd_succ = J_n + H_n * init_Js[None, :]

        dlogM = tl.flip(tl.where(col_idx >= Q_CHUNK_SIZE, tl.gather(dlogM_qd_succ, col1_idx, 1), tl.gather(dlogM_qd_prev, col0_idx, 1)), 1)
        dlogM_rowsum = tl.sum(dlogM, axis=0)
        dlogM_colsum = tl.sum(dlogM, axis=1)
        dtaus += tl.sum(dlogM_rowsum, axis=0)
        dM = (dlogM * tl.exp2(rtau-logM_succ))

        tl.store(dlm_ptrs, dlogM)
        dlm_ptrs = tl.advance(dlm_ptrs, (0, -Q_CHUNK_SIZE))

        succ_soft_k_fp32 = succ_soft_k.to(tl.float32)
        

        dK = succ_soft_k_fp32 * (tl.dot(tl.trans(dM), soft_q_fp32 * (1 - SEPS * N_VOCAB) + SEPS, input_precision='ieee')  - dlogM_rowsum[:,None])
        dk_desc.atomic_add(((idx_q - idx_d) * Q_CHUNK_SIZE, 0), dK)
        dQ_accum += tl.dot(dM, succ_soft_k_fp32 * (1 - SEPS * N_VOCAB) + SEPS, input_precision='ieee') - dlogM_colsum[:,None]

        logM_succ = logM_succ_next # [QCS, QCS]
        dQK_prev = dQK_succ # [QCS, QCS]
        dlogM_qd_prev = dlogM_qd_succ # [QCS, QCS]
        succ_soft_k = prev_soft_k # [QCS, VOC]

        V_ptr += uvh_stride3 * Q_CHUNK_SIZE
        J_ptr += uvh_stride3 * Q_CHUNK_SIZE

    #dtaus *= -rtau * rtau * (1.0/(RCP_LN2*RCP_LN2))
    tl.atomic_add((dtau_ptr + (pid_bh % N_HEADS))[None], dtaus)
    dQ_accum = dQ_accum * soft_q_fp32
    dq_ptrs = tl.make_block_ptr(
        dq,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    )
    tl.store(dq_ptrs, dQ_accum)

@triton.jit
def perprocess_kernel_hh(
    logq, logk, # [B, H, N, C]
    V_ptr, H_ptr, # [B, H, Q_CHUNKS, N]
    tau_ptr, # [H]
    qk_stride0, qk_stride1, qk_stride2, qk_stride3,
    uvh_stride0, uvh_stride1, uvh_stride2, uvh_stride3,
    N_BATCH, N_HEADS, N_CTX, Q_CHUNKS, D_CHUNKS, N_VOCAB: tl.constexpr,
    Q_CHUNK_SIZE: tl.constexpr, D_CHUNK_SIZE: tl.constexpr, SEPS :tl.constexpr):
    RCP_LN2: tl.constexpr = 1.4426950216

    pid_bh = tl.program_id(0)
    pid_qd = tl.program_id(1)

    logq = logq + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    logk = logk + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)

    idx_q, idx_d = pid_qd // D_CHUNKS, pid_qd % D_CHUNKS

    if idx_q < idx_d:
        return

    V_ptr = V_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * idx_q + uvh_stride3 * idx_d * D_CHUNK_SIZE
    H_ptr = H_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * idx_q + uvh_stride3 * idx_d * D_CHUNK_SIZE
    
    rtau = tl.load(tau_ptr + (pid_bh % N_HEADS))
    rtau = RCP_LN2 * rtau

    logq_ptrs = tl.make_block_ptr(
        logq,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    )
    logk_ptrs_perv = tl.make_block_ptr(
        logk,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE - ((idx_d + 1) * D_CHUNK_SIZE), 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    )
    logk_ptrs_succ = tl.make_block_ptr(
        logk,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = ((idx_q + 1) * Q_CHUNK_SIZE - ((idx_d + 1) * D_CHUNK_SIZE), 0),
        block_shape = (D_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    ) 

    q_soft = tl.load(logq_ptrs) # [Q_CHUNK_SIZE, N_VOCAB]
    logk_prevs = tl.load(logk_ptrs_perv, boundary_check=(0,), padding_option='zero') # [Q_CHUNK_SIZE, N_VOCAB]

    m_qk_prev, soft_k_prev = compute_qd_fused_partial_log_gemm(q_soft, logk_prevs, rtau, SEPS)
    

    logk_succs = tl.load(logk_ptrs_succ, boundary_check=(0,), padding_option='zero')
    m_qk_succ, soft_k_succ = compute_qd_fused_partial_log_gemm(q_soft, logk_succs, rtau, SEPS)
    d_idx = tl.arange(0, D_CHUNK_SIZE)[None]

    V2, C2 = recompute_qd_block_lastrow(idx_q, idx_d, m_qk_prev, m_qk_succ, Q_CHUNK_SIZE, D_CHUNK_SIZE)
    
    tl.store(V_ptr + d_idx, V2)
    tl.store(H_ptr + d_idx, C2)



@triton.jit
def perprocess_kernel_hh_hard(
    hard_q, hard_k, # [B, H, N]
    V_ptr, H_ptr, # [B, H, Q_CHUNKS, N]
    tau_ptr, # [H]
    qk_stride0, qk_stride1,
    uvh_stride0, uvh_stride1, uvh_stride2, uvh_stride3,
    N_BATCH, N_HEADS, N_CTX, Q_CHUNKS, D_CHUNKS,
    Q_CHUNK_SIZE: tl.constexpr, D_CHUNK_SIZE: tl.constexpr):
    RCP_LN2: tl.constexpr = 1.4426950216

    pid_bh = tl.program_id(0)
    pid_qd = tl.program_id(1)

    hard_q = hard_q + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    hard_k = hard_k + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)

    idx_q, idx_d = pid_qd // D_CHUNKS, pid_qd % D_CHUNKS

    if idx_q < idx_d:
        return

    V_ptr = V_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * idx_q + uvh_stride3 * idx_d * D_CHUNK_SIZE
    H_ptr = H_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * idx_q + uvh_stride3 * idx_d * D_CHUNK_SIZE
    
    rtau = tl.load(tau_ptr + (pid_bh % N_HEADS))
    rtau = RCP_LN2 * rtau

    logq_ptrs = tl.make_block_ptr(
        hard_q,
        shape = (N_CTX,),
        strides = (1,),
        offsets = (idx_q * Q_CHUNK_SIZE,),
        block_shape = (Q_CHUNK_SIZE,),
        order = (0,)
    )
    logk_ptrs_perv = tl.make_block_ptr(
        hard_k,
        shape = (N_CTX,),
        strides = (1,),
        offsets = (idx_q * Q_CHUNK_SIZE - ((idx_d + 1) * D_CHUNK_SIZE), ),
        block_shape = (Q_CHUNK_SIZE, ),
        order = (0, )
    )
    logk_ptrs_succ = tl.make_block_ptr(
        hard_k,
        shape = (N_CTX, ),
        strides = (1, ),
        offsets = ((idx_q + 1) * Q_CHUNK_SIZE - ((idx_d + 1) * D_CHUNK_SIZE), ),
        block_shape = (D_CHUNK_SIZE, ),
        order = (0, )
    ) 

    q_soft = tl.load(logq_ptrs) # [Q_CHUNK_SIZE, ]
    logk_prevs = tl.load(logk_ptrs_perv, boundary_check=(0,), padding_option='zero') # [Q_CHUNK_SIZE, ]

    m_qk_prev = compute_qd_fused_partial_hard_gemm(q_soft, logk_prevs, rtau)
    

    logk_succs = tl.load(logk_ptrs_succ, boundary_check=(0,), padding_option='zero')
    m_qk_succ = compute_qd_fused_partial_hard_gemm(q_soft, logk_succs, rtau)
    d_idx = tl.arange(0, D_CHUNK_SIZE)[None]

    V2, C2 = recompute_qd_block_lastrow(idx_q, idx_d, m_qk_prev, m_qk_succ, Q_CHUNK_SIZE, D_CHUNK_SIZE)
    
    tl.store(V_ptr + d_idx, V2)
    tl.store(H_ptr + d_idx, C2)




@triton.jit
def chunk_passing_kernel(
    V_ptr, H_ptr, # [B, H, Q_CHUNKS, N]
    uvh_stride0, uvh_stride1, uvh_stride2, uvh_stride3,
    N_BATCH, N_HEADS, N_CTX, Q_CHUNKS, D_CHUNK_SIZE:tl.constexpr
):
    
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    V_ptr = V_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS)
    H_ptr = H_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS)
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr,
        shape = (Q_CHUNKS, N_CTX),
        strides = (uvh_stride2, uvh_stride3),
        offsets=(0, D_CHUNK_SIZE * pid_d),
        block_shape=(1, D_CHUNK_SIZE),
        order=(1,0)
    )
    H_block_ptr = tl.make_block_ptr(
        H_ptr,
        shape = (Q_CHUNKS, N_CTX),
        strides = (uvh_stride2, uvh_stride3),
        offsets=(0, D_CHUNK_SIZE * pid_d),
        block_shape=(1, D_CHUNK_SIZE),
        order=(1,0)
    )

    V_last = tl.full((1, D_CHUNK_SIZE), -1e5, dtype = tl.float32)

    for i in tl.range(0, Q_CHUNKS):
        Vs = tl.load(V_block_ptr)
        Hs = tl.load(H_block_ptr)

        #Z = tl.maximum(V_last + Hs, Vs)
        Vnew = lse_scan(V_last + Hs, Vs) #tl.log2(tl.exp2(Vs - Z) + tl.exp2(V_last + Hs - Z)) + Z

        tl.store(V_block_ptr, Vnew)
        
        V_last = Vnew
        V_block_ptr = tl.advance(V_block_ptr, (1, 0))
        H_block_ptr = tl.advance(H_block_ptr, (1, 0))

@triton.jit
def attn_fwd_kernel_hh(
    logq, logk, values, output, rmax, # [B, H, N, C]
    V_ptr, # [B, H, Q_CHUNKS, N]
    tau_ptr, # [H]
    qk_stride0, qk_stride1, qk_stride2, qk_stride3,
    vo_stride0, vo_stride1, vo_stride2, vo_stride3,
    uvh_stride0, uvh_stride1, uvh_stride2, uvh_stride3,
    N_BATCH, N_HEADS, N_CTX, Q_CHUNKS, N_VOCAB: tl.constexpr,
    Q_CHUNK_SIZE: tl.constexpr, N_HEADDIM: tl.constexpr, SEPS: tl.constexpr
):
    pid_bh = tl.program_id(0)
    idx_q = tl.program_id(1)

    logq = logq + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    logk = logk + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    values = values + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)
    output = output + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)

    V_ptr = V_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q - 1) + uvh_stride3 * (tl.arange(0, Q_CHUNK_SIZE) + Q_CHUNK_SIZE * (idx_q - 1))
    rmax_ptr = rmax + (N_HEADS * N_CTX) * (pid_bh // N_HEADS) + N_CTX * (pid_bh % N_HEADS) + idx_q * Q_CHUNK_SIZE + tl.arange(0, Q_CHUNK_SIZE)
    mask = idx_q * Q_CHUNK_SIZE + tl.arange(0, Q_CHUNK_SIZE) < N_CTX

    rtau = tl.load(tau_ptr + (pid_bh % N_HEADS))
    RCP_LN2: tl.constexpr = 1.4426950216
    rtau = RCP_LN2 * rtau

    logq_ptrs = tl.make_block_ptr(
        logq,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    )
    soft_q = tl.load(logq_ptrs) # [Q_CHUNK_SIZE, N_VOCAB]
    

    logk_ptrs = tl.make_block_ptr(
        logk,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (0, 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    ) 
    v_block_ptr = tl.make_block_ptr(
        values,
        shape = (N_CTX, N_HEADDIM),
        strides = (vo_stride2, vo_stride3),
        offsets = (0, 0),
        block_shape = (Q_CHUNK_SIZE, N_HEADDIM),
        order = (1, 0)
    )

    o_block_ptr = tl.make_block_ptr(
        output,
        shape = (Q_CHUNK_SIZE, N_HEADDIM),
        strides = (vo_stride2, vo_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape=(Q_CHUNK_SIZE, N_HEADDIM),
        order = (1, 0)
    )

    # prepare qd_succ
    
    prev_logM = tl.full((Q_CHUNK_SIZE, Q_CHUNK_SIZE), float("-inf"), dtype = tl.float32)
    logk_vals = tl.load(logk_ptrs, boundary_check=(0,), padding_option='zero')
    succ_logM, succ_soft_k = compute_qd_fused_partial_log_gemm(soft_q, logk_vals, rtau, SEPS)
    succ_attn_Vs, prev_logM = recompute_qd_block(idx_q, idx_q, prev_logM, succ_logM, tl.full((Q_CHUNK_SIZE,),float("-inf"), dtype = tl.float32), Q_CHUNK_SIZE, Q_CHUNK_SIZE, RET_PREV=False)

    attn_out = tl.load(o_block_ptr).to(tl.float32)
    rmax_out = tl.load(rmax_ptr[:,None], mask=mask[:,None], other=0) * RCP_LN2

    col_idx = tl.arange(0, Q_CHUNK_SIZE)[:, None] + tl.arange(0, Q_CHUNK_SIZE)[None, :] + 1
    col0_idx = tl.minimum(col_idx, Q_CHUNK_SIZE - 1)
    col1_idx = tl.maximum(col_idx - Q_CHUNK_SIZE, 0)

    for idx_d in tl.range(idx_q, -1, -1, num_stages=2):
        logk_ptrs = tl.advance(logk_ptrs, (Q_CHUNK_SIZE, 0))

        v_succs = tl.load(v_block_ptr) # [Q_CHUNK_SIZE, N_HEADDIM]
        
        V_n = tl.full((Q_CHUNK_SIZE, Q_CHUNK_SIZE), float("-inf"), dtype = tl.float32)

        if idx_d != 0:
            init_Vs = tl.load(V_ptr)
            logk_vals = tl.load(logk_ptrs, boundary_check=(0,), padding_option='zero')
            succ_logM, soft_k_succ = compute_qd_fused_partial_log_gemm(soft_q, logk_vals, rtau, SEPS)
            V_n, prev_logM = recompute_qd_block(idx_q, idx_d - 1, prev_logM, succ_logM, init_Vs, Q_CHUNK_SIZE, Q_CHUNK_SIZE, RET_PREV=False)

        attns_Vs = tl.flip(tl.where(col_idx >= Q_CHUNK_SIZE, tl.gather(succ_attn_Vs, col1_idx, 1), tl.gather(V_n, col0_idx, 1)), 1)
        
        new_rmax = tl.maximum(rmax_out, tl.max(attns_Vs, axis = 1, keep_dims = True))
        attn_out = tl.dot(tl.exp2(attns_Vs - new_rmax).to(tl.bfloat16), v_succs, attn_out * tl.exp2(rmax_out - new_rmax))
        rmax_out = new_rmax

        succ_attn_Vs = V_n

        v_block_ptr = tl.advance(v_block_ptr, (Q_CHUNK_SIZE, 0))
        V_ptr -= uvh_stride3 * Q_CHUNK_SIZE
    
    

    rsqrt_weight = tl.rsqrt(1e-7 + (1.0/N_HEADDIM) * tl.sum(attn_out * attn_out, axis=1, keep_dims=True))
    attn_out = attn_out * rsqrt_weight

    tl.store(o_block_ptr, attn_out.to(output.dtype.element_ty))
    tl.store(rmax_ptr[:,None], rmax_out - tl.log2(rsqrt_weight), mask=mask[:,None])

@triton.jit
def attn_fwd_kernel_hard_hh(
    logq, logk, values, output, rmax, # [B, H, N, C]
    V_ptr, # [B, H, Q_CHUNKS, N]
    tau_ptr, # [H]
    qk_stride0, qk_stride1, 
    vo_stride0, vo_stride1, vo_stride2, vo_stride3,
    uvh_stride0, uvh_stride1, uvh_stride2, uvh_stride3,
    N_BATCH, N_HEADS, N_CTX, Q_CHUNKS, 
    Q_CHUNK_SIZE: tl.constexpr, N_HEADDIM: tl.constexpr
):
    pid_bh = tl.program_id(0)
    idx_q = tl.program_id(1)

    logq = logq + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    logk = logk + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    values = values + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)
    output = output + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)

    V_ptr = V_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q - 1) + uvh_stride3 * (tl.arange(0, Q_CHUNK_SIZE) + Q_CHUNK_SIZE * (idx_q - 1))
    rmax_ptr = rmax + (N_HEADS * N_CTX) * (pid_bh // N_HEADS) + N_CTX * (pid_bh % N_HEADS) + idx_q * Q_CHUNK_SIZE + tl.arange(0, Q_CHUNK_SIZE)
    mask = idx_q * Q_CHUNK_SIZE + tl.arange(0, Q_CHUNK_SIZE) < N_CTX

    rtau = tl.load(tau_ptr + (pid_bh % N_HEADS))
    RCP_LN2: tl.constexpr = 1.4426950216
    rtau = RCP_LN2 * rtau

    logq_ptrs = tl.make_block_ptr(
        logq,
        shape = (N_CTX, ),
        strides = (1,),
        offsets = (idx_q * Q_CHUNK_SIZE,),
        block_shape = (Q_CHUNK_SIZE,),
        order = (0,)
    )
    soft_q = tl.load(logq_ptrs) # [Q_CHUNK_SIZE, N_VOCAB]
    

    logk_ptrs = tl.make_block_ptr(
        logk,
        shape = (N_CTX, ),
        strides = (1,),
        offsets = (0, ),
        block_shape = (Q_CHUNK_SIZE,),
        order = (0,)
    ) 
    v_block_ptr = tl.make_block_ptr(
        values,
        shape = (N_CTX, N_HEADDIM),
        strides = (vo_stride2, vo_stride3),
        offsets = (0, 0),
        block_shape = (Q_CHUNK_SIZE, N_HEADDIM),
        order = (1, 0)
    )

    o_block_ptr = tl.make_block_ptr(
        output,
        shape = (Q_CHUNK_SIZE, N_HEADDIM),
        strides = (vo_stride2, vo_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape=(Q_CHUNK_SIZE, N_HEADDIM),
        order = (1, 0)
    )

    # prepare qd_succ
    
    prev_logM = tl.full((Q_CHUNK_SIZE, Q_CHUNK_SIZE), float("-inf"), dtype = tl.float32)
    logk_vals = tl.load(logk_ptrs, boundary_check=(0,), padding_option='zero')
    succ_logM = compute_qd_fused_partial_hard_gemm(soft_q, logk_vals, rtau)
    succ_attn_Vs, prev_logM = recompute_qd_block(idx_q, idx_q, prev_logM, succ_logM, tl.full((Q_CHUNK_SIZE,),float("-inf"), dtype = tl.float32), Q_CHUNK_SIZE, Q_CHUNK_SIZE, RET_PREV=False)

    attn_out = tl.load(o_block_ptr).to(tl.float32)
    rmax_out = tl.load(rmax_ptr[:,None], mask=mask[:,None], other=0) * RCP_LN2

    col_idx = tl.arange(0, Q_CHUNK_SIZE)[:, None] + tl.arange(0, Q_CHUNK_SIZE)[None, :] + 1
    col0_idx = tl.minimum(col_idx, Q_CHUNK_SIZE - 1)
    col1_idx = tl.maximum(col_idx - Q_CHUNK_SIZE, 0)

    for idx_d in tl.range(idx_q, -1, -1, num_stages=2):
        logk_ptrs = tl.advance(logk_ptrs, (Q_CHUNK_SIZE, ))

        v_succs = tl.load(v_block_ptr) # [Q_CHUNK_SIZE, N_HEADDIM]
        
        V_n = tl.full((Q_CHUNK_SIZE, Q_CHUNK_SIZE), float("-inf"), dtype = tl.float32)

        if idx_d != 0:
            init_Vs = tl.load(V_ptr)
            logk_vals = tl.load(logk_ptrs, boundary_check=(0,), padding_option='zero')
            succ_logM = compute_qd_fused_partial_hard_gemm(soft_q, logk_vals, rtau)
            V_n, prev_logM = recompute_qd_block(idx_q, idx_d - 1, prev_logM, succ_logM, init_Vs, Q_CHUNK_SIZE, Q_CHUNK_SIZE, RET_PREV=False)

        attns_Vs = tl.flip(tl.where(col_idx >= Q_CHUNK_SIZE, tl.gather(succ_attn_Vs, col1_idx, 1), tl.gather(V_n, col0_idx, 1)), 1)
        
        new_rmax = tl.maximum(rmax_out, tl.max(attns_Vs, axis = 1, keep_dims = True))
        attn_out = tl.dot(tl.exp2(attns_Vs - new_rmax).to(tl.bfloat16), v_succs, attn_out * tl.exp2(rmax_out - new_rmax))
        rmax_out = new_rmax

        succ_attn_Vs = V_n

        v_block_ptr = tl.advance(v_block_ptr, (Q_CHUNK_SIZE, 0))
        V_ptr -= uvh_stride3 * Q_CHUNK_SIZE
    
    

    rsqrt_weight = tl.rsqrt(1e-7 + (1.0/N_HEADDIM) * tl.sum(attn_out * attn_out, axis=1, keep_dims=True))
    attn_out = attn_out * rsqrt_weight

    tl.store(o_block_ptr, attn_out.to(output.dtype.element_ty))
    tl.store(rmax_ptr[:,None], rmax_out - tl.log2(rsqrt_weight), mask=mask[:,None])


def parallel_attn_bwd(logq: torch.Tensor, logk: torch.Tensor, values: torch.Tensor, temp: torch.Tensor, do: torch.Tensor, o: torch.Tensor, rmax: torch.Tensor, Vs: torch.Tensor):
    assert logq.shape == logk.shape, "Shape of logq/k must be the same"

    BATCH, HEADS, N, N_VOCAB = logq.shape
    N_HEADDIM = values.shape[-1]

    seps = 1e-3
    Q_CHUNK_SIZE = 32
    D_CHUNK_SIZE = 32

    dV = torch.zeros_like(values, dtype = torch.float32, device = logq.device)
    dq = torch.zeros_like(logq, dtype = torch.float32, device = logq.device)
    dk = torch.zeros_like(logq, dtype = torch.float32, device = logq.device)
    dlm = torch.zeros((BATCH, HEADS, N, N), dtype = torch.float32, device = logq.device)
    dtemp = torch.zeros_like(temp, device = logq.device)
    J = torch.zeros((BATCH, HEADS, N // Q_CHUNK_SIZE, N,), dtype = torch.float32, device = logq.device)
    H = torch.zeros((BATCH, HEADS, N // Q_CHUNK_SIZE, N,), dtype = torch.float32, device = logq.device)
    
    attn_bwd_kernel_hh[(BATCH * HEADS,(N // Q_CHUNK_SIZE))](
        logq, logk, values, rmax, do, dV, dlm, Vs, H, J, temp, 
        logq.stride(0), logq.stride(1), logq.stride(2), logq.stride(3), 
        o.stride(0), o.stride(1), o.stride(2), o.stride(3), 
        Vs.stride(0), Vs.stride(1), Vs.stride(2), Vs.stride(3),
        dlm.stride(0), dlm.stride(1), dlm.stride(2), dlm.stride(3),
        BATCH, HEADS,N,
        N_VOCAB=N_VOCAB, Q_CHUNK_SIZE=Q_CHUNK_SIZE, N_HEADDIM=N_HEADDIM, SEPS=seps)
    
    chunk_passing_kernel_bwd[(BATCH * HEADS,(N // D_CHUNK_SIZE))](
        H, J, 
        Vs.stride(0), Vs.stride(1), Vs.stride(2), Vs.stride(3), 
        BATCH, HEADS, N, (N // D_CHUNK_SIZE), D_CHUNK_SIZE=D_CHUNK_SIZE)
    
    attn_bwd_kernel_post_hh[(BATCH * HEADS,(N // Q_CHUNK_SIZE))](
        logq,logk,values,rmax,do,dq,dk,dlm,Vs,J,temp,dtemp,
        logq.stride(0), logq.stride(1), logq.stride(2), logq.stride(3), 
        o.stride(0), o.stride(1), o.stride(2), o.stride(3), 
        Vs.stride(0), Vs.stride(1), Vs.stride(2), Vs.stride(3), 
        dlm.stride(0), dlm.stride(1), dlm.stride(2), dlm.stride(3), 
        BATCH, HEADS,N,(N // Q_CHUNK_SIZE),
        N_VOCAB=N_VOCAB, Q_CHUNK_SIZE=Q_CHUNK_SIZE, N_HEADDIM=N_HEADDIM, SEPS=seps)
    
    return dq, dk, dV, dtemp

def parallel_attn_fwd(logq: torch.Tensor, logk: torch.Tensor, values: torch.Tensor, betas: torch.Tensor, fallback_o: torch.Tensor, temp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert logq.shape == logk.shape, "Shape of logq/k must be the same"

    BATCH, HEADS, N, N_VOCAB = logq.shape
    N_HEADDIM = values.shape[-1]

    seps = 1e-3
    Q_CHUNK_SIZE = 32
    D_CHUNK_SIZE = 32
    V = torch.full((BATCH, HEADS, N // Q_CHUNK_SIZE, N,), float("-inf"), dtype = torch.float32, device = logq.device)
    H = torch.zeros((BATCH, HEADS, N // Q_CHUNK_SIZE, N,), dtype = torch.float32, device = logq.device)
    o = torch.clone(fallback_o)
    rmax = torch.clone(betas)
    perprocess_kernel_hh[(BATCH * HEADS, (N // Q_CHUNK_SIZE) * (N // D_CHUNK_SIZE))](
        logq, logk, V, H, temp, 
        logq.stride(0), logq.stride(1), logq.stride(2), logq.stride(3), 
        V.stride(0), V.stride(1), V.stride(2), V.stride(3), 
        BATCH, HEADS, N, (N // Q_CHUNK_SIZE), (N // D_CHUNK_SIZE), 
        N_VOCAB=N_VOCAB, Q_CHUNK_SIZE=Q_CHUNK_SIZE, D_CHUNK_SIZE=D_CHUNK_SIZE, SEPS=seps)
    chunk_passing_kernel[(BATCH * HEADS, (N // D_CHUNK_SIZE))](
        V, H, 
        V.stride(0), V.stride(1), V.stride(2), V.stride(3), 
        BATCH, HEADS, N, (N // Q_CHUNK_SIZE), D_CHUNK_SIZE)
    
    attn_fwd_kernel_hh[(BATCH * HEADS, (N // Q_CHUNK_SIZE))](
        logq, logk, values, o, rmax, V, temp, 
        logq.stride(0), logq.stride(1),logq.stride(2), logq.stride(3), 
        o.stride(0), o.stride(1), o.stride(2), o.stride(3), 
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        BATCH, HEADS,N,(N // Q_CHUNK_SIZE),
        N_VOCAB=N_VOCAB, Q_CHUNK_SIZE=Q_CHUNK_SIZE, N_HEADDIM=N_HEADDIM, SEPS=seps)
         
    return o, rmax, V


def parallel_attn_hard_fwd(idx_q: torch.Tensor, idx_k: torch.Tensor, values: torch.Tensor, betas: torch.Tensor, fallback_o: torch.Tensor, temp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert idx_q.shape == idx_k.shape, "Shape of logq/k must be the same"

    BATCH, HEADS, N = idx_q.shape
    N_HEADDIM = values.shape[-1]

    seps = 1e-3
    Q_CHUNK_SIZE = 32
    D_CHUNK_SIZE = 32
    V = torch.full((BATCH, HEADS, N // Q_CHUNK_SIZE, N,), float("-inf"), dtype = torch.float32, device = idx_q.device)
    H = torch.zeros((BATCH, HEADS, N // Q_CHUNK_SIZE, N,), dtype = torch.float32, device = idx_q.device)
    o = torch.clone(fallback_o)
    rmax = torch.clone(betas)
    perprocess_kernel_hh_hard[(BATCH * HEADS, (N // Q_CHUNK_SIZE) * (N // D_CHUNK_SIZE))](
        idx_q, idx_k, V, H, temp, 
        idx_q.stride(0), idx_q.stride(1), 
        V.stride(0), V.stride(1), V.stride(2), V.stride(3), 
        BATCH, HEADS, N, (N // Q_CHUNK_SIZE), (N // D_CHUNK_SIZE), 
        Q_CHUNK_SIZE=Q_CHUNK_SIZE, D_CHUNK_SIZE=D_CHUNK_SIZE)
    chunk_passing_kernel[(BATCH * HEADS, (N // D_CHUNK_SIZE))](
        V, H, 
        V.stride(0), V.stride(1), V.stride(2), V.stride(3), 
        BATCH, HEADS, N, (N // Q_CHUNK_SIZE), D_CHUNK_SIZE)
    
    attn_fwd_kernel_hard_hh[(BATCH * HEADS, (N // Q_CHUNK_SIZE))](
        idx_q, idx_k, values, o, rmax, V, temp, 
        idx_q.stride(0), idx_q.stride(1), 
        o.stride(0), o.stride(1), o.stride(2), o.stride(3), 
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        BATCH, HEADS,N,(N // Q_CHUNK_SIZE),
        Q_CHUNK_SIZE=Q_CHUNK_SIZE, N_HEADDIM=N_HEADDIM)
         
    return o, rmax, V

class ParallelHardDiscreteAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.Function, idx_q, idx_k, values, betas, fallback_o, temp):
        ctx.dtype = logq.dtype

        o, rmax, Vs = parallel_attn_hard_fwd(idx_q, idx_k, values, betas, fallback_o, temp)
        return o
    
    @staticmethod
    def backward(ctx, do):
        return None, None, None, None, None, None
    
class ParallelNonHouseholderAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.Function, logq, logk, values, betas, fallback_o, temp):
        ctx.dtype = logq.dtype

        q = torch.softmax(logq, -1).half()
        k = torch.softmax(logk, -1).half()

        o, rmax, Vs = parallel_attn_fwd(q, k, values, betas, fallback_o, temp)

        ctx.save_for_backward(q, k, values, temp, o, rmax, Vs, betas, fallback_o)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, values, temp, o, rmax, Vs, betas, fallback_o = ctx.saved_tensors

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda:0")

        triton.set_allocator(alloc_fn)

        do = (do - (1.0/values.shape[-1]) * torch.sum(do * o, -1, keepdim=True) * o) # RMSNorm bp

        dq, dk, dv, dtemp = parallel_attn_bwd(q, k, values, temp, do, o, rmax, Vs)
        coef = torch.exp2(betas * 1.4426950216 - rmax)
        dbetas = torch.sum(do * fallback_o, -1) * coef
        dfallback_o = do * coef[...,None]

        return dq, dk, dv, dbetas, dfallback_o, dtemp

if __name__ == "__main__":
    torch.set_default_device('cuda:0')

    BATCH = 64
    HEADS = 4
    N = 256
    N_HEADDIM = 64
    N_VOCAB = 64
    seps = 1e-4

    betas = torch.full((BATCH, HEADS, N), -999, dtype = torch.float32).requires_grad_(True)
    fallback_o = torch.randn((BATCH, HEADS, N, N_HEADDIM), dtype = torch.bfloat16).requires_grad_(True)
    rcptaus = torch.nn.Parameter(torch.full((HEADS,), 5.0, dtype = torch.float32), requires_grad=True)
    values = torch.randn((BATCH, HEADS, N, N_HEADDIM), dtype = torch.bfloat16).requires_grad_(True)
    logq, logk = torch.randn((BATCH, HEADS, N, N_VOCAB), dtype = torch.bfloat16).requires_grad_(True), torch.randn((BATCH, HEADS, N, N_VOCAB), dtype = torch.bfloat16).requires_grad_(True)
    
    logq_pv, logk_pv = logq.detach().clone().requires_grad_(True), logk.detach().clone().requires_grad_(True)
    values_pv = values.detach().clone().requires_grad_(True)
    rcptaus_pv = rcptaus.detach().clone().requires_grad_(True)
    betas_pv = betas.detach().clone().requires_grad_(True)
    fallback_o_pv = fallback_o.detach().clone().requires_grad_(True)

    #sq, sk = torch.argmax(logq, -1), torch.argmax(logk, -1)
    #o_disc = ParallelHardDiscreteAttention.apply(sq, sk, values_pv, betas_pv, fallback_o_pv, rcptaus_pv)
    #o_refk = std_rosa_batched_hard(sq, sk, values.float(), betas.float(), fallback_o.float(), rcptaus.float())
    
    #print((o_disc - o_refk).abs().mean())
    #o_ref = make_rosa_attn_noc_fusedUV_batched(logq.float(), logk.float(), values.float(), betas.float(), fallback_o.float(), rcptaus.float())
    o_ref = std_rosa_batched(logq.float(), logk.float(), values.float(), betas.float(), fallback_o.float(), rcptaus.float())
    o = ParallelNonHouseholderAttention.apply(logq_pv, logk_pv, values_pv, betas_pv, fallback_o_pv, rcptaus_pv)

    print((o - o_ref).abs().mean())

    do = torch.randn_like(o_ref)

    o_ref.backward(do)
    o.backward(do.to(o.dtype))

    print(f"Q grad error = {(logq.grad - logq_pv.grad).abs().mean().item():8.5f}")
    print(f"K grad error = {(logk.grad - logk_pv.grad).abs().mean().item():8.5f}")
    print(f"V grad error = {(values.grad - values_pv.grad).abs().mean().item():8.5f}")
    print(f"tau grad error = {(rcptaus.grad - rcptaus_pv.grad).abs().mean().item():8.5f}")

    print(rcptaus.grad, rcptaus_pv.grad)

    print(f"Beta grad error = {((betas.grad - betas_pv.grad).abs().mean().item()):8.5f}")

    print(f"Fallback grad error = {((fallback_o.grad - fallback_o_pv.grad).abs().mean().item()):8.5f}")

    exit(0)
