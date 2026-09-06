import math
import sys
import os
from typing import Callable, Tuple
from itertools import product
sys.path.append('build/linux/x86_64/release')
os.environ['TRITON_INTERPRET'] = '1'

import torch
import dism_C

import triton
import triton.language as tl

def log21p(x: torch.Tensor):
    LN2 = 0.69314718055994530941723212145818
    return torch.log1p(x) / LN2

def lse(x: torch.Tensor, y: torch.Tensor): # logsumexp(x, y)
    return log21p(torch.exp2(-torch.abs(x - y))) + torch.maximum(x, y)

def neg_lse(x: torch.Tensor, y: torch.Tensor): # logsumexp(x, y)
    return -log21p(torch.exp2(-torch.abs(x - y))) + torch.minimum(x, y)

def diag_reduce_ex(
    logM: torch.Tensor,
    top_initials: torch.Tensor,
    left_initials: torch.Tensor,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    *B, R, C = logM.shape

    rights = torch.zeros(logM.shape[:-2] + (R,), dtype=logM.dtype, device=logM.device)

    prev_next = top_initials  # (..., C)
    for r in range(R):
        cur = prev_next  # (..., C)
        if r >= 1:
            cur0 = left_initials[..., r - 1]  # (...,)
            cur = torch.cat([cur0.unsqueeze(-1), cur[..., 1:]], dim=-1)

        passed = fn(cur, logM[..., r, :])  # (..., C)

        rights[..., r] = passed[..., C - 1]

        prev_next = torch.cat(
            [torch.zeros_like(passed[..., :1]), passed[..., :-1]],
            dim=-1,
        )
    prev_next[..., 0] = left_initials[..., -1]

    bottom = prev_next
    return bottom, rights

def diag_scan_ex(
    logM: torch.Tensor,
    top_initials: torch.Tensor,
    left_initials: torch.Tensor,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    *B, R, C = logM.shape
    device = logM.device

    acc = torch.empty_like(logM)
    rights = torch.zeros(logM.shape[:-2] + (R, ), dtype = logM.dtype, device = logM.device)
    prev_next = top_initials  # (..., C)
    for r in range(R):
        cur = prev_next  # (..., C)
        if r >= 1:
            cur0 = left_initials[..., r - 1]          # (...,)
            cur = torch.cat([cur0.unsqueeze(-1), cur[..., 1:]], dim=-1)
        acc[..., r, :] = cur
        passed = fn(cur, logM[..., r, :])  # (..., C)

        rights[..., r] = passed[..., C - 1]
        prev_next = torch.cat(
            [torch.zeros_like(passed[..., :1]), passed[..., :-1]],
            dim=-1,
        )

    prev_next[..., 0] = left_initials[..., -1]
    return acc, prev_next, rights

def diag_acc_new(logM: torch.Tensor, top_H: torch.Tensor, top_V: torch.Tensor, left_H: torch.Tensor, left_V: torch.Tensor):
    N, M = logM.shape[-2:]
    assert left_H.shape[-1] == N and left_V.shape[-1] == N
    assert top_H.shape[-1] == M and top_V.shape[-1] == M

    logM_acc, bottom_H, right_H = diag_scan_ex(logM, top_H, left_H, lambda x,y:x+y)
    bottom_V, right_V = diag_reduce_ex(logM_acc, -top_V, -left_V, neg_lse)
    bottom_V = -bottom_V
    right_V = -right_V

    return bottom_V, bottom_H, right_V, right_H

@triton.jit
def logsumexp(x: tl.tensor, y: tl.tensor):
    return tl.maximum(x,y) + tl.log2(1 + tl.exp2(-tl.abs(x - y)))

@triton.jit
def fwd_passing_kernel_fixed(v_buffer, h_buffer, HEADS, T, VH_SIZE, CHECKPOINT_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_group = tl.program_id(0)
    bh_id = tl.program_id(1)
    batch = bh_id // HEADS
    head = bh_id % HEADS

    PP_WARPS: tl.constexpr = 4
    PP_QBLOCK_SIZE: tl.constexpr = (CHECKPOINT_SIZE * PP_WARPS)

    tl.static_assert(BLOCK_SIZE % PP_QBLOCK_SIZE == 0)

    v_buffer = v_buffer + batch * HEADS * VH_SIZE + head * VH_SIZE
    h_buffer = h_buffer + batch * HEADS * VH_SIZE + head * VH_SIZE
    
    v_current = tl.full((BLOCK_SIZE, ), -999.0, dtype = tl.float32)
    
    vh_offset = tl.arange(0, BLOCK_SIZE) + CHECKPOINT_SIZE - BLOCK_SIZE + 1
    
    for i in tl.range(0, (tl.cdiv(T, BLOCK_SIZE) - i_group) * (BLOCK_SIZE // CHECKPOINT_SIZE)):
        vh_line_start = i_group * (BLOCK_SIZE // CHECKPOINT_SIZE) + i
        vh_qblock_idx = vh_line_start // PP_WARPS
        col_size = (vh_qblock_idx + 1) * PP_QBLOCK_SIZE
        vh_block_offset = ((vh_qblock_idx + 1) * vh_qblock_idx // 2) * (PP_QBLOCK_SIZE * 4) + PP_QBLOCK_SIZE * (vh_line_start % PP_WARPS) * (vh_qblock_idx + 1)

        vh_load_offset = (vh_offset % col_size) + vh_block_offset
        mask = vh_offset > 1
        v_data = tl.load(v_buffer + vh_load_offset, mask = mask, other=-999.0)
        h_data = tl.load(h_buffer + vh_load_offset, mask = mask, other=0.0)

        v_current = logsumexp(h_data + v_current, v_data)

        tl.store(v_buffer + vh_load_offset, v_current, mask = mask)

        vh_offset = vh_offset + CHECKPOINT_SIZE
    
def launch_fwd_passing_kernel_fixed(v_buffer: torch.Tensor, h_buffer: torch.Tensor, T: int):
    B, H, X, A = v_buffer.shape
    BLOCK_SIZE = 128
    fwd_passing_kernel_fixed[(triton.cdiv(T, BLOCK_SIZE), B * H)](v_buffer, h_buffer, H, T, X * A, 32, BLOCK_SIZE)

def test_fwd_preprocess(q, k, v, rcptau, CHECKPOINT_SIZE: int = 32):
    BATCH, SEQLEN, HEAD, QK_DIM = q.shape
    
    state = dism_C.BaselineNoPEAttnState(q, k, v, rcptau)
    state.invoke_fwd_preprocess()

    QK_EPS = 1e-3
    QK_BIAS = math.log2(1 - QK_EPS * QK_DIM)
    QK_LOG_OFFSET = QK_EPS / (1 - QK_EPS * QK_DIM)

    Q_BLOCK_SIZE = CHECKPOINT_SIZE * 4 # 4 warps

    VH_ATOM_SIZE = state.fwd_v_buffer.shape[-1]

    errors = []

    for CHECK_BATCH, CHECK_HEAD in product(range(BATCH), range(HEAD)):
        for q_iter in range(SEQLEN // CHECKPOINT_SIZE):
            q_blk = q_iter // 4
            q_slice = slice((q_iter*CHECKPOINT_SIZE),(q_iter+1)*CHECKPOINT_SIZE)
            k_slice = slice(0, (q_blk + 1) * Q_BLOCK_SIZE)
            
            q_start = ((q_blk + 1) * q_blk // 2) * 4 * (Q_BLOCK_SIZE // VH_ATOM_SIZE) + (q_iter % 4) * (q_blk + 1) * (Q_BLOCK_SIZE // VH_ATOM_SIZE)

            vh_slice = slice(q_start, q_start + (q_blk + 1) * (Q_BLOCK_SIZE // VH_ATOM_SIZE))

            warp0_M = q[CHECK_BATCH,q_slice,CHECK_HEAD,:].float() @ k[CHECK_BATCH,k_slice,CHECK_HEAD,:].T.float()
            warp0_logM = torch.log2(QK_LOG_OFFSET + warp0_M) + (rcptau[CHECK_HEAD] + QK_BIAS)

            top_H = torch.zeros((warp0_logM.shape[-1],), dtype = torch.float32)
            top_V = torch.full((warp0_logM.shape[-1],), -99, dtype = torch.float32)

            left_H = torch.zeros((warp0_logM.shape[-2],), dtype = torch.float32)
            left_V = torch.full((warp0_logM.shape[-2],), -99, dtype = torch.float32)
            bottom_V, bottom_H, right_V, right_H = diag_acc_new(warp0_logM, top_H, top_V, left_H, left_V)

            bottom_gt, right_gt = bottom_V + bottom_H, right_V + right_H

            fwd_v = state.fwd_v_buffer[CHECK_BATCH, CHECK_HEAD, vh_slice].flatten()
            error = fwd_v - bottom_gt
            error[0] = fwd_v[0] - right_gt[-1]
            errors.append(error)
    
    total_err = torch.cat(errors, dim=-1)
    print(f"Mean error = {total_err.abs().mean().item():8.5f}, max error = {total_err.abs().max().item():8.5f}")

@triton.jit
def bwd_preprocess(o, do, drms, fallback_o, d_fallback_o, fwd_max, betas, d_betas, SEQLEN, HEAD, HEADDIM: tl.constexpr, BLOCK_M: tl.constexpr):
    i_bh = tl.program_id(1)
    batch = i_bh / HEAD
    head = i_bh % HEAD
    o = o + batch * HEAD * SEQLEN * HEADDIM + head * HEADDIM
    do = do + batch * HEAD * SEQLEN * HEADDIM + head * HEADDIM
    drms = drms + batch * HEAD * SEQLEN * HEADDIM + head * HEADDIM
    fallback_o = fallback_o + batch * HEAD * SEQLEN * HEADDIM + head * HEADDIM
    betas = betas + batch * HEAD * SEQLEN + head * SEQLEN
    d_betas = d_betas + batch * HEAD * SEQLEN + head * SEQLEN
    fwd_max = fwd_max + batch * HEAD * SEQLEN + head * SEQLEN

    i_idx = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    v_idx = tl.arange(0, HEADDIM)
    mask = i_idx < SEQLEN
    o_val = tl.load(o + i_idx[:,None] * HEAD * HEADDIM + v_idx[None, :], mask = mask[:,None]).to(tl.float32)
    do_val = tl.load(do + i_idx[:, None] * HEAD * HEADDIM + v_idx[None, :], mask = mask[:,None]).to(tl.float32)
    fallback_o_val = tl.load(fallback_o + i_idx[:, None] * HEAD * HEADDIM + v_idx[None, :], mask = mask[:,None]).to(tl.float32)
    max_val = tl.load(fwd_max + i_idx, mask = mask)
    betas_val = tl.load(betas + i_idx, mask = mask)

    d_rms_in = (do_val - (1.0/HEADDIM) * tl.sum(do_val * o_val, -1, keep_dims=True) * o_val) # [N, D]
    tl.store(drms + i_idx[:,None] * HEAD * HEADDIM + v_idx[None, :], d_rms_in.to(drms.dtype), mask = mask[:, None])

    fallback_weight = tl.exp2(betas_val - max_val)
    tl.store(d_fallback_o + i_idx[:,None] * HEAD * HEADDIM + v_idx[None, :], (d_rms_in * fallback_weight[:, None]).to(d_fallback_o.dtype), mask = mask[:, None])

    d_betas_val = tl.sum(d_rms_in * fallback_o_val, dim=-1, keepdim=False)
    tl.store(d_betas + i_idx, d_betas_val, mask = mask)


def test_fwd(q, k, v, rcptau, betas, fallback_o):
    BATCH, SEQLEN, HEAD, QK_DIM = q.shape

    state = dism_C.BaselineNoPEAttnState(q, k, v, rcptau, betas)
    state.invoke_fwd_preprocess()

    state.fwd_output.copy_(fallback_o)

    launch_fwd_passing_kernel_fixed(state.fwd_v_buffer, state.fwd_h_buffer, SEQLEN)
    
    state.fwd_h_buffer.zero_()
    state.invoke_fwd()

    do = torch.randn_like(state.fwd_output)
    dv = torch.empty_like(v)
    state.invoke_bwd_preprocess(do, dv)

    errors = []

    for CHECK_BATCH, CHECK_HEAD in product(range(BATCH), range(HEAD)):
        QK_EPS = 1e-3
        QK_BIAS = math.log2(1 - QK_EPS * QK_DIM)
        QK_LOG_OFFSET = QK_EPS / (1 - QK_EPS * QK_DIM)

        warp0_M = q[CHECK_BATCH,:,CHECK_HEAD,:].float() @ k[CHECK_BATCH,:,CHECK_HEAD,:].T.float()
        warp0_logM = torch.log2(QK_LOG_OFFSET + warp0_M) + (rcptau[CHECK_HEAD] + QK_BIAS)

        
        for i in range(1, warp0_logM.shape[-2]):
            last_state = warp0_logM[i-1,:]
            last_state = torch.roll(last_state, 1)
            last_state[0] = float('-inf')
            warp0_logM[i,:] += lse(last_state, torch.zeros_like(last_state))

        

        tmask = torch.tril(torch.ones_like(warp0_logM, dtype = torch.bool))
        warp0_logM = torch.where(tmask, warp0_logM, torch.full_like(warp0_logM, float("-inf")))
        
        
        #print((warp0_logM[96:128, 0:16] - Zf[96:128]).T)
        #print(state.fwd_v_buffer[CHECK_BATCH, CHECK_HEAD, 18:19])
        #print(state.fwd_h_buffer[CHECK_BATCH, CHECK_HEAD, 1])

        Zf = torch.maximum(warp0_logM.max(-1, keepdim=True).values, betas[CHECK_BATCH, CHECK_HEAD, :, None])
        attn = torch.exp2(warp0_logM - Zf)

        #print(torch.exp2(warp0_logM[96:128, 0:16] - Zf[96:128,:]).T)
        v_grad_error = (attn.T[:,:] @ do[CHECK_BATCH, :, CHECK_HEAD, :].float() - dv[CHECK_BATCH, :, CHECK_HEAD, :].float())
        #print(v_grad_error.abs().mean(-1,keepdim=True))
        print(f"Batch {CHECK_BATCH} Head {CHECK_HEAD} v_grad_error mean = {v_grad_error[0:128].abs().mean().item():8.5f}")
        o = (attn @ v[CHECK_BATCH,:,CHECK_HEAD,:].float()) + torch.exp2(betas[CHECK_BATCH, CHECK_HEAD, :, None] - Zf) * fallback_o[CHECK_BATCH, :, CHECK_HEAD, :]
        o = torch.nn.functional.rms_norm(o, (HEADDIM,), eps=1e-4)

        error = state.fwd_output[CHECK_BATCH,:,CHECK_HEAD,:].float() - o
        errors.append(error.flatten())

    total_err = torch.cat(errors, -1)
    print(f"Mean error = {total_err.abs().mean().item():8.5f}, max error = {total_err.abs().max().item():8.5f}")


if __name__ == "__main__":
    torch.set_printoptions(threshold=100000, precision=4, linewidth=100000, sci_mode=False)
    torch.set_default_device('cuda:0')
    BATCH, SEQLEN, HEAD, QK_DIM, HEADDIM = 1, 512, 2, 64, 64

    q_logits = torch.randn((BATCH, SEQLEN, HEAD, QK_DIM), dtype = torch.float32)
    k_logits = torch.randn((BATCH, SEQLEN, HEAD, QK_DIM), dtype = torch.float32)
    v = torch.randn((BATCH, SEQLEN, HEAD, QK_DIM), dtype = torch.bfloat16)
    betas = torch.randn((BATCH, HEAD, SEQLEN), dtype = torch.float32)
    fallback_o = torch.randn((BATCH, SEQLEN, HEAD, HEADDIM), dtype = torch.bfloat16)

    q = torch.softmax(q_logits, dim = -1).half()
    k = torch.softmax(k_logits, dim = -1).half()
    rcptau = torch.rand((HEAD, ), dtype = torch.float32) + 3.0

    test_fwd(q, k, v, rcptau, betas, fallback_o)