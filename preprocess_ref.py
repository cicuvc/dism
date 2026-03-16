import math
from pathlib import Path
import sys
from typing import Callable, Tuple
sys.path.append('build/linux/x86_64/release')

import torch
from einops import rearrange
import dism_C

from tqdm import tqdm


def log21p(x: torch.Tensor):
    LN2 = 0.69314718055994530941723212145818
    return torch.log1p(x) / LN2

@torch.compile
def lse0(x: torch.Tensor): # logsumexp(x, 0)
    return log21p(torch.exp2(-torch.abs(x))) + torch.clamp_min(x, 0)

@torch.compile
def lse(x: torch.Tensor, y: torch.Tensor): # logsumexp(x, y)
    return log21p(torch.exp2(-torch.abs(x - y))) + torch.maximum(x, y)

@torch.compile
def neg_lse(x: torch.Tensor, y: torch.Tensor): # logsumexp(x, y)
    return -log21p(torch.exp2(-torch.abs(x - y))) + torch.minimum(x, y)

def diag_scan(logM: torch.Tensor, top_initials: torch.Tensor, left_initials: torch.Tensor, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    R, C = logM.shape[-2:]
    acc_logM = torch.empty_like(logM)
    finals = torch.zeros(logM.shape[:-2] + (R + C, ), dtype = logM.dtype, device = logM.device)
    
    for i in range(C):
        current = top_initials[..., i]
        for j in range(min(C - i, R)):
            acc_logM[..., j, i + j] = current
            current = fn(current, logM[..., j, i + j])
        finals[..., C - 1 - i] = current
    
    for i in range(R):
        current = left_initials[..., i]
        for j in range(min(C, R - i - 1)):
            acc_logM[..., i + j + 1, j] = current
            current = fn(current, logM[..., i + j + 1, j])
        finals[..., C + i] = current

    return acc_logM, torch.flip(finals[..., (-C):], (-1,)), finals[..., :R]

@torch.compile
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


def diag_reduce(logM: torch.Tensor, top_initials: torch.Tensor, left_initials: torch.Tensor, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    R, C = logM.shape[-2:]
    finals = torch.zeros(logM.shape[:-2] + (R + C, ), dtype = logM.dtype, device = logM.device)
    
    for i in range(C):
        current = top_initials[..., i]
        for j in range(min(C - i, R)):
            current = fn(current, logM[..., j, i + j])
        finals[..., C - 1 - i] = current
    
    for i in range(R):
        current = left_initials[..., i]
        for j in range(min(C, R - i - 1)):
            current = fn(current, logM[..., i + j + 1, j])
        finals[..., C + i] = current

    return torch.flip(finals[..., (-C):], (-1,)), finals[..., :R]

@torch.compile
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

@torch.compile
def diag_acc_new(logM: torch.Tensor, top_H: torch.Tensor, top_V: torch.Tensor, left_H: torch.Tensor, left_V: torch.Tensor):
    N, M = logM.shape[-2:]
    assert left_H.shape[-1] == N and left_V.shape[-1] == N
    assert top_H.shape[-1] == M and top_V.shape[-1] == M

    logM_acc, bottom_H, right_H = diag_scan_ex(logM, top_H, left_H, lambda x,y:x+y)
    bottom_V, right_V = diag_reduce_ex(logM_acc, -top_V, -left_V, neg_lse)
    bottom_V = -bottom_V
    right_V = -right_V

    return bottom_V, bottom_H, right_V, right_H

def scan_single_matrix(logM: torch.Tensor):
    R, C = logM.shape[-2:]
    top_H = torch.zeros(logM.shape[:-2] + (C,), dtype = torch.float32)
    top_V = torch.full(logM.shape[:-2] + (C,), -9999, dtype = torch.float32)
    left_H = torch.zeros(logM.shape[:-2] + (R,), dtype = torch.float32)
    left_V = torch.full(logM.shape[:-2] + (R,), -9999, dtype = torch.float32)

    return diag_acc_new(logM, top_H, top_V, left_H, left_V)

def lower_tri_ids_flat(R: int, C: int, device=None):
    r = torch.arange(R, device=device)[:, None]
    c = torch.arange(C, device=device)[None, :]
    return (r * C + c)[c <= r]

def preprocess_val_ref(bottom_V_dut: torch.Tensor, bottom_H_dut: torch.Tensor, q: torch.Tensor, k: torch.Tensor, EPS: float = 1e-3, Q_BLOCK: int = 128):
    BATCH, SEQLEN, HEAD, QK_DIM = q.shape
    QK_LOG_BIAS = EPS / (1 - EPS * QK_DIM)
    QK_BIAS = math.log2(1 - EPS * QK_DIM)
    WARPGROUP_WARPS = 4
    
    full_map = torch.log2(torch.einsum('bnhc,bmhc->bhnm', q.float(), k.float()) + QK_LOG_BIAS) + rcptau[None, :, None, None] + QK_BIAS
    segment_map = rearrange(full_map, 'b h (k wg_size qb) m -> b h k wg_size qb m', wg_size = WARPGROUP_WARPS, qb = Q_BLOCK // WARPGROUP_WARPS)
    bottom_V, bottom_H, right_V, right_H = scan_single_matrix(segment_map) # [B, H, q_blocks, wg_size, M]
    ids = lower_tri_ids_flat(SEQLEN // Q_BLOCK, SEQLEN // Q_BLOCK, q.device)
    
    (bottom_V, bottom_H) = (rearrange(t, 'b h q w (k kb) -> b h (q k) (w kb)', kb = Q_BLOCK)[:,:,ids] for t in [bottom_V, bottom_H])
    #bottom_ref = bottom_V + bottom_H
    qids = torch.arange(0, SEQLEN // Q_BLOCK) * (torch.arange(0, SEQLEN // Q_BLOCK) + 1) // 2
    error_V = (bottom_V - bottom_V_dut)
    error_H = (bottom_H - bottom_H_dut)
    error_V[:,:,qids,::128] = 0
    error_H[:,:,qids,::128] = 0

    return bottom_V, bottom_H, error_V, error_H

def calc_error(x: torch.Tensor, y: torch.Tensor):
    avg_error = torch.abs(x - y).mean().item()
    max_error = torch.abs(x - y).max().item()
    rel_avg_error = (torch.abs(x - y) / (torch.abs(y) + 1e-7)).mean().item()
    rel_max_error = (torch.abs(x - y) / (torch.abs(y) + 1e-7)).max().item()
    return avg_error, max_error, rel_avg_error, rel_max_error

if __name__ == "__main__":
    torch.set_printoptions(threshold=100000, precision=4, linewidth=100000, sci_mode=False)
    torch.set_default_device('cuda:0')
    BATCH, SEQLEN, HEAD, QK_DIM = 2, 1024, 2, 64

    for epochs in tqdm(range(1000)):
        q_logits = torch.randn((BATCH, SEQLEN, HEAD, QK_DIM), dtype = torch.float32)
        k_logits = torch.randn((BATCH, SEQLEN, HEAD, QK_DIM), dtype = torch.float32)
        v = torch.randn((BATCH, SEQLEN, HEAD, QK_DIM), dtype = torch.bfloat16)

        q = torch.softmax(q_logits, dim = -1).half()
        k = torch.softmax(k_logits, dim = -1).half()
        rcptau = torch.rand((HEAD, ), dtype = torch.float32) + 1.0

        state = dism_C.BaselineNoPEAttnState(q, k, v, rcptau)
        state.invoke_fwd_preprocess()

        
        ref_V, ref_H, v_error, h_error = preprocess_val_ref( -state.fwd_v_buffer,  state.fwd_h_buffer, q, k)

        torch.testing.assert_close(ref_H + h_error, ref_H, rtol = 1e-3, atol = 1e-3)
        torch.testing.assert_close(ref_V + v_error, ref_V, rtol = 1e-3, atol = 1e-3)
        
        avg_error, max_error, rel_avg_error, rel_max_error = calc_error(ref_H + h_error, ref_H)
        print(f"The avg_error, max_error, rel_avg_error, rel_max_error of H = {avg_error:.5f}, {max_error:.5f}, {(rel_avg_error * 100):.2f}%, {(rel_max_error * 100):.2f}%")
        avg_error, max_error, rel_avg_error, rel_max_error = calc_error(ref_V + v_error, ref_V)
        print(f"The avg_error, max_error, rel_avg_error, rel_max_error of V = {avg_error:.5f}, {max_error:.5f}, {(rel_avg_error * 100):.2f}%, {(rel_max_error * 100):.2f}%")
        
    print("✅ All test passed!")