import math
import os
from typing import Tuple

import tqdm
os.environ['TRITON_INTERPRET'] = '0'

from torch.nn.attention import SDPBackend, sdpa_kernel

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

def logits_fused(logq, logk, eps = 1e-7):
    rowmax_q = torch.max(logq, dim = -1, keepdim=True).values
    rowmax_k = torch.max(logk, dim = -1, keepdim=True).values
    q_scaled = torch.exp(logq - rowmax_q)
    k_sclaed = torch.exp(logk - rowmax_k)
    q_logsum = torch.log(q_scaled.sum(-1, keepdim=True))
    k_logsum = torch.log(k_sclaed.sum(-1, keepdim=True))
    soft_q = q_scaled / q_scaled.sum(-1, keepdim=True)
    soft_k = k_sclaed / k_sclaed.sum(-1, keepdim=True)

    return torch.log(q_scaled @ k_sclaed.T + eps) - q_logsum - k_logsum.T, soft_q, soft_k

def logits_fused_eps(logq, logk, eps = 1e-4):
    dim = logq.shape[-1]
    rowmax_q = torch.max(logq, dim = -1, keepdim=True).values
    rowmax_k = torch.max(logk, dim = -1, keepdim=True).values
    q_scaled = torch.exp(logq - rowmax_q)
    k_sclaed = torch.exp(logk - rowmax_k)
    q_sum = q_scaled.sum(-1, keepdim=True)
    k_sum = k_sclaed.sum(-1, keepdim=True)
    soft_q = (1 - dim * eps) * q_scaled / q_sum + eps
    soft_k = k_sclaed / k_sum

    return torch.log(soft_q @ soft_k.T), soft_q, soft_k


def make_rosa_attn_scores(logq, logk, values:torch.Tensor, sel_tau):
    N, _ = logq.shape
    M, sq, sk = logits_fused_eps(logq, logk)
    logf = torch.full((N,), float("-inf"), dtype = torch.float32, device = logq.device)
    attn_scores = [torch.zeros((0, N), dtype = torch.float32, device = logq.device)]
    for i in range(0, N):
        d_indices = torch.arange(0, i + 1, device = logq.device)
        Zf = torch.maximum(logf[d_indices].detach(), torch.zeros((i+1,),dtype = torch.float32, device = logq.device))
        logf[d_indices] = M[i,i-d_indices] + 1 / sel_tau + torch.log(torch.exp(-Zf) + torch.exp(logf[d_indices] - Zf)) + Zf
        Zf = torch.clamp_min(torch.max(logf[d_indices].detach(), dim = -1, keepdim=True).values, 0)
        attn_scores.append(torch.nn.functional.pad(torch.exp(logf[None, i - d_indices] - Zf), (0, N - (i + 1))))
    
    attn = torch.cat(attn_scores, dim = -2)
    return torch.nn.functional.rms_norm(attn @ values, (values.shape[-1],), eps = 1e-7)

def make_rosa_attn_noc(logq: torch.Tensor, logk: torch.Tensor, values: torch.Tensor, sel_tau: float):
    N, _ = logq.shape
    logM, soft_q, soft_k = logits_fused_eps(logq, logk) 
    globals()['logM'] = logM
    logM.retain_grad()
    globals()['M'] = torch.exp(logM)
    globals()['M'].retain_grad()
    logM += 1 / sel_tau

    logM_qd = torch.zeros_like(logM)
    for q in range(N):
        logM_qd[q, slice(0, q + 1)] = torch.flip(logM[q, slice(0, q + 1)], (0,))

    A_Us = torch.zeros((N, N), dtype = torch.float32)
    A_Vs = torch.full((N, N), 0, dtype = torch.float32)
    globals()['logM_qd'] = logM_qd
    logM_qd.retain_grad()
    
    for d in range(0, N):
        qidx = torch.arange(d, N, device = logq.device)
        NL = logM_qd[qidx, d] # NL_(d, d+1, ..., N-1)

        Cs_list = [NL[0]]
        Ds_list = [NL[0]]

        for i in range(1, N - d):
            C, D = NL[i] + Cs_list[-1], max(Ds_list[-1] + NL[i], NL[i])
            Cs_list.append(C)
            Ds_list.append(D)

        Vs = torch.stack(Ds_list)

        Us = [torch.ones(tuple(), device = logq.device)] # U_(d, d+1, ..., N-1)

        idx = torch.arange(-1, N - d - 1)
        Alpha = torch.exp(torch.clamp_max(Vs, 0))[idx] # Alpha_(d, d+1, ..., N-1)
        Beta = torch.exp(-torch.clamp_min(Vs, 0))[idx]

        for i in range(1, N - d):
            Us.append(Us[-1] * Alpha[i] + Beta[i])
        
        A_Us[slice(d, N), d] = torch.stack(Us)
        A_Vs[slice(d, N), d] = Vs
    
    globals()['A_Us'] = A_Us
    globals()['A_Vs'] = A_Vs
    A_Us.retain_grad(), A_Vs.retain_grad()

    attns = torch.zeros((N, N), dtype = torch.float)
    
    Zf = A_Vs.max(1, keepdim=True).values.detach()
    inv_attns = A_Us * torch.exp(A_Vs - Zf)
    for q in range(N):
        attns[q, slice(0, q + 1)] = torch.flip(inv_attns[q, slice(0, q + 1)], (0,))

    attns.retain_grad()
    globals()['attns'] = attns

    return torch.nn.functional.rms_norm(attns @ values, (values.shape[-1],), eps = 1e-7)

def make_rosa_attn_noc_fusedUV(logq: torch.Tensor, logk: torch.Tensor, values: torch.Tensor, sel_tau: float):
    N, _ = logq.shape
    logM, soft_q, soft_k = logits_fused_eps(logq, logk) 
    globals()['logM'] = logM
    logM.retain_grad()
    globals()['M'] = torch.exp(logM)
    globals()['M'].retain_grad()
    logM += 1 / sel_tau[0]

    logM_qd = torch.zeros_like(logM)
    for q in range(N):
        logM_qd[q, slice(0, q + 1)] = torch.flip(logM[q, slice(0, q + 1)], (0,))

    A_Vs = torch.full((N, N), 0, dtype = torch.float32)
    globals()['logM_qd'] = logM_qd
    logM_qd.retain_grad()
    
    for d in range(0, N):
        qidx = torch.arange(d, N, device = logq.device)
        NL = logM_qd[qidx, d] # NL_(d, d+1, ..., N-1)

        Cs_list = [NL[0]]
        Ds_list = [NL[0]]

        for i in range(1, N - d):
            C, D = NL[i] + Cs_list[-1], max(Ds_list[-1] + NL[i], NL[i])
            Cs_list.append(C)
            Ds_list.append(D)

        Vs = torch.stack(Ds_list)

        Us = [torch.ones(tuple(), device = logq.device)] # U_(d, d+1, ..., N-1)

        idx = torch.arange(-1, N - d - 1)
        Alpha = torch.exp(torch.clamp_max(Vs, 0))[idx] # Alpha_(d, d+1, ..., N-1)
        Beta = torch.exp(-torch.clamp_min(Vs, 0))[idx]

        for i in range(1, N - d):
            Us.append(Us[-1] * Alpha[i] + Beta[i])
        
        A_Vs[slice(d, N), d] = Vs + torch.log(torch.stack(Us))
    
    globals()['A_Vs'] = A_Vs
    A_Vs.retain_grad()

    attns = torch.zeros((N, N), dtype = torch.float)
    
    Zf = A_Vs.max(1, keepdim=True).values.detach()
    inv_attns = torch.exp(A_Vs - Zf)
    for q in range(N):
        attns[q, slice(0, q + 1)] = torch.flip(inv_attns[q, slice(0, q + 1)], (0,))

    attns.retain_grad()
    globals()['attns'] = attns

    return torch.nn.functional.rms_norm(attns @ values, (values.shape[-1],), eps = 1e-7)


@torch.no_grad
def make_rosa_attn_grad_noc(logq: torch.Tensor, logk: torch.Tensor, values: torch.Tensor, sel_tau: float, do: torch.Tensor):
    N, _ = logq.shape
    logM, soft_q, soft_k = logits_fused_eps(logq, logk) 

    logM += 1 / sel_tau[0]

    logM_qd = torch.zeros_like(logM)
    for q in range(N):
        logM_qd[q, slice(0, q + 1)] = torch.flip(logM[q, slice(0, q + 1)], (0,))

    A_Vs = torch.full((N, N), 0, dtype = torch.float32)
    
    for d in range(0, N):
        qidx = torch.arange(d, N, device = logq.device)
        NL = logM_qd[qidx, d] # NL_(d, d+1, ..., N-1)

        Cs_list = [NL[0]]
        Ds_list = [NL[0]]

        for i in range(1, N - d):
            C, D = NL[i] + Cs_list[-1], max(Ds_list[-1] + NL[i], NL[i])
            Cs_list.append(C)
            Ds_list.append(D)

        Vs = torch.stack(Ds_list)

        Us = [torch.ones(tuple(), device = logq.device)] # U_(d, d+1, ..., N-1)

        idx = torch.arange(-1, N - d - 1)
        Alpha = torch.exp(torch.clamp_max(Vs, 0))[idx] # Alpha_(d, d+1, ..., N-1)
        Beta = torch.exp(-torch.clamp_min(Vs, 0))[idx]

        for i in range(1, N - d):
            Us.append(Us[-1] * Alpha[i] + Beta[i])
        
        A_Vs[slice(d, N), d] = Vs + torch.log(torch.stack(Us))

    attns = torch.zeros((N, N), dtype = torch.float)
    
    Zf = A_Vs.max(1, keepdim=True).values.detach()
    inv_attns = torch.exp(A_Vs - Zf)
    for q in range(N):
        attns[q, slice(0, q + 1)] = torch.flip(inv_attns[q, slice(0, q + 1)], (0,))

    out = attns @ values # [N, N] @ [N, D]
    norm_out = torch.nn.functional.rms_norm(out, (values.shape[-1],), eps = 1e-7)

    # start grad calculation
    out_denom = torch.sqrt(1e-7 + (out * out).mean(-1))
    Zf -= torch.log(1 / out_denom)[...,None]

    dout = (do - (1.0/values.shape[-1]) * torch.sum(do * norm_out, -1, keepdim=True) * norm_out) # [N, D]
    globals()['dout'] = dout
    dattns = dout @ values.T # [N, N]

    attns_dV = torch.zeros((N, N), dtype = torch.float)
    inv_attns_dV = torch.exp(A_Vs - Zf)
    for q in range(N):
        attns_dV[q, slice(0, q + 1)] = torch.flip(inv_attns_dV[q, slice(0, q + 1)], (0,))

    dvalues = attns_dV.T @ dout

    globals()['attns_dV'] = attns_dV

    inv_dattns = torch.zeros_like(inv_attns)
    for q in range(N):
        inv_dattns[q, slice(0, q + 1)] = torch.flip(dattns[q, slice(0, q + 1)], (0,))

    globals()['dattns_qd'] = inv_dattns

    current_J = torch.zeros((N,), dtype = torch.float32)
    J = torch.zeros((N, N), dtype = torch.float32)
    for i in range(N - 1, -1, -1):
        alpha = torch.sigmoid(A_Vs[i,:])
        beta = torch.exp(A_Vs[i,:]-Zf[i,:])
        current_J = current_J * alpha + inv_dattns[i,:] * beta
        J[i,slice(0,i+1)] = current_J[slice(0,i+1)]
    dlogM_qd = J

    globals()['dlogM_qd'] = J

    dM_qd = dlogM_qd / torch.exp(logM_qd - 1/sel_tau)
    dlogM = torch.full_like(dM_qd, 0)
    dM = torch.full_like(dM_qd, 0)
    for q in range(N):
        dM[q, slice(0, q + 1)] = torch.flip(dM_qd[q, slice(0, q + 1)], (0,))
        dlogM[q, slice(0, q + 1)] = torch.flip(dlogM_qd[q, slice(0, q + 1)], (0,))

    globals()['dlogM'] = dlogM
    globals()['dM'] = dM

    dlogq = (dM @ soft_k - dlogM.sum(-1)[:,None]) * soft_q
    dlogk = (dM.T @ soft_q - dlogM.sum(0)[:,None]) * soft_k
    print(f'Grad error: {(globals()['logM_qd'].grad - dlogM_qd).abs().max().item():.7f}')
    #print(f'dlogq Grad error: {(globals()['logq'].grad - dlogq).abs().max().item():.7f}')
    #print(f'dlogk Grad error: {(globals()['logk'].grad - dlogk).abs().max().item():.7f}')

def logits_fused_eps_batched(logq, logk, eps=1e-4):
    """
    logq, logk: [B, H, N, D]
    Returns: logM [B, H, N, N], soft_q [B, H, N, D], soft_k [B, H, N, D]
    """
    dim = logq.shape[-1]
    rowmax_q = torch.max(logq, dim=-1, keepdim=True).values
    rowmax_k = torch.max(logk, dim=-1, keepdim=True).values
    q_scaled = torch.exp(logq - rowmax_q)
    k_scaled = torch.exp(logk - rowmax_k)
    q_sum = q_scaled.sum(-1, keepdim=True)
    k_sum = k_scaled.sum(-1, keepdim=True)
    soft_q = (1 - dim * eps) * q_scaled / q_sum + eps
    soft_k = k_scaled / k_sum
    
    # soft_q @ soft_k.T for last two dimensions
    logM = torch. log(torch.matmul(soft_q, soft_k.transpose(-2, -1)))  # [B, H, N, N]
    
    return logM, soft_q, soft_k


def make_rosa_attn_noc_fusedUV_batched(logq: torch.Tensor, logk: torch.Tensor, values: torch. Tensor, sel_tau: torch.Tensor):
    """
    logq, logk: [B, H, N, D]
    values: [B, H, N, C]
    sel_tau: [H]
    Returns:  [B, H, N, C]
    """
    B, H, N, _ = logq.shape
    
    
    logM, soft_q, soft_k = logits_fused_eps_batched(logq, logk)  # [B, H, N, N]
    globals()['logM'] = logM
    logM.retain_grad()
    globals()['M'] = torch.exp(logM)
    globals()['M'].retain_grad()
    
    # sel_tau: [H] -> [1, H, 1, 1]
    logM = logM + 1 / sel_tau[None, :, None, None]
    
    logM_qd = torch. zeros_like(logM)  # [B, H, N, N]
    for q in range(N):
        logM_qd[:, :, q, slice(0, q + 1)] = torch.flip(logM[:, :, q, slice(0, q + 1)], (-1,))
    
    A_Vs = torch.zeros((B, H, N, N), dtype=torch.float32, device=logq.device)  # 改为zeros
    globals()['logM_qd'] = logM_qd
    logM_qd.retain_grad()
    
    for d in range(0, N):
        qidx = torch.arange(d, N, device=logq.device)
        NL = logM_qd[:, :, qidx, d]  # [B, H, N-d]
        
        Cs_list = [NL[:, :, 0]]  # [B, H]
        Ds_list = [NL[:, :, 0]]  # [B, H]
        
        for i in range(1, N - d):
            C = NL[:, :, i] + Cs_list[-1]  # [B, H]
            D = torch.maximum(Ds_list[-1] + NL[:, : , i], NL[:, : , i])  # 使用 maximum
            Cs_list.append(C)
            Ds_list.append(D)
        
        Vs = torch.stack(Ds_list, dim=-1)  # [B, H, N-d]
        
        Us_list = [torch. ones((B, H), device=logq.device)]  # [B, H]
        
        # 关键修正：索引逻辑
        for i in range(1, N - d):
            # Alpha 和 Beta 使用 Vs[..., i-1]
            Alpha_i = torch.exp(torch.clamp_max(Vs[..., i-1], 0))  # [B, H]
            Beta_i = torch. exp(-torch.clamp_min(Vs[..., i-1], 0))  # [B, H]
            Us_list.append(Us_list[-1] * Alpha_i + Beta_i)  # [B, H]
        
        Us_stacked = torch.stack(Us_list, dim=-1)  # [B, H, N-d]
        A_Vs[: , :, slice(d, N), d] = Vs + torch.log(Us_stacked)
    
    globals()['A_Vs'] = A_Vs
    A_Vs.retain_grad()
    
    attns = torch.zeros((B, H, N, N), dtype=torch.float, device=logq.device)
    
    Zf = A_Vs. max(-1, keepdim=True).values.detach()  # [B, H, N, 1]
    inv_attns = torch.exp(A_Vs - Zf)  # [B, H, N, N]
    for q in range(N):
        attns[:, :, q, slice(0, q + 1)] = torch.flip(inv_attns[: , :, q, slice(0, q + 1)], (-1,))
    
    attns. retain_grad()
    globals()['attns'] = attns
    
    out = torch.matmul(attns, values)  # [B, H, N, N] @ [B, H, N, C] -> [B, H, N, C]
    C = values. shape[-1]
    return torch.nn.functional.rms_norm(out, (C,), eps=1e-7)


@torch.no_grad()
def make_rosa_attn_grad_noc_batched(logq: torch.Tensor, logk: torch.Tensor, values: torch.Tensor, sel_tau: torch.Tensor, do: torch.Tensor):
    """
    logq, logk: [B, H, N, D]
    values: [B, H, N, C]
    sel_tau: [H]
    do: [B, H, N, C]
    """
    B, H, N, _ = logq.shape
    
    
    logM, soft_q, soft_k = logits_fused_eps_batched(logq, logk)  # [B, H, N, N]
    
    # sel_tau: [H] -> [1, H, 1, 1]
    logM = logM + 1 / sel_tau[None, : , None, None]
    
    logM_qd = torch.zeros_like(logM)  # [B, H, N, N]
    for q in range(N):
        logM_qd[:, :, q, slice(0, q + 1)] = torch.flip(logM[:, :, q, slice(0, q + 1)], (-1,))
    
    A_Vs = torch. zeros((B, H, N, N), dtype=torch.float32, device=logq.device)  # 改为zeros
    
    for d in range(0, N):
        qidx = torch.arange(d, N, device=logq.device)
        NL = logM_qd[:, :, qidx, d]  # [B, H, N-d]
        
        Cs_list = [NL[:, :, 0]]  # [B, H]
        Ds_list = [NL[:, :, 0]]  # [B, H]
        
        for i in range(1, N - d):
            C = NL[:, :, i] + Cs_list[-1]  # [B, H]
            D = torch.maximum(Ds_list[-1] + NL[:, : , i], NL[:, : , i])  # 使用 maximum
            Cs_list.append(C)
            Ds_list.append(D)
        
        Vs = torch.stack(Ds_list, dim=-1)  # [B, H, N-d]
        
        Us_list = [torch.ones((B, H), device=logq.device)]  # [B, H]
        
        # 关键修正：索引逻辑
        for i in range(1, N - d):
            # Alpha 和 Beta 使用 Vs[..., i-1]
            Alpha_i = torch.exp(torch.clamp_max(Vs[..., i-1], 0))  # [B, H]
            Beta_i = torch.exp(-torch.clamp_min(Vs[..., i-1], 0))  # [B, H]
            Us_list.append(Us_list[-1] * Alpha_i + Beta_i)  # [B, H]
        
        Us_stacked = torch.stack(Us_list, dim=-1)  # [B, H, N-d]
        A_Vs[:, :, slice(d, N), d] = Vs + torch.log(Us_stacked)
    
    attns = torch.zeros((B, H, N, N), dtype=torch.float, device=logq.device)
    
    Zf = A_Vs. max(-1, keepdim=True).values.detach()  # [B, H, N, 1]
    inv_attns = torch.exp(A_Vs - Zf)  # [B, H, N, N]
    for q in range(N):
        attns[:, :, q, slice(0, q + 1)] = torch.flip(inv_attns[: , :, q, slice(0, q + 1)], (-1,))
    
    out = torch.matmul(attns, values)  # [B, H, N, N] @ [B, H, N, C] -> [B, H, N, C]

    C = values.shape[-1]
    norm_out = torch.nn.functional.rms_norm(out, (C,), eps=1e-7)
    
    # start grad calculation
    out_denom = torch.sqrt(1e-7 + (out * out).mean(-1))  # [B, H, N]
    Zf = Zf - torch.log(1 / out_denom)[..., None]  # [B, H, N, 1]
    
    dout = (do - (1.0 / C) * torch.sum(do * norm_out, -1, keepdim=True) * norm_out)  # [B, H, N, C]
    globals()['dout'] = dout
    dattns = torch.matmul(dout, values. transpose(-2, -1))  # [B, H, N, C] @ [B, H, C, N] -> [B, H, N, N]
    
    attns_dV = torch.zeros((B, H, N, N), dtype=torch.float, device=logq. device)
    inv_attns_dV = torch.exp(A_Vs - Zf)  # [B, H, N, N]
    for q in range(N):
        attns_dV[: , :, q, slice(0, q + 1)] = torch.flip(inv_attns_dV[:, :, q, slice(0, q + 1)], (-1,))
    
    dvalues = torch.matmul(attns_dV. transpose(-2, -1), dout)  # [B, H, N, N] @ [B, H, N, C] -> [B, H, N, C]
    
    globals()['attns_dV'] = attns_dV
    
    inv_dattns = torch.zeros_like(inv_attns)  # [B, H, N, N]
    for q in range(N):
        inv_dattns[:, :, q, slice(0, q + 1)] = torch.flip(dattns[:, :, q, slice(0, q + 1)], (-1,))
    
    globals()['dattns_qd'] = inv_dattns
    
    current_J = torch.zeros((B, H, N), dtype=torch.float32, device=logq.device)
    J = torch.zeros((B, H, N, N), dtype=torch.float32, device=logq.device)
    for i in range(N - 1, -1, -1):
        alpha = torch.sigmoid(A_Vs[: , :, i, : ])  # [B, H, N]
        beta = torch. exp(A_Vs[:, : , i, :] - Zf[:, :, i, :])  # [B, H, N]
        current_J = current_J * alpha + inv_dattns[:, :, i, :] * beta  # [B, H, N]
        J[:, :, i, slice(0, i + 1)] = current_J[: , :, slice(0, i + 1)]
    dlogM_qd = J
    
    globals()['dlogM_qd'] = J
    
    # 修正：梯度计算
    dM_qd = dlogM_qd * torch.exp(logM_qd)  # [B, H, N, N]
    dM = torch.zeros_like(dM_qd)
    dlogM = torch.zeros_like(dM_qd)
    for q in range(N):
        dM[:, :, q, slice(0, q + 1)] = torch.flip(dM_qd[:, :, q, slice(0, q + 1)], (-1,))
        dlogM[:, :, q, slice(0, q + 1)] = torch.flip(dlogM_qd[:, :, q, slice(0, q + 1)], (-1,))
    
    globals()['dlogM'] = dlogM
    globals()['dM'] = dM
    
    dlogq = (torch.matmul(dM, soft_k) - dlogM. sum(-1)[..., None]) * soft_q  # [B, H, N, D]
    dlogk = (torch.matmul(dM. transpose(-2, -1), soft_q) - dlogM.sum(-2)[..., None]) * soft_k  # [B, H, N, D]
    
    print(f'Grad error: {(globals()["logM_qd"]. grad - dlogM_qd).abs().max().item():.7f}')
    
    return dlogq, dlogk, dvalues

@triton.jit
def add_mul_scan(la: tl.tensor, lb: tl.tensor, ra: tl.tensor, rb: tl.tensor):
    return la * ra, libdevice.fma(lb, ra, rb)

@triton.jit
def vs_scan(lc: tl.tensor, ld: tl.tensor, rc: tl.tensor, rd: tl.tensor):
    # Cs[i] + Cs[i - 1], max(Ds[i - 1] + Cs[i], Ds[i])
    return lc + rc, tl.maximum(ld + rc, rd)

@triton.jit
def max_scan(lc: tl.tensor, rc: tl.tensor):
    return tl.maximum(lc, rc)

@triton.jit
def compute_qd_fused_partial_log_gemm(q_soft: tl.tensor, logk: tl.tensor, rtau: tl.tensor, eps: tl.constexpr = 1e-4):
    RCP_LN2: tl.constexpr = 1.4426950216
    
    dim = q_soft.shape[-1]
    rowmax = tl.max(logk, axis=1, keep_dims=True)
    k_sclaed = tl.exp2(logk * RCP_LN2 - rowmax)
    k_sum = tl.sum(k_sclaed, 1, keep_dims=True)
    k_soft = (k_sclaed / k_sum).to(q_soft.dtype)
    m_qk = rtau + (tl.log2(1-dim*eps) + (eps / (1-dim*eps))) + libdevice.fast_log2f(tl.dot(q_soft, tl.trans(k_soft))) # [QCS, DCS]
    return m_qk, k_soft


@triton.jit
def tril(N: tl.constexpr, dtype: tl.dtype) -> tl.tensor:
    idx = tl.arange(0, N)
    return (idx[:, None] >= idx[None, :]).to(dtype)

@triton.jit
def triu(N: tl.constexpr, dtype: tl.dtype) -> tl.tensor:
    idx = tl.arange(0, N)
    return (idx[:, None] <= idx[None, :]).to(dtype)


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
    NL_D = tl.where(avail_mask, NL, float("-inf"))

    # [QCS, DCS]
    NL_P = tl.dot(tril(Q_CHUNK_SIZE, NL_C.dtype), NL_C)
    NL_PC = tl.associative_scan(NL_D - NL_P, axis = 0, combine_fn=max_scan)
    V_n = tl.maximum(NL_PC, init_Vs[None, :]) + NL_P
    C = NL_P

    idx = (tl.arange(0, Q_CHUNK_SIZE) - 1)[:,None].broadcast_to(Q_CHUNK_SIZE, D_CHUNK_SIZE)
    
    V_nexp = tl.exp2(tl.where(idx < 0, init_Vs[None, :], tl.gather(V_n, idx, axis = 0)))
    alpha = tl.minimum(V_nexp, 1)
    beta = tl.minimum(tl.fdiv(1.0, V_nexp), 1)
    alpha = tl.where(avail_mask, alpha, 1)
    beta = tl.where(avail_mask, beta, 0)
    A, B = tl.associative_scan((alpha, beta), axis = 0, combine_fn=add_mul_scan)
    U_n = A + B
    V_n += libdevice.fast_log2f(U_n)

    return V_n, tl.where(RET_PREV, logM_prev, logM_succ), C, NL


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

    NL_C = tl.where(avail_mask, NL, 0).to(tl.bfloat16)
    
    NL_G = tl.dot(triu(Q_CHUNK_SIZE, NL_C.dtype), NL_C)
    colmax_G = tl.max(NL_G, axis = 0, keep_dims=True) 
    C2 = tl.gather(NL_G, tl.zeros((1, D_CHUNK_SIZE), dtype = tl.int32), axis = 0)
    NL_G = tl.where(avail_mask, NL_G, float("-inf"))
    red_all = tl.sum(tl.exp2(NL_G - colmax_G), axis = 0) #+ tl.squeeze(tl.exp2(prev_row - colmax_G), dim=0)
    V2 = libdevice.log2(red_all) + colmax_G
    

    return V2, tl.where(RET_PREV, logM_prev, logM_succ), C2, NL


@triton.jit
def attn_bwd_kernel_hh(
    logq, logk, values, output, rmax, do, dV, # [B, H, N, C]
    V_ptr, H_ptr, J_ptr, # [B, H, Q_CHUNKS, N]
    tau_ptr, # [H]
    qk_stride0, qk_stride1, qk_stride2, qk_stride3,
    vo_stride0, vo_stride1, vo_stride2, vo_stride3,
    uvh_stride0, uvh_stride1, uvh_stride2, uvh_stride3,
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
    output = output + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)

    V_ptr = V_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q - 1) + uvh_stride3 * tl.arange(0, Q_CHUNK_SIZE)
    H_ptr = H_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q) + uvh_stride3 * tl.arange(0, Q_CHUNK_SIZE)
    J_ptr = J_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q) + uvh_stride3 * tl.arange(0, Q_CHUNK_SIZE)
    rmax_ptr = rmax + (N_HEADS * N_CTX) * (pid_bh // N_HEADS) + N_CTX * (pid_bh % N_HEADS) + idx_q * Q_CHUNK_SIZE + tl.arange(0, Q_CHUNK_SIZE)
    
    dV_desc = tl.make_tensor_descriptor(dV, (N_CTX, N_HEADDIM), (vo_stride2, vo_stride3), (Q_CHUNK_SIZE, N_HEADDIM))

    tau = tl.load(tau_ptr + (pid_bh % N_HEADS))
    RCP_LN2: tl.constexpr = 1.4426950216
    rtau = RCP_LN2 / tau

    logq_ptrs = tl.make_block_ptr(
        logq,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    )
    logq_values = tl.load(logq_ptrs) # [Q_CHUNK_SIZE, N_VOCAB]
    soft_q = tl.softmax(logq_values, dim=-1, keep_dims=True).to(tl.bfloat16)
    
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
    o_block_ptr = tl.make_block_ptr(
        output,
        shape = (Q_CHUNK_SIZE, N_HEADDIM),
        strides = (vo_stride2, vo_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape=(Q_CHUNK_SIZE, N_HEADDIM),
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

    o = tl.load(o_block_ptr) # [QCS, D]
    do = tl.load(do_block_ptr) # [QCS, D]
    Zf = tl.load(rmax_ptr) # [QCS]

    dAV = (do - (1.0/N_HEADDIM) * tl.sum(do * o, -1, keep_dims=True) * o).to(tl.float16) # [N, D]

    logM_succ, succ_soft_k = compute_qd_fused_partial_log_gemm(soft_q, tl.load(logk_ptrs), rtau, SEPS)

    prev_attn_Vs = tl.full((Q_CHUNK_SIZE, Q_CHUNK_SIZE), float("-inf"), dtype = tl.float32)

    v_succs = tl.load(v_block_ptr).to(tl.float16) # [Q_CHUNK_SIZE, N_HEADDIM]
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
        V_n, logM_succ, Csum, logM_qd = recompute_qd_block(idx_q, idx_d, logM_prev, logM_succ, init_Vs, Q_CHUNK_SIZE, Q_CHUNK_SIZE, RET_PREV=True)

        attns_Vs = tl.flip(tl.where(col_idx >= Q_CHUNK_SIZE, tl.gather(V_n, col1_idx, 1), tl.gather(prev_attn_Vs, col0_idx, 1)), 1)
        attns = tl.exp2(attns_Vs - Zf[:,None]).to(tl.float16)
        dV_out = tl.dot(tl.trans(attns), dAV)
        dV_desc.atomic_add(((idx_q - idx_d) * Q_CHUNK_SIZE, 0), dV_out)

        v_prev = tl.load(v_block_ptr, boundary_check=(0,), padding_option='zero').to(tl.float16) # [Q_CHUNK_SIZE, N_HEADDIM]
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
    logq, logk, values, output, rmax, do, dq, dk, # [B, H, N, C]
    V_ptr, J_ptr, # [B, H, Q_CHUNKS, N]
    tau_ptr, dtau_ptr, # [H]
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
    dk = dk + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)
    dq = dq + qk_stride0 * (pid_bh // N_HEADS) + qk_stride1 * (pid_bh % N_HEADS)

    values = values + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)
    output = output + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)
    do = do + vo_stride0 * (pid_bh // N_HEADS) + vo_stride1 * (pid_bh % N_HEADS)

    V_ptr = V_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q - 1) + uvh_stride3 * tl.arange(0, Q_CHUNK_SIZE)
    J_ptr = J_ptr + uvh_stride0 * (pid_bh // N_HEADS) + uvh_stride1 * (pid_bh % N_HEADS) + uvh_stride2 * (idx_q + 1) + uvh_stride3 * tl.arange(0, Q_CHUNK_SIZE)
    rmax_ptr = rmax + (N_HEADS * N_CTX) * (pid_bh // N_HEADS) + N_CTX * (pid_bh % N_HEADS) + idx_q * Q_CHUNK_SIZE + tl.arange(0, Q_CHUNK_SIZE)
    
    dk_desc = tl.make_tensor_descriptor(dk, (N_CTX, N_VOCAB), (qk_stride2, qk_stride3), (Q_CHUNK_SIZE, N_VOCAB))

    tau = tl.load(tau_ptr + (pid_bh % N_HEADS))
    RCP_LN2: tl.constexpr = 1.4426950216
    rtau = RCP_LN2 / tau

    logq_ptrs = tl.make_block_ptr(
        logq,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    )
    logq_values = tl.load(logq_ptrs) # [Q_CHUNK_SIZE, N_VOCAB]
    soft_q = tl.softmax(logq_values, dim=-1, keep_dims=True).to(tl.bfloat16)
    
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
    o_block_ptr = tl.make_block_ptr(
        output,
        shape = (Q_CHUNK_SIZE, N_HEADDIM),
        strides = (vo_stride2, vo_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape=(Q_CHUNK_SIZE, N_HEADDIM),
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

    o = tl.load(o_block_ptr) # [QCS, D]
    do = tl.load(do_block_ptr) # [QCS, D]
    Zf = tl.load(rmax_ptr) # [QCS]

    dAV = (do - (1.0/N_HEADDIM) * tl.sum(do * o, -1, keep_dims=True) * o).to(tl.float16) # [N, D]

    logM_succ, succ_soft_k = compute_qd_fused_partial_log_gemm(soft_q, tl.load(logk_ptrs), rtau, SEPS)

    v_succs = tl.load(v_block_ptr).to(tl.float16) # [Q_CHUNK_SIZE, N_HEADDIM]
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

    for idx_d in tl.range(0, idx_q + 1):
        init_Vs = tl.load(V_ptr, mask=idx_q>0, other=float("-inf"))
        init_Js = tl.load(J_ptr, mask=idx_q<Q_CHUNKS-1, other=0)

        logk_ptrs = tl.advance(logk_ptrs, (-Q_CHUNK_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (-Q_CHUNK_SIZE, 0))

        logk_vals = tl.load(logk_ptrs, boundary_check=(0,), padding_option='zero')
        logM_prev, prev_soft_k = compute_qd_fused_partial_log_gemm(soft_q, logk_vals, rtau, SEPS)
        V_n, logM_succ_next, Csum, logM_qd = recompute_qd_block(idx_q, idx_d, logM_prev, logM_succ, init_Vs, Q_CHUNK_SIZE, Q_CHUNK_SIZE, RET_PREV=True)

        v_prev = tl.load(v_block_ptr, boundary_check=(0,), padding_option='zero') # [Q_CHUNK_SIZE, N_HEADDIM]
        dQK_succ = tl.flip(tl.dot(dAV, tl.trans(v_prev.to(tl.float16))), 1)

        dQK = tl.where(inv_col_idx >= Q_CHUNK_SIZE, tl.gather(dQK_succ, inv_col1_idx, 1), tl.gather(dQK_prev, inv_col0_idx, 1))
        beta = dQK * tl.exp2(V_n - Zf[:, None])
        alpha = tl.fdiv(1., 1. + tl.exp2(-V_n))

        H_n, J_n = tl.associative_scan((alpha, beta), axis = 0, reverse=True, combine_fn=add_mul_scan) # [QCS, QCS]
        dlogM_qd_succ = J_n + H_n * init_Js[None, :]

        dlogM = tl.flip(tl.where(col_idx >= Q_CHUNK_SIZE, tl.gather(dlogM_qd_succ, col1_idx, 1), tl.gather(dlogM_qd_prev, col0_idx, 1)), 1)
        dlogM_rowsum = tl.sum(dlogM, axis=0)
        dlogM_colsum = tl.sum(dlogM, axis=1)
        dtaus += tl.sum(dlogM_rowsum, axis=0)
        dM = (dlogM * tl.exp2(rtau-logM_succ)).to(tl.bfloat16)

        dK = succ_soft_k * (tl.dot(tl.trans(dM), soft_q * (1 - SEPS * N_VOCAB) + SEPS)  - dlogM_rowsum[:,None])
        dk_desc.atomic_add(((idx_q - idx_d) * Q_CHUNK_SIZE, 0), dK)
        dQ_accum += tl.dot(dM, succ_soft_k * (1 - SEPS * N_VOCAB) + SEPS) - dlogM_colsum[:,None]

        logM_succ = logM_succ_next # [QCS, QCS]
        dQK_prev = dQK_succ # [QCS, QCS]
        dlogM_qd_prev = dlogM_qd_succ # [QCS, QCS]
        succ_soft_k = prev_soft_k # [QCS, VOC]

        V_ptr += uvh_stride3 * Q_CHUNK_SIZE
        J_ptr += uvh_stride3 * Q_CHUNK_SIZE

    dtaus *= -rtau * rtau * (1.0/(RCP_LN2*RCP_LN2))
    tl.atomic_add((dtau_ptr + (pid_bh % N_HEADS))[None], dtaus)
    dQ_accum = dQ_accum * soft_q
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
    
    tau = tl.load(tau_ptr + (pid_bh % N_HEADS))
    rtau = RCP_LN2 / tau

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

    logq_values = tl.load(logq_ptrs) # [Q_CHUNK_SIZE, N_VOCAB]
    logk_prevs = tl.load(logk_ptrs_perv, boundary_check=(0,), padding_option='zero') # [Q_CHUNK_SIZE, N_VOCAB]

    q_soft = tl.softmax(logq_values, dim=-1, keep_dims=True).to(tl.bfloat16)
    m_qk_prev, soft_k_prev = compute_qd_fused_partial_log_gemm(q_soft, logk_prevs, rtau, SEPS)
    

    logk_succs = tl.load(logk_ptrs_succ, boundary_check=(0,), padding_option='zero')
    m_qk_succ, soft_k_succ = compute_qd_fused_partial_log_gemm(q_soft, logk_succs, rtau, SEPS)
    d_idx = tl.arange(0, D_CHUNK_SIZE)[None]

    V2, _, C2, _ = recompute_qd_block_lastrow(idx_q, idx_d, m_qk_prev, m_qk_succ, Q_CHUNK_SIZE, D_CHUNK_SIZE)
    
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

        Z = tl.maximum(V_last + Hs, Vs)
        Vnew = tl.log2(tl.exp2(Vs - Z) + tl.exp2(V_last + Hs - Z)) + Z

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
    
    tau = tl.load(tau_ptr + (pid_bh % N_HEADS))

    logq_ptrs = tl.make_block_ptr(
        logq,
        shape = (N_CTX, N_VOCAB),
        strides = (qk_stride2, qk_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape = (Q_CHUNK_SIZE, N_VOCAB),
        order = (1, 0)
    )
    logq_values = tl.load(logq_ptrs) # [Q_CHUNK_SIZE, N_VOCAB]
    soft_q = tl.softmax(logq_values, dim=-1, keep_dims=True).to(tl.bfloat16)
    

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

    # prepare qd_succ
    RCP_LN2: tl.constexpr = 1.4426950216
    rtau = RCP_LN2 / tau
    prev_logM = tl.full((Q_CHUNK_SIZE, Q_CHUNK_SIZE), float("-inf"), dtype = tl.float32)
    logk_vals = tl.load(logk_ptrs, boundary_check=(0,), padding_option='zero')
    succ_logM, succ_soft_k = compute_qd_fused_partial_log_gemm(soft_q, logk_vals, rtau, SEPS)
    succ_attn_Vs, prev_logM, Csum, logM_qd = recompute_qd_block(idx_q, idx_q, prev_logM, succ_logM, tl.full((Q_CHUNK_SIZE,),float("-inf"), dtype = tl.float32), Q_CHUNK_SIZE, Q_CHUNK_SIZE, RET_PREV=False)

    attn_out = tl.zeros((Q_CHUNK_SIZE, N_HEADDIM), dtype = tl.float32)
    rmax_out = tl.zeros((Q_CHUNK_SIZE, 1), dtype = tl.float32)

    col_idx = tl.arange(0, Q_CHUNK_SIZE)[:, None] + tl.arange(0, Q_CHUNK_SIZE)[None, :] + 1
    col0_idx = tl.minimum(col_idx, Q_CHUNK_SIZE - 1)
    col1_idx = tl.maximum(col_idx - Q_CHUNK_SIZE, 0)

    for idx_d in tl.range(idx_q, -1, -1, num_stages=2):
        logk_ptrs = tl.advance(logk_ptrs, (Q_CHUNK_SIZE, 0))

        v_succs = tl.load(v_block_ptr).to(tl.float16) # [Q_CHUNK_SIZE, N_HEADDIM]
        
        V_n = tl.full((Q_CHUNK_SIZE, Q_CHUNK_SIZE), float("-inf"), dtype = tl.float32)

        if idx_d != 0:
            init_Vs = tl.load(V_ptr)
            logk_vals = tl.load(logk_ptrs, boundary_check=(0,), padding_option='zero')
            succ_logM, soft_k_succ = compute_qd_fused_partial_log_gemm(soft_q, logk_vals, rtau, SEPS)
            V_n, prev_logM, Csum, logM_qd = recompute_qd_block(idx_q, idx_d - 1, prev_logM, succ_logM, init_Vs, Q_CHUNK_SIZE, Q_CHUNK_SIZE, RET_PREV=False)

        attns_Vs = tl.flip(tl.where(col_idx >= Q_CHUNK_SIZE, tl.gather(succ_attn_Vs, col1_idx, 1), tl.gather(V_n, col0_idx, 1)), 1)
        
        new_rmax = tl.maximum(rmax_out, tl.max(attns_Vs, axis = 1, keep_dims = True))
        new_attn_out = tl.dot(tl.exp2(attns_Vs - new_rmax).to(tl.float16), v_succs)
        attn_out = attn_out * tl.exp2(rmax_out - new_rmax) + new_attn_out
        rmax_out = new_rmax

        succ_attn_Vs = V_n

        v_block_ptr = tl.advance(v_block_ptr, (Q_CHUNK_SIZE, 0))
        V_ptr -= uvh_stride3 * Q_CHUNK_SIZE
    
    o_block_ptr = tl.make_block_ptr(
        output,
        shape = (Q_CHUNK_SIZE, N_HEADDIM),
        strides = (vo_stride2, vo_stride3),
        offsets = (idx_q * Q_CHUNK_SIZE, 0),
        block_shape=(Q_CHUNK_SIZE, N_HEADDIM),
        order = (1, 0)
    )
    rsqrt_weight = tl.rsqrt(1e-7 + (1.0/N_HEADDIM) * tl.sum(attn_out * attn_out, axis=1, keep_dims=True))
    attn_out = attn_out * rsqrt_weight

    tl.store(o_block_ptr, attn_out)
    tl.store(rmax_ptr[:,None], rmax_out - tl.log2(rsqrt_weight))

def parallel_attn_bwd(logq: torch.Tensor, logk: torch.Tensor, values: torch.Tensor, temp: torch.Tensor, do: torch.Tensor, o: torch.Tensor, rmax: torch.Tensor, Vs: torch.Tensor):
    assert logq.shape == logk.shape, "Shape of logq/k must be the same"

    BATCH, HEADS, N, N_VOCAB = logq.shape
    N_HEADDIM = values.shape[-1]

    seps = 1e-4
    Q_CHUNK_SIZE = 32
    D_CHUNK_SIZE = 32

    dV = torch.zeros_like(values, dtype = torch.float32, device = logq.device)
    dq = torch.zeros_like(logq, dtype = torch.float32, device = logq.device)
    dk = torch.zeros_like(logq, dtype = torch.float32, device = logq.device)
    dtemp = torch.zeros_like(temp, device = logq.device)
    J = torch.zeros((BATCH, HEADS, N // Q_CHUNK_SIZE, N,), dtype = torch.float32, device = logq.device)
    H = torch.zeros((BATCH, HEADS, N // Q_CHUNK_SIZE, N,), dtype = torch.float32, device = logq.device)
    
    attn_bwd_kernel_hh[(BATCH * HEADS,(N // Q_CHUNK_SIZE))](
        logq, logk, values, o, rmax, do, dV, Vs, H, J, temp, 
        logq.stride(0), logq.stride(1), logq.stride(2), logq.stride(3), 
        o.stride(0), o.stride(1), o.stride(2), o.stride(3), 
        Vs.stride(0), Vs.stride(1), Vs.stride(2), Vs.stride(3),
        BATCH, HEADS,N,
        N_VOCAB=N_VOCAB, Q_CHUNK_SIZE=Q_CHUNK_SIZE, N_HEADDIM=N_HEADDIM, SEPS=seps)
    
    chunk_passing_kernel_bwd[(BATCH * HEADS,(N // D_CHUNK_SIZE))](
        H, J, 
        Vs.stride(0), Vs.stride(1), Vs.stride(2), Vs.stride(3), 
        BATCH, HEADS, N, (N // D_CHUNK_SIZE), D_CHUNK_SIZE=D_CHUNK_SIZE)
    
    attn_bwd_kernel_post_hh[(BATCH * HEADS,(N // Q_CHUNK_SIZE))](
        logq,logk,values,o,rmax,do,dq,dk,Vs,J,temp,dtemp,
        logq.stride(0), logq.stride(1), logq.stride(2), logq.stride(3), 
        o.stride(0), o.stride(1), o.stride(2), o.stride(3), 
        Vs.stride(0), Vs.stride(1), Vs.stride(2), Vs.stride(3), 
        BATCH, HEADS,N,(N // Q_CHUNK_SIZE),
        N_VOCAB=N_VOCAB, Q_CHUNK_SIZE=Q_CHUNK_SIZE, N_HEADDIM=N_HEADDIM, SEPS=seps)
    
    return dq, dk, dV, dtemp

def parallel_attn_fwd(logq: torch.Tensor, logk: torch.Tensor, values: torch.Tensor, temp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert logq.shape == logk.shape, "Shape of logq/k must be the same"

    BATCH, HEADS, N, N_VOCAB = logq.shape
    N_HEADDIM = values.shape[-1]

    seps = 1e-4
    Q_CHUNK_SIZE = 32
    D_CHUNK_SIZE = 32
    V = torch.full((BATCH, HEADS, N // Q_CHUNK_SIZE, N,), float("-inf"), dtype = torch.float32, device = logq.device)
    H = torch.zeros((BATCH, HEADS, N // Q_CHUNK_SIZE, N,), dtype = torch.float32, device = logq.device)
    o = torch.empty((BATCH, HEADS, N, N_HEADDIM), dtype = torch.float32, device = logq.device)
    rmax = torch.empty((BATCH, HEADS, N, ), dtype = torch.float32, device = logq.device)
    perprocess_kernel_hh[(BATCH * HEADS, (N // Q_CHUNK_SIZE) * (N // D_CHUNK_SIZE))](
        logq, logk, V, H, temp, 
        logq.stride(0), logq.stride(1), logq.stride(2), logq.stride(3), 
        V.stride(0), V.stride(1), V.stride(2), V.stride(3), 
        BATCH, HEADS, N, (N // Q_CHUNK_SIZE), (N // D_CHUNK_SIZE), 
        N_VOCAB=N_VOCAB, Q_CHUNK_SIZE=Q_CHUNK_SIZE, D_CHUNK_SIZE=D_CHUNK_SIZE, SEPS=seps)
    
    """
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
    """
    return o, rmax, V

class ParallelNonHouseholderAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.Function, logq, logk, values, temp):
        ctx.dtype = logq.dtype

        

        o, rmax, Vs = parallel_attn_fwd(logq, logk, values, temp)

        ctx.save_for_backward(logq, logk, values, temp, o, rmax, Vs)
        return o.to(logq.dtype)

    @staticmethod
    def backward(ctx, do):
        logq, logk, values, temp, o, rmax, Vs = ctx.saved_tensors

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda:0")

        triton.set_allocator(alloc_fn)

        dq, dk, dv, dtemp = parallel_attn_bwd(logq, logk, values, temp, do, o, rmax, Vs)

        return dq.to(logq), dk.to(logk), dv.to(values), dtemp.to(temp)

if __name__ == "__main__":
    torch.set_default_device('cuda:0')

    BATCH = 2
    HEADS = 16
    N = 4096
    N_HEADDIM = 64
    N_VOCAB = 64
    seps = 1e-4
    taus = torch.nn.Parameter(torch.rand((HEADS,), dtype = torch.float32) * 0.2 + 0.3, requires_grad=True)
    values = torch.randn((BATCH, HEADS, N, N_HEADDIM), dtype = torch.bfloat16).requires_grad_(True)
    logq, logk = torch.randn((BATCH, HEADS, N, N_VOCAB), dtype = torch.bfloat16).requires_grad_(True), torch.randn((BATCH, HEADS, N, N_VOCAB), dtype = torch.bfloat16).requires_grad_(True)
    logq_pv, logk_pv = logq.detach().clone().requires_grad_(True), logk.detach().clone().requires_grad_(True)
    values_pv = values.detach().clone().requires_grad_(True)
    taus_pv = taus.detach().clone().requires_grad_(True)
    for i in tqdm.tqdm(range(10000)):



        #with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        #    o = torch.nn.functional.scaled_dot_product_attention(logq_pv, logk_pv, values, is_causal=True)
        #o,x = parallel_path_attention(logq_pv.transpose(1,2),logk_pv.transpose(1,2),values_pv.transpose(1,2),logk_pv.transpose(1,2),torch.zeros((BATCH, N, HEADS), dtype = torch.bfloat16, device='cuda:0'))
        o = ParallelNonHouseholderAttention.apply(logq_pv, logk_pv, values_pv, taus_pv)

        #do = torch.randn_like(o)
        
        #o.backward(do)


        if False:
            v_o = make_rosa_attn_noc_fusedUV_batched(logq.float(), logk.float(), values.float(), taus)
            v_o.backward(do)
            #make_rosa_attn_grad_noc_batched(logq, logk, values, taus, do)
            o_error = (o - v_o).abs().max().item()
            dq_error = (logq.grad - logq_pv.grad).abs().mean().item()
            dk_error = (logk.grad - logk_pv.grad).abs().mean().item()
            dv_error = (values.grad - values_pv.grad).abs().mean().item()
            dtaus_error = ((taus.grad - taus_pv.grad) / (taus.grad)).abs().mean().item()
            print(f"Test errors: o = {o_error:.5f}, dq = {dq_error:.5f}, dk = {dk_error:.5f}, dv = {dv_error:.5f}, dte = {dtaus_error:.5f}")
            
            logq.grad = None
            logk.grad = None
            values.grad = None
            taus.grad = None
    pass
