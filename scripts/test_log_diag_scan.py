# Test if the scan algorithm works in preprocess stage

import sys
sys.path.append('build/linux/x86_64/release')
from typing import Callable
import torch
import dism_C


def log21p(x: torch.Tensor):
    LN2 = 0.69314718055994530941723212145818
    return torch.log1p(x) / LN2

def lse0(x: torch.Tensor): # logsumexp(x, 0)
    return log21p(torch.exp2(-torch.abs(x))) + torch.clamp_min(x, 0)

def lse(x: torch.Tensor, y: torch.Tensor): # logsumexp(x, y)
    return log21p(torch.exp2(-torch.abs(x - y))) + torch.maximum(x, y)

def neg_lse(x: torch.Tensor, y: torch.Tensor): # logsumexp(x, y)
    return -log21p(torch.exp2(-torch.abs(x - y))) + torch.minimum(x, y)


def line_scan(logX: torch.Tensor, initial: torch.Tensor): # Y_{n} = X_{n} * (1 + Y_{n-1}), Y_{-1} = initial
    logY = torch.empty_like(logX)
    for i in range(logX.shape[-1]):
        logY[..., i] = (initial := lse0(initial) + logX[..., i])
    return logY, initial

def inv_diag_scan(logM: torch.Tensor):
    R, C = logM.shape
    acc_logM = torch.empty_like(logM)
    finals = torch.zeros((R + C, ), dtype = logM.dtype, device = logM.device)

    for i in range(C - 1):
        current = torch.zeros_like(logM[0,0])
        for j in range(min(i + 1, R)):
            acc_logM[R - 1 - j, i - j] = current
            current = current + logM[R - 1 - j, i - j]
        finals[R + C - 2 - i] = current

    for i in range(R):
        current = torch.zeros_like(logM[0,0])
        for j in range(min(i + 1, C)):
            acc_logM[i - j, C - 1 - j] = current
            current = current + logM[i - j, C - 1 - j]
        finals[i] = current

    return acc_logM, torch.flip(finals[..., :C], (-1,)), finals[..., C:]


def diag_scan(logM: torch.Tensor, top_initials: torch.Tensor, left_initials: torch.Tensor, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    R, C = logM.shape
    acc_logM = torch.empty_like(logM)
    finals = torch.zeros((R + C, ), dtype = logM.dtype, device = logM.device)
    
    for i in range(C):
        current = torch.zeros_like(logM[0,0])
        for j in range(min(C - i, R)):
            acc_logM[j, i + j] = current
            current = fn(current, logM[j, i + j])
        finals[C - 1 - i] = current
    
    for i in range(R):
        current = torch.zeros_like(logM[0,0])
        for j in range(min(C, R - i - 1)):
            acc_logM[i + j + 1, j] = current
            current = fn(current, logM[i + j + 1, j])
        finals[C + i] = current

    return acc_logM, torch.flip(finals[..., (-C):], (-1,)), finals[..., :R]

def diag_reduce(logM: torch.Tensor, top_initials: torch.Tensor, left_initials: torch.Tensor, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    R, C = logM.shape
    finals = torch.zeros((R + C, ), dtype = logM.dtype, device = logM.device)
    
    for i in range(C):
        current = top_initials[..., i]
        for j in range(min(C - i, R)):
            current = fn(current, logM[j, i + j])
        finals[C - 1 - i] = current
    
    for i in range(R):
        current = left_initials[..., i]
        for j in range(min(C, R - i - 1)):
            current = fn(current, logM[i + j + 1, j])
        finals[C + i] = current

    return torch.flip(finals[..., (-C):], (-1,)), finals[..., :R]

def tl_to_br(top: torch.Tensor, left: torch.Tensor):
    R, C = left.shape[-1], top.shape[-1]
    finals = torch.cat([torch.flip(top, (-1,)), left], -1)
    return torch.flip(finals[..., (-C):], (-1,)), finals[..., :R]

def diag_accumulation(logM: torch.Tensor, top_initial: torch.Tensor, left_initial: torch.Tensor):
    N, M = logM.shape
    assert left_initial.shape[-1] == N
    assert top_initial.shape[-1] == M

    logM_acc, e_bottom, e_right = diag_scan(logM, torch.zeros_like(top_initial), torch.zeros_like(left_initial), lambda x,y: x+y)

    x_bottom, x_right = diag_reduce(logM_acc, -top_initial, -left_initial, neg_lse)

    return -x_bottom + e_bottom, -x_right + e_right

def extract_diag(x: torch.Tensor, diag: int):
    R, C = x.shape[-2:]
    if diag > 0:
        length = min(R, C - diag)
        return x.as_strided(x.shape[:-2] + (length, ), x.stride()[:-2] + (x.stride()[-2] + x.stride()[-1],), x.stride()[-1] * diag)
    else:
        length = min(C, R + diag)
        return x.as_strided(x.shape[:-2] + (length, ), x.stride()[:-2] + (x.stride()[-2] + x.stride()[-1],), -x.stride()[-2] * diag)

def py_test(logM: torch.Tensor, t: torch.Tensor, l: torch.Tensor):
    R, C = logM.shape[-2:]
    bottom, left = diag_accumulation(logM, t, l)

    initials = torch.cat([torch.flip(t, (-1,)), l], dim = -1)
    finals = torch.cat([left, torch.flip(bottom, (-1,))], dim = -1)
    ref_finals = torch.empty_like(finals)
    for i in range(-R, C):
        ref_finals[C - i - 1] = line_scan(extract_diag(o, i), initials[C - i - 1])[-1] # -(R-1) => R + C - 1
    
    torch.testing.assert_close(finals, ref_finals, rtol = 1e-3, atol = 1e-3)

    return torch.stack((bottom, left))


if __name__ == "__main__":
    torch.set_printoptions(threshold=100000, linewidth=100000)
    torch.set_default_device('cuda:0')

    for i in range(32):
        N = 16
        o = torch.randn((N, N),dtype = torch.float)
        t = torch.randn((2, N, ), dtype = torch.float)
        ref = py_test(o, t[0], t[1])

        dism_C.test_scan(o, t)

        torch.testing.assert_close(ref, t, atol = 3e-4, rtol = 1e-2)

    print("All tests passed!")