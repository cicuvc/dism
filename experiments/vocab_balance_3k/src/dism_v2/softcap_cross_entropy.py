"""Triton CE of c*tanh(logits/c), with fused softcap backward.

2D CUDA logits [tokens,vocab], integer targets [tokens]. FP32 loss/statistics;
input-dtype gradients. Only row statistics are materialized, not capped logits
or probabilities. Softcap is a positive non-learnable Python scalar. No label
smoothing, class weighting, z-loss or tensor parallelism in this initial API.
Invalid non-ignored target IDs yield NaN loss (never an out-of-bounds load).
All-ignored mean is NaN with zero gradients, matching torch.cross_entropy.
"""
import math

import torch
from torch import nn
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    # User selected the approximate path for the production50257-class case.
    # Strict small-vocabulary oracle failures remain visible in the tests.
    return tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", constraints="=f,f",
                                   args=[x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _exp(x):
    return tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", constraints="=f,f",
                                   args=[x * 1.4426950408889634], dtype=tl.float32,
                                   is_pure=True, pack=1)


@triton.jit
def _row_forward(X, Y, LSE, LOSS, PARTIAL, stride_x, stride_y,
                 V: tl.constexpr, CAP: tl.constexpr, IGNORE: tl.constexpr,
                 SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(X + row * stride_x + col, col < V, other=0).to(tl.float32)
    capped = CAP * _tanh(raw / CAP)
    capped = tl.where(col < V, capped, -float('inf'))
    maximum = tl.max(capped, axis=0)
    lse = maximum + tl.log(tl.sum(_exp(capped - maximum), axis=0))
    if SPLITS > 1:
        tl.store(PARTIAL + row * SPLITS + part, lse)
    else:
        target = tl.load(Y + row * stride_y).to(tl.int64)
        valid = (target >= 0) & (target < V)
        chosen = tl.load(X + row * stride_x + target, valid, other=0).to(tl.float32)
        loss = lse - CAP * _tanh(chosen / CAP)
        loss = tl.where(valid, loss, float('nan'))
        tl.store(LOSS + row, tl.where(target == IGNORE, 0., loss))
        tl.store(LSE + row, lse)


@triton.jit
def _merge_forward(X, Y, PARTIAL, LSE, LOSS, stride_x, stride_y,
                   ROWS: tl.constexpr, V: tl.constexpr, CAP: tl.constexpr,
                   IGNORE: tl.constexpr, SPLITS: tl.constexpr, P: tl.constexpr,
                   R: tl.constexpr):
    row = (tl.program_id(0) * R + tl.arange(0, R)).to(tl.int64)
    part = tl.arange(0, P)
    value = tl.load(PARTIAL + row[:, None] * SPLITS + part[None, :],
                    (row[:, None] < ROWS) & (part[None, :] < SPLITS), other=-float('inf'))
    maximum = tl.max(value, axis=1)
    maximum = tl.where(row < ROWS, maximum, 0.)
    lse = maximum + tl.log(tl.sum(_exp(value - maximum[:, None]), axis=1))
    target = tl.load(Y + row * stride_y, row < ROWS, other=IGNORE).to(tl.int64)
    valid = (target >= 0) & (target < V)
    chosen = tl.load(X + row * stride_x + target, (row < ROWS) & valid, other=0).to(tl.float32)
    loss = tl.where(valid, lse - CAP * _tanh(chosen / CAP), float('nan'))
    loss = tl.where(target == IGNORE, 0., loss)
    tl.store(LSE + row, lse, row < ROWS)
    tl.store(LOSS + row, loss, row < ROWS)


@triton.jit
def _reduce_rows(LOSS, Y, SUMS, COUNTS, stride_y, ROWS: tl.constexpr,
                 IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    losses = tl.load(LOSS + row, row < ROWS, other=0.)
    target = tl.load(Y + row * stride_y, row < ROWS, other=IGNORE)
    tl.store(SUMS + tl.program_id(0), tl.sum(losses, axis=0))
    tl.store(COUNTS + tl.program_id(0), tl.sum(((row < ROWS) & (target != IGNORE)).to(tl.int32), axis=0))


@triton.jit
def _finish_reduction(SUMS, COUNTS, OUT, INV, PARTS: tl.constexpr,
                      MEAN: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    total = tl.sum(tl.load(SUMS + i, i < PARTS, other=0.), axis=0)
    count = tl.sum(tl.load(COUNTS + i, i < PARTS, other=0), axis=0).to(tl.float32)
    inv = tl.where(count > 0, 1. / count, 0.) if MEAN else 1.
    result = tl.where(count > 0, total * inv, float('nan')) if MEAN else total
    tl.store(OUT, result)
    tl.store(INV, inv)


@triton.jit
def _backward(X, Y, LSE, UPSTREAM, INV, DX, stride_x, stride_y, stride_up,
               V: tl.constexpr, CAP: tl.constexpr, IGNORE: tl.constexpr,
               NONE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(X + row * stride_x + col, col < V, other=0).to(tl.float32)
    t = _tanh(raw / CAP)
    lse = tl.load(LSE + row)
    target = tl.load(Y + row * stride_y).to(tl.int64)
    p = _exp(CAP * t - lse)
    if NONE:
        scale = tl.load(UPSTREAM + row * stride_up).to(tl.float32)
    else:
        scale = tl.load(UPSTREAM).to(tl.float32) * tl.load(INV)
    # dz/dx = 1-tanh(x/c)^2: the c from z cancels the inner1/c.
    dx = (p - (col == target).to(tl.float32)) * tl.fma(-t, t, 1.) * scale
    dx = tl.where((target >= 0) & (target < V), dx, float('nan'))
    dx = tl.where(target == IGNORE, 0., dx)
    tl.store(DX + row * V + col, dx, col < V)


class _SoftcapCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, cap, ignore, reduction):
        rows, vocab = logits.shape
        losses = torch.empty(rows, device=logits.device, dtype=torch.float32)
        lse = torch.empty_like(losses)
        inv = torch.ones((), device=logits.device, dtype=torch.float32)
        if rows:
            block = min(triton.next_power_of_2(vocab), 4096)
            splits = triton.cdiv(vocab, block)
            partial = torch.empty((rows, splits), device=logits.device, dtype=torch.float32) if splits > 1 else lse
            _row_forward[(rows, splits)](logits, target, lse, losses, partial, logits.stride(0), target.stride(0),
                vocab, cap, ignore, splits, block, num_warps=4 if block < 2048 else 8)
            if splits > 1:
                _merge_forward[(triton.cdiv(rows, 32),)](logits, target, partial, lse, losses,
                    logits.stride(0), target.stride(0), rows, vocab, cap, ignore, splits,
                    triton.next_power_of_2(splits), 32, num_warps=4)
            if reduction != 'none':
                parts = triton.cdiv(rows, 1024)
                sums = torch.empty(parts, device=logits.device, dtype=torch.float32)
                counts = torch.empty(parts, device=logits.device, dtype=torch.int32)
                result = torch.empty((), device=logits.device, dtype=torch.float32)
                _reduce_rows[(parts,)](losses, target, sums, counts, target.stride(0), rows, ignore, 1024)
                _finish_reduction[(1,)](sums, counts, result, inv, parts, reduction == 'mean',
                    triton.next_power_of_2(parts))
            else:
                result = losses
        else:
            result = losses if reduction == 'none' else torch.full((), float('nan') if reduction == 'mean' else 0.,
                                                                  device=logits.device, dtype=torch.float32)
        ctx.save_for_backward(logits, target, lse, inv)
        ctx.cap, ctx.ignore, ctx.reduction = cap, ignore, reduction
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        logits, target, lse, inv = ctx.saved_tensors
        dx = torch.empty(logits.shape, device=logits.device, dtype=logits.dtype)
        if logits.shape[0]:
            _backward[(logits.shape[0], triton.cdiv(logits.shape[1], 1024))](
                logits, target, lse, grad, inv, dx, logits.stride(0), target.stride(0),
                grad.stride(0) if ctx.reduction == 'none' else 0,
                logits.shape[1], ctx.cap, ctx.ignore, ctx.reduction == 'none', 1024, num_warps=4)
        return dx, None, None, None, None


def softcap_cross_entropy(logits, target, softcap, *, ignore_index=-100, reduction='mean'):
    """CE(c*tanh(x/c), target), with FP32 output and original-logit gradient.

    No input is mutated. Last logit stride must be1; padded row strides and
    strided target/upstream vectors are supported. First-order autograd only.
    NVIDIA CUDA backend; concurrent benchmark results are not meaningful.
    """
    if logits.ndim != 2 or target.shape != logits.shape[:1] or logits.shape[1] == 0:
        raise ValueError('Expected logits[N,V>0], target[N]')
    if not logits.is_cuda or target.device != logits.device:
        raise ValueError('logits and target must be on the same CUDA device')
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32) or target.dtype not in (torch.int32, torch.int64):
        raise TypeError('Expected floating logits and int32/int64 targets')
    if logits.stride(1) != 1:
        raise ValueError('Last logit stride must be1')
    if reduction not in ('none', 'sum', 'mean'):
        raise ValueError('reduction must be none/sum/mean')
    if not isinstance(softcap, (int, float)) or isinstance(softcap, bool) or not math.isfinite(softcap) or not 0 < softcap <= torch.finfo(torch.float32).max:
        raise ValueError('softcap must be a finite positive FP32-representable Python scalar')
    if softcap < torch.finfo(torch.float32).tiny:
        raise ValueError('Subnormal softcap is unsupported')
    if not isinstance(ignore_index, int):
        raise TypeError('ignore_index must be an integer')
    return _SoftcapCE.apply(logits, target, float(softcap), ignore_index, reduction)


class FusedSoftcapCrossEntropyLoss(nn.Module):
    def __init__(self, softcap, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.softcap, self.ignore_index, self.reduction = softcap, ignore_index, reduction

    def forward(self, logits, target):
        return softcap_cross_entropy(logits, target, self.softcap,
                                     ignore_index=self.ignore_index, reduction=self.reduction)
