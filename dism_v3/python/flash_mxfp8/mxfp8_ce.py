"""Triton LSE merge and CE with independent row/column MXFP8 gradients.

Losses and dZ are unnormalized; labels must be valid vocabulary indices or
ignore_index. Ignored rows have zero loss and zero gradients in both directions.
No BF16 dZ tensor is materialized. Quantization rounds dZ to BF16 in registers
and follows cuda/mxfp8_quantize.cuh (E8M0 exponent offset 7).
With softcap, logits remain raw BF16; partial LSE must already describe the
FP32 transformed logits. The chain rule is applied before quantization.
"""

import math
import struct

import torch
import triton
import triton.language as tl


@triton.jit
def _softcap_tanh(z, SOFTCAP: tl.constexpr):
    # Same instruction and FP32 division as the CUDA partial-LSE epilogue.
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", constraints="=f,f",
        args=[tl.div_rn(z, SOFTCAP)], dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _merge_lse(Z, Partial, Target, LSE, Loss, Weight, M: tl.constexpr,
               V: tl.constexpr, P: tl.constexpr, BP: tl.constexpr,
               IGNORE_INDEX: tl.constexpr, HAS_WEIGHT: tl.constexpr,
               SOFTCAP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    p = tl.arange(0, BP)
    local = tl.load(Partial + row * P + p, p < P, float('-inf'))
    maximum = tl.max(local, 0)
    lse = maximum + tl.log(tl.sum(tl.exp(local - maximum), 0))
    y = tl.load(Target + row)
    ignored = y == IGNORE_INDEX
    valid = (y >= 0) & (y < V) & ~ignored
    tl.device_assert(ignored | ((y >= 0) & (y < V)), "target outside vocabulary")
    safe_y = tl.where(valid, y, 0)
    zy = tl.load(Z + row * V + safe_y, valid, float('nan')).to(tl.float32)
    if SOFTCAP is not None:
        zy = SOFTCAP * _softcap_tanh(zy, SOFTCAP)
    tl.store(LSE + row, lse)
    loss = tl.where(ignored, 0., lse - zy)
    if HAS_WEIGHT:
        loss *= tl.load(Weight + row)
    tl.store(Loss + row, loss)


@triton.jit
def _ce_dual_quantize(Z, LSE, Target, Weight, Row, RowScale, Col, ColScale,
                      M: tl.constexpr, V: tl.constexpr, MP: tl.constexpr,
                      MP16: tl.constexpr, VP16: tl.constexpr,
                      IGNORE_INDEX: tl.constexpr, HAS_WEIGHT: tl.constexpr,
                      SOFTCAP: tl.constexpr):
    # Each program owns one 32x32 tile: complete groups in both directions.
    tm = tl.program_id(0).to(tl.int64)
    tv = tl.program_id(1).to(tl.int64)
    rows = tm * 32 + tl.arange(0, 32)
    cols = tv * 32 + tl.arange(0, 32)
    y = tl.load(Target + rows, rows < M, IGNORE_INDEX)
    active = (rows < M) & (y != IGNORE_INDEX)
    z = tl.load(Z + rows[:, None] * V + cols[None, :],
                active[:, None] & (cols[None, :] < V), 0).to(tl.float32)
    lse = tl.load(LSE + rows, active, 0)
    if SOFTCAP is not None:
        t = _softcap_tanh(z, SOFTCAP)
        z = SOFTCAP * t
    g = tl.exp(z - lse[:, None]) - (y[:, None] == cols[None, :]).to(tl.float32)
    if SOFTCAP is not None:
        g *= 1. - t * t
    # Zero before either scale reduction, including when ignore_index is a
    # vocabulary ID. Ignored rows must not affect a column's quantization scale.
    g = tl.where(active[:, None] & (cols[None, :] < V), g, 0)
    if HAS_WEIGHT:
        weight = tl.load(Weight + rows, rows < M, 0)
        g *= weight[:, None]
    # Match the existing BF16 -> MXFP8 quantizer, including its rounding point.
    g = g.to(tl.bfloat16).to(tl.float32)
    exponent = (g.to(tl.int32, bitcast=True) >> 23) & 255
    er = tl.maximum(tl.max(exponent, 1), 7) - 7
    ec = tl.maximum(tl.max(exponent, 0), 7) - 7
    rr = ((254 - er) << 23).to(tl.float32, bitcast=True)
    rc = ((254 - ec) << 23).to(tl.float32, bitcast=True)
    qr = (g * rr[:, None]).to(tl.bfloat16).to(tl.float8e4nv)
    qc = (g * rc[None, :]).to(tl.bfloat16).to(tl.float8e4nv)
    tl.store(Row + rows[:, None] * V + cols[None, :], qr,
             (rows[:, None] < M) & (cols[None, :] < V))
    tl.store(Col + cols[None, :] * MP + rows[:, None], qc, cols[None, :] < V)
    # Disjoint byte stores, never an int32 read-modify-write across programs.
    tl.store(RowScale + (tv // 4 * MP16 + rows) * 4 + tv % 4,
             er.to(tl.uint8), rows < MP16)
    tl.store(ColScale + (tm // 4 * VP16 + cols) * 4 + tm % 4,
             ec.to(tl.uint8), cols < VP16)


def merge_lse(logits, partial_lse, target, *, lse=None, loss=None, ignore_index=-100, row_weight=None, softcap=None):
    """Merge partials from the same softcap setting; return FP32 LSE/loss[M]."""
    m, v = logits.shape
    if (logits.dtype != torch.bfloat16 or not logits.is_cuda or
            not logits.is_contiguous() or m <= 0 or v <= 0):
        raise ValueError("Expected contiguous CUDA BF16 logits[M,V]")
    if (partial_lse.ndim != 2 or partial_lse.shape[0] != m or
            partial_lse.shape[1] <= 0 or partial_lse.dtype != torch.float32 or
            not partial_lse.is_contiguous() or partial_lse.device != logits.device):
        raise ValueError("Expected contiguous FP32 partial_lse[M,P]")
    _validate_target(target, m, logits.device)
    _validate_row_weight(row_weight, m, logits.device)
    _validate_ignore_index(ignore_index)
    softcap = _validate_softcap(softcap)
    if lse is None:
        lse = torch.empty(m, device=logits.device, dtype=torch.float32)
    if loss is None:
        loss = torch.empty_like(lse)
    for t in (lse, loss):
        if t.shape != (m,) or t.dtype != torch.float32 or t.device != logits.device or not t.is_contiguous():
            raise ValueError("Expected contiguous FP32 output[M]")
    p = partial_lse.shape[1]
    with torch.cuda.device(logits.device):
        _merge_lse[(m,)](logits, partial_lse, target, lse, loss, row_weight, m, v, p,
                         triton.next_power_of_2(p), ignore_index, row_weight is not None,
                         softcap, enable_fp_fusion=False)
    return lse, loss


def _validate_target(target, m, device):
    if (target.shape != (m,) or target.dtype != torch.int64 or
            target.device != device or not target.is_contiguous()):
        raise ValueError("Expected contiguous CUDA int64 target[M]")


def _validate_ignore_index(ignore_index):
    if type(ignore_index) is not int or not -(2**63) <= ignore_index < 2**63:
        raise ValueError("ignore_index must be a signed int64 integer")


def _validate_softcap(softcap):
    """Canonicalize the host constant so CUDA and Triton use identical FP32 s."""
    if softcap is None:
        return None
    if type(softcap) not in (int, float):
        raise ValueError("softcap must be None or a positive finite FP32 constant")
    try:
        value = struct.unpack("f", struct.pack("f", softcap))[0]
    except (OverflowError, struct.error):
        value = float('inf')
    if not math.isfinite(value) or value <= 0:
        raise ValueError("softcap must be None or a positive finite FP32 constant")
    return value


def _validate_row_weight(row_weight, m, device):
    if row_weight is not None and (
        row_weight.shape != (m,) or row_weight.dtype != torch.float32 or
        row_weight.device != device or not row_weight.is_contiguous()
    ):
        raise ValueError("Expected contiguous FP32 row_weight[M] on logits device")


class CEPlan:
    """Reusable quantization workspace compatible with CUDA dgrad/wgrad plans."""
    def __init__(self, m, v, device):
        if m <= 0 or v <= 0:
            raise ValueError("Positive dimensions required")
        self.m, self.v = m, v
        self.mp = triton.cdiv(m, 128) * 128
        self.mp16, self.vp16 = triton.cdiv(m, 16) * 16, triton.cdiv(v, 16) * 16
        self.row_fp8 = torch.empty((m, v), device=device, dtype=torch.float8_e4m3fn)
        self.column_fp8 = torch.empty((v, self.mp), device=device, dtype=torch.float8_e4m3fn)
        self.row_scale = torch.zeros((triton.cdiv(v, 128), self.mp16), device=device, dtype=torch.int32)
        self.column_scale = torch.zeros((self.mp // 128, self.vp16), device=device, dtype=torch.int32)

    def __call__(self, logits, lse, target, *, ignore_index=-100, row_weight=None, softcap=None):
        if (logits.shape != (self.m, self.v) or logits.dtype != torch.bfloat16 or
                logits.device != self.row_fp8.device or not logits.is_contiguous()):
            raise ValueError("Expected contiguous BF16 logits matching the CE plan")
        if (lse.shape != (self.m,) or lse.dtype != torch.float32 or
                lse.device != logits.device or not lse.is_contiguous()):
            raise ValueError("Expected contiguous FP32 LSE[M]")
        _validate_target(target, self.m, logits.device)
        _validate_row_weight(row_weight, self.m, logits.device)
        _validate_ignore_index(ignore_index)
        softcap = _validate_softcap(softcap)
        with torch.cuda.device(logits.device):
            _ce_dual_quantize[(self.mp // 32, triton.cdiv(self.v, 32))](
                logits, lse, target, row_weight, self.row_fp8, self.row_scale.view(torch.uint8),
                self.column_fp8, self.column_scale.view(torch.uint8),
                self.m, self.v, self.mp, self.mp16, self.vp16, ignore_index, row_weight is not None,
                softcap, num_warps=4,
                enable_fp_fusion=False,
            )
        return self.row_fp8, self.row_scale, self.column_fp8, self.column_scale
