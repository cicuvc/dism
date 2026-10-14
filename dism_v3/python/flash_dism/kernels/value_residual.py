"""Fused Triton value-residual mix (forward + backward) for sm120.

The reference helper (``flash_dism.module.value_residual_mix``) materialises
four extra FP32 tensors around the elementwise mix:

.. code-block:: python

    g = torch.sigmoid(gate.float()).unsqueeze(-1)      # [B,N,H,1] fp32
    mixed = v_first.float() + (v.float() - v_first.float()) * g
    out = mixed.to(v.dtype)

``value_residual_mix`` is called once per layer on the packed training row, so
``v``/``v_first``/``out`` dominate the traffic and the casts/sigmoid are pure
overhead.  This module performs the whole mix in registers and writes the final
dtype directly, and it fuses the three backward gradients into one kernel that
reduces ``dgate`` inside the CTA (no atomics; compiler reduction scratch only).

Math and precision contract
---------------------------
With ``a = FP32(v)``, ``b = FP32(v_first)``, ``s = sigmoid(FP32(gate))``,
``delta = a - b`` and an FP32 upstream gradient ``dy``:

* forward  ``out = cast_v_dtype(b + delta * s)``
* ``dv       = dy * s``
* ``dvfirst  = dy - (dy * s)``
* ``dgate    = (sum_DV(dy * delta) * (1 - s)) * s``

All internal arithmetic, including the ``DV`` reduction, is FP32; only the
final store rounds to the output dtype.  The backward ordering is not a free
choice: it reproduces the exact FP32 operation order of the eager autograd
graph (``dvfirst`` as ``dy - fl(dy*s)`` rather than ``dy * (1 - s)``, and the
``DV`` reduction *before* the sigmoid derivative, which PyTorch evaluates as
``grad * (1 - y) * y``). Sigmoid uses hardware tanh(gate/2)/2 + 1/2, as explicitly
approved for this operator. This introduces approximation error relative to
torch.sigmoid; tests bound it rather than requiring bitwise equality. Negative
tails saturate earlier due to cancellation in 1 + tanh; no relative tail
accuracy is claimed. The remaining operations use round-to-nearest intrinsics.

Layout: ``v, v_first: [B, N, H, DV]`` and ``gate: [B, N, H]`` in the natural
training strides.  No large tensor is made contiguous: the three input strides
and the ``dout`` strides are passed to the kernel, so sliced/transposed views
are handled directly.  Every output is freshly allocated and contiguous, and
the kernels store it with flat row-major offsets.

Divergence from the eager helper (measured, see docs/VALUE_RESIDUAL_FUSION.md):
the helper spells ``v_first.float()`` twice, so for BF16/F16 inputs autograd
rounds *each* of the two contributions to ``v_first`` back to the input dtype
before accumulating them, i.e. ``dvfirst = bf16(dy) - bf16(dy*s)``.  The fused
kernel keeps that path in FP32 (``dvfirst = FP32(dy - dy*s)``, one final
rounding), as the contract requires. For FP32 inputs only the tanh approximation
and reduction order distinguish the implementations; BF16/F16 additionally
have the documented duplicated-cast difference. See the accuracy report.

Limitations (explicitly validated in the tests, not silently papered over):

* CUDA only, ``numel < 2**31``; CPU and other combinations keep the eager
  reference path through ``value_residual_supported``.
* first-order gradients only; no second-order support is claimed.
* ``dtype`` limited to BF16/F16/FP32 inputs (FP64 is a math reference only and
  falls back to eager rather than silently narrowing to FP32).
"""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# Dtypes accepted by the fused kernels.  FP64 stays on the eager path on
# purpose: the task contract forbids quietly computing FP64 in FP32.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# ``DISM_VALUE_RESIDUAL_KERNEL`` selects the common-helper implementation:
#   "fused"     (default) fused kernel where eligible, eager reference otherwise
#   "reference" always use the eager reference helper
# Read once at import time so the decision is a Dynamo constant, not a host
# call inside a compiled graph.
_KERNEL_MODE = os.environ.get("DISM_VALUE_RESIDUAL_KERNEL", "fused").strip().lower()


@triton.jit
def _sigmoid_tanh(gate):
    t = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", constraints="=f,f",
        args=[gate * 0.5], dtype=tl.float32, is_pure=True, pack=1)
    return tl.fma(t, 0.5, 0.5)


@triton.jit
def _value_residual_fwd_kernel(
    V, VF, G, O,
    v_sb, v_sn, v_sh, v_sd, f_sb, f_sn, f_sh, f_sd, g_sb, g_sn, g_sh,
    total_rows, nh, heads,
    DV: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < total_rows
    batch = rows // nh
    rem = rows - batch * nh
    seq = rem // heads
    head = rem - seq * heads
    # Logical indices fit int32 by the numel guard. Widen only addresses,
    # after division, to avoid out-of-line int64 division for runtime H/N.
    batch, seq, head = batch.to(tl.int64), seq.to(tl.int64), head.to(tl.int64)
    cols = tl.arange(0, BLOCK_DV).to(tl.int64)
    elem_mask = row_mask[:, None] & (cols < DV)[None, :]
    off_v = (batch[:, None] * v_sb + seq[:, None] * v_sn
             + head[:, None] * v_sh + cols[None, :] * v_sd)
    off_f = (batch[:, None] * f_sb + seq[:, None] * f_sn
             + head[:, None] * f_sh + cols[None, :] * f_sd)
    # Outputs are freshly allocated contiguous [B,N,H,DV]; flat row-major.
    off_o = rows[:, None] * DV + cols[None, :]
    off_g = batch * g_sb + seq * g_sn + head * g_sh
    gate = tl.load(G + off_g, mask=row_mask, other=0.0).to(tl.float32)
    # Hardware tanh approximation is intentional. Keep the remaining FP32
    # rounding/mixture contract; forward and backward recompute the same s.
    s = _sigmoid_tanh(gate)
    a = tl.load(V + off_v, mask=elem_mask, other=0.0).to(tl.float32)
    b = tl.load(VF + off_f, mask=elem_mask, other=0.0).to(tl.float32)
    delta = libdevice.sub_rn(a, b)
    mixed = libdevice.add_rn(b, libdevice.mul_rn(delta, s[:, None]))
    tl.store(O + off_o, mixed, mask=elem_mask)


@triton.jit
def _value_residual_bwd_kernel(
    V, VF, G, DO, DV_OUT, DF_OUT, DG_OUT,
    v_sb, v_sn, v_sh, v_sd, f_sb, f_sn, f_sh, f_sd, g_sb, g_sn, g_sh,
    do_sb, do_sn, do_sh, do_sd,
    total_rows, nh, heads,
    DV: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < total_rows
    batch = rows // nh
    rem = rows - batch * nh
    seq = rem // heads
    head = rem - seq * heads
    batch, seq, head = batch.to(tl.int64), seq.to(tl.int64), head.to(tl.int64)
    cols = tl.arange(0, BLOCK_DV).to(tl.int64)
    elem_mask = row_mask[:, None] & (cols < DV)[None, :]
    off_v = (batch[:, None] * v_sb + seq[:, None] * v_sn
             + head[:, None] * v_sh + cols[None, :] * v_sd)
    off_f = (batch[:, None] * f_sb + seq[:, None] * f_sn
             + head[:, None] * f_sh + cols[None, :] * f_sd)
    off_do = (batch[:, None] * do_sb + seq[:, None] * do_sn
              + head[:, None] * do_sh + cols[None, :] * do_sd)
    off_o = rows[:, None] * DV + cols[None, :]
    off_g = batch * g_sb + seq * g_sn + head * g_sh
    gate = tl.load(G + off_g, mask=row_mask, other=0.0).to(tl.float32)
    s = _sigmoid_tanh(gate)
    a = tl.load(V + off_v, mask=elem_mask, other=0.0).to(tl.float32)
    b = tl.load(VF + off_f, mask=elem_mask, other=0.0).to(tl.float32)
    dy = tl.load(DO + off_do, mask=elem_mask, other=0.0).to(tl.float32)
    # Padded lanes load as zero, so their delta is exactly zero and they cannot
    # perturb the DV reduction.
    delta = libdevice.sub_rn(a, b)
    dv = libdevice.mul_rn(dy, s[:, None])
    df = libdevice.sub_rn(dy, dv)
    # Reduce over DV first: the eager graph sums the broadcast dimension of
    # ``(a-b)*s`` before the sigmoid derivative multiplies by (1-s)*s.
    dgate_pre = tl.sum(libdevice.mul_rn(dy, delta), axis=1)
    dgate = libdevice.mul_rn(libdevice.mul_rn(dgate_pre, libdevice.sub_rn(1.0, s)), s)
    tl.store(DV_OUT + off_o, dv, mask=elem_mask)
    tl.store(DF_OUT + off_o, df, mask=elem_mask)
    # dgate is a fresh contiguous [B,N,H] tensor: flat row index.
    tl.store(DG_OUT + rows, dgate, mask=row_mask)


def _launch_config(rows, dv):
    """Deterministic launch shape (tuning is deferred while the GPU is shared).

    One CTA covers ``BLOCK_ROWS`` token/head rows and the full, masked ``DV``
    axis so the ``dgate`` reduction never crosses CTAs.
    """
    block_dv = max(16, triton.next_power_of_2(int(dv)))
    if dv <= 64:
        block_rows = 8
    else:
        block_rows = 4
    return block_rows, block_dv, 4


def _check_operands(v, v_first, gate):
    if v.ndim != 4 or gate.ndim != 3:
        raise ValueError("value residual expects v/v_first [B,N,H,DV] and gate [B,N,H]")
    if v.shape != v_first.shape or gate.shape != v.shape[:-1]:
        raise ValueError(
            "value residual expects v_first with v's shape and gate [B,N,H]")
    if v.numel() >= 2 ** 31:
        raise ValueError("value residual fused kernel requires numel < 2**31")
    for name, tensor in (("v", v), ("v_first", v_first), ("gate", gate)):
        if tensor.device.type != "cuda":
            raise ValueError(f"value residual fused kernel requires CUDA, got {name} on CPU")
        if tensor.device != v.device:
            raise ValueError("value residual inputs must be on the same CUDA device")
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"value residual fused kernel does not support {name} dtype {tensor.dtype}")


def value_residual_fused(v, v_first, gate):
    """Fused forward.  Differentiable through its registered backward."""
    return _value_residual_forward_op(v, v_first, gate)


def torch_fwd_value_residual(v, v_first, gate):
    """Raw forward launch; straight CUDA, no autograd bookkeeping."""
    _check_operands(v, v_first, gate)
    # Contiguous output: the kernel stores it with flat row-major offsets while
    # reading the inputs through their own strides.
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    shape = v.shape
    rows = shape[0] * shape[1] * shape[2]
    if rows == 0:
        return out
    dv = shape[3]
    block_rows, block_dv, num_warps = _launch_config(rows, dv)
    _value_residual_fwd_kernel[(triton.cdiv(rows, block_rows),)](
        v, v_first, gate, out,
        *v.stride(), *v_first.stride(),
        gate.stride(0), gate.stride(1), gate.stride(2),
        rows, shape[1] * shape[2], shape[2],
        DV=dv, BLOCK_ROWS=block_rows, BLOCK_DV=block_dv, num_warps=num_warps)
    return out


def torch_bwd_value_residual(v, v_first, gate, dout):
    """Raw backward launch; returns FP32 grads rounded to each input's dtype."""
    _check_operands(v, v_first, gate)
    if dout.shape != v.shape:
        raise ValueError("value residual backward expects dout with v's shape")
    if dout.device != v.device:
        raise ValueError("value residual dout must be on the input CUDA device")
    if dout.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"value residual backward does not support dout dtype {dout.dtype}")
    dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    df = torch.empty(v_first.shape, dtype=v_first.dtype, device=v_first.device)
    dg = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    shape = v.shape
    rows = shape[0] * shape[1] * shape[2]
    if rows == 0:
        return dv, df, dg
    dv_last = shape[3]
    block_rows, block_dv, num_warps = _launch_config(rows, dv_last)
    _value_residual_bwd_kernel[(triton.cdiv(rows, block_rows),)](
        v, v_first, gate, dout, dv, df, dg,
        *v.stride(), *v_first.stride(),
        gate.stride(0), gate.stride(1), gate.stride(2),
        *dout.stride(),
        rows, shape[1] * shape[2], shape[2],
        DV=dv_last, BLOCK_ROWS=block_rows, BLOCK_DV=block_dv, num_warps=num_warps)
    return dv, df, dg


def value_residual_supported(v, v_first, gate):
    """Eligibility for the fused path; also the Dynamo-time dispatch guard.

    Metadata-only (no data inspection, no sync), so it stays inside a compiled
    graph.  Anything unsupported keeps the eager reference helper.
    """
    if _KERNEL_MODE != "fused":
        return False
    if not (isinstance(v, torch.Tensor) and isinstance(v_first, torch.Tensor)
            and isinstance(gate, torch.Tensor)):
        return False
    if v.ndim != 4 or gate.ndim != 3:
        return False
    if v.shape != v_first.shape or gate.shape != v.shape[:-1]:
        return False
    if min(v.shape) < 1:
        return False
    if v.numel() >= 2 ** 31:
        return False
    for tensor in (v, v_first, gate):
        if tensor.device.type != "cuda" or tensor.dtype not in _SUPPORTED_DTYPES:
            return False
        if tensor.device != v.device:
            return False
    return True


# Versioned operator identity mirrors ``flash_dism::voc_forward_hybrid_v1``: a
# signature/semantics change must bump the name so stale Inductor caches are
# never reused.
@torch.library.custom_op("flash_dism::value_residual_forward_v3", mutates_args=())
def _value_residual_forward_op(
    v: torch.Tensor, v_first: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return torch_fwd_value_residual(v, v_first, gate)


@_value_residual_forward_op.register_fake
def _value_residual_forward_fake(v, v_first, gate):
    return v.new_empty(v.shape)


@torch.library.custom_op("flash_dism::value_residual_backward_v3", mutates_args=())
def _value_residual_backward_op(
    v: torch.Tensor, v_first: torch.Tensor, gate: torch.Tensor, dout: torch.Tensor
) -> list[torch.Tensor]:
    return list(torch_bwd_value_residual(v, v_first, gate, dout))


@_value_residual_backward_op.register_fake
def _value_residual_backward_fake(v, v_first, gate, dout):
    return [v.new_empty(v.shape), v_first.new_empty(v_first.shape),
            gate.new_empty(gate.shape)]


def _value_residual_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _value_residual_backward(ctx, dout):
    v, v_first, gate = ctx.saved_tensors
    return tuple(_value_residual_backward_op(v, v_first, gate, dout))


_value_residual_forward_op.register_autograd(
    _value_residual_backward, setup_context=_value_residual_setup)
