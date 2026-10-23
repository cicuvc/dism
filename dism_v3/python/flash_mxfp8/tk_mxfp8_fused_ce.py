"""Experimental chunked MXFP8 LM head + cross entropy on SM120.

By default returns a scalar SUM of non-ignored token losses. balance="token"
averages valid tokens; balance="document" averages the per-document valid-token
means, excluding empty/all-ignored documents. No bias or smoothing option.
Forward computes first-order dX/dW
chunk by chunk; backward only scales the saved gradients. Z is BF16, LSE/loss
are FP32, and dZ is independently MXFP8-quantized along both GEMM reduction axes.

Example::

    criterion = MXFP8FusedLinearCrossEntropy(chunk_size=1024, balance="token")
    loss = criterion(hidden_bf16, targets_int64, lm_head_weight)
    loss.backward()

Weights may be FP32 masters or BF16 (including a tied embedding parameter).
Optional quantized_weight=(E4M3, E8M0) avoids requantizing W; the caller must keep
that cache synchronized. Quantization uses the existing linear's STE convention.
Build CUDA dependencies as for MXFP8Linear, plus python -m flash_mxfp8.build --only lse.
"""

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .core import quantize_weight
from .mxfp8_ce import CEPlan, merge_lse, _validate_target, _validate_ignore_index, _validate_softcap
from .mxfp8_balance import document_weights, reduce_balanced_loss, validate_cu_seqlens
from .tk_mxfp8_lse import Library as LSELibrary, CONFIGS
from .tk_mxfp8_dgrad import Library as DgradLibrary
from .tk_mxfp8_wgrad import Library as WgradLibrary
from .tk_silu_bwd import SiluBwdQuantizeLibrary


def _validate_quantized_input(x, chunks, chunk_size):
    """Check packed chunk layouts without host reads or GPU synchronization."""
    m, h = x.shape
    if not isinstance(chunks, (tuple, list)) or len(chunks) != (m+chunk_size-1)//chunk_size:
        raise ValueError("quantized_input must contain one four-tensor tuple per CE chunk")
    for start, tensors in zip(range(0, m, chunk_size), chunks):
        n = min(chunk_size, m-start)
        mp, mp16 = (n+127)//128*128, (n+15)//16*16
        shapes = ((n,h), ((h+127)//128,mp16), (mp,h), (mp//128,(h+15)//16*16))
        dtypes = (torch.float8_e4m3fn, torch.int32, torch.float8_e4m3fn, torch.int32)
        if not isinstance(tensors, (tuple, list)) or len(tensors) != 4:
            raise ValueError("expected row/scales/column/scales for each CE chunk")
        for tensor, shape, dtype in zip(tensors, shapes, dtypes):
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype
                    or tensor.device != x.device or not tensor.is_contiguous() or tensor.data_ptr()%16
                    or tensor.requires_grad):
                raise ValueError("invalid prequantized activation shape/dtype/device/alignment")


class _ChunkState:
    def __init__(self, x, w, ws, module, quantized=None):
        self.xq = module._quant.plan_direct(x) if quantized is None else None
        row, row_scale, column, column_scale = (quantized if quantized is not None else
            (self.xq.row_fp8, self.xq.row_scale, self.xq.column_fp8, self.xq.column_scale))
        self.gemm = module._gemm.plan(
            row, w, row_scale, ws,
            config=module.forward_config, raster_group=module.raster_group,
            softcap=module.softcap,
        )
        m, h = x.shape
        v = w.shape[0]
        self.lse = torch.empty(m, device=x.device, dtype=torch.float32)
        self.loss = torch.empty_like(self.lse)
        self.ce = CEPlan(m, v, x.device)
        self.dx = module._dgrad.plan(
            self.ce.row_fp8, w, self.ce.row_scale, ws,
            config=module.dx_config, raster_group=module.raster_group,
        )
        self.dw = module._wgrad.plan(
            self.ce.column_fp8, column,
            self.ce.column_scale, column_scale,
            config=module.dw_config, raster_group=module.raster_group,
        )

    def __call__(self, x, target, w, ws, ignore_index, row_weight=None, quantized=None):
        if quantized is None:
            self.xq.rebind(x)
            row, row_scale, column, column_scale = (self.xq.row_fp8, self.xq.row_scale,
                                                   self.xq.column_fp8, self.xq.column_scale)
        else:
            row, row_scale, column, column_scale = quantized
        self.gemm.rebind(row, w, row_scale, ws)
        self.dx.rebind(self.ce.row_fp8, w, self.ce.row_scale, ws)
        if quantized is not None:
            self.dw.rebind(self.ce.column_fp8, column, self.ce.column_scale, column_scale)
        try:
            if self.xq is not None:
                self.xq()
            logits, partial = self.gemm()
            merge_lse(logits, partial, target, lse=self.lse, loss=self.loss,
                      ignore_index=ignore_index, row_weight=row_weight, softcap=self.gemm.softcap)
            self.ce(logits, self.lse, target, ignore_index=ignore_index, row_weight=row_weight,
                    softcap=self.gemm.softcap)
            return self.loss, self.dx(), self.dw()
        finally:
            # Cached workspaces must not retain a previous activation graph or W.
            # The descriptors are rebound before their next launch.
            if self.xq is not None:
                self.xq.x = None
            self.gemm.refs = ()
            self.dx.refs = []
            self.dw.refs = []


class _FusedCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, weight, wq, ws, module, cu_seqlens, quantized_input):
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        losses = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        row_weight = doc_counts = None
        with torch.cuda.device(x.device):
            if module.balance in ("document", "document_sum"):
                row_weight, doc_counts = document_weights(
                    target, cu_seqlens, ignore_index=module.ignore_index,
                )
            for start in range(0, x.shape[0], module.chunk_size):
                end = min(start + module.chunk_size, x.shape[0])
                c_x = x[start:end]
                quantized = None if quantized_input is None else quantized_input[start // module.chunk_size]
                state = module._state(c_x, wq, ws, quantized)
                c_weight = None if row_weight is None else row_weight[start:end]
                c_loss, c_dx, c_dw = state(c_x, target[start:end], wq, ws, module.ignore_index, c_weight, quantized)
                losses[start:end].copy_(c_loss)
                dx[start:end].copy_(c_dx)
                dw.add_(c_dw)
        # Independent storage: future forwards may immediately reuse every plan.
        # Accumulate dW in FP32 and cast only after all chunks have contributed.
        if module.balance in (None, "document_sum"):
            loss, inv = losses.sum(), None
        else:
            loss, inv = reduce_balanced_loss(
                losses, target, balance=module.balance, doc_counts=doc_counts,
                ignore_index=module.ignore_index,
            )
        ctx.save_for_backward(dx, dw.to(weight.dtype), inv)
        if module.balance == "document_sum":
            count = (doc_counts > 0).sum()
            ctx.mark_non_differentiable(count)
            return loss, count
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_count=None):
        dx, dw, inv = ctx.saved_tensors
        # No host read of grad_output, and no mutation of saved tensors: CUDA
        # Graph capture and repeated backward with different scalars both work.
        scale = grad_output if inv is None else grad_output * inv
        return dx * scale, None, dw * scale, None, None, None, None, None


class MXFP8FusedLinearCrossEntropy(nn.Module):
    """Reusable sequential-stream criterion, with bounded per-chunk workspaces.

    X must be contiguous BF16 [M,H], W contiguous FP32/BF16 [V,H], and targets
    contiguous int64 [M] in [0,V) or equal to ignore_index (default -100).
    Ignored rows contribute zero loss and gradients; all-ignored inputs return
    zero loss and gradients as well. H/V must be multiples of 64 and satisfy the
    selected GEMM panels. Default configs require multiples of 128. A final
    short token chunk is supported via the quantizer's zero padding.

    balance=None preserves summed loss. "token" divides by valid token count;
    "document" uses cu_seqlens[D+1] (CUDA int32/int64) to weight each nonempty
    document equally by its valid-token mean. Boundaries cover [0,M], are
    nondecreasing, and describe documents, not arbitrary attention segments.
    Repeated boundaries (empty documents) are allowed. Normalization is over
    this invocation, not across gradient-accumulation steps or distributed ranks.
    "document_sum" instead returns (sum of document means, valid document count)
    without division by that count, for global optimizer-step normalization.

    softcap=None disables softcapping. A positive finite constant s applies
    s*tanh(l/s) in FP32 to the raw BF16 logits l. CUDA and Triton both use
    tanh.approx.f32. The chain-rule factor (1-tanh(l/s)**2) is applied before
    BF16 rounding and dual MXFP8 gradient quantization. No extra buffer is used.

    quantized_input optionally supplies one (row, row_scale, column, column_scale)
    tuple per chunk, as returned by RMSNormMXFP8. These detached buffers MUST
    represent x and remain unmodified until forward has finished on its stream.
    Gradients still flow through the BF16 x using the same STE as direct
    quantization. No activation quantization kernel is launched on this path.

    Warm up once for each shape before CUDA Graph capture. This object owns
    mutable scratch buffers and must not be called concurrently on two streams.
    Only first-order differentiation is supported, as in the existing linear.
    """
    def __init__(self, *, chunk_size=1024, forward_config=8, dx_config=12,
                 dw_config=12, raster_group=13, ignore_index=-100, balance=None,
                 softcap=None):
        super().__init__()
        self.softcap = _validate_softcap(softcap)
        if balance not in (None, "token", "document", "document_sum"):
            raise ValueError("balance must be None, 'token', 'document', or 'document_sum'")
        self.balance = balance
        _validate_ignore_index(ignore_index)
        self.ignore_index = ignore_index
        if type(chunk_size) is not int or chunk_size <= 0 or chunk_size % 128:
            raise ValueError("chunk_size must be a positive multiple of 128")
        if type(forward_config) is not int or forward_config not in CONFIGS:
            raise ValueError("Unsupported forward LSE config")
        self.chunk_size = chunk_size
        self.forward_config, self.dx_config, self.dw_config = forward_config, dx_config, dw_config
        self.raster_group = raster_group
        self._states = {}
        self._quant = self._gemm = self._dgrad = self._wgrad = None

    def _state(self, x, w, ws, quantized=None):
        if self._gemm is None:
            self._quant, self._gemm = SiluBwdQuantizeLibrary(), LSELibrary()
            self._dgrad, self._wgrad = DgradLibrary(), WgradLibrary()
        key = (x.device, tuple(x.shape), tuple(w.shape), self.forward_config,
               self.dx_config, self.dw_config, self.raster_group, self.softcap, quantized is not None)
        if key not in self._states:
            self._states[key] = _ChunkState(x, w, ws, self, quantized)
        return self._states[key]

    def clear_workspace(self):
        """Drop cached per-shape workspaces outside CUDA Graph capture/replay."""
        self._states.clear()

    @torch.compiler.disable
    def forward(self, x, target, weight, *, quantized_weight=None, cu_seqlens=None,
                quantized_input=None):
        if (x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda or
                not x.is_contiguous() or min(x.shape) <= 0):
            raise ValueError("Expected nonempty contiguous CUDA BF16 X[M,H]")
        if (weight.ndim != 2 or weight.shape[1] != x.shape[1] or weight.shape[0] <= 0 or
                weight.dtype not in (torch.bfloat16, torch.float32) or
                weight.device != x.device or not weight.is_contiguous()):
            raise ValueError("Expected contiguous FP32/BF16 W[V,H] on X's device")
        if weight.shape[0] % 64 or x.shape[1] % 64:
            raise ValueError("V and H must be multiples of 64")
        if torch.cuda.get_device_capability(x.device) != (12, 0):
            raise RuntimeError("SM120 required")
        _validate_target(target, x.shape[0], x.device)
        if self.balance in ("document", "document_sum") or cu_seqlens is not None:
            validate_cu_seqlens(cu_seqlens, x.device)
        if quantized_input is not None:
            _validate_quantized_input(x, quantized_input, self.chunk_size)
        if quantized_weight is None:
            with torch.no_grad():
                wq, ws = quantize_weight(weight.detach().float())
        else:
            wq, ws = quantized_weight
            if (wq.shape != weight.shape or wq.dtype != torch.float8_e4m3fn or
                    ws.shape != (weight.shape[0] // 32, weight.shape[1] // 32) or
                    ws.dtype != torch.uint8 or wq.device != x.device or ws.device != x.device or
                    not wq.is_contiguous() or not ws.is_contiguous()):
                raise ValueError("Invalid quantized weight cache")
        return _FusedCEFunction.apply(x, target, weight, wq, ws, self, cu_seqlens, quantized_input)


def fused_linear_cross_entropy_loss(x, target, weight, *, chunk_size=1024,
                                    forward_config=8, dx_config=12, dw_config=12,
                                    raster_group=13, quantized_weight=None, ignore_index=-100,
                                    balance=None, cu_seqlens=None, softcap=None, quantized_input=None):
    """One-shot scalar CE with balance/softcap; use the module to reuse plans."""
    return MXFP8FusedLinearCrossEntropy(
        chunk_size=chunk_size, forward_config=forward_config, dx_config=dx_config,
        dw_config=dw_config, raster_group=raster_group, ignore_index=ignore_index, balance=balance,
        softcap=softcap,
    )(x, target, weight, quantized_weight=quantized_weight, cu_seqlens=cu_seqlens,
      quantized_input=quantized_input)
