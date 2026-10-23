"""Final RMSNorm with BF16 output and chunk-ready dual MXFP8 activations."""
import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .tk_rmsnorm import RMSNormDualQuantPlan, RMSNormBackwardPlan


class _RMSNormMXFP8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor, eps, chunk_size):
        # Each invocation owns its outputs/inverse, including multiple forwards
        # before backward and retained-graph backward. CUDA Graph owns the pool
        # allocations during replay; no mutable Python plan is saved in ctx.
        plan = RMSNormDualQuantPlan(x, factor, eps=eps, write_bf16=True,
                                   chunk_size=chunk_size)
        row, rs, column, cs, inverse = plan()
        ctx.save_for_backward(x, factor, inverse)
        ctx.mark_non_differentiable(row, rs, column, cs)
        return plan.y_bf16, row, rs, column, cs

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, _row, _rs, _column, _cs):
        x, factor, inverse = ctx.saved_tensors
        dx, df = RMSNormBackwardPlan(dy.contiguous(), x, factor, inverse)()
        return dx, df.to(factor.dtype), None, None


class RMSNormMXFP8(nn.Module):
    """RMSNorm plus dual quantization, retaining BF16 y as the gradient input.

    Returns (y, chunks). Each chunk contains (row, row_scale, column,
    column_scale), ready for MXFP8FusedLinearCrossEntropy.quantized_input.
    Quantized tensors are detached STE representations of y, not independent
    differentiable inputs. The caller supplies the existing norm parameter,
    so this module adds no parameters or persistent checkpoint buffers.
    """
    def __init__(self, *, eps=1e-6, chunk_size=1024):
        super().__init__()
        if type(chunk_size) is not int or chunk_size <= 0 or chunk_size % 128:
            raise ValueError('chunk_size must be a positive multiple of 128')
        self.eps, self.chunk_size = float(eps), chunk_size

    @torch.compiler.disable
    def forward(self, x, factor, *, eps=None, chunk_size=None):
        eps = self.eps if eps is None else float(eps)
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        if type(chunk_size) is not int or chunk_size <= 0 or chunk_size % 128:
            raise ValueError('chunk_size must be a positive multiple of 128')
        y, row, rs, column, cs = _RMSNormMXFP8Function.apply(
            x, factor, eps, chunk_size)
        m, k = x.shape
        chunks = []
        scales = rs.view(-1)
        for start in range(0, m, chunk_size):
            n = min(chunk_size, m-start)
            mp, mp16 = (n+127)//128*128, (n+15)//16*16
            offset = start*((k+127)//128)
            chunk_rs = scales[offset:offset+((k+127)//128)*mp16].view((k+127)//128, mp16)
            chunks.append((row[start:start+n], chunk_rs, column[start:start+mp],
                           cs[start//128:(start+mp)//128]))
        return y, chunks
