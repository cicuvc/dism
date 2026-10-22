"""Fused residual-add RMSNorm with BF16 and dual-direction MXFP8 outputs."""

import torch
from torch import nn

from .tk_rmsnorm import FusedAddRMSNormBackwardPlan, FusedAddRMSNormPlan


class _FusedAddRMSNormMXFP8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, factor, owner, eps):
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        residual = residual.reshape(-1, residual.shape[-1])
        plan = owner._forward_plan(x, residual, factor)
        inverse, _, y_bf16, row, row_scale, column, column_scale = plan()
        ctx.owner = owner
        ctx.input_shape = shape
        ctx.factor_dtype = factor.dtype
        ctx.save_for_backward(x, residual, factor, inverse)
        ctx.mark_non_differentiable(row, row_scale, column, column_scale)
        return y_bf16.reshape(shape), row, row_scale, column, column_scale

    @staticmethod
    def backward(ctx, dy, _drow, _drow_scale, _dcolumn, _dcolumn_scale):
        x, residual, factor, inverse = ctx.saved_tensors
        dy = dy.reshape(-1, x.shape[-1]).contiguous()
        plan = ctx.owner._backward_plan(dy, x, residual, factor, inverse)
        dx, _, factor_grad = plan()
        dx = dx.reshape(ctx.input_shape)
        return dx, dx, factor_grad.to(ctx.factor_dtype), None, None


class FusedAddRMSNormMXFP8(nn.Module):
    """RMSNorm(x + residual) producing BF16 and row/column MXFP8 layouts."""

    def __init__(self, width, *, eps=1e-6):
        super().__init__()
        self.width = int(width)
        self.eps = float(eps)
        self._forward_plans = {}
        self._backward_plans = {}

    @staticmethod
    def _key(tensors):
        return (
            tensors[0].device.index,
            tuple(tensors[0].shape),
            tuple(tensor.dtype for tensor in tensors),
        )

    def _forward_plan(self, x, residual, factor):
        key = self._key((x, residual, factor))
        plan = self._forward_plans.get(key)
        if plan is None:
            plan = FusedAddRMSNormPlan(
                x,
                residual,
                factor,
                eps=self.eps,
                write_sum=False,
                y_mode="bf16_mxfp8",
            )
            self._forward_plans[key] = plan
        else:
            plan.rebind(x, residual, factor)
        return plan

    def _backward_plan(self, dy, x, residual, factor, inverse):
        key = self._key((dy, x, residual, factor))
        plan = self._backward_plans.get(key)
        if plan is None:
            plan = FusedAddRMSNormBackwardPlan(
                dy, x, residual, factor, inverse
            )
            self._backward_plans[key] = plan
        else:
            plan.rebind(dy, x, residual, factor, inverse)
        return plan

    @torch.compiler.disable
    def forward(self, x, residual, factor):
        if (
            x.shape[-1] != self.width
            or x.shape != residual.shape
            or factor.shape != (self.width,)
        ):
            raise ValueError("invalid fused RMSNorm input shapes")
        return _FusedAddRMSNormMXFP8Function.apply(x, residual, factor, self, self.eps)
