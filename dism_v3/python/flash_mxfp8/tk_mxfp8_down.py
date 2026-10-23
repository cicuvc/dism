"""Pure-MXFP8 down projection assembled from existing kernels."""

import math

import torch
from torch import nn

from .core import quantize_weight
from .tk_mxfp8_dgrad import Library as DgradLibrary
from .tk_mxfp8_gemm import Library as GemmLibrary
from .tk_mxfp8_wgrad import Library as WgradLibrary
from .tk_silu_bwd import SiluBwdQuantizeLibrary


class _ForwardState:
    def __init__(
        self,
        hidden,
        weight_fp8,
        weight_scale,
        forward_config,
        raster_group,
    ):
        self.quantize = SiluBwdQuantizeLibrary().plan_direct(hidden)
        self.gemm = GemmLibrary().plan(
            self.quantize.row_fp8,
            weight_fp8,
            self.quantize.row_scale,
            weight_scale,
            config=forward_config,
            raster_group=raster_group,
        )

    def __call__(self, hidden, weight_fp8, weight_scale):
        self.quantize.rebind(hidden)
        self.gemm.rebind(
            self.quantize.row_fp8,
            weight_fp8,
            self.quantize.row_scale,
            weight_scale,
        )
        self.quantize()
        return self.gemm()


class _BackwardState:
    def __init__(
        self,
        dy,
        hidden,
        hidden_row,
        hidden_row_scale,
        hidden_column,
        hidden_column_scale,
        weight_fp8,
        weight_scale,
        dx_config,
        dw_config,
        raster_group,
    ):
        self.dy_quantize = SiluBwdQuantizeLibrary().plan_direct(
            dy, transpose=True
        )
        self.dx = DgradLibrary().plan(
            self.dy_quantize.row_fp8,
            weight_fp8,
            self.dy_quantize.row_scale,
            weight_scale,
            config=dx_config,
            raster_group=raster_group,
        )
        self.dw = WgradLibrary().plan(
            self.dy_quantize.column_fp8,
            hidden_column,
            self.dy_quantize.column_scale,
            hidden_column_scale,
            config=dw_config,
            raster_group=raster_group,
        )
        self.hidden = hidden
        self.hidden_row = hidden_row
        self.hidden_row_scale = hidden_row_scale
        self.device = dy.device
        self.dx_shape = (dy.shape[0], hidden_column.shape[1])
        self.dw_shape = (weight_fp8.shape[0], hidden_column.shape[1])

    def __call__(
        self,
        dy,
        hidden_column,
        hidden_column_scale,
        weight_fp8,
        weight_scale,
    ):
        self.dy_quantize.reallocate_outputs(dy)
        self.dy_quantize.rebind(dy)
        self.dx.rebind(
            self.dy_quantize.row_fp8,
            weight_fp8,
            self.dy_quantize.row_scale,
            weight_scale,
            out=torch.empty(self.dx_shape, device=self.device, dtype=torch.bfloat16),
        )
        self.dw.rebind(
            self.dy_quantize.column_fp8,
            hidden_column,
            self.dy_quantize.column_scale,
            hidden_column_scale,
            out=torch.empty(self.dw_shape, device=self.device, dtype=torch.bfloat16),
        )
        self.dy_quantize()
        dx = self.dx()
        dw = self.dw()
        self.dy_quantize.release()
        self.dx.release()
        self.dw.release()
        return dx, dw


class _DownFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, module):
        module._sync_if_needed()
        state = module._forward_state(hidden)
        out = state(hidden, module._weight_fp8, module._weight_scale)
        ctx.module = module
        ctx.save_for_backward(
            hidden,
            weight,
            module._weight_fp8,
            module._weight_scale,
            state.quantize.row_fp8,
            state.quantize.row_scale,
            state.quantize.column_fp8,
            state.quantize.column_scale,
        )
        return out

    @staticmethod
    def backward(ctx, dy):
        (
            hidden,
            weight,
            weight_fp8,
            weight_scale,
            hidden_row,
            hidden_row_scale,
            hidden_column,
            hidden_column_scale,
        ) = ctx.saved_tensors
        dy = dy.contiguous()
        if dy.dtype != torch.bfloat16:
            dy = dy.to(torch.bfloat16)
        state = ctx.module._backward_state(
            dy,
            hidden,
            hidden_row,
            hidden_row_scale,
            hidden_column,
            hidden_column_scale,
            weight_fp8,
            weight_scale,
        )
        dx, dw = state(
            dy,
            hidden_column,
            hidden_column_scale,
            weight_fp8,
            weight_scale,
        )
        return dx, dw.float(), None


class MXFP8DownProjection(nn.Module):
    """No-bias down projection with pure MXFP8 forward and backward GEMMs."""

    def __init__(
        self,
        in_features,
        out_features,
        *,
        device=None,
        forward_config=70,
        dx_config=12,
        dw_config=12,
        raster_group=13,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.forward_config = forward_config
        self.dx_config = dx_config
        self.dw_config = dw_config
        self.raster_group = raster_group
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "_weight_fp8",
            torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_weight_scale",
            torch.empty(0, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self._synced_version = -1
        self._fp8_dirty = False
        self._forward_states = {}
        self._backward_states = {}
        self.reset_parameters()
        if self.weight.device.type == "cuda":
            self.refresh_fp8_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def refresh_fp8_weight(self):
        if not self.weight.is_cuda:
            raise RuntimeError("MXFP8DownProjection requires CUDA")

        with torch.no_grad():
            self._weight_fp8, self._weight_scale = quantize_weight(
                self.weight.detach()
            )
        self._synced_version = self.weight._version
        self._fp8_dirty = False
        return self

    def refresh_fp8_weights(self):
        return self.refresh_fp8_weight()

    def mark_fp8_dirty(self):
        self._fp8_dirty = True
        return self

    def fp8_parameter_bindings(self):
        return ((self.weight, self._weight_fp8, self._weight_scale),)

    def mark_fp8_synced(self):
        self._synced_version = self.weight._version
        self._fp8_dirty = False
        return self

    def _sync_if_needed(self):
        if self._fp8_dirty or self._synced_version != self.weight._version:
            self.refresh_fp8_weight()

    @property
    def quantized_weight(self):
        self._sync_if_needed()
        return self._weight_fp8, self._weight_scale

    def _forward_state(self, hidden):
        self._sync_if_needed()
        key = (
            hidden.device.index,
            tuple(hidden.shape),
            self.forward_config,
            self.raster_group,
        )
        # Eager eval owns temporary workspaces per invocation. Training and
        # graph capture retain the existing stable-buffer reuse contract.
        retain = self.training or torch.cuda.is_current_stream_capturing()
        state = self._forward_states.get(key) if retain else None
        if state is None:
            state = _ForwardState(
                hidden,
                self._weight_fp8,
                self._weight_scale,
                self.forward_config,
                self.raster_group,
            )
            if retain:
                self._forward_states[key] = state
        return state

    def _backward_state(
        self,
        dy,
        hidden,
        hidden_row,
        hidden_row_scale,
        hidden_column,
        hidden_column_scale,
        weight_fp8,
        weight_scale,
    ):
        key = (
            dy.device.index,
            tuple(dy.shape),
            tuple(hidden_column.shape),
            self.dx_config,
            self.dw_config,
            self.raster_group,
        )
        state = self._backward_states.get(key)
        if state is None:
            state = _BackwardState(
                dy,
                hidden,
                hidden_row,
                hidden_row_scale,
                hidden_column,
                hidden_column_scale,
                weight_fp8,
                weight_scale,
                self.dx_config,
                self.dw_config,
                self.raster_group,
            )
            self._backward_states[key] = state
        return state

    @torch.compiler.disable
    def forward(self, hidden):
        if hidden.ndim != 2 or hidden.shape[1] != self.in_features:
            raise ValueError(
                f"expected hidden [M,{self.in_features}], got {tuple(hidden.shape)}"
            )
        if hidden.dtype != torch.bfloat16 or not hidden.is_cuda:
            raise TypeError("expected CUDA BF16 hidden")
        return _DownFunction.apply(hidden.contiguous(), self.weight, self)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"forward_config={self.forward_config}, dx_config={self.dx_config}, "
            f"dw_config={self.dw_config}"
        )
