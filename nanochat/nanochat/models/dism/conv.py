"""Optional conv frontend: torch Linear + FLA ShortConvolution(SiLU).

Drop-in replacement for ``flash_dism.kernels.conv1d.CausalShortConv1d`` used to
compare the fused Triton (linear + depthwise causal conv + SiLU) kernel against
two library ops. Parameter count matches the fused module
(``weight`` [out, in] plus the depthwise kernel).

``bypass_compiled_conv`` tells the DISM branch to call this module directly
instead of the fused ``flash_dism::conv_forward`` custom op.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LinearShortConvSiLU(nn.Module):
    bypass_compiled_conv = True

    def __init__(self, in_channels, out_channels, kernel_size, backend="cuda"):
        super().__init__()
        from fla.modules import ShortConvolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.conv = ShortConvolution(out_channels, kernel_size, activation="silu", backend=backend)

    def forward(self, x, cu_seqlens=None, max_seqlen=None, input_state=None):
        if input_state is not None:
            raise NotImplementedError("linear_fla conv does not support cached decoding state")
        hidden = F.linear(x, self.weight)
        output, _ = self.conv(hidden, cu_seqlens=cu_seqlens)
        return output
