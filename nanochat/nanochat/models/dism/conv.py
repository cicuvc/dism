"""Linear + direct CUDA causal_conv1d(SiLU), with legacy FLA parameter layout.

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


@torch.library.custom_op('nanochat_dism::shortconv_cuda_v1', mutates_args=())
def _cuda_shortconv(hidden: torch.Tensor, weight: torch.Tensor,
                    seq_idx: torch.Tensor | None) -> torch.Tensor:
    # Functional boundary: the upstream op mutates a noncontiguous out= tensor,
    # which Dynamo cannot trace inside its autograd.Function on this Torch build.
    from causal_conv1d.cpp_functions import causal_conv1d_fwd_function
    return causal_conv1d_fwd_function(hidden.transpose(1, 2), weight, None,
                                      seq_idx, None, None, True).transpose(1, 2).contiguous()


@_cuda_shortconv.register_fake
def _cuda_shortconv_fake(hidden, weight, seq_idx):
    return torch.empty_like(hidden, memory_format=torch.contiguous_format)


@torch.library.custom_op('nanochat_dism::shortconv_cuda_backward_v1', mutates_args=())
def _cuda_shortconv_backward(hidden: torch.Tensor, weight: torch.Tensor,
                             dout: torch.Tensor, seq_idx: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
    dx, dw, _, _ = causal_conv1d_bwd_function(
        hidden.transpose(1, 2), weight, None, dout.contiguous().transpose(1, 2),
        seq_idx, None, None, None, False, True)
    return dx.transpose(1, 2).contiguous(), dw


@_cuda_shortconv_backward.register_fake
def _cuda_shortconv_backward_fake(hidden, weight, dout, seq_idx):
    return (torch.empty_like(hidden, memory_format=torch.contiguous_format),
            torch.empty_like(weight))


def _shortconv_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _shortconv_autograd(ctx, dout):
    hidden, weight, seq_idx = ctx.saved_tensors
    dx, dw = _cuda_shortconv_backward(hidden, weight, dout, seq_idx)
    return dx, dw, None


_cuda_shortconv.register_autograd(_shortconv_autograd, setup_context=_shortconv_context)


def packed_conv_sequence_ids(cu_seqlens, tokens, device):
    """Build [1,T] int32 IDs entirely on-device, including empty intervals.

    No scalar extraction or pointer cache: callers may overwrite boundary
    storage between batches. One result can be shared by all convolution layers.
    Training's native pack validator still owns boundary/alignment validation.
    """
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError('cu_seqlens must be a one-dimensional boundary vector')
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError('cu_seqlens must be int32 or int64')
    boundaries = cu_seqlens.to(device=device).contiguous()
    positions = torch.arange(tokens, device=device, dtype=boundaries.dtype)
    return (torch.searchsorted(boundaries, positions, right=True, out_int32=True)-1).unsqueeze(0)


class LinearShortConvSiLU(nn.Module):
    bypass_compiled_conv = True
    supports_seq_idx = True

    def __init__(self, in_channels, out_channels, kernel_size, backend="cuda"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if backend == 'cuda':
            # Same parameter keys, shapes, ordering and initialization RNG as
            # ShortConvolution (which subclasses nn.Conv1d). No wrapper forward.
            self.conv = nn.Conv1d(out_channels, out_channels, kernel_size,
                                  groups=out_channels, bias=False, padding=kernel_size-1)
            self.conv.backend = 'cuda'
        elif backend == 'triton':
            from fla.modules import ShortConvolution
            self.conv = ShortConvolution(out_channels, kernel_size, activation='silu', backend=backend)
        else:
            raise ValueError(f'unsupported convolution backend {backend!r}')

    def project_linear(self, x):
        """Linear (input projection) part only; shared between DISM and GDN q/k."""
        return F.linear(x, self.weight)

    def depthwise_silu(self, hidden, cu_seqlens=None, seq_idx=None):
        """Depthwise causal conv + SiLU on an already-projected tensor."""
        if self.conv.backend == 'cuda':
            if cu_seqlens is not None and hidden.shape[0] != 1:
                raise ValueError('packed convolution requires batch=1')
            if seq_idx is None and cu_seqlens is not None:
                seq_idx = packed_conv_sequence_ids(cu_seqlens, hidden.shape[1], hidden.device)
            if seq_idx is not None:
                if (seq_idx.shape != hidden.shape[:2] or seq_idx.dtype != torch.int32
                        or seq_idx.device != hidden.device):
                    raise ValueError('seq_idx must be same-device int32 [B,T]')
                seq_idx = seq_idx.contiguous()
            # Preserve channel-last layout required by the CUDA seq_idx path.
            hidden = hidden.contiguous()
            return _cuda_shortconv(hidden, self.conv.weight.squeeze(1), seq_idx)
        output, _ = self.conv(hidden, cu_seqlens=cu_seqlens)
        return output

    def forward(self, x, cu_seqlens=None, max_seqlen=None, input_state=None, seq_idx=None):
        if input_state is not None:
            raise NotImplementedError("linear_fla conv does not support cached decoding state")
        return self.depthwise_silu(self.project_linear(x), cu_seqlens=cu_seqlens, seq_idx=seq_idx)
