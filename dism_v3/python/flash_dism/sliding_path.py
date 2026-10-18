"""Sliding-window PaTH mixers matching the DISM hybrid interfaces."""

import torch
from torch import nn

from fla.modules.l2norm import l2norm

from .kernels.conv1d import CausalShortConv1d
from .module import DismAttention, value_residual_mix
from .ops.sliding_delta_net.varlen import prepare_varlen_metadata
from .ops.sliding_path import SlidingWindowPaTH, sliding_window_path
from .sliding_delta import _project


@torch.compiler.disable
def sliding_path_raw(module, x, cu_seqlens, max_seqlen, v_first=None,
                     linear_q=None, linear_k=None, conv_seq_idx=None):
    """Differentiable packed sliding-window PaTH branch."""
    q = _project(module.path_q_conv, module.head_dim, x, cu_seqlens, max_seqlen,
                 linear_q, conv_seq_idx)
    k = _project(module.path_k_conv, module.head_dim, x, cu_seqlens, max_seqlen,
                 linear_k, conv_seq_idx)
    v = _project(module.path_v_conv, module.value_dim, x, cu_seqlens, max_seqlen,
                 conv_seq_idx=conv_seq_idx)
    if module.value_residual and v_first is not None:
        v = value_residual_mix(v, v_first, module.path_v_residual_gate(x))
    beta = module.path_b_proj(x).float().sigmoid().transpose(1, 2).contiguous()
    q, k = l2norm(q), l2norm(k)
    return sliding_window_path(
        q, k, v, beta, module.window_size,
        chunk_size=module.sliding_path_chunk_size,
        scale=module.head_dim ** -0.5,
        cu_seqlens=cu_seqlens,
    )


class PureSlidingPathAttention(SlidingWindowPaTH):
    """Repo-facing adapter around the vendored PaTH layer."""

    is_pure_gdn = False
    is_pure_mixer = True

    def __init__(self, width, heads, *, head_dim=64, value_dim=64, conv_size=4,
                 value_residual=False, window_size=128,
                 sliding_path_chunk_size=32, **kwargs):
        if head_dim != 64 or value_dim != 64:
            raise ValueError('sliding PaTH currently requires Q/K and V head_dim=64')
        super().__init__(
            hidden_size=width, expand_v=1, head_dim=head_dim,
            num_heads=heads, num_v_heads=heads, use_gate=True,
            use_short_conv=True, allow_neg_eigval=False,
            conv_size=conv_size, window_size=window_size,
            chunk_size=sliding_path_chunk_size)
        self.value_residual = bool(value_residual)
        if self.value_residual:
            self.path_v_residual_gate = nn.Linear(width, heads)
            self.path_v_residual_gate._value_residual_gate = True

    @property
    def norm(self):
        return self.o_norm

    def mixer_value_first(self, x, *, cu_seqlens=None, max_seqlen=None,
                          conv_seq_idx=None):
        metadata = prepare_varlen_metadata(
            cu_seqlens, batch=x.shape[0], total_tokens=x.shape[1],
            chunk_size=self.chunk_size, device=x.device)
        conv_indices = None
        if metadata is not None:
            conv_metadata = prepare_varlen_metadata(
                cu_seqlens, batch=1, total_tokens=x.shape[1],
                chunk_size=64, device=x.device)
            conv_indices = conv_metadata.chunk_indices
        _, _, value, _ = self._features(
            x, None if metadata is None else metadata.cu_seqlens, conv_indices)
        return value

    @torch.compiler.disable
    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, **kwargs):
        if (attention_mask is not None or past_key_values is not None
                or use_cache):
            raise NotImplementedError('Packed sliding PaTH prefill only')
        self._validate_input(hidden_states)
        cu_seqlens = kwargs.get('cu_seqlens')
        q, k, v, beta = self._project_sequence(hidden_states, cu_seqlens)
        v_first = kwargs.get('v_first')
        if self.value_residual and v_first is not None:
            v = value_residual_mix(v, v_first, self.path_v_residual_gate(hidden_states))
        output = sliding_window_path(
            q, k, v, beta, self.window_size, chunk_size=self.chunk_size,
            scale=self.scale, cu_seqlens=cu_seqlens)
        return self._output(output, hidden_states), None, past_key_values


class DismSlidingPathAttention(DismAttention):
    """DISM output plus a sliding PaTH branch sharing the Q/K linear."""

    def __init__(self, width, heads, **kwargs):
        window_size = int(kwargs.pop('window_size', 128))
        sliding_path_chunk_size = int(kwargs.pop('sliding_path_chunk_size', 32))
        super().__init__(width, heads, **kwargs)
        self.path_q_conv = CausalShortConv1d(
            width, heads*self.head_dim, self.q_conv.kernel_size)
        self.path_k_conv = CausalShortConv1d(
            width, heads*self.head_dim, self.k_conv.kernel_size)
        self.path_v_conv = CausalShortConv1d(
            width, heads*self.value_dim, self.v_conv.kernel_size)
        self.path_b_proj = nn.Linear(width, heads, bias=False)
        if isinstance(window_size, bool) or window_size <= 0:
            raise ValueError('window_size must be a positive integer')
        self.window_size = window_size
        self.sliding_path_chunk_size = sliding_path_chunk_size
        if self.sliding_path_chunk_size not in (16, 32, 64):
            raise ValueError('sliding_path_chunk_size must be 16, 32, or 64')
        if self.value_residual:
            self.path_v_residual_gate = nn.Linear(width, heads)
            self.path_v_residual_gate._value_residual_gate = True
        self.tie_branch_projections()

    def tie_branch_projections(self):
        self.path_q_conv.weight = self.q_conv.weight
        self.path_k_conv.weight = self.k_conv.weight

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.tie_branch_projections()
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.tie_branch_projections()
        return result

    def _combine_cuda(self, output, x, v, cu_seqlens, max_seqlen, v_first=None,
                      linear_q=None, linear_k=None, conv_seq_idx=None):
        return output + sliding_path_raw(
            self, x, cu_seqlens, max_seqlen, v_first,
            linear_q, linear_k, conv_seq_idx)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, **kwargs):
        if use_cache or past_key_values is not None:
            raise NotImplementedError('DISM + sliding PaTH uses packed prefill only')
        if not hidden_states.is_cuda or hidden_states.shape[1] % 256:
            raise ValueError('DISM + sliding PaTH requires CUDA and 256-aligned input')
        return super().forward(hidden_states, attention_mask=attention_mask, **kwargs)
