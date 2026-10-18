"""FLA-style torch module for sliding-window PaTH with tied keys/reflectors.

Uses the same parameterization and parameter names as SlidingWindowDeltaNet:
bias-free Q/K/V/beta projections, optional causal short convolutions with
SiLU, per-head Q/K L2 normalization (epsilon 1e-6), optional grouped value
heads, and head-wise RMSNorm with an optional SiLU output gate. There is no
state decay gate. Requires PyTorch, Triton, and flash-linear-attention (fla).

The attention operator is softmax over EXACTLY max(doc_start,i-w+1) <= j <= i
of scale*q_i^T H_i...H_{j+1}k_j, with H_t=I-beta_t*k_t*k_t^T. Values are raw
v_j; no write_beta option. K is also the Householder vector. Both full and
decode paths share the same projections and normalization.

Defaults match the DeltaNet module, including scale=None -> head_dim**-0.5.
Here scale controls the softmax logits and therefore its concentration; set
scale=1.0 to use the original unscaled formula on the normalized Q/K features.
Beta defaults to sigmoid(b_proj(x)); allow_neg_eigval=True doubles it.

forward returns a tensor, supports fixed-length and packed varlen training
(cu_seqlens requires B=1), and does not update a cache. decode accepts one
token per batch row, is forward-only, and updates its explicit state in place
on the current stream. Prefix tokens can be fed one by one. State includes
the PaTH KV ring, GPU valid-length/cursor counters, and Q/K/V conv histories.
This is a single-layer state API, not a Transformers Cache interface.

At the operator boundary Q/K/V are BF16, beta and recurrent keys are FP32;
the kernel's TF32x3 solve/WY precision is preserved. FP32 parameters with or
without BF16 autocast, and BF16 parameters with BF16 inputs, are supported.
Packed metadata has the core operator's first-use host synchronization and
no-varlen-CUDA-Graph constraint. Convolutions reset at document boundaries
and extend the raw-input receptive field by up to conv_size-1 tokens.

Example:
    layer = SlidingWindowPaTH(hidden_size=512, num_heads=4, head_dim=64,
                              window_size=128, scale=1.0).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        y = layer(x)                              # [B,T,512]
        packed_y = layer(packed_x, cu_seqlens=cu)  # B=1
        state = layer.create_decode_state(x.shape[0])
        token_y = layer.decode(x[:, :1], state)
    state.reset_()                                # start fresh sequences
"""

from dataclasses import dataclass

import torch

from ..sliding_delta_net.layer import SlidingWindowDeltaNet, SlidingWindowDeltaNetState
from .decode import (
    SlidingPathDecodeState, create_path_decode_state, sliding_window_path_decode,
)
from .triton import sliding_window_path


@dataclass(frozen=True)
class SlidingWindowPaTHState:
    """One layer's mutable PaTH ring and optional BF16 Q/K/V conv histories.

    Allocate after moving the layer to its target device. path.lengths and
    path.cursor are kernel-owned GPU tensors. Use separate states for
    independent streams, and order calls sharing a state on one stream.
    """

    path: SlidingPathDecodeState
    conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None

    @torch.no_grad()
    def reset_(self):
        self.path.reset_()
        if self.conv_state is not None:
            for tensor in self.conv_state:
                tensor.zero_()
        return self


class SlidingWindowPaTH(SlidingWindowDeltaNet):
    """PaTH softmax operator with the existing FLA-style feature pipeline.

    Parameter shapes/names match SlidingWindowDeltaNet for the same config;
    state_dict can be copied strictly between the layers. Their decoding
    states are different types because PaTH must exclude empty ring slots
    from its softmax denominator.
    """

    def __init__(
        self, hidden_size=2048, expand_v=2, head_dim=256, num_heads=6,
        num_v_heads=None, use_gate=True, use_short_conv=True,
        allow_neg_eigval=False, conv_size=4, conv_bias=False,
        layer_idx=None, norm_eps=1e-5, *, window_size=128, chunk_size=32,
        scale=None, matmul_precision="bf16",
    ):
        super().__init__(
            hidden_size=hidden_size, expand_v=expand_v, head_dim=head_dim,
            num_heads=num_heads, num_v_heads=num_v_heads, use_gate=use_gate,
            use_short_conv=use_short_conv, allow_neg_eigval=allow_neg_eigval,
            conv_size=conv_size, conv_bias=conv_bias, layer_idx=layer_idx,
            norm_eps=norm_eps, window_size=window_size, chunk_size=chunk_size,
            write_beta=False, scale=scale, matmul_precision=matmul_precision,
        )

    def forward(self, hidden_states, *, cu_seqlens=None):
        """Return [B,T,hidden_size] with differentiable window-local softmax.

        cu_seqlens may be CPU or same-device CUDA int32/int64 offsets, with
        physical B=1. Empty documents are supported; total T must be positive.
        """
        q, k, v, beta = self._project_sequence(hidden_states, cu_seqlens)
        o = sliding_window_path(
            q, k, v, beta, self.window_size, chunk_size=self.chunk_size,
            scale=self.scale, matmul_precision=self.matmul_precision,
            cu_seqlens=cu_seqlens)
        return self._output(o, hidden_states)

    def create_decode_state(self, batch_size):
        """Allocate an empty PaTH ring and conv histories on the layer device."""
        device = self.q_proj.weight.device
        path = create_path_decode_state(
            batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim,
            self.window_size, device=device)
        conv_state = None
        if self.use_short_conv:
            conv_state = tuple(torch.zeros(batch_size, width, self.conv_size,
                                           device=device, dtype=torch.bfloat16)
                               for width in (self.key_dim, self.key_dim, self.value_dim))
        return SlidingWindowPaTHState(path, conv_state)

    def _validate_state(self, state, x):
        # Validate every buffer before _features mutates convolution histories.
        if not isinstance(state, SlidingWindowPaTHState):
            raise TypeError("state must be a SlidingWindowPaTHState")
        if not isinstance(state.path, SlidingPathDecodeState):
            raise TypeError("state.path must be a SlidingPathDecodeState")
        if state.path.write_beta:
            raise ValueError("PaTH state must store raw values (write_beta=False)")
        super()._validate_state(SlidingWindowDeltaNetState(state.path, state.conv_state), x)
        lengths = state.path.lengths
        if (not isinstance(lengths, torch.Tensor)
                or lengths.shape != (x.shape[0], self.num_v_heads)
                or lengths.dtype != torch.int32 or lengths.device != x.device
                or not lengths.is_contiguous() or lengths.requires_grad):
            raise ValueError("state lengths must be contiguous int32 [B,H] on the input device")
        other_buffers = (x, state.path.keys, state.path.values, state.path.cursor)
        if state.conv_state is not None:
            other_buffers += state.conv_state
        if any(lengths.untyped_storage().data_ptr() == z.untyped_storage().data_ptr()
               for z in other_buffers):
            raise ValueError("state lengths must not alias input or other state buffers")

    @torch.no_grad()
    def decode(self, hidden_states, state):
        """Return [B,1,hidden_size], updating ring, lengths, and conv histories.

        Inference-only even when hidden_states requires gradients. The state
        is not returned separately; reset it via state.reset_().
        """
        self._validate_input(hidden_states)
        if hidden_states.shape[1] != 1:
            raise ValueError("decode expects exactly one token per batch row")
        self._validate_state(state, hidden_states)
        q, k, v, beta = self._features(hidden_states, state=state)
        o = sliding_window_path_decode(q, k, v, beta, state.path, scale=self.scale)
        return self._output(o, hidden_states)
