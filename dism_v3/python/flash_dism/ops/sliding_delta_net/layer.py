"""FLA-style parameterization of the gate-free sliding-window DeltaNet.

Requires PyTorch, Triton, and flash-linear-attention (``fla``). The projections,
short convolutions, head-wise RMSNorm and SiLU output gate follow FLA's
``fla.layers.gated_deltanet.GatedDeltaNet``. There is no decay projection,
``A_log``, or ``dt_bias``. ``use_gate`` controls only the OUTPUT gate.

Unlike FLA's Transformers layer interface, forward returns a tensor and uses
an explicit single-layer decode state, not a Transformers Cache:

    layer = SlidingWindowDeltaNet(hidden_size=512, num_heads=4, head_dim=64,
                                 window_size=128).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        y = layer(x)                         # [B,T,512], differentiable
        packed_y = layer(packed_x, cu_seqlens=cu)  # B=1, independent documents
        state = layer.create_decode_state(batch_size=x.shape[0])
        token_y = layer.decode(x[:, :1], state)   # no grad; updates state

Q/K/V are BF16 at the operator boundary; beta and the recurrent key cache
are FP32. FP32 parameters work with or without BF16 autocast; a BF16 module
also works with BF16 inputs. The existing kernel's TF32x3 inverse is retained.
Default query scale is head_dim**-0.5, as in FLA. Default write_beta=False
preserves the original sliding-window formula (writes v, not beta*v).

Packed short convolutions reset at every document boundary, including empty
documents. The convolution extends the raw-input receptive field by up to
conv_size-1 tokens beyond the attention window. Packed metadata validation
has the same synchronization / CUDA Graph constraints as the core operator.
Decoding handles one token per batch row; prompt tokens can be fed in order.
It is forward-only and state is mutated on the current stream. No automatic
training/eval dispatch or implicit cache allocation happens in forward.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.modules.conv.causal_conv1d import causal_conv1d
from fla.modules.l2norm import l2norm

from .decode import (
    SlidingDeltaDecodeState,
    create_decode_state,
    sliding_window_delta_net_decode,
)
from .triton import sliding_window_delta_net
from .varlen import prepare_varlen_metadata


@dataclass(frozen=True)
class SlidingWindowDeltaNetState:
    """One layer's KV ring and optional Q/K/V short-convolution histories.

    conv_state entries have shape [B, projection_width, conv_size] and BF16
    dtype, following FLA's ShortConvolution convention. State is not a model
    buffer; allocate it after moving the model to its target device.
    """

    delta: SlidingDeltaDecodeState
    conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None

    @torch.no_grad()
    def reset_(self):
        self.delta.reset_()
        if self.conv_state is not None:
            for tensor in self.conv_state:
                tensor.zero_()
        return self


class SlidingWindowDeltaNet(nn.Module):
    """Sliding attention with FLA GatedDeltaNet's feature parameterization.

    num_v_heads may be an integer multiple of num_heads (grouped value
    attention); Q/K are shared within each group of value heads. expand_v
    determines head_v_dim=head_dim*expand_v, which must be integral.
    allow_neg_eigval changes beta=sigmoid(b_proj(x)) to 2*sigmoid(...).
    All linear projections are bias-free, matching FLA. Output RMSNorm has
    one learned vector shared across heads. use_short_conv=False still uses
    SiLU on all three projected features, matching FLA.
    """

    def __init__(
        self, hidden_size=2048, expand_v=2, head_dim=256, num_heads=6,
        num_v_heads=None, use_gate=True, use_short_conv=True,
        allow_neg_eigval=False, conv_size=4, conv_bias=False,
        layer_idx=None, norm_eps=1e-5, *, window_size=128, chunk_size=32,
        write_beta=False, scale=None, matmul_precision="bf16",
    ):
        super().__init__()
        num_v_heads = num_heads if num_v_heads is None else num_v_heads
        for name, value in (("hidden_size", hidden_size), ("head_dim", head_dim),
                            ("num_heads", num_heads), ("num_v_heads", num_v_heads),
                            ("window_size", window_size), ("conv_size", conv_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(expand_v) or expand_v <= 0:
            raise ValueError("expand_v must be finite and positive")
        head_v_dim = int(head_dim * expand_v)
        if head_v_dim < 1 or not math.isclose(head_v_dim, head_dim * expand_v):
            raise ValueError("head_dim * expand_v must be a positive integer")
        if num_v_heads % num_heads:
            raise ValueError("num_v_heads must be an integer multiple of num_heads")
        if chunk_size not in (16, 32, 64):
            raise ValueError("chunk_size must be 16, 32, or 64")
        if matmul_precision not in ("bf16", "tf32x3"):
            raise ValueError("matmul_precision must be 'bf16' or 'tf32x3'")
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        if scale is not None and not math.isfinite(scale):
            raise ValueError("scale must be finite")
        for name, flag in (("use_gate", use_gate), ("use_short_conv", use_short_conv),
                           ("allow_neg_eigval", allow_neg_eigval),
                           ("conv_bias", conv_bias), ("write_beta", write_beta)):
            if not isinstance(flag, bool):
                raise TypeError(f"{name} must be bool")

        self.hidden_size, self.expand_v = hidden_size, expand_v
        self.head_dim = self.head_k_dim = head_dim
        self.head_v_dim = head_v_dim
        self.num_heads, self.num_v_heads = num_heads, num_v_heads
        self.key_dim = num_heads * head_dim
        self.value_dim = num_v_heads * head_v_dim
        self.use_gate, self.use_short_conv = use_gate, use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size, self.conv_bias = conv_size, conv_bias
        self.layer_idx, self.norm_eps = layer_idx, norm_eps
        self.window_size, self.chunk_size = window_size, chunk_size
        self.write_beta, self.matmul_precision = write_beta, matmul_precision
        self.scale = head_dim ** -0.5 if scale is None else float(scale)

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_v_heads, bias=False)
        if use_short_conv:
            for name, width in (("q", self.key_dim), ("k", self.key_dim),
                                ("v", self.value_dim)):
                setattr(self, name + "_conv1d", ShortConvolution(
                    width, conv_size, bias=conv_bias, activation="silu", backend="triton"))
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(head_v_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _validate_input(self, x):
        if x.ndim != 3 or x.shape[-1] != self.hidden_size or min(x.shape[:2]) < 1:
            raise ValueError("hidden_states must be nonempty [B,T,hidden_size]")
        if x.device.type != "cuda" or x.device != self.q_proj.weight.device:
            raise ValueError("hidden_states and the module must be on the same CUDA device")
        if x.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("hidden_states must be FP32 or BF16")

    def _features(self, x, cu_seqlens=None, conv_chunk_indices=None, state=None):
        features = []
        for index, name in enumerate(("q", "k", "v")):
            z = getattr(self, name + "_proj")(x).to(torch.bfloat16)
            if self.use_short_conv:
                conv = getattr(self, name + "_conv1d")
                if state is None:
                    # Use the differentiable full-sequence entry even for T=1.
                    # ShortConvolution.forward dispatches T=1 to an in-place
                    # inference update and can misclassify packed empty docs.
                    z, _ = causal_conv1d(
                        z, conv.weight.squeeze(1), conv.bias,
                        activation="silu", backend="triton", cu_seqlens=cu_seqlens,
                        chunk_indices=conv_chunk_indices)
                else:
                    z, _ = conv.step(z, residual=None, cache=state.conv_state[index])
            else:
                z = F.silu(z)
            features.append(z)
        b, t = x.shape[:2]
        q, k = (l2norm(z.reshape(b, t, self.num_heads, self.head_k_dim))
                for z in features[:2])
        repeats = self.num_v_heads // self.num_heads
        if repeats != 1:
            q, k = (z.repeat_interleave(repeats, dim=2) for z in (q, k))
        v = features[2].reshape(b, t, self.num_v_heads, self.head_v_dim)
        beta = self.b_proj(x).float().sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2
        return q, k, v, beta.transpose(1, 2).contiguous()

    def _output(self, o, x):
        if self.use_gate:
            g = self.g_proj(x).to(o.dtype).reshape_as(o)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        # Linear follows the parameter dtype without autocast, and the active
        # autocast dtype under AMP. This also permits ordinary FP32 modules.
        return self.o_proj(o.flatten(2).to(self.o_proj.weight.dtype))

    def forward(self, hidden_states, *, cu_seqlens=None):
        """Return [B,T,hidden_size]; packed input uses B=1 and integer offsets.

        Both short convolution and sliding attention obey the same document
        boundaries. No state is read or updated by this differentiable path.
        """
        self._validate_input(hidden_states)
        metadata = prepare_varlen_metadata(
            cu_seqlens, batch=hidden_states.shape[0], total_tokens=hidden_states.shape[1],
            chunk_size=self.chunk_size, device=hidden_states.device)
        # Give the convolution an owned, validated GPU snapshot as well. This
        # protects its pending backward against changes to the caller's offsets.
        cu = None if metadata is None else metadata.cu_seqlens
        conv_indices = None
        if cu is not None and self.use_short_conv:
            # FLA's convolution uses 64-token chunks. Supply our metadata:
            # its default prepare_chunk_indices renumbers nonempty documents,
            # producing wrong document IDs when cu contains repeated offsets.
            conv_metadata = prepare_varlen_metadata(
                cu_seqlens, batch=1, total_tokens=hidden_states.shape[1],
                chunk_size=64, device=hidden_states.device)
            conv_indices = conv_metadata.chunk_indices
        q, k, v, beta = self._features(hidden_states, cu, conv_indices)
        o = sliding_window_delta_net(
            q, k, v, beta, self.window_size, chunk_size=self.chunk_size,
            scale=self.scale, write_beta=self.write_beta,
            matmul_precision=self.matmul_precision, cu_seqlens=cu_seqlens)
        return self._output(o, hidden_states)

    def create_decode_state(self, batch_size):
        """Allocate empty ring/conv histories on this layer's device."""
        device = self.q_proj.weight.device
        delta = create_decode_state(batch_size, self.num_v_heads, self.head_k_dim,
                                    self.head_v_dim, self.window_size,
                                    device=device, write_beta=self.write_beta)
        conv_state = None
        if self.use_short_conv:
            conv_state = tuple(torch.zeros(batch_size, width, self.conv_size,
                                           device=device, dtype=torch.bfloat16)
                               for width in (self.key_dim, self.key_dim, self.value_dim))
        return SlidingWindowDeltaNetState(delta, conv_state)

    def _validate_state(self, state, x):
        # Validate all buffers before mutating any convolution history.
        if not isinstance(state, SlidingWindowDeltaNetState):
            raise TypeError("state must be a SlidingWindowDeltaNetState")
        delta = state.delta
        if not isinstance(delta, SlidingDeltaDecodeState) or delta.write_beta != self.write_beta:
            raise ValueError("state has an incompatible write convention")
        b, h, w = x.shape[0], self.num_v_heads, self.window_size
        buffers = [(delta.keys, (b, h, w, self.head_k_dim), torch.float32),
                   (delta.values, (b, h, w, self.head_v_dim),
                    torch.float32 if self.write_beta else torch.bfloat16),
                   (delta.cursor, (b, h), torch.int32)]
        if self.use_short_conv:
            if not isinstance(state.conv_state, tuple) or len(state.conv_state) != 3:
                raise ValueError("state must contain three convolution histories")
            buffers.extend((z, (b, width, self.conv_size), torch.bfloat16)
                           for z, width in zip(state.conv_state,
                                               (self.key_dim, self.key_dim, self.value_dim)))
        elif state.conv_state is not None:
            raise ValueError("state contains convolution histories but short convolution is disabled")
        for z, shape, dtype in buffers:
            if (not isinstance(z, torch.Tensor) or z.shape != shape or z.dtype != dtype
                    or z.device != x.device or not z.is_contiguous() or z.requires_grad):
                raise ValueError("state buffer shape/dtype/device/layout does not match the layer")

    @torch.no_grad()
    def decode(self, hidden_states, state):
        """Process [B,1,hidden_size], updating ring and conv histories in place.

        Returns a tensor, without gradients. Call state.reset_() to begin new
        sequences. For packed training followed by decoding, build one state
        row per document and feed its prefix tokens separately.
        """
        self._validate_input(hidden_states)
        if hidden_states.shape[1] != 1:
            raise ValueError("decode expects exactly one token per batch row")
        self._validate_state(state, hidden_states)
        q, k, v, beta = self._features(hidden_states, state=state)
        o = sliding_window_delta_net_decode(q, k, v, beta, state.delta, scale=self.scale)
        return self._output(o, hidden_states)
