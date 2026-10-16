"""Trainable DISM v3 attention; residual addition and PreNorm belong to the caller."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from .kernels.conv1d import CausalShortConv1d
from .kernels.random_bool import triton_rand_bool
from .kernels.value_residual import value_residual_fused, value_residual_supported


def value_residual_mix_reference(v, v_first, gate):
    """Eager reference: v_first + (v - v_first) * sigmoid(gate), per head.

    v, v_first: [B, N, H, DV]; gate: [B, N, H].

    Kept as the numerical and performance reference for the fused kernel; it is
    also the fallback for every layout/dtype the fused operator does not cover.
    """
    g = torch.sigmoid(gate.float()).unsqueeze(-1)
    mixed = v_first.float() + (v.float() - v_first.float()) * g
    return mixed.to(v.dtype)


def value_residual_mix(v, v_first, gate):
    """Single common helper for the value-residual mix.

    Dispatches to the fused Triton operator when the packed CUDA training
    layout is supported, and to :func:`value_residual_mix_reference` otherwise.
    Set ``DISM_VALUE_RESIDUAL_KERNEL=reference`` to force the eager path.
    """
    if value_residual_supported(v, v_first, gate):
        return value_residual_fused(v, v_first, gate)
    return value_residual_mix_reference(v, v_first, gate)


class TransVQMap(nn.Module):
    """TransVQ codebook map P_phi: a shared, dense-Jacobian transform of the codebook.

    The base codebook is a frozen reference; only the map is trained, so every
    codeword receives gradient through the map's dense Jacobian and the codebook
    does not collapse. ``kind`` selects the transform:
      - "linear_attn": one linear-attention layer over the V codewords.
      - "rank": C' = C + U (V^T C) with U,V in R^{V x rank}; low-rank dense
                cross-codeword mixing shared across heads.
      - "diagonal": C' = C * s (per-dimension scale; NO cross-codeword coupling,
                a negative control for the kernel hypothesis).
      - "frozen_random": as "rank" but U,V are frozen random; only s is learned.
    An output RMSNorm with elementwise scaling sets the exit scale.
    """

    def __init__(self, dim, vocab, mlp_ratio=2.0, dropout=0.0, lite=False,
                 kind="linear_attn", rank=4):
        super().__init__()
        self.kind = str(kind)
        self.rank = int(rank)
        self.out_norm = nn.RMSNorm(dim)
        if self.kind == "linear_attn":
            self.lite = bool(lite)
            self.norm1 = nn.LayerNorm(dim)
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.q_norm = nn.RMSNorm(dim)
            self.k_norm = nn.RMSNorm(dim)
            if not self.lite:
                self.v_proj = nn.Linear(dim, dim, bias=False)
                self.out_proj = nn.Linear(dim, dim, bias=False)
                self.norm2 = nn.LayerNorm(dim)
                hidden = max(1, int(dim * mlp_ratio))
                self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        elif self.kind in ("rank", "frozen_random"):
            self.u = nn.Parameter(torch.randn(vocab, self.rank) * 0.02)
            self.v = nn.Parameter(torch.randn(vocab, self.rank) * 0.02)
            self.s = nn.Parameter(torch.ones(self.rank))
            if self.kind == "frozen_random":
                self.u.requires_grad_(False)
                self.v.requires_grad_(False)
        elif self.kind == "mean_shift":
            self.m = nn.Parameter(torch.zeros(dim))
        elif self.kind == "diagonal":
            self.s = nn.Parameter(torch.ones(dim))
        else:
            raise ValueError(f"unknown vocab_transvq_map kind {kind!r}")

    def forward(self, codebook):
        # codebook: [heads, V, dim]
        if self.kind == "linear_attn":
            h = self.norm1(codebook)
            q = self.q_norm(F.elu(self.q_proj(h)) + 1.0)
            k = self.k_norm(F.elu(self.k_proj(h)) + 1.0)
            v = h if self.lite else self.v_proj(h)
            kv = torch.einsum("hvd,hve->hde", k, v)
            normalizer = k.sum(dim=1)
            numerator = torch.einsum("hvd,hde->hve", q, kv)
            denominator = 1.0 + torch.einsum("hvd,hd->hv", q, normalizer).unsqueeze(-1)
            attn = numerator / denominator
            if not self.lite:
                attn = self.out_proj(attn)
            x = codebook + attn
            if not self.lite:
                x = x + self.mlp(self.norm2(x))
            return self.out_norm(x)
        if self.kind in ("rank", "frozen_random"):
            projected = torch.einsum("vr,hvd->hrd", self.v, codebook)
            mixed = torch.einsum("vr,hrd->hvd", self.u * self.s, projected)
            return self.out_norm(codebook + mixed)
        if self.kind == "mean_shift":
            return self.out_norm(codebook + self.m)
        return self.out_norm(codebook * self.s)


def _salt_hard_seed(seed, layer_idx):
    """XOR with a SplitMix64 layer salt; tensor seeds stay on their device."""
    if layer_idx is None:
        return seed
    mask = (1 << 64) - 1
    salt = (layer_idx + 0x9E3779B97F4A7C15) & mask
    salt = ((salt ^ (salt >> 30)) * 0xBF58476D1CE4E5B9) & mask
    salt = ((salt ^ (salt >> 27)) * 0x94D049BB133111EB) & mask
    salt ^= salt >> 31
    salt = salt if salt < (1 << 63) else salt - (1 << 64)
    if isinstance(seed, torch.Tensor):
        if seed.ndim != 0 or seed.dtype not in (torch.int32, torch.int64):
            raise ValueError("hard_seed must be a scalar integer tensor")
        return seed.to(torch.int64).bitwise_xor(salt)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("hard_seed must be an integer or a scalar integer tensor")
    if not -(1 << 63) <= seed < (1 << 63):
        raise ValueError("hard_seed must fit in int64")
    return seed ^ salt


class DismAttention(nn.Module):
    """v2-style projections/shortconv/gate with v3 signed soft readout.

    Inputs/outputs are [B,N,width]. Use CUDA BF16 autocast with FP32 master
    parameters. N and all packed document boundaries must be multiples of 256.
    Random direction is one global choice per call. Hard decisions are per row.
    A separate Triton sampler generates flags; RNG is not fused into DISM.
    """
    _decode_extra_states = 0

    def __init__(self, width, heads, *, head_dim=64, value_dim=64,
                 readout_dim=32, vocab_size=512, conv_size=4, layer_idx=None,
                 rope_theta=10000.0, qknorm_eps=1e-6, readout_l2_norm=False, soft_k_l2_norm=False,
                 value_residual=False, vocab_temperature=1.0,
                 vocab_transvq=False, vocab_transvq_lite=False,
                 vocab_transvq_map="linear_attn", vocab_transvq_rank=4, gqa_query_channels: int = 1,
                 kv_group_size: int = 1, hard_tau_grad_scale: float = 1.0):
        super().__init__()
        if min(width, heads, vocab_size, conv_size) <= 0:
            raise ValueError("width, heads, vocab_size and conv_size must be positive")
        if head_dim not in (32, 64) or value_dim not in (32, 64) or readout_dim not in (16, 32):
            raise ValueError("supported dimensions: D/DV=32/64, R=16/32")
        from fla.modules import FusedRMSNormGated
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(qknorm_eps) or qknorm_eps <= 0:
            raise ValueError("qknorm_eps must be finite and positive")
        self.rope_theta = float(rope_theta)
        self.qknorm_eps = float(qknorm_eps)
        self.readout_l2_norm = bool(readout_l2_norm)
        self.soft_k_l2_norm = bool(soft_k_l2_norm)
        if not math.isfinite(vocab_temperature) or vocab_temperature <= 0:
            raise ValueError("vocab_temperature must be finite and positive")
        # Host schedule value, not a parameter or device buffer.
        self.vocab_temperature = float(vocab_temperature)

        if type(gqa_query_channels) is not int or gqa_query_channels not in (1, 2):
            raise ValueError("gqa_query_channels must be 1 or 2")
        self.gqa_query_channels = gqa_query_channels
        if type(kv_group_size) is not int or kv_group_size not in (1, 2):
            raise ValueError("kv_group_size must be 1 or 2")
        if kv_group_size > 1 and gqa_query_channels > 1:
            # Two different ways to share one kv across two queries; not composable.
            raise ValueError("kv_group_size and gqa_query_channels are mutually exclusive")
        if heads % kv_group_size:
            raise ValueError("heads must be divisible by kv_group_size")
        self.kv_group_size = kv_group_size
        if not (0.0 <= float(hard_tau_grad_scale) <= 1.0):
            raise ValueError("hard_tau_grad_scale must be in [0,1]")
        # 1.0 keeps rtau trained by both paths (unchanged); 0.0 detaches it on
        # hard rows so only the soft relaxation sets it.
        self.hard_tau_grad_scale = float(hard_tau_grad_scale)
        self.width, self.heads = width, heads
        # kv_group_size=2 gives heads//2 key/value heads, each queried by
        # kv_group_size distinct query heads (classic GQA head grouping).
        self.kv_heads = heads // kv_group_size
        if layer_idx is not None and (isinstance(layer_idx, bool) or not isinstance(layer_idx, int)
                                      or not 0 <= layer_idx < (1 << 63)):
            raise ValueError("layer_idx must be a nonnegative int64 integer or None")
        self.layer_idx = layer_idx
        self.head_dim, self.value_dim, self.readout_dim = head_dim, value_dim, readout_dim
        self.q_conv = CausalShortConv1d(width, heads * head_dim, conv_size)
        self.k_conv = CausalShortConv1d(width, self.kv_heads * head_dim, conv_size)
        self.v_conv = CausalShortConv1d(width, self.kv_heads * value_dim, conv_size)
        self.sq_proj = nn.Linear(width, heads * readout_dim, bias=False)
        self.sk_proj = nn.Linear(width, self.kv_heads * readout_dim, bias=False)
        # Store effective codewords directly: SiLU shapes only initialization,
        # not the trainable parameter space or its backward Jacobian.
        self.q_vocab = nn.Parameter(F.silu(torch.randn(heads, vocab_size, head_dim, dtype=torch.float32)))
        self.k_vocab = nn.Parameter(F.silu(torch.randn(self.kv_heads, vocab_size, head_dim, dtype=torch.float32)))
        self.log_sel_tau = nn.Parameter(torch.empty(heads).uniform_(-4, 4))
        for parameter in (self.q_vocab, self.k_vocab, self.log_sel_tau):
            parameter._no_weight_decay = True
        self.g_proj_down = nn.Linear(width, max(1, width // 8))
        self.g_proj_up = nn.Linear(max(1, width // 8), heads * value_dim)
        self.norm = FusedRMSNormGated(value_dim, eps=1e-5)
        self.o_proj = nn.Linear(heads * value_dim, width)
        self.value_residual = bool(value_residual)
        if self.value_residual:
            # Per-head, data-dependent gate: v_first + (v - v_first) * sigmoid(g).
            self.v_residual_gate = nn.Linear(width, heads)
            self.v_residual_gate._value_residual_gate = True
        self.vocab_transvq = bool(vocab_transvq)
        if self.vocab_transvq:
            # C' = P_phi(C): train only the map, keep the base codebook frozen.
            self.q_vocab_map = TransVQMap(head_dim, vocab_size, lite=vocab_transvq_lite,
                                          kind=vocab_transvq_map, rank=vocab_transvq_rank)
            self.k_vocab_map = TransVQMap(head_dim, vocab_size, lite=vocab_transvq_lite,
                                          kind=vocab_transvq_map, rank=vocab_transvq_rank)
            self.q_vocab.requires_grad_(False)
            self.k_vocab.requires_grad_(False)

        if self.gqa_query_channels == 2:
            self.gqa_q_conv = CausalShortConv1d(width, heads * head_dim, conv_size)
            self.gqa_sq_proj = nn.Linear(width, heads * readout_dim, bias=False)
            self.gqa_mix = nn.Parameter(torch.full((heads, 2), 0.5))

    @staticmethod
    def _expand_gqa(tensor, head_axis, factor=2):
        """Duplicate each shared head in h*factor+g order; backward sums over g."""
        shape = list(tensor.shape)
        expanded = shape[:head_axis + 1] + [factor] + shape[head_axis + 1:]
        shape[head_axis] *= factor
        return tensor.unsqueeze(head_axis + 1).expand(expanded).reshape(shape).contiguous()

    def _mix_gqa(self, output):
        # The core has already normalized each channel independently.
        batch, length = output.shape[:2]
        return (output.reshape(batch, length, self.heads, 2, self.value_dim)
                * self.gqa_mix.view(1, 1, self.heads, 2, 1)).sum(3)

    def _rotary_tables(self, length, device, *, offset=0, dtype=torch.float32, dimension=None):
        """Split-half RoPE tables for the hybrid SWA branch."""
        dimension = self.readout_dim if dimension is None else dimension
        frequencies = self.rope_theta ** (-torch.arange(0, dimension, 2,
                                                        device=device, dtype=dtype) / dimension)
        positions = torch.arange(offset, offset + length, device=device, dtype=dtype)
        phase = positions[:, None] * frequencies[None, :]
        return phase.cos(), phase.sin()

    def _combine_cuda(self, output, x, v, cu_seqlens, max_seqlen, v_first=None,
                      linear_q=None, linear_k=None, conv_seq_idx=None):
        return output

    def _activate_readout(self, projected, *, is_key=False, heads=None):
        features = F.silu(projected).unflatten(-1, (self.heads if heads is None else heads,
                                                    self.readout_dim))
        if self.readout_l2_norm or (is_key and self.soft_k_l2_norm):
            # FP32 norm/reduction for BF16 training; unit L2 per head, not RMS.
            dtype = features.dtype
            work = features.float() if dtype in (torch.float16, torch.bfloat16) else features
            features = F.normalize(work, p=2, dim=-1, eps=1e-6).to(dtype)
        return features.contiguous()

    def _combine_torch(self, output, x, cache, previous_extra):
        return output, ()

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        """FLA interface: returns (output, None, past_key_values).

        DISM options (hard_prob, hard, hard_seed, direction, generator, cu_seqlens,
        max_seqlen) are keyword arguments. Set layer_idx when supplying a FLA
        Cache. Cached inference uses Torch, requires eval(), and accepts any
        positive token count. This layer does not construct a Cache implicitly.
        """
        from fla.layers.utils import get_layer_cache, update_layer_cache
        x = hidden_states
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError("hidden_states must have shape [B,N,width]")
        last_state = get_layer_cache(self, past_key_values)
        options = {name: kwargs.get(name) for name in
                   ('hard_prob', 'direction', 'hard', 'hard_seed', 'generator', 'vocab_temperature')}
        v_first = kwargs.get('v_first')
        probability = options['hard_prob']
        probability = (0.0 if self.training else 1.0) if probability is None else float(probability)
        if options['hard'] is None and options['hard_seed'] is not None and probability not in (0.0, 1.0):
            options['hard_seed'] = _salt_hard_seed(options['hard_seed'], self.layer_idx)
        cu_seqlens, max_seqlen = kwargs.get('cu_seqlens'), kwargs.get('max_seqlen')
        cached = bool(use_cache) or last_state is not None
        if self.training and cached:
            raise ValueError("cached DISM inference requires eval()")
        torch_path = cached or (not self.training and (x.device.type != 'cuda' or x.shape[1] % 256 != 0))
        if attention_mask is not None:
            if (attention_mask.ndim != 2 or attention_mask.shape[0] != x.shape[0]
                    or attention_mask.shape[1] < x.shape[1]):
                raise ValueError("attention_mask must be [B,T] with T >= new token count")
            if cu_seqlens is not None:
                raise ValueError("provide attention_mask or cu_seqlens, not both")
            mask = attention_mask[:, -x.shape[1]:].to(device=x.device)
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("attention_mask must contain only 0/1")
        else:
            mask = None
        if self.gqa_query_channels > 1 and (torch_path or past_key_values is not None):
            raise NotImplementedError("GQA-DISM G>1 supports only uncached aligned CUDA execution")
        if self.kv_group_size > 1 and (torch_path or past_key_values is not None):
            raise NotImplementedError("shared-KV DISM supports only uncached aligned CUDA execution")
        if torch_path:
            if v_first is not None:
                raise NotImplementedError("value residual is only implemented for packed CUDA training/eval")
            if cu_seqlens is not None or (mask is not None and not bool(mask.all())):
                raise NotImplementedError("Torch cached decoding currently requires equal-length unpadded batches")
            from .decoding import forward_torch
            output, recurrent, conv = forward_torch(self, x, last_state, **options)
            if use_cache:
                update_layer_cache(self, past_key_values, recurrent_state=recurrent,
                                   conv_state=conv, offset=x.shape[1])
            return output, None, past_key_values

        indices = None
        batch, length, _ = x.shape
        if mask is not None:
            from fla.layers.utils import get_unpad_data, index_first_axis
            indices, cu_seqlens, max_seqlen = get_unpad_data(mask)
            x = index_first_axis(x.reshape(-1, self.width), indices).unsqueeze(0)
            hard = options['hard']
            if hard is not None:
                if hard.shape != (batch, self.heads):
                    raise ValueError("CUDA hard must be bool [B,H], not a token mask")
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                if not bool((lengths == lengths[0]).all()):
                    raise NotImplementedError("compact hard rows require equal unpadded training-row lengths")
                options['hard'] = hard.contiguous()
            direction = options['direction']
            if direction is not None:
                if direction.shape != (batch, self.heads) or not torch.equal(direction, direction[:1].expand_as(direction)):
                    raise ValueError("packed attention_mask path requires the same direction across batch")
                options['direction'] = direction[:1].contiguous()
        output = self._forward_cuda(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                                    v_first=v_first, conv_seq_idx=kwargs.get('conv_seq_idx'),
                                    share_branch_boundaries=kwargs.get('_share_branch_boundaries', False),
                                    direction_true=kwargs.get('direction_true', False), **options)
        if indices is not None:
            from fla.layers.utils import pad_input
            output = pad_input(output.squeeze(0), indices, batch, length)
        return output, None, past_key_values

    def _forward_cuda(self, x, hard_prob=None, *, direction=None, hard=None,
                hard_seed=None, generator=None, cu_seqlens=None, max_seqlen=None,
                v_first=None, vocab_temperature=None, conv_seq_idx=None, direction_true=False,
                share_branch_boundaries=False):
        """Default: soft training, hard evaluation; explicit flags override sampling.

        direction: bool [B,H] (True selects k-to-qemb, the query-LSE path).
        direction_true: explicit fixed-True CUDA specialization, overriding the
        direction tensor; not an inference from its device contents.
        hard: bool [training_rows,H]; soft uses CUDA forward/backward, hard uses
        integer CUDA forward and asynchronous event backward. Pass both flags for replay. A supplied
        generator must target the same device as x. No stateful annealing here.
        cu_seqlens: int32 packed boundaries, with batch=1 and 256 alignment.
        max_seqlen: optional upper bound for fused convolution launch sizing.
        conv_seq_idx: optional precomputed int32 [B,N] convolution document IDs,
        matching cu_seqlens; consumed only by frontends with supports_seq_idx.
        share_branch_boundaries: internal adapter contract: boundaries are an
        owned, immutable per-forward snapshot shared across layers, never a
        dataloader buffer that can be overwritten between forwards.
        """
        from . import voc_dism, VarlenLayout

        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError("x must have shape [B,N,width]")
        if not x.is_cuda:
            raise ValueError("DismAttention requires CUDA inputs")
        batch, length, _ = x.shape
        if length <= 0 or length % 256:
            raise ValueError("module sequence length must be positive and 256-aligned")
        # Validate packed boundaries before convolution; reuse the native layout.
        layout = None
        compiled_layout = None
        # Auxiliary branches share the caller's boundary identity across layers.
        # prepare_pack returns a functional clone for DISM; passing that clone
        # to FLA defeats its identity-keyed chunk-index cache at every layer.
        branch_cu_seqlens = cu_seqlens
        compiling = torch.compiler.is_compiling()
        if cu_seqlens is not None:
            if batch != 1:
                raise ValueError("packed execution requires batch=1")
            if compiling:
                from .compiler import prepare_pack
                # A total-token upper bound is safe and independent of the
                # number/lengths of documents; explicit bounds avoid overlaunch.
                max_seqlen = length if max_seqlen is None else max_seqlen
                cu_seqlens, compiled_layout = prepare_pack(cu_seqlens, length, max_seqlen)
            else:
                layout = VarlenLayout.from_cu_seqlens(cu_seqlens, length)
                longest = max(layout.lengths, default=0)
                if max_seqlen is None:
                    max_seqlen = longest
                elif isinstance(max_seqlen, bool) or not isinstance(max_seqlen, int) or max_seqlen < longest:
                    raise ValueError("max_seqlen must be an integer >= longest document")
            cu_seqlens = cu_seqlens.to(device=x.device).contiguous()
            branch_cu_seqlens = branch_cu_seqlens.to(device=x.device).contiguous()
        elif max_seqlen is not None:
            raise ValueError("max_seqlen requires cu_seqlens")
        if hard is None:
            probability = (0.0 if self.training else 1.0) if hard_prob is None else float(hard_prob)
            if not 0.0 <= probability <= 1.0:
                raise ValueError("hard_prob must be in [0,1]")
            if probability in (0.0, 1.0):
                hard = torch.full((batch, self.heads), bool(probability), device=x.device)
            else:
                seed = hard_seed
                if seed is None:
                    seed = torch.randint(0, torch.iinfo(torch.int64).max, (),
                                         device=x.device, dtype=torch.int64, generator=generator)
                hard = triton_rand_bool((batch, self.heads), probability,
                                       device=x.device, seed=seed)
        if direction is None:
            direction = torch.randint(2, (), device=x.device, generator=generator,
                                      dtype=torch.int32).bool().expand(batch, self.heads).contiguous()

        def project(convolution, channels, linear=None, heads=None):
            conv_options = ({'seq_idx': conv_seq_idx}
                            if getattr(convolution, 'supports_seq_idx', False) else {})
            if linear is not None:
                values = convolution.depthwise_silu(linear, cu_seqlens=cu_seqlens, **conv_options)
            elif compiling and not getattr(convolution, "bypass_compiled_conv", False):
                from .compiler import conv_forward
                dtype = torch.bfloat16 if torch.is_autocast_enabled('cuda') else x.dtype
                values = conv_forward(x.to(dtype), convolution.weight.to(dtype),
                                      convolution.conv_weight.to(dtype), cu_seqlens, max_seqlen)
            else:
                values = convolution(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **conv_options)
            return values.reshape(batch, length, self.heads if heads is None else heads,
                                  channels).contiguous()

        # torch Linear + FLA conv: the DISM and GDN branches share the q/k input
        # projections when the linear weights are aliased, so compute the tied
        # projection once and apply each branch's independent depthwise conv on
        # top. kv_group_size>1 breaks the k tie (different output widths), so the
        # two branches are decided independently rather than off one flag.
        tied = bool(getattr(self, "share_qk_linear", False))
        gdn_q_conv = getattr(self, "gdn_q_conv", None)
        gdn_k_conv = getattr(self, "gdn_k_conv", None)
        share_q = tied and gdn_q_conv is not None and self.q_conv.weight is gdn_q_conv.weight
        share_k = tied and gdn_k_conv is not None and self.k_conv.weight is gdn_k_conv.weight
        linear_q = self.q_conv.project_linear(x) if share_q else None
        linear_k = self.k_conv.project_linear(x) if share_k else None
        q = project(self.q_conv, self.head_dim, linear_q)
        k = project(self.k_conv, self.head_dim, linear_k, heads=self.kv_heads)
        v = project(self.v_conv, self.value_dim, heads=self.kv_heads)
        if self.kv_group_size > 1:
            # One shared key side queried by kv_group_size distinct query heads.
            k, v = (self._expand_gqa(t, 2, self.kv_group_size) for t in (k, v))
        if self.value_residual and v_first is not None:
            v = value_residual_mix(v, v_first, self.v_residual_gate(x))
        if q.dtype != torch.bfloat16:
            raise ValueError("use CUDA BF16 autocast; the production kernels require BF16")
        sq = self._activate_readout(self.sq_proj(x))
        sk = self._activate_readout(self.sk_proj(x), is_key=True, heads=self.kv_heads)
        if self.kv_group_size > 1:
            sk = self._expand_gqa(sk, 2, self.kv_group_size)
        # No runtime codebook activation; interpolation owns the BF16 cast.
        if self.vocab_transvq:
            with torch.autocast('cuda', enabled=False):
                eq = self.q_vocab_map(self.q_vocab.float())
                ek = self.k_vocab_map(self.k_vocab.float())
        else:
            eq, ek = self.q_vocab.float(), self.k_vocab.float()
        if self.kv_group_size > 1:
            # The shared key anchors one symbol set for the whole group; the
            # query side keeps an independent codebook per query head.
            ek = self._expand_gqa(ek, 0, self.kv_group_size)
        tau = F.softplus(self.log_sel_tau.float()).contiguous()
        if self.hard_tau_grad_scale != 1.0 and hard is not None:
            # Per (row, head) hard flags; rtau is one scalar per head, so the
            # flags must agree across rows for this to be well defined.
            row = hard if hard.ndim == 1 else hard[0]
            if hard.ndim > 1 and not torch.equal(hard, row.expand_as(hard)):
                raise ValueError("hard_tau_grad_scale requires uniform hard flags across rows")
            row = row.to(device=tau.device)
            if bool(row.any()):
                scaled = tau * self.hard_tau_grad_scale + tau.detach() * (1.0 - self.hard_tau_grad_scale)
                tau = torch.where(row, scaled, tau)
        dism_v = v
        if self.gqa_query_channels == 2:
            q_extra = project(self.gqa_q_conv, self.head_dim)
            sq_extra = self._activate_readout(self.gqa_sq_proj(x))
            q = torch.cat((q.unsqueeze(3), q_extra.unsqueeze(3)), dim=3).flatten(2, 3)
            sq = torch.cat((sq.unsqueeze(3), sq_extra.unsqueeze(3)), dim=3).flatten(2, 3)
            k, sk, dism_v = (self._expand_gqa(t, 2) for t in (k, sk, v))
            # Optional shared-head vocabularies arrive as [V,D].
            eq = eq.unsqueeze(0).expand(self.heads, -1, -1) if eq.ndim == 2 else eq
            ek = ek.unsqueeze(0).expand(self.heads, -1, -1) if ek.ndim == 2 else ek
            eq, ek, tau = (self._expand_gqa(t, 0) for t in (eq, ek, tau))
            hard, direction = (self._expand_gqa(t, 1) for t in (hard, direction))
        from .reinforce import enabled, sampled_voc
        beta = self.vocab_temperature if vocab_temperature is None else vocab_temperature
        if enabled():
            if self.gqa_query_channels > 1:
                raise NotImplementedError("GQA-DISM G>1 does not support independent categorical K sampling")
            output = sampled_voc(q, k, sq, sk, dism_v, eq, ek, tau, direction=direction,
                                 hard=hard, layout=layout, beta=beta)
        elif compiling or direction_true:
            from .compiler import voc_forward
            if not compiling and layout is not None:
                compiled_layout = layout.table.flatten()
            output = voc_forward(q, k, sq, sk, dism_v, eq, ek, tau, beta, direction, hard,
                                 cu_seqlens, compiled_layout, direction_true=direction_true)[0]
        else:
            output = voc_dism(q, k, sq, sk, dism_v, eq, ek, tau,
                              direction=direction, hard=hard, layout=layout, beta=beta)
        if self.gqa_query_channels == 2:
            output = self._mix_gqa(output)
        output = self._combine_cuda(output, x, v,
                                    branch_cu_seqlens if share_branch_boundaries else cu_seqlens,
                                    max_seqlen, v_first=v_first,
                                    linear_q=linear_q, linear_k=linear_k, conv_seq_idx=conv_seq_idx)
        gate = self.g_proj_up(self.g_proj_down(x)).reshape(batch, length, self.heads, self.value_dim)
        if compiling:
            # Let Inductor fuse this expression. FLA's current custom backward
            # queries Triton device properties from inside the traced graph.
            values = output.float()
            values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.norm.eps)
            values = values * self.norm.weight.float() * F.silu(gate.float())
            output = values.to(output.dtype).flatten(2)
        else:
            output = self.norm(output, gate).flatten(2)
        return self.o_proj(output)
