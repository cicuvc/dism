"""Trainable DISM v3 attention; residual addition and PreNorm belong to the caller."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from .kernels.conv1d import CausalShortConv1d
from .kernels.random_bool import triton_rand_bool


def value_residual_mix(v, v_first, gate):
    """v_first + (v - v_first) * sigmoid(gate), per head.

    v, v_first: [B, N, H, DV]; gate: [B, N, H].
    """
    g = torch.sigmoid(gate.float()).unsqueeze(-1)
    mixed = v_first.float() + (v.float() - v_first.float()) * g
    return mixed.to(v.dtype)


class TransVQMap(nn.Module):
    """TransVQ codebook map P_phi: one linear-attention transformer layer.

    The base codebook is a frozen reference; only this map is trained, so the
    embedding loss updates every transformed codeword and the codebook does not
    collapse (arXiv 2602.18896, \"TransVQ\"). Linear attention keeps the codebook
    interaction bounded, and an output RMSNorm with elementwise scaling bounds the
    magnitude of the transformed codebook.
    """

    def __init__(self, dim, mlp_ratio=2.0, dropout=0.0, lite=False):
        super().__init__()
        self.lite = bool(lite)
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(dim)  # q = norm(silu(linear(h)))
        self.k_norm = nn.RMSNorm(dim)
        if not self.lite:
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)
            self.norm2 = nn.LayerNorm(dim)
            hidden = max(1, int(dim * mlp_ratio))
            self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.out_norm = nn.RMSNorm(dim)  # elementwise learnable scale at the exit

    def forward(self, codebook):
        # codebook: [heads, V, dim]; linear attention runs over V per head.
        # Feature map is elu(x)+1 (non-negative), then an RMSNorm on q and k;
        # non-negative features keep the linear-attention denominator positive.
        h = self.norm1(codebook)
        q = self.q_norm(F.elu(self.q_proj(h)) + 1.0)
        k = self.k_norm(F.elu(self.k_proj(h)) + 1.0)
        v = h if self.lite else self.v_proj(h)
        kv = torch.einsum("hvd,hve->hde", k, v)
        normalizer = k.sum(dim=1)
        numerator = torch.einsum("hvd,hde->hve", q, kv)
        # 1 + sum_j (q_i . k_j) keeps the denominator >= 1; numerators unchanged.
        denominator = 1.0 + torch.einsum("hvd,hd->hv", q, normalizer).unsqueeze(-1)
        attn = numerator / denominator
        if not self.lite:
            attn = self.out_proj(attn)
        x = codebook + attn
        if not self.lite:
            x = x + self.mlp(self.norm2(x))
        return self.out_norm(x)


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
                 value_residual=False, vocab_transvq=False, vocab_transvq_lite=False):
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

        self.width, self.heads = width, heads
        if layer_idx is not None and (isinstance(layer_idx, bool) or not isinstance(layer_idx, int)
                                      or not 0 <= layer_idx < (1 << 63)):
            raise ValueError("layer_idx must be a nonnegative int64 integer or None")
        self.layer_idx = layer_idx
        self.head_dim, self.value_dim, self.readout_dim = head_dim, value_dim, readout_dim
        self.q_conv = CausalShortConv1d(width, heads * head_dim, conv_size)
        self.k_conv = CausalShortConv1d(width, heads * head_dim, conv_size)
        self.v_conv = CausalShortConv1d(width, heads * value_dim, conv_size)
        self.sq_proj = nn.Linear(width, heads * readout_dim, bias=False)
        self.sk_proj = nn.Linear(width, heads * readout_dim, bias=False)
        # Store effective codewords directly: SiLU shapes only initialization,
        # not the trainable parameter space or its backward Jacobian.
        self.q_vocab = nn.Parameter(F.silu(torch.randn(heads, vocab_size, head_dim, dtype=torch.float32)))
        self.k_vocab = nn.Parameter(F.silu(torch.randn(heads, vocab_size, head_dim, dtype=torch.float32)))
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
            self.q_vocab_map = TransVQMap(head_dim, lite=vocab_transvq_lite)
            self.k_vocab_map = TransVQMap(head_dim, lite=vocab_transvq_lite)
            self.q_vocab.requires_grad_(False)
            self.k_vocab.requires_grad_(False)

    def _rotary_tables(self, length, device, *, offset=0, dtype=torch.float32, dimension=None):
        """Split-half RoPE tables for the hybrid SWA branch."""
        dimension = self.readout_dim if dimension is None else dimension
        frequencies = self.rope_theta ** (-torch.arange(0, dimension, 2,
                                                        device=device, dtype=dtype) / dimension)
        positions = torch.arange(offset, offset + length, device=device, dtype=dtype)
        phase = positions[:, None] * frequencies[None, :]
        return phase.cos(), phase.sin()

    def _combine_cuda(self, output, x, v, cu_seqlens, max_seqlen, v_first=None,
                      linear_q=None, linear_k=None):
        return output

    def _activate_readout(self, projected, *, is_key=False):
        features = F.silu(projected).unflatten(-1, (self.heads, self.readout_dim))
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
                   ('hard_prob', 'direction', 'hard', 'hard_seed', 'generator')}
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
                options['hard'] = hard.transpose(1, 2).reshape(-1, self.heads)[indices].T.unsqueeze(0).contiguous()
            direction = options['direction']
            if direction is not None:
                if direction.shape != (batch, self.heads) or not torch.equal(direction, direction[:1].expand_as(direction)):
                    raise ValueError("packed attention_mask path requires the same direction across batch")
                options['direction'] = direction[:1].contiguous()
        output = self._forward_cuda(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, v_first=v_first, **options)
        if indices is not None:
            from fla.layers.utils import pad_input
            output = pad_input(output.squeeze(0), indices, batch, length)
        return output, None, past_key_values

    def _forward_cuda(self, x, hard_prob=None, *, direction=None, hard=None,
                hard_seed=None, generator=None, cu_seqlens=None, max_seqlen=None,
                v_first=None):
        """Default: soft training, hard evaluation; explicit flags override sampling.

        direction: bool [B,H] (True selects k-to-qemb, the query-LSE path).
        hard: bool [B,H,N]. Pass both for reproducible comparisons. A supplied
        generator must target the same device as x. No stateful annealing here.
        cu_seqlens: int32 packed boundaries, with batch=1 and 256 alignment.
        max_seqlen: optional upper bound for fused convolution launch sizing.
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
        elif max_seqlen is not None:
            raise ValueError("max_seqlen requires cu_seqlens")
        if hard is None:
            probability = (0.0 if self.training else 1.0) if hard_prob is None else float(hard_prob)
            if not 0.0 <= probability <= 1.0:
                raise ValueError("hard_prob must be in [0,1]")
            if probability in (0.0, 1.0):
                hard = torch.full((batch, self.heads, length), bool(probability), device=x.device)
            else:
                seed = hard_seed
                if seed is None:
                    seed = torch.randint(0, torch.iinfo(torch.int64).max, (),
                                         device=x.device, dtype=torch.int64, generator=generator)
                hard = triton_rand_bool((batch, self.heads, length), probability,
                                       device=x.device, seed=seed)
        if direction is None:
            direction = torch.randint(2, (), device=x.device, generator=generator,
                                      dtype=torch.int32).bool().expand(batch, self.heads).contiguous()

        def project(convolution, channels, linear=None):
            if linear is not None:
                values = convolution.depthwise_silu(linear, cu_seqlens=cu_seqlens)
            elif compiling and not getattr(convolution, "bypass_compiled_conv", False):
                from .compiler import conv_forward
                dtype = torch.bfloat16 if torch.is_autocast_enabled('cuda') else x.dtype
                values = conv_forward(x.to(dtype), convolution.weight.to(dtype),
                                      convolution.conv_weight.to(dtype), cu_seqlens, max_seqlen)
            else:
                values = convolution(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            return values.reshape(batch, length, self.heads, channels).contiguous()

        # torch Linear + FLA conv: the DISM and GDN branches share the q/k input
        # projection (tied linear weight), so compute it once and apply each
        # branch's independent depthwise conv on top.
        share_qk_linear = (getattr(self, "share_qk_linear", False)
                           and getattr(self, "gdn_q_conv", None) is not None
                           and self.q_conv.weight is self.gdn_q_conv.weight)
        linear_q = self.q_conv.project_linear(x) if share_qk_linear else None
        linear_k = self.k_conv.project_linear(x) if share_qk_linear else None
        q = project(self.q_conv, self.head_dim, linear_q)
        k = project(self.k_conv, self.head_dim, linear_k)
        v = project(self.v_conv, self.value_dim)
        if self.value_residual and v_first is not None:
            v = value_residual_mix(v, v_first, self.v_residual_gate(x))
        if q.dtype != torch.bfloat16:
            raise ValueError("use CUDA BF16 autocast; the production kernels require BF16")
        sq = self._activate_readout(self.sq_proj(x))
        sk = self._activate_readout(self.sk_proj(x), is_key=True)
        # No runtime codebook activation; interpolation owns the BF16 cast.
        if self.vocab_transvq:
            with torch.autocast('cuda', enabled=False):
                eq = self.q_vocab_map(self.q_vocab.float())
                ek = self.k_vocab_map(self.k_vocab.float())
        else:
            eq, ek = self.q_vocab.float(), self.k_vocab.float()
        tau = F.softplus(self.log_sel_tau.float()).contiguous()
        if compiling:
            from .compiler import voc_forward
            output = voc_forward(q, k, sq, sk, v, eq, ek, tau, direction, hard,
                                 cu_seqlens, compiled_layout)[0]
        else:
            output = voc_dism(q, k, sq, sk, v, eq, ek, tau,
                              direction=direction, hard=hard, layout=layout)
        output = self._combine_cuda(output, x, v, cu_seqlens, max_seqlen, v_first=v_first,
                                    linear_q=linear_q, linear_k=linear_k)
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
