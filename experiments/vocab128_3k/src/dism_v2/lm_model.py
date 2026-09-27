"""~50M untied decoder LM: copy-task DISM + parallel RoPE sliding attention."""
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F
from flash_attn import flash_attn_func
from .softcap_cross_entropy import FusedSoftcapCrossEntropyLoss


@dataclass
class LMConfig:
    vocab_size: int = 50257
    width: int = 256
    layers: int = 15
    heads: int = 4
    head_dim: int = 64
    qk_vocab: int = 512
    ffn_hidden: int = 1024
    context: int = 2048
    window: int = 128  # Includes the current token:127 left + self.
    rope_theta: float = 10000.0
    softcap: float | None = None
    architecture: str = "hybrid"
    dism_tie_qk_vocab: bool = False
    dism_vocab_groups: int | None = None  # None=heads; contiguous heads share a group.
    dism_activation: str = 'baseline'  # baseline / vocab_silu / no_qk_silu
    dism_output_gla: bool = False
    dism_vocab_norm: bool = False  # Per-codeword FP32 RMSNorm, no affine parameters.


def vocab_group_count(c):
    if c.dism_activation not in ('baseline', 'vocab_silu', 'no_qk_silu'):
        raise ValueError('Unknown DISM activation parameterization')
    groups = c.heads if c.dism_vocab_groups is None else c.dism_vocab_groups
    if type(groups) is not int or groups <= 0 or c.heads % groups:
        raise ValueError('dism_vocab_groups must be a positive divisor of heads')
    if type(c.dism_tie_qk_vocab) is not bool:
        raise ValueError('dism_tie_qk_vocab must be bool')
    return groups


class DismAttention(nn.Module):
    """Copy-task projections, causal short convolutions, vocab and output gate."""
    def __init__(self, c):
        super().__init__()
        from fla.modules import FusedRMSNormGated, ShortConvolution
        self.c = c
        self.q_proj = nn.Linear(c.width, c.width, bias=False)
        self.k_proj = nn.Linear(c.width, c.width, bias=False)
        self.v_proj = nn.Linear(c.width, c.width, bias=False)
        self.o_proj = nn.Linear(c.width, c.width)
        qk_activation = None if c.dism_activation == 'no_qk_silu' else 'swish'
        self.qd_conv = ShortConvolution(c.width, 4, activation=qk_activation)
        self.kd_conv = ShortConvolution(c.width, 4, activation=qk_activation)
        self.v_dism = ShortConvolution(c.width, 4, activation="silu")
        self.norm = FusedRMSNormGated(c.width, eps=1e-5)
        self.vocab_groups = vocab_group_count(c)
        # Tied mode has exactly one registered owner, not two state_dict aliases.
        self.q_voc = nn.Parameter(torch.randn(self.vocab_groups, c.qk_vocab, c.head_dim))
        self.k_voc = (None if c.dism_tie_qk_vocab else
                      nn.Parameter(torch.randn(self.vocab_groups, c.qk_vocab, c.head_dim)))
        self.log_sel_tau = nn.Parameter(torch.empty(c.heads).uniform_(-4, 4))
        for p in (self.q_voc, self.k_voc, self.log_sel_tau):
            if p is not None:
                p._no_weight_decay = True
        self.g_proj_down = nn.Linear(c.width, c.width // 8)
        self.g_proj_up = nn.Linear(c.width // 8, c.width)

    def expanded_vocabularies(self, dtype):
        """Existing CUDA ABI [H,V,D]; reductions happen in master FP32.

        Expand BEFORE casting, separately on the Q/K paths. Casting a shared
        group tensor once would accumulate head/branch gradients in BF16.
        This is parameter sharing, not K/V-cache GQA or a bandwidth optimization.
        """
        def expand(vocab):
            # Differentiable effective codebook; apply to FP32 master before
            # separate Q/K expansion and BF16 casts. All interpolation uses it.
            if self.c.dism_activation == 'vocab_silu':
                vocab = F.silu(vocab)
            if self.c.dism_vocab_norm:
                vocab = vocab.float()
                vocab = vocab * torch.rsqrt(vocab.square().mean(dim=-1, keepdim=True) + 1e-6)
            per_group = self.c.heads // self.vocab_groups
            full = vocab if per_group == 1 else vocab.repeat_interleave(per_group, dim=0)
            return full.to(dtype).contiguous()
        return expand(self.q_voc), expand(self.q_voc if self.k_voc is None else self.k_voc)

    def forward(self, x, hard_prob, generator):
        from .autograd import voc_dism
        b, n, _ = x.shape
        def project(proj, conv):
            return conv(proj(x))[0].reshape(b, n, self.c.heads, self.c.head_dim).transpose(1, 2).contiguous()
        q = project(self.q_proj, self.qd_conv)
        k = project(self.k_proj, self.kd_conv)
        v = project(self.v_proj, self.v_dism)
        tau = F.softplus(self.log_sel_tau.float()).contiguous()
        qvoc, kvoc = self.expanded_vocabularies(q.dtype)
        out = voc_dism(q, k, v, tau, qvoc, kvoc, hard_prob=hard_prob,
                       direction="random", generator=generator, sm_scale=1.,
                       embedding_backend="cuda", embedding_backward_backend="cuda")
        out = out.transpose(1, 2).reshape(b, n, self.c.width)
        gate = self.g_proj_up(self.g_proj_down(x))
        return self.o_proj(self.norm(out, gate))


class SlidingAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)
        freq = c.rope_theta ** (-torch.arange(0, c.head_dim, 2, dtype=torch.float32) / c.head_dim)
        angles = torch.outer(torch.arange(c.context, dtype=torch.float32), freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def rope(self, x):
        cos = self.cos[:x.shape[1]][None, :, None, :].to(x.dtype)
        sin = self.sin[:x.shape[1]][None, :, None, :].to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.c.heads, self.c.head_dim).unbind(2)
        window = (-1, -1) if self.c.architecture == 'full_attention' else (self.c.window - 1, 0)
        out = flash_attn_func(self.rope(q), self.rope(k), v.contiguous(),
                              causal=True, window_size=window, dropout_p=0.)
        return self.out(out.reshape(b, n, self.c.width))


class SharedDismAttention(DismAttention):
    """Shared QKV projections; separate causal convs; sum before shared gate/output."""
    def __init__(self, c):
        super().__init__(c)
        from fla.modules import ShortConvolution
        self.q_swa_conv = ShortConvolution(c.width, 4, activation='swish')
        self.k_swa_conv = ShortConvolution(c.width, 4, activation='swish')
        self.v_swa_conv = ShortConvolution(c.width, 4, activation='silu')
        freq = c.rope_theta ** (-torch.arange(0, c.head_dim, 2, dtype=torch.float32) / c.head_dim)
        angles = torch.outer(torch.arange(c.context, dtype=torch.float32), freq)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    rope = SlidingAttention.rope

    def forward(self, x, hard_prob, generator):
        from .autograd import voc_dism
        b, n, _ = x.shape
        # Each projection executes exactly once; both branches backprop into it.
        q0, k0, v0 = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        def convolve(conv, projected):
            return conv(projected)[0].reshape(b, n, self.c.heads, self.c.head_dim)
        q = convolve(self.qd_conv, q0).transpose(1, 2).contiguous()
        k = convolve(self.kd_conv, k0).transpose(1, 2).contiguous()
        v = convolve(self.v_dism, v0).transpose(1, 2).contiguous()
        tau = F.softplus(self.log_sel_tau.float()).contiguous()
        qvoc, kvoc = self.expanded_vocabularies(q.dtype)
        dism = voc_dism(q, k, v, tau, qvoc, kvoc, hard_prob=hard_prob,
                        direction='random', generator=generator, sm_scale=1.,
                        embedding_backend='cuda', embedding_backward_backend='cuda')
        qs = self.rope(convolve(self.q_swa_conv, q0))
        ks = self.rope(convolve(self.k_swa_conv, k0))
        vs = convolve(self.v_swa_conv, v0).contiguous()
        swa = flash_attn_func(qs, ks, vs, causal=True,
                              window_size=(self.c.window - 1, 0), dropout_p=0.)
        out = (dism.transpose(1, 2) + swa).reshape(b, n, self.c.width)
        gate = self.g_proj_up(self.g_proj_down(x))
        return self.o_proj(self.norm(out, gate))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.dism = (SharedDismAttention(c) if c.architecture == 'hybrid_shared' else
                     (DismAttentionWithGLA(c) if c.dism_output_gla else DismAttention(c))
                     if c.architecture == 'hybrid' else None)
        self.swa = None if c.architecture == 'hybrid_shared' else SlidingAttention(c)
        self.up = nn.Linear(c.width, 2 * c.ffn_hidden)
        self.down = nn.Linear(c.ffn_hidden, c.width)

    def forward(self, x, hard_prob, generator):
        h = self.norm1(x)
        if self.swa is None:
            x = x + self.dism(h, hard_prob, generator)
        elif self.dism is not None:
            x = x + self.dism(h, hard_prob, generator) + self.swa(h)
        else:
            x = x + self.swa(h)
        gate, value = self.up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * value)


class DecoderLM(nn.Module):
    def __init__(self, c=LMConfig()):
        super().__init__()
        if c.width != c.heads * c.head_dim or c.head_dim != 64:
            raise ValueError("Expected width=heads*64")
        if c.architecture not in ("hybrid", "hybrid_shared", "swa_only", "full_attention"):
            raise ValueError("Unknown architecture")
        vocab_group_count(c)
        if c.dism_output_gla and c.architecture != 'hybrid':
            raise ValueError('Output GLA currently requires architecture=hybrid')
        if c.dism_vocab_norm and c.architecture not in ('hybrid', 'hybrid_shared'):
            raise ValueError('Vocabulary normalization requires a hybrid architecture')
        if c.architecture in ('swa_only', 'full_attention') and (c.dism_tie_qk_vocab or c.dism_vocab_groups is not None or c.dism_activation != 'baseline'):
            raise ValueError('DISM vocabulary options require a hybrid architecture')
        if c.architecture == 'full_attention' and c.window != -1:
            raise ValueError('Full attention configuration must record window=-1')
        self.config = c
        self.embedding = nn.Embedding(c.vocab_size, c.width)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.final_norm = nn.LayerNorm(c.width)
        self.lm_head = nn.Linear(c.width, c.vocab_size, bias=False)
        # Deliberately untied. DISM vocabulary/conv/gate initialization follows copy_task.
        nn.init.normal_(self.embedding.weight, std=.02)
        nn.init.normal_(self.lm_head.weight, std=.02)
        if c.softcap is None:
            from fla.modules.fused_cross_entropy import FusedCrossEntropyLoss
        self.loss_fn = (FusedCrossEntropyLoss(reduction="mean", inplace_backward=True) if c.softcap is None
                        else FusedSoftcapCrossEntropyLoss(c.softcap, reduction="mean"))

    def forward_features(self, input_ids, hard_prob=1., generator=None):
        if input_ids.shape[1] > self.config.context:
            raise ValueError("Sequence exceeds configured RoPE/context length")
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x, hard_prob, generator)
        return self.final_norm(x)

    def forward(self, input_ids, labels, hard_prob, generator):
        x = self.forward_features(input_ids, hard_prob, generator)
        logits = self.lm_head(x).reshape(-1, self.config.vocab_size)
        return self.loss_fn(logits, labels.reshape(-1))


def parameter_count(config):
    # No GPU work or large real allocations while preparing a control run.
    with torch.device("meta"):
        model = DecoderLM(config)
    return sum(p.numel() for p in model.parameters())


def matched_swa_config(reference):
    """Keep depth/width/window; use a uniform64-aligned FFN to match the budget."""
    if reference.architecture not in ('hybrid', 'hybrid_shared'):
        raise ValueError("Expected a hybrid reference")
    # Count the removed copy-task branch analytically so a pure-SWA remote
    # deployment need not import DISM's CUDA/FLA dependencies, even on meta.
    w, g = reference.width, reference.width // 8
    # FLA ShortConvolution defaults to bias=False.
    dism_per_layer = (4 * w * w + w + 3 * (4 * w) + w
                      + (1 if reference.dism_tie_qk_vocab else 2) * vocab_group_count(reference) * reference.qk_vocab * reference.head_dim
                      + reference.heads + 2 * w * g + g + w)
    if reference.architecture == 'hybrid_shared':
        dism_per_layer += 3 * (4 * w) - 4 * w * w
    # SwiGLU up[2F,W],bias[2F],down[W,F],bias[W].
    hidden = reference.ffn_hidden + dism_per_layer / (3 * w + 2)
    hidden = max(64, int(hidden / 64 + .5) * 64)
    return replace(reference, architecture='swa_only', ffn_hidden=hidden, dism_activation='baseline',
                   dism_tie_qk_vocab=False, dism_vocab_groups=None, dism_output_gla=False,
                   dism_vocab_norm=False)


def parameter_groups(model, weight_decay):
    """Respect explicit metadata, norms, and biases; tied aliases cannot duplicate."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 or getattr(p, "_no_weight_decay", False) else decay).append(p)
    assert len({id(p) for p in decay + no_decay}) == len(decay) + len(no_decay)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.}]


def hard_probability(step, total_steps):
    """Zero-based optimizer update: first pure soft, final pure hard."""
    return min(max(step / max(total_steps - 1, 1), 0.), 1.)


def learning_rate(step, total_steps, warmup, peak, minimum_ratio=.1):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = min((step - warmup) / max(total_steps - warmup - 1, 1), 1.)
    return peak * (minimum_ratio + (1 - minimum_ratio) * .5 * (1 + math.cos(math.pi * progress)))


class DismAttentionWithGLA(nn.Module):
    """Copy-task projections, causal short convolutions, vocab and output gate."""
    def __init__(self, c):
        super().__init__()
        from fla.modules import FusedRMSNormGated, ShortConvolution
        from fla.ops.simple_gla import chunk_simple_gla
        
        self.chunk_simple_gla = chunk_simple_gla

        self.c = c
        self.q_proj = nn.Linear(c.width, c.width, bias=False)
        self.k_proj = nn.Linear(c.width, c.width, bias=False)
        self.v_proj = nn.Linear(c.width, c.width, bias=False)
        self.o_proj = nn.Linear(c.width, c.width)
        qk_activation = None if c.dism_activation == 'no_qk_silu' else 'swish'
        self.qd_conv = ShortConvolution(c.width, 4, activation=qk_activation)
        self.kd_conv = ShortConvolution(c.width, 4, activation=qk_activation)
        self.v_dism = ShortConvolution(c.width, 4, activation="silu")
        self.norm = FusedRMSNormGated(c.width, eps=1e-5)
        self.vocab_groups = vocab_group_count(c)
        # Tied mode has exactly one registered owner, not two state_dict aliases.
        self.q_voc = nn.Parameter(torch.randn(self.vocab_groups, c.qk_vocab, c.head_dim))
        self.k_voc = (None if c.dism_tie_qk_vocab else
                      nn.Parameter(torch.randn(self.vocab_groups, c.qk_vocab, c.head_dim)))
        self.log_sel_tau = nn.Parameter(torch.empty(c.heads).uniform_(-4, 4))
        for p in (self.q_voc, self.k_voc, self.log_sel_tau):
            if p is not None:
                p._no_weight_decay = True
        self.g_proj_down = nn.Linear(c.width, c.width // 8)
        self.g_proj_up = nn.Linear(c.width // 8, c.width)




        # mamba2-style gating
        self.a_proj = nn.Linear(c.width, c.heads, bias=False)
        A = torch.empty(c.heads, dtype=torch.float32).uniform_(0, 16).clamp_min_(1e-6)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(c.heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

    def expanded_vocabularies(self, dtype):
        """Existing CUDA ABI [H,V,D]; reductions happen in master FP32.

        Expand BEFORE casting, separately on the Q/K paths. Casting a shared
        group tensor once would accumulate head/branch gradients in BF16.
        This is parameter sharing, not K/V-cache GQA or a bandwidth optimization.
        """
        def expand(vocab):
            # Differentiable effective codebook; apply to FP32 master before
            # separate Q/K expansion and BF16 casts. All interpolation uses it.
            if self.c.dism_activation == 'vocab_silu':
                vocab = F.silu(vocab)
            if self.c.dism_vocab_norm:
                vocab = vocab.float()
                vocab = vocab * torch.rsqrt(vocab.square().mean(dim=-1, keepdim=True) + 1e-6)
            per_group = self.c.heads // self.vocab_groups
            full = vocab if per_group == 1 else vocab.repeat_interleave(per_group, dim=0)
            return full.to(dtype).contiguous()
        return expand(self.q_voc), expand(self.q_voc if self.k_voc is None else self.k_voc)

    def forward(self, x, hard_prob, generator):
        from .autograd import voc_dism
        b, n, _ = x.shape
        def project(proj, conv):
            u = conv(proj(x))[0].reshape(b, n, self.c.heads, self.c.head_dim)
            return u.transpose(1, 2).contiguous(), u
        q, q_bnhc = project(self.q_proj, self.qd_conv)
        k, k_bnhc = project(self.k_proj, self.kd_conv)
        v, _ = project(self.v_proj, self.v_dism)
        tau = F.softplus(self.log_sel_tau.float()).contiguous()
        qvoc, kvoc = self.expanded_vocabularies(q.dtype)
        out = voc_dism(q, k, v, tau, qvoc, kvoc, hard_prob=hard_prob,
                       direction="random", generator=generator, sm_scale=1.,
                       embedding_backend="cuda", embedding_backward_backend="cuda")

        out_bnhc = out.transpose(1, 2).contiguous()
        g_bnh = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)
        gla_out, _ = self.chunk_simple_gla(q_bnhc, k_bnhc, out_bnhc, g_bnh,
                                         scale=self.c.head_dim ** -.5)
        out = (out_bnhc + gla_out).reshape(b, n, self.c.width)
        gate = self.g_proj_up(self.g_proj_down(x))
        return self.o_proj(self.norm(out, gate))
