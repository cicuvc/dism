"""A compact Llama 2 causal LM adapted to nanochat pretraining.

The module layout intentionally follows Transformers' LlamaForCausalLM so a
matching Hugging Face state dict can be loaded directly for numerical tests.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.flash_attention import flash_attn
from nanochat.optim import MuonAdamW


@dataclass
class Llama2Config:
    sequence_len: int = 4096
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 32
    n_embd: int = 4096
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("Llama 2 head_dim must be even for RoPE")


class Linear(nn.Linear):
    """Keep master weights in fp32 and cast only for the matmul."""

    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon).to(input_dtype)
        return self.weight.to(input_dtype) * hidden_states


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.sequence_len = config.sequence_len
        self.q_proj = Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

    def forward(self, hidden_states, cos_sin, cu_seqlens=None, segment_ids=None):
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.n_kv_head, self.head_dim)
        q, k = apply_rotary_pos_emb(q, k, *cos_sin)
        if cu_seqlens is None:
            output = flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=(self.sequence_len, 0)
            )
        else:
            output = flash_attn.flash_attn_varlen_func(
                q, k, v,
                cu_seqlens=cu_seqlens,
                max_seqlen=T,
                segment_ids=segment_ids,
                causal=True,
                window_size=(self.sequence_len, 0),
            )
        return self.o_proj(output.contiguous().view(B, T, -1))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.n_embd, config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.n_embd, config.rms_norm_eps)

    def forward(self, hidden_states, cos_sin, cu_seqlens=None, segment_ids=None):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), cos_sin,
            cu_seqlens=cu_seqlens, segment_ids=segment_ids,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class LlamaBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.n_layer)
        ])
        self.norm = LlamaRMSNorm(config.n_embd, config.rms_norm_eps)


class Llama2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaBackbone(config)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary(config.sequence_len, head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _precompute_rotary(sequence_len, head_dim, theta, device=None):
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        positions = torch.arange(sequence_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, :].to(COMPUTE_DTYPE)
        sin = emb.sin()[None, :, None, :].to(COMPUTE_DTYPE)
        return cos, sin

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LlamaRMSNorm):
                nn.init.ones_(module.weight)
        cos, sin = self._precompute_rotary(
            self.config.sequence_len,
            self.config.n_embd // self.config.n_head,
            self.config.rope_theta,
            device=self.model.embed_tokens.weight.device,
        )
        self.cos, self.sin = cos, sin
        if COMPUTE_DTYPE != torch.float16:
            self.model.embed_tokens.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.model.embed_tokens.weight.device

    def num_matmul_params(self):
        return sum(module.weight.numel() for module in self.modules() if isinstance(module, Linear))

    def estimate_flops(self):
        head_dim = self.config.n_embd // self.config.n_head
        attention = self.config.n_layer * 12 * self.config.n_head * head_dim * self.config.sequence_len
        return 6 * self.num_matmul_params() + attention

    def num_scaling_params(self):
        embedding = self.model.embed_tokens.weight.numel()
        lm_head = self.lm_head.weight.numel()
        layer_matrices = sum(
            parameter.numel()
            for layer in self.model.layers
            for parameter in layer.parameters()
            if parameter.ndim >= 2
        )
        norms = sum(
            parameter.numel() for name, parameter in self.named_parameters()
            if "layernorm" in name or name == "model.norm.weight"
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        assert total == embedding + lm_head + layer_matrices + norms
        return {
            "embedding": embedding,
            "lm_head": lm_head,
            "layer_matrices": layer_matrices,
            "norms": norms,
            "total": total,
        }

    def scaling_parameter_count(self):
        counts = self.num_scaling_params()
        return counts["layer_matrices"] + counts["lm_head"]

    def setup_pretraining_optimizer(self, config):
        named_parameters = dict(self.named_parameters())
        embedding = [named_parameters.pop("model.embed_tokens.weight")]
        lm_head = [named_parameters.pop("lm_head.weight")]
        matrix = [parameter for parameter in named_parameters.values() if parameter.ndim >= 2]
        scalar = [parameter for parameter in named_parameters.values() if parameter.ndim < 2]
        model_dim_scale = (self.config.n_embd / 4096) ** -0.5
        print0(f"Scaling Llama AdamW LRs by 1/sqrt({self.config.n_embd}/4096) = {model_dim_scale:.6f}")
        groups = [
            dict(kind="adamw", params=lm_head, lr=config.unembedding_lr * model_dim_scale,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
            dict(kind="adamw", params=embedding, lr=config.embedding_lr * model_dim_scale,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
            dict(kind="adamw", params=scalar, lr=config.scalar_lr * model_dim_scale,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        ]
        for shape in sorted({parameter.shape for parameter in matrix}):
            groups.append(dict(
                kind="muon",
                params=[parameter for parameter in matrix if parameter.shape == shape],
                lr=config.matrix_lr,
                momentum=0.95,
                ns_steps=5,
                beta2=0.9,
                weight_decay=config.weight_decay,
            ))
        optimizer = MuonAdamW(groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, input_ids, targets=None, *, cu_seqlens=None, segment_ids=None,
                loss_reduction="mean"):
        if (cu_seqlens is None) != (segment_ids is None):
            raise ValueError("cu_seqlens and segment_ids must be provided together")
        _, T = input_ids.shape
        if T > self.config.sequence_len:
            raise ValueError(f"Sequence length {T} exceeds configured {self.config.sequence_len}")
        hidden_states = self.model.embed_tokens(input_ids).to(COMPUTE_DTYPE)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states, cos_sin,
                cu_seqlens=cu_seqlens, segment_ids=segment_ids,
            )
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=-1, reduction=loss_reduction,
        )
