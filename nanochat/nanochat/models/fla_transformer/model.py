"""FLA attention/RoPE, postnorm, MXFP8 SwiGLU and document-balanced CE.

Only metadata preparation differs from FLA Attention.forward: callers can supply
fixed-storage RoPE chunk indices and a host max document length for CUDA Graphs.
No GDN, DISM selector, convolution, value-residual or attention gate is retained.
"""
import torch
from torch import nn
import triton
from fla.layers.attn import Attention, flash_attn_varlen_func
from fla.utils import get_multiprocessor_count
from flash_dism.modeling_dism import DismMLP
from flash_dism.kernels.fused_cross_entropy import fused_cross_entropy_unreduced
from flash_mxfp8 import FusedAddRMSNormMXFP8, refresh_fp8_modules
from nanochat.models.dism.model import packed_document_ce, DismLM


def rotary_block_size(tokens, device):
    # Must match fla.modules.rotary.rotary_embedding_fwdbwd.
    return min(128, triton.next_power_of_2(triton.cdiv(tokens, get_multiprocessor_count(device.index))))


class PackedAttention(Attention):
    def forward(self, hidden_states, *, cu_seqlens, max_seqlen, rotary_chunks=None):
        shape = hidden_states.shape
        q = self.q_proj(hidden_states).view(*shape[:-1], self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(*shape[:-1], self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(*shape[:-1], self.num_kv_heads, self.head_dim)
        q, k = self.rotary(q, k, cu_seqlens=cu_seqlens,
                           max_seqlen=shape[1], chunk_indices=rotary_chunks)
        output = flash_attn_varlen_func(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, causal=True,
            dropout_p=0.0, window_size=(-1, -1))
        return self.o_proj(output.reshape(shape))


class Block(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        from types import SimpleNamespace
        self.attn = PackedAttention(hidden_size=config.n_embd, num_heads=config.n_head,
                                    num_kv_heads=config.n_head, rope_theta=config.rope_theta,
                                    layer_idx=index)
        self.attn_norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.fused_attn_norm = FusedAddRMSNormMXFP8(config.n_embd, eps=config.norm_eps)
        self.mlp = DismMLP(SimpleNamespace(mxfp8=True, hidden_size=config.n_embd,
                                         intermediate_size=config.intermediate_size))
        self.mlp_norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, x, **kwargs):
        branch = self.attn(x, **kwargs)
        x, row, rs, col, cs = self.fused_attn_norm(x, branch, self.attn_norm.weight)
        return self.mlp_norm(x + self.mlp.forward_quantized(x, row, rs, col, cs)).to(torch.bfloat16)


class FLATransformer(nn.Module):
    document_balanced_loss = True
    count_normalized_loss = True
    aligned_segment_ends = True
    loss_sum_reduction = 'document_sum'
    loss_normalization = 'global_valid_document_mean_v1'
    supports_packed_niah = True
    # Share the existing AdamW recipe, grouping and cache-refresh behavior.
    setup_pretraining_optimizer = DismLM.setup_pretraining_optimizer

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False, dtype=torch.bfloat16)
        self.mxfp8_ce = self.mxfp8_ce_norm = None
        if config.mxfp8_ce:
            from flash_mxfp8 import MXFP8FusedLinearCrossEntropy, RMSNormMXFP8
            self.mxfp8_ce = MXFP8FusedLinearCrossEntropy(ignore_index=-1, balance='document_sum', softcap=config.softcap)
            self.mxfp8_ce_norm = RMSNormMXFP8(eps=config.norm_eps, chunk_size=self.mxfp8_ce.chunk_size)

    def get_device(self):
        return self.embedding.weight.device

    @torch.no_grad()
    def init_weights(self):
        for name, p in self.named_parameters():
            if p.ndim >= 2:
                nn.init.normal_(p, std=.02)
            else:
                nn.init.ones_(p)
        # to_empty(meta) also materializes FLA's nonpersistent frequency buffer.
        for layer in self.layers:
            rotary = layer.attn.rotary
            rotary.inv_freq.copy_(1.0 / (self.config.rope_theta ** (
                torch.arange(0, rotary.dim, 2, device=self.get_device(), dtype=torch.float32) / rotary.dim)))
        if self.get_device().type == 'cuda':
            refresh_fp8_modules(self)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.get_device().type == 'cuda':
            refresh_fp8_modules(self)
        return result

    def num_scaling_params(self):
        embedding = self.embedding.weight.numel()
        head = self.lm_head.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        return dict(embedding=embedding, lm_head=head, layers=total-embedding-head,
                    total=total, matched_total=total)

    def scaling_parameter_count(self):
        return self.num_scaling_params()['total'] - self.embedding.weight.numel()

    def estimate_flops(self):
        return 6*self.scaling_parameter_count() + 12*self.config.n_layer*self.config.n_embd*self.config.sequence_len

    def forward(self, input_ids, targets=None, *, cu_seqlens, segment_ids=None,
                loss_reduction='mean', rotary_chunks=None):
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.sequence_len:
            raise ValueError('expected bounded [rows, length] inputs')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = self.embedding(input_ids.reshape(1, -1))
            for layer in self.layers:
                x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=self.config.sequence_len,
                          rotary_chunks=rotary_chunks)
            quantized = None
            if self.mxfp8_ce_norm is not None and torch.is_grad_enabled() and targets is not None and loss_reduction in ('mean','document_sum'):
                x, quantized = self.mxfp8_ce_norm(x.reshape(-1, x.shape[-1]).contiguous(), self.norm.weight,
                    eps=self.norm.eps, chunk_size=self.mxfp8_ce.chunk_size)
            else:
                x = self.norm(x).to(torch.bfloat16)
            if targets is not None:
                with torch.autocast('cuda', enabled=False):
                    if loss_reduction in ('mean', 'document_sum'):
                        if self.mxfp8_ce is not None and torch.is_grad_enabled():
                            loss, count = self.mxfp8_ce(x.reshape(-1,x.shape[-1]).contiguous(),
                                targets.reshape(-1).contiguous(), self.lm_head.weight,
                                cu_seqlens=cu_seqlens, quantized_input=quantized)
                            return (loss, count) if loss_reduction=='document_sum' else loss/count.clamp_min(1)
                        return packed_document_ce(x.reshape(1,-1,x.shape[-1]), self.lm_head.weight,
                            targets, cu_seqlens, self.config.sequence_len, self.config.softcap, loss_reduction)
                    if loss_reduction == 'none' and not torch.is_grad_enabled():
                        return fused_cross_entropy_unreduced(x, self.lm_head.weight,
                            targets.reshape(1,-1), ignore_index=-1, softcap=self.config.softcap).flatten()
                    raise ValueError('unsupported loss reduction')
            logits = self.lm_head(x).float()
            if self.config.softcap > 0:
                logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
            return logits.reshape(*input_ids.shape, -1)
