"""Parameter-matched pure SWA control, sharing hybrid's QK/RoPE and LM loss."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import DismLM
from flash_dism.hybrid import _compiled_varlen_swa
from flash_dism.kernels.linear_rmsnorm_rope import FusedLinearRMSNormRoPE
from flash_dism.kernels.fused_cross_entropy import fused_cross_entropy


class SwaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width, heads, dim = config.n_embd, config.n_head, config.head_dim
        self.attn_norm = nn.RMSNorm(width, eps=1e-6)
        self.mlp_norm = nn.RMSNorm(width, eps=1e-6)
        self.q_proj = FusedLinearRMSNormRoPE(width, heads, dim, eps=1e-6)
        self.k_proj = FusedLinearRMSNormRoPE(width, heads, dim, eps=1e-6)
        self.v_proj = nn.Linear(width, heads*dim, bias=False)
        self.o_proj = nn.Linear(heads*dim, width, bias=False)
        self.gate_proj = nn.Linear(width, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(width, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, width, bias=False)

    def forward(self, x, cu_seqlens, cos, sin):
        c = self.config
        z = self.attn_norm(x).contiguous()
        options = dict(cu_seqlens=cu_seqlens, max_seqlen=c.sequence_len)
        q = self.q_proj(z, cos, sin, **options)
        k = self.k_proj(z, cos, sin, **options)
        v = self.v_proj(z).reshape(-1, c.n_head, c.head_dim)
        out = _compiled_varlen_swa(q.squeeze(0), k.squeeze(0), v,
                                  cu_seqlens, c.sequence_len, c.window_size)
        x = x + self.o_proj(out.reshape(1, -1, c.n_head*c.head_dim))
        z = self.mlp_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(z)) * self.up_proj(z))


class SwaLM(DismLM):
    """Reuse optimizer/counting conventions, but no DISM parameters or RNG."""
    def _tie_vocabularies(self):
        # DismLM invokes this after device moves and load_state_dict. SWA has
        # neither codebooks nor shared DISM/GDN projection parameters.
        pass

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        if config.sequence_len % 256 or config.head_dim % 2 or config.window_size < 1:
            raise ValueError('Expected aligned context, even head_dim and positive window')
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(SwaBlock(config) for _ in range(config.n_layer))
        self.norm = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, FusedLinearRMSNormRoPE)):
                nn.init.normal_(module.weight, std=.02)
            if isinstance(module, FusedLinearRMSNormRoPE):
                nn.init.ones_(module.rms_weight)
            if isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

    def set_training_step(self, step):
        pass  # No stochastic hard/direction selection in this control.

    def estimate_flops(self):
        c = self.config
        return 6*self.scaling_parameter_count() + 12*c.n_layer*c.n_head*c.head_dim*c.window_size

    def forward(self, input_ids, targets=None, *, cu_seqlens=None, segment_ids=None, loss_reduction='mean'):
        batch, length = input_ids.shape
        c = self.config
        if length > c.sequence_len or cu_seqlens is None:
            raise ValueError('SWA control requires packed metadata and bounded context')
        frequencies = 10000. ** (-torch.arange(0, c.head_dim, 2, device=input_ids.device,
                                              dtype=torch.float32)/c.head_dim)
        phase = torch.arange(c.sequence_len, device=input_ids.device)[:, None]*frequencies
        cos, sin = phase.cos(), phase.sin()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = self.embedding(input_ids.reshape(1, -1))
            for layer in self.layers:
                x = layer(x, cu_seqlens, cos, sin)
            x = self.norm(x).to(torch.bfloat16)
            if targets is not None and loss_reduction == 'mean':
                with torch.autocast('cuda', enabled=False):
                    return fused_cross_entropy(x, self.lm_head.weight.to(torch.bfloat16),
                                               targets.reshape(1, -1), ignore_index=-1, softcap=c.softcap)
            logits = self.lm_head(x).float()
            logits = c.softcap * torch.tanh(logits/c.softcap)
            if targets is None:
                return logits.reshape(batch, length, -1)
            return F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1),
                                   ignore_index=-1, reduction=loss_reduction)
