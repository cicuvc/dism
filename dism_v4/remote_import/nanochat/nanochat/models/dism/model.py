"""nanochat adapter for flash_dism hybrid; no cross-step KV state in training."""
import math
import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl
from flash_dism import DismConfig as FlashConfig
from flash_dism.modeling_dism import DismBlock
from flash_dism.kernels.conv1d import CausalShortConv1d
from flash_dism.kernels.linear_rmsnorm_rope import FusedLinearRMSNormRoPE
from flash_dism.kernels.fused_cross_entropy import fused_cross_entropy


@triton.jit
def _flags(hard, direction, counter, probability, N: tl.constexpr, H: tl.constexpr,
           SEED: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    seed = tl.load(counter).to(tl.uint64) + SEED
    tl.store(hard + offsets, tl.rand(seed, offsets) < tl.load(probability), offsets < N*H)
    if tl.program_id(0) == 0:
        choice = tl.rand(seed, tl.full((), N*H, tl.uint32)) < 0.5
        tl.store(direction + tl.arange(0, BLOCK), choice, tl.arange(0, BLOCK) < H)


@torch.library.custom_op('nanochat_dism::flags', mutates_args=())
def sample_flags(counter: torch.Tensor, probability: torch.Tensor, n: int, heads: int,
                 seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    hard = torch.empty((1, heads, n), device=counter.device, dtype=torch.bool)
    direction = torch.empty((1, heads), device=counter.device, dtype=torch.bool)
    _flags[(triton.cdiv(n*heads, 256),)](hard, direction, counter, probability, n, heads, seed, 256)
    return hard, direction


@sample_flags.register_fake
def _fake_flags(counter, probability, n, heads, seed):
    return (counter.new_empty((1, heads, n), dtype=torch.bool),
            counter.new_empty((1, heads), dtype=torch.bool))


class DismLM(nn.Module):
    aligned_segment_ends = True
    exact_optimizer_hyperparameters = True
    dynamic_compile = True

    def __init__(self, config, *, attention_type='hybrid'):
        super().__init__()
        self.config = config
        self.v4 = attention_type == 'hybrid_v4'
        if not 0.0 <= config.hard_prob_max <= 1.0:
            raise ValueError('hard_prob_max must be in [0,1]')
        if config.sequence_len % 256:
            raise ValueError('DISM context must be256-aligned')
        c = FlashConfig(hidden_size=config.n_embd, num_hidden_layers=config.n_layer,
                        num_heads=config.n_head, head_dim=config.head_dim,
                        value_dim=config.value_dim, readout_dim=config.readout_dim,
                        qk_vocab_size=config.qk_vocab_size, intermediate_size=config.intermediate_size,
                        attention_type=attention_type, window_size=config.window_size,
                        vocab_size=config.vocab_size, bos_token_id=1, eos_token_id=2,
                        soft_k_l2_norm=config.soft_k_l2_norm)
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(DismBlock(c, i) for i in range(config.n_layer))
        self.norm = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.register_buffer('training_step', torch.zeros((), dtype=torch.int64))
        self.register_buffer('rng_counter', torch.zeros((), dtype=torch.int64))
        self.register_buffer('hard_probability', torch.zeros((), dtype=torch.float32))

    @torch.no_grad()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding, CausalShortConv1d, FusedLinearRMSNormRoPE)):
                nn.init.normal_(m.weight, std=.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, CausalShortConv1d):
                nn.init.uniform_(m.conv_weight, -1/math.sqrt(m.kernel_size), 1/math.sqrt(m.kernel_size))
            if isinstance(m, FusedLinearRMSNormRoPE):
                nn.init.ones_(m.rms_weight)
            if isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)
        for layer in self.layers:
            attn = layer.attn
            for table in (attn.q_vocab, attn.k_vocab):
                table.copy_(F.silu(torch.randn_like(table)))
            attn.log_sel_tau.uniform_(-4, 4)
            attn.norm.weight.fill_(1)
        self.training_step.zero_()
        self.rng_counter.zero_()
        self.hard_probability.zero_()

    @torch.no_grad()
    def set_training_step(self, step):
        self.training_step.fill_(step)
        progress = min(max(step / max(1, self.config.anneal_steps-1), 0.), 1.)
        self.hard_probability.fill_(self.config.hard_prob_max * progress)

    def get_device(self):
        return self.embedding.weight.device

    def num_scaling_params(self):
        embedding = self.embedding.weight.numel()
        head = self.lm_head.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        return dict(embedding=embedding, lm_head=head, layers=total-embedding-head, total=total)

    def scaling_parameter_count(self):
        return self.num_scaling_params()['total'] - self.embedding.weight.numel()

    def estimate_flops(self):
        return 6*self.scaling_parameter_count() + 12*self.config.n_layer*self.config.n_head*64*self.config.sequence_len

    def setup_pretraining_optimizer(self, config):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            exempt = (p.ndim < 2 or name.endswith(('.bias', 'q_vocab', 'k_vocab', 'log_sel_tau', 'rms_weight'))
                      or getattr(p, '_no_weight_decay', False))
            (no_decay if exempt else decay).append(p)
        groups = [dict(params=decay, weight_decay=config.weight_decay, kind='adamw'),
                  dict(params=no_decay, weight_decay=0., kind='adamw')]
        optimizer = torch.optim.AdamW(groups, lr=config.matrix_lr, betas=(.9, .95), eps=1e-8,
                                     fused=self.get_device().type == 'cuda')
        for g in optimizer.param_groups:
            g['initial_lr'] = g['lr']
        return optimizer

    def forward(self, input_ids, targets=None, *, cu_seqlens=None, segment_ids=None, loss_reduction='mean'):
        batch, length = input_ids.shape
        if length > self.config.sequence_len or cu_seqlens is None:
            raise ValueError('DISM nanochat adapter requires packed varlen metadata and bounded context')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = self.embedding(input_ids.reshape(1, -1))
            if self.training:
                # Copy before mutation: saved random identity belongs to this
                # microbatch, not a mutable counter read during backward.
                counter = self.rng_counter.clone()
                self.rng_counter.add_(1)
            for i, layer in enumerate(self.layers):
                if self.training:
                    hard, direction = sample_flags(counter, self.hard_probability,
                        batch*length, self.config.n_head, self.config.rng_seed + i*104729)
                else:
                    hard = torch.ones((1, self.config.n_head, batch*length), device=x.device, dtype=torch.bool)
                    direction = torch.ones((1, self.config.n_head), device=x.device, dtype=torch.bool)
                gate_options = {}
                if self.v4:
                    if self.training:
                        # Same device-resident annealing probability, independent
                        # reproducible stream; no CPU scalar extraction or recompile.
                        delta_hard, _ = sample_flags(counter, self.hard_probability,
                            batch*length, self.config.n_head,
                            (self.config.rng_seed + i*104729) ^ 0x36A9F12B)
                    else:
                        delta_hard = hard
                    gate_options['delta_hard'] = delta_hard
                x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=self.config.sequence_len,
                          hard=hard, direction=direction, use_cache=False, **gate_options)
            x = self.norm(x).to(torch.bfloat16)
            if targets is not None and loss_reduction == 'mean':
                # nanochat targets are ALREADY shifted, unlike HF labels.
                with torch.autocast('cuda', enabled=False):
                    return fused_cross_entropy(x, self.lm_head.weight.to(torch.bfloat16), targets.reshape(1, -1),
                                               ignore_index=-1, softcap=self.config.softcap)
            logits = self.lm_head(x).float()
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
            if targets is None:
                return logits.reshape(batch, length, -1)
            return F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1),
                                   ignore_index=-1, reduction=loss_reduction)
