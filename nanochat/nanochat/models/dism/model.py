"""nanochat adapter for flash_dism hybrid; no cross-step KV state in training."""
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl
from flash_dism import DismConfig as FlashConfig
from flash_dism.modeling_dism import DismBlock
from flash_dism.kernels.conv1d import CausalShortConv1d
from flash_dism.kernels.linear_rmsnorm_rope import FusedLinearRMSNormRoPE
from flash_dism.kernels.fused_cross_entropy import fused_cross_entropy, fused_cross_entropy_unreduced


@torch.compiler.disable
def _gdn_value_first(attn, x, cu_seqlens, max_seqlen):
    """Value vector of the first GDN mixer, reused by later layers (value residual)."""
    return attn.gdn_v_conv(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).unflatten(
        -1, (attn.heads, attn.value_dim))


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
        self.eval_hard = True
        if not 0. <= config.hard_start_fraction < 1.:
            raise ValueError("hard_start_fraction must be in [0,1)")
        if not config.hard_start_fraction < config.hard_end_fraction <= 1.:
            raise ValueError("hard_end_fraction must be > start and <=1")
        if config.shared_readout or config.readout_mode != 'default' or config.soft_qk_rope or config.split_branch_output:
            raise ValueError('This source branch retains the baseline readout/output path only')
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
                        soft_k_l2_norm=config.soft_k_l2_norm,
                        value_residual=config.value_residual,
                        post_norm=config.post_norm)
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        if config.alternating_gdn or config.rear_half_dism:
            if config.alternating_gdn and config.rear_half_dism:
                raise ValueError('Choose only one mixed-layer layout')
            if attention_type != 'hybrid_gdn' or config.n_layer%2:
                raise ValueError('Alternating GDN requires an even number of GDN/hybrid layers')
            blocks=[]
            for i in range(config.n_layer):
                layer_config=copy.copy(c)
                if (i < config.n_layer//2 if config.rear_half_dism else i%2==0):
                    layer_config.attention_type='pure_gdn'
                blocks.append(DismBlock(layer_config,i))
            self.layers=nn.ModuleList(blocks)
        else:
            self.layers = nn.ModuleList(DismBlock(c, i) for i in range(config.n_layer))
        if config.value_residual and not getattr(self.layers[0].attn, "is_pure_gdn", False):
            raise ValueError('value_residual requires a GDN-only first mixer layer')
        self._tie_vocabularies()
        self.norm = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.register_buffer('training_step', torch.zeros((), dtype=torch.int64))
        self.register_buffer('rng_counter', torch.zeros((), dtype=torch.int64))
        self.register_buffer('hard_probability', torch.zeros((), dtype=torch.float32))

    def _tie_vocabularies(self):
        first_hybrid=None
        for i, layer in enumerate(self.layers):
            attn = layer.attn
            if getattr(attn,"is_pure_gdn",False):continue
            if first_hybrid is None:first_hybrid=attn
            if hasattr(attn, "tie_branch_projections"):
                attn.tie_branch_projections()
            if self.config.vocab_share_heads:
                for name in ('q_vocab', 'k_vocab'):
                    table = getattr(attn, name)
                    if table.ndim == 3:
                        setattr(attn, name, nn.Parameter(table[0].detach().clone()))
            if self.config.vocab_share_layers and attn is not first_hybrid:
                attn.q_vocab = first_hybrid.q_vocab
                attn.k_vocab = first_hybrid.k_vocab
            if self.config.vocab_share_qk:
                attn.k_vocab = attn.q_vocab
            attn.q_vocab._no_weight_decay = True
            attn.k_vocab._no_weight_decay = True

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._tie_vocabularies()
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._tie_vocabularies()
        return result

    @torch.no_grad()
    def init_weights(self):
        initialized_weights = set()
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding, CausalShortConv1d, FusedLinearRMSNormRoPE)):
                if id(m.weight) not in initialized_weights:
                    nn.init.normal_(m.weight, std=.02)
                    initialized_weights.add(id(m.weight))
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, CausalShortConv1d):
                nn.init.uniform_(m.conv_weight, -1/math.sqrt(m.kernel_size), 1/math.sqrt(m.kernel_size))
            if isinstance(m, FusedLinearRMSNormRoPE):
                nn.init.ones_(m.rms_weight)
            if isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)
        initialized_tables = set()
        for layer in self.layers:
            attn = layer.attn
            if getattr(attn,"is_pure_gdn",False):
                attn.norm.weight.fill_(1)
                attn.init_gdn_parameters()
                continue
            for table in (attn.q_vocab, attn.k_vocab):
                # Consume the original per-head RNG stream, but initialize each shared parameter once.
                draw = F.silu(torch.randn((attn.heads, attn.q_vocab.shape[-2], attn.head_dim),
                                          device=table.device, dtype=table.dtype))
                if id(table) not in initialized_tables:
                    table.copy_(draw[0] if table.ndim == 2 else draw)
                    initialized_tables.add(id(table))
            attn.log_sel_tau.uniform_(-4, 4)
            attn.norm.weight.fill_(1)
            if hasattr(attn, "init_gdn_parameters"):
                attn.init_gdn_parameters()
        self.training_step.zero_()
        self.rng_counter.zero_()
        self.hard_probability.zero_()
        if getattr(self.config, 'value_residual', False):
            gate_bias = float(getattr(self.config, 'value_residual_gate_bias', 2.0))
            for module in self.modules():
                if getattr(module, '_value_residual_gate', False):
                    nn.init.constant_(module.bias, gate_bias)

    @torch.no_grad()
    def set_training_step(self, step):
        self.training_step.fill_(step)
        progress = min(max(step / max(1, self.config.anneal_steps-1), 0.), 1.)
        progress = min(1., max(0., (progress-self.config.hard_start_fraction) / (self.config.hard_end_fraction-self.config.hard_start_fraction)))
        self.hard_probability.fill_(self.config.hard_prob_max * progress)

    def get_device(self):
        return self.embedding.weight.device

    def num_scaling_params(self):
        embedding = self.embedding.weight.numel()
        head = self.lm_head.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        return dict(embedding=embedding, lm_head=head, layers=total-embedding-head, total=total,
                    matched_total=total)

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

    def forward(self, input_ids, targets=None, *, cu_seqlens=None, segment_ids=None, loss_reduction='mean', hard_mode=None, paired_counter=None, direction_flip=False):
        batch, length = input_ids.shape
        if length > self.config.sequence_len or cu_seqlens is None:
            raise ValueError('DISM nanochat adapter requires packed varlen metadata and bounded context')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = self.embedding(input_ids.reshape(1, -1))
            if self.training:
                # Copy before mutation: saved random identity belongs to this
                # microbatch, not a mutable counter read during backward.
                if paired_counter is None:
                    counter = self.rng_counter.clone()
                    self.rng_counter.add_(1)
                else:
                    counter = paired_counter
            v_first = None
            for i, layer in enumerate(self.layers):
                if getattr(layer.attn,"is_pure_gdn",False):
                    if self.config.value_residual and v_first is None:
                        v_first = _gdn_value_first(layer.attn, x, cu_seqlens, self.config.sequence_len)
                        x=layer(x,cu_seqlens=cu_seqlens,max_seqlen=self.config.sequence_len,use_cache=False)
                    elif self.config.value_residual:
                        x=layer(x,cu_seqlens=cu_seqlens,max_seqlen=self.config.sequence_len,use_cache=False,v_first=v_first)
                    else:
                        x=layer(x,cu_seqlens=cu_seqlens,max_seqlen=self.config.sequence_len,use_cache=False)
                    continue
                if self.training:
                    hard, direction = sample_flags(counter, self.hard_probability,
                        batch*length, self.config.n_head, self.config.rng_seed + i*104729)
                else:
                    hard = torch.full((1, self.config.n_head, batch*length), self.eval_hard, device=x.device, dtype=torch.bool)
                    direction = torch.ones((1, self.config.n_head), device=x.device, dtype=torch.bool)
                if self.training:
                    from .routing import direction_for_policy
                    direction = direction_for_policy(direction, counter, self.training_step,
                        i-self.config.n_layer//2, self.config.direction_policy,
                        self.config.direction_freeze_step, direction_flip)
                    if hard_mode is None and self.config.hard_granularity == 'sequence':
                        # One hard flag per original packed training row and head.
                        row_hard, _ = sample_flags(counter, self.hard_probability,
                            batch, self.config.n_head, self.config.rng_seed+i*104729)
                        hard = row_hard.unsqueeze(-1).expand(1,self.config.n_head,batch,length).reshape(1,self.config.n_head,batch*length).contiguous()
                if self.training and hard_mode is not None:
                    hard = torch.full_like(hard, hard_mode)
                if self.config.value_residual:
                    x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=self.config.sequence_len,
                              hard=hard, direction=direction, use_cache=False, v_first=v_first)
                else:
                    x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=self.config.sequence_len,
                              hard=hard, direction=direction, use_cache=False)
            x = self.norm(x).to(torch.bfloat16)
            if targets is not None and loss_reduction == 'mean':
                # nanochat targets are ALREADY shifted, unlike HF labels.
                with torch.autocast('cuda', enabled=False):
                    return fused_cross_entropy(x, self.lm_head.weight.to(torch.bfloat16), targets.reshape(1, -1),
                                               ignore_index=-1, softcap=self.config.softcap)
            if targets is not None and loss_reduction == 'none' and not torch.is_grad_enabled():
                # BPB needs unreduced token losses, not document means. The
                # fused training kernel already computes these without logits.
                with torch.autocast('cuda', enabled=False):
                    return fused_cross_entropy_unreduced(
                        x, self.lm_head.weight.to(torch.bfloat16), targets.reshape(1, -1),
                        ignore_index=-1, softcap=self.config.softcap).flatten()
            logits = self.lm_head(x).float()
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
            if targets is None:
                return logits.reshape(batch, length, -1)
            return F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1),
                                   ignore_index=-1, reduction=loss_reduction)
