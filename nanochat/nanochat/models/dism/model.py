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
from .conv import LinearShortConvSiLU, packed_conv_sequence_ids
from flash_dism.kernels.fused_cross_entropy import fused_cross_entropy_unreduced
from flash_dism.kernels.fused_ce_chunked import fused_ce_chunked, fused_ce_chunked_sum


def packed_document_ce(hidden, weight, targets, cu_seqlens, max_seqlen, softcap,
                       reduction='mean', balance='document'):
    """Targets are already shifted; boundaries define loss documents, not hard rows."""
    if balance not in ('document', 'token'):
        raise ValueError('loss_balance must be document or token')
    entry = fused_ce_chunked_sum if reduction in ('document_sum', 'token_sum') else fused_ce_chunked
    return entry(hidden, weight.to(torch.bfloat16), targets.reshape(1, -1),
                 ignore_index=-1, softcap=softcap,
                 cu_seqlens=(cu_seqlens.to(device=hidden.device).contiguous()
                             if balance == 'document' else None),
                 max_seqlen=max_seqlen if balance == 'document' else None)


@torch.compiler.disable
def _gdn_value_first(attn, x, cu_seqlens, max_seqlen, conv_seq_idx=None):
    """Value vector of the first GDN mixer, reused by later layers (value residual)."""
    options = {'seq_idx': conv_seq_idx} if getattr(attn.gdn_v_conv, 'supports_seq_idx', False) else {}
    return attn.gdn_v_conv(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **options).unflatten(
        -1, (attn.heads, attn.value_dim))


def _replace_convs_with_linear_fla(model, backend):
    """Install the checkpoint-compatible linear + depthwise frontend."""
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, CausalShortConv1d):
                setattr(parent, name, LinearShortConvSiLU(
                    child.in_channels, child.out_channels, child.kernel_size, backend))
    for module in model.modules():
        if getattr(module, "gdn_q_conv", None) is not None and getattr(module, "q_conv", None) is not None:
            # DISM/GDN share the tied q/k input projection; compute it once if tied.
            module.share_qk_linear = True


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
    document_balanced_loss = True
    count_normalized_loss = True

    def __init__(self, config, *, attention_type='hybrid'):
        super().__init__()
        self.config = config
        if config.loss_balance not in ('document', 'token'):
            raise ValueError('loss_balance must be document or token')
        self.document_balanced_loss = config.loss_balance == 'document'
        self.loss_sum_reduction = config.loss_balance + '_sum'
        self.loss_normalization = 'global_valid_' + config.loss_balance + '_mean_v1'
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
                        vocab_temperature=getattr(config, 'vocab_temperature', 1.0),
                        vocab_transvq=getattr(config, "vocab_transvq", False),
                        vocab_transvq_lite=getattr(config, "vocab_transvq_lite", False),
                        vocab_transvq_map=getattr(config, "vocab_transvq_map", "linear_attn"),
                        vocab_transvq_rank=getattr(config, "vocab_transvq_rank", 4),
                        post_norm=config.post_norm)
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        if getattr(config, 'layer_layout', ''):
            # Dedicated-layer layout: no layer mixes DISM and GDN internally.
            spec = {'g': 'pure_gdn', 'd': 'dism', 'h': 'hybrid_gdn'}
            layout = str(config.layer_layout)
            if attention_type != 'hybrid_gdn':
                raise ValueError('layer_layout requires the hybrid_gdn architecture')
            if len(layout) != config.n_layer or any(ch not in spec for ch in layout):
                raise ValueError('layer_layout must have n_layer entries drawn from g/d/h')
            blocks = []
            for i, ch in enumerate(layout):
                layer_config = copy.copy(c)
                layer_config.attention_type = spec[ch]
                blocks.append(DismBlock(layer_config, i))
            self.layers = nn.ModuleList(blocks)
        elif config.alternating_gdn or config.rear_half_dism:
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
        if config.conv_impl not in ("fused", "linear_fla"):
            raise ValueError(f"unsupported conv_impl {config.conv_impl!r}")
        if config.conv_impl == "linear_fla":
            _replace_convs_with_linear_fla(self, config.conv_backend)
        self._tie_vocabularies()
        self.norm = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.register_buffer('training_step', torch.zeros((), dtype=torch.int64))
        self.register_buffer('rng_counter', torch.zeros((), dtype=torch.int64))
        self.register_buffer('hard_probability', torch.zeros((), dtype=torch.float32))
        # Host schedule value, reconstructed from config + restored training step.
        self.vocab_temperature = float(getattr(config, 'vocab_temperature', 1.0))

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
                # Share the base table but keep separate q/k maps if requested
                # (C_q = H_q C, C_k = H_k C for different drift directions).
                if (getattr(attn, "q_vocab_map", None) is not None
                        and not getattr(self.config, "vocab_transvq_separate_qk", False)):
                    attn.k_vocab_map = attn.q_vocab_map
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
            if isinstance(m, (nn.Linear, nn.Embedding, CausalShortConv1d, FusedLinearRMSNormRoPE, LinearShortConvSiLU)):
                if id(m.weight) not in initialized_weights:
                    nn.init.normal_(m.weight, std=.02)
                    initialized_weights.add(id(m.weight))
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, CausalShortConv1d):
                nn.init.uniform_(m.conv_weight, -1/math.sqrt(m.kernel_size), 1/math.sqrt(m.kernel_size))
            if isinstance(m, LinearShortConvSiLU):
                nn.init.uniform_(m.conv.weight, -1/math.sqrt(m.kernel_size), 1/math.sqrt(m.kernel_size))
            if isinstance(m, FusedLinearRMSNormRoPE):
                nn.init.ones_(m.rms_weight)
            if isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
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
        # Temperature is host-only and absent from state_dict. Initialization
        # restores its config value; the training loop reapplies the step schedule.
        default_beta = float(getattr(self.config, 'vocab_temperature', 1.0))
        self.vocab_temperature = default_beta
        for layer in self.layers:
            attn = layer.attn
            if hasattr(attn, 'vocab_temperature'):
                attn.vocab_temperature = default_beta
        if getattr(self.config, 'value_residual', False):
            gate_bias = float(getattr(self.config, 'value_residual_gate_bias', 2.0))
            for module in self.modules():
                if getattr(module, '_value_residual_gate', False):
                    nn.init.constant_(module.bias, gate_bias)
        if getattr(self.config, 'vocab_transvq', False):
            # Initialize the map's exit RMSNorm elementwise factor so the transformed
            # codebook RMS matches the base codebook RMS instead of being normalized to 1.
            for layer in self.layers:
                attn = layer.attn
                if getattr(attn, 'q_vocab_map', None) is None:
                    continue
                base_rms = attn.q_vocab.detach().float().pow(2).mean().sqrt()
                for mapper in (attn.q_vocab_map, getattr(attn, 'k_vocab_map', None)):
                    if mapper is None:
                        continue
                    if hasattr(mapper, 'u'):
                        nn.init.normal_(mapper.u, std=0.02)
                        nn.init.normal_(mapper.v, std=0.02)
                        nn.init.ones_(mapper.s)
                    elif getattr(mapper, 'kind', None) == 'diagonal':
                        nn.init.ones_(mapper.s)
                    elif getattr(mapper, 'kind', None) == 'mean_shift':
                        nn.init.zeros_(mapper.m)
                    if getattr(mapper, 'out_norm', None) is not None:
                        mapper.out_norm.weight.fill_(base_rms)

    @torch.no_grad()
    def set_training_step(self, step):
        self.training_step.fill_(step)
        progress = min(max(step / max(1, self.config.anneal_steps-1), 0.), 1.)
        hard_progress = min(1., max(0., (progress-self.config.hard_start_fraction) / (self.config.hard_end_fraction-self.config.hard_start_fraction)))
        hp_start = float(getattr(self.config, 'hard_prob_start', 0.0))
        self.hard_probability.fill_(hp_start + (self.config.hard_prob_max - hp_start) * hard_progress)
        # Cosine vocabulary-temperature schedule (start==end keeps it constant).
        beta_start = float(getattr(self.config, 'vocab_temperature_start', 1.0))
        beta_end = float(getattr(self.config, 'vocab_temperature_end', 1.0))
        self.vocab_temperature = beta_start + 0.5 * (beta_end - beta_start) * (1.0 - math.cos(math.pi * progress))

    @torch.no_grad()
    def set_vocab_temperature(self, value):
        """Set the vocabulary-softmax temperature used by the soft DISM branch."""
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('vocab_temperature must be finite and positive')
        self.vocab_temperature = value

    def get_device(self):
        return self.embedding.weight.device

    def num_scaling_params(self):
        embedding = self.embedding.weight.numel()
        head = self.lm_head.weight.numel()
        # The TransVQ codebook map is a training-time reparameterization, not
        # model capacity, so it is excluded from the reported parameter counts.
        transvq_map = sum(p.numel() for name, p in self.named_parameters() if 'vocab_map' in name)
        total = sum(p.numel() for p in self.parameters()) - transvq_map
        return dict(embedding=embedding, lm_head=head, layers=total-embedding-head, total=total,
                    matched_total=total, transvq_map=transvq_map)

    def scaling_parameter_count(self):
        return self.num_scaling_params()['total'] - self.embedding.weight.numel()

    def estimate_flops(self):
        return 6*self.scaling_parameter_count() + 12*self.config.n_layer*self.config.n_head*64*self.config.sequence_len

    def setup_pretraining_optimizer(self, config):
        # Optional selector-only AdamW betas. The DISM selector path is gated:
        # on hard rows its gradient is exactly zero for q_vocab/k_vocab and the
        # DISM short-conv filters, so those tensors see fewer active rows and a
        # noisier gradient. A higher selector beta1 (longer momentum) reduces
        # that noise. Disabled when selector_beta1 <= 0.
        sel_b1 = float(getattr(self.config, 'selector_beta1', 0.0) or 0.0)
        sel_b2 = float(getattr(self.config, 'selector_beta2', 0.0) or 0.0)
        use_sel = sel_b1 > 0.0
        decay, no_decay, map_decay, map_no_decay = [], [], [], []
        sel_decay, sel_no_decay = [], []
        for name, p in self.named_parameters():
            if 'vocab_map' in name:
                # TransVQ map is a training-time reparameterization; lock its LR to
                # 1/5 of the main matrix LR to keep the codebook moving slowly.
                (map_no_decay if p.ndim < 2 else map_decay).append(p)
                continue
            exempt = (p.ndim < 2 or name.endswith(('.bias', 'q_vocab', 'k_vocab', 'log_sel_tau', 'rms_weight'))
                      or getattr(p, '_no_weight_decay', False))
            is_selector = (name.endswith(('q_vocab', 'k_vocab'))
                           or name.endswith(('.q_conv.conv_weight', '.k_conv.conv_weight')))
            if use_sel and is_selector:
                (sel_no_decay if exempt else sel_decay).append(p)
                continue
            (no_decay if exempt else decay).append(p)
        groups = [dict(params=decay, weight_decay=config.weight_decay, kind='adamw'),
                  dict(params=no_decay, weight_decay=0., kind='adamw')]
        if map_decay or map_no_decay:
            groups += [dict(params=map_decay, weight_decay=config.weight_decay, kind='adamw', lr_scale=0.2),
                       dict(params=map_no_decay, weight_decay=0., kind='adamw', lr_scale=0.2)]
        if use_sel:
            sel_betas = (sel_b1, sel_b2 if sel_b2 > 0 else 0.95)
            if sel_decay:
                groups.append(dict(params=sel_decay, weight_decay=config.weight_decay,
                                   kind='adamw', betas=sel_betas))
            if sel_no_decay:
                groups.append(dict(params=sel_no_decay, weight_decay=0.,
                                   kind='adamw', betas=sel_betas))
        optimizer = torch.optim.AdamW(groups, lr=config.matrix_lr, betas=(.9, .95), eps=1e-8,
                                     fused=self.get_device().type == 'cuda')
        for g in optimizer.param_groups:
            if 'lr_scale' in g:
                g['lr'] = config.matrix_lr * g['lr_scale']
            g['initial_lr'] = g['lr']
        return optimizer

    def forward(self, input_ids, targets=None, *, cu_seqlens=None, segment_ids=None, loss_reduction='mean', hard_mode=None, paired_counter=None, direction_flip=False):
        batch, length = input_ids.shape
        if length > self.config.sequence_len or cu_seqlens is None:
            raise ValueError('DISM nanochat adapter requires packed varlen metadata and bounded context')
        if self.config.hard_granularity != 'sequence':
            raise ValueError('CUDA DISM supports only sequence-level hard flags')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = self.embedding(input_ids.reshape(1, -1))
            conv_seq_idx = (packed_conv_sequence_ids(cu_seqlens, batch*length, x.device)
                            if self.config.conv_impl == 'linear_fla' and self.config.conv_backend == 'cuda'
                            else None)
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
                        v_first = _gdn_value_first(layer.attn, x, cu_seqlens, self.config.sequence_len, conv_seq_idx)
                        x=layer(x,cu_seqlens=cu_seqlens,max_seqlen=self.config.sequence_len,use_cache=False,
                                conv_seq_idx=conv_seq_idx)
                    elif self.config.value_residual:
                        x=layer(x,cu_seqlens=cu_seqlens,max_seqlen=self.config.sequence_len,use_cache=False,v_first=v_first,
                                conv_seq_idx=conv_seq_idx)
                    else:
                        x=layer(x,cu_seqlens=cu_seqlens,max_seqlen=self.config.sequence_len,use_cache=False,
                                conv_seq_idx=conv_seq_idx)
                    continue
                if self.training:
                    hard, direction = sample_flags(counter, self.hard_probability,
                        batch*length, self.config.n_head, self.config.rng_seed + i*104729)
                else:
                    hard = torch.full((batch, self.config.n_head), self.eval_hard, device=x.device, dtype=torch.bool)
                    direction = torch.ones((1, self.config.n_head), device=x.device, dtype=torch.bool)
                if self.training:
                    from .routing import direction_for_policy
                    direction = direction_for_policy(direction, counter, self.training_step,
                        i-self.config.n_layer//2, self.config.direction_policy,
                        self.config.direction_freeze_step, direction_flip)
                    if self.config.hard_granularity == 'sequence':
                        # One hard flag per original packed training row and head.
                        row_hard, _ = sample_flags(counter, self.hard_probability,
                            batch, self.config.n_head, self.config.rng_seed+i*104729)
                        hard = row_hard.squeeze(0).T.contiguous()
                if self.training and hard_mode is not None:
                    hard = torch.full_like(hard, hard_mode)
                fixed_true = (not self.training or
                              (self.config.direction_policy == 'true' and not direction_flip))
                if self.config.value_residual:
                    x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=self.config.sequence_len,
                              hard=hard, direction=direction, use_cache=False, v_first=v_first,
                              vocab_temperature=self.vocab_temperature, conv_seq_idx=conv_seq_idx,
                              direction_true=fixed_true)
                else:
                    x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=self.config.sequence_len,
                              hard=hard, direction=direction, use_cache=False,
                              vocab_temperature=self.vocab_temperature, conv_seq_idx=conv_seq_idx,
                              direction_true=fixed_true)
            x = self.norm(x).to(torch.bfloat16)
            if targets is not None and loss_reduction == 'reinforce':
                # Optionally-sampled DISM path: differentiable mean NLL plus a
                # detached per-token NLL used as the score-function reward/baseline.
                with torch.autocast('cuda', enabled=False):
                    direct = packed_document_ce(x, self.lm_head.weight, targets,
                        cu_seqlens, self.config.sequence_len, self.config.softcap,
                        balance=self.config.loss_balance)
                    with torch.no_grad():
                        tokens = fused_cross_entropy_unreduced(
                            x, self.lm_head.weight.to(torch.bfloat16), targets.reshape(1, -1),
                            ignore_index=-1, softcap=self.config.softcap).flatten()
                    return direct, tokens
            if targets is not None and loss_reduction in ('mean', 'document_sum', 'token_sum'):
                # nanochat targets are ALREADY shifted, unlike HF labels.
                with torch.autocast('cuda', enabled=False):
                    return packed_document_ce(x, self.lm_head.weight, targets,
                        cu_seqlens, self.config.sequence_len, self.config.softcap, loss_reduction,
                        balance=self.config.loss_balance)
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
