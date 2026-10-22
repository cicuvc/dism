"""nanochat adapter for the selected DISM-GDN pretraining architecture."""
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
from flash_dism.kernels.fused_cross_entropy import fused_cross_entropy_unreduced
from flash_dism.kernels.fused_ce_chunked import fused_ce_chunked, fused_ce_chunked_sum
from fla.modules import ShortConvolution

from .conv import (
    LinearShortConvSiLU,
    MXFP8LinearShortConvSiLU,
    packed_conv_sequence_ids,
)


def packed_document_ce(hidden, weight, targets, cu_seqlens, max_seqlen, softcap,
                       reduction="mean"):
    """Chunked CE with one equal-weight contribution per nonempty document."""
    if reduction not in ("mean", "document_sum"):
        raise ValueError("loss_reduction must be mean or document_sum")
    entry = fused_ce_chunked_sum if reduction == "document_sum" else fused_ce_chunked
    return entry(
        hidden,
        weight.to(torch.bfloat16),
        targets.reshape(1, -1),
        ignore_index=-1,
        softcap=softcap,
        cu_seqlens=cu_seqlens.to(device=hidden.device).contiguous(),
        max_seqlen=max_seqlen,
    )


@torch.compiler.disable
def _mixer_value_first(attn, x, cu_seqlens, max_seqlen, conv_seq_idx=None):
    """Value vector of the first recurrent mixer, reused by later DISM layers."""
    if hasattr(attn, "mixer_value_first"):
        return attn.mixer_value_first(
            x,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            conv_seq_idx=conv_seq_idx,
        )
    conv = attn.gdn_v_conv
    options = {"seq_idx": conv_seq_idx} if conv.supports_seq_idx else {}
    return conv(
        x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **options
    ).unflatten(-1, (attn.heads, attn.value_dim))


def _replace_convs_with_linear_fla(model, *, mxfp8=False):
    """Install the selected linear+depthwise FLA convolution frontend."""
    conv_type = MXFP8LinearShortConvSiLU if mxfp8 else LinearShortConvSiLU
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, CausalShortConv1d):
                setattr(
                    parent,
                    name,
                    conv_type(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        "cuda",
                    ),
                )
    for module in model.modules():
        if getattr(module, "gdn_q_conv", None) is not None:
            module.share_qk_linear = True


@triton.jit
def _hard_flags(hard, counter, probability, N: tl.constexpr, H: tl.constexpr,
                SEED: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    seed = tl.load(counter).to(tl.uint64) + SEED
    tl.store(
        hard + offsets,
        tl.rand(seed, offsets) < tl.load(probability),
        offsets < N * H,
    )


@torch.library.custom_op("nanochat_dism::hard_flags", mutates_args=())
def sample_hard_flags(counter: torch.Tensor, probability: torch.Tensor, n: int,
                      heads: int, seed: int) -> torch.Tensor:
    hard = torch.empty((1, heads, n), device=counter.device, dtype=torch.bool)
    _hard_flags[(triton.cdiv(n * heads, 256),)](
        hard, counter, probability, n, heads, seed, 256
    )
    return hard


@sample_hard_flags.register_fake
def _fake_hard_flags(counter, probability, n, heads, seed):
    return counter.new_empty((1, heads, n), dtype=torch.bool)


@triton.jit
def _hard_flags_selectors(hard, direction, tau, log_tau, counter, probability,
                          N: tl.constexpr, H: tl.constexpr,
                          SEED: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N * H
    seed = tl.load(counter).to(tl.uint64) + SEED
    row, head = offsets // H, offsets % H
    # Preserve the historical sampler's [head, row] RNG coordinates and flags
    # while writing the compact [row, head] layout directly.
    rng_index = head * N + row
    tl.store(
        hard + offsets,
        tl.rand(seed, rng_index) < tl.load(probability),
        mask,
    )
    head_mask = offsets < H
    tl.store(direction + offsets, True, head_mask)
    value = tl.load(log_tau + offsets, mask=head_mask, other=0.0)
    tl.store(tau + offsets, tl.log(1.0 + tl.exp(value)), head_mask)


@torch.library.custom_op(
    "nanochat_dism::hard_flags_selectors", mutates_args=()
)
def sample_hard_flags_selectors(
    counter: torch.Tensor,
    probability: torch.Tensor,
    log_tau: torch.Tensor,
    n: int,
    heads: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hard = torch.empty((n, heads), device=counter.device, dtype=torch.bool)
    direction = torch.empty((1, heads), device=counter.device, dtype=torch.bool)
    tau = torch.empty_like(log_tau)
    _hard_flags_selectors[(triton.cdiv(n * heads, 256),)](
        hard,
        direction,
        tau,
        log_tau,
        counter,
        probability,
        n,
        heads,
        seed,
        256,
    )
    return hard, direction, tau


@sample_hard_flags_selectors.register_fake
def _fake_hard_flags_selectors(counter, probability, log_tau, n, heads, seed):
    return (
        counter.new_empty((n, heads), dtype=torch.bool),
        counter.new_empty((1, heads), dtype=torch.bool),
        torch.empty_like(log_tau),
    )


def _hard_flags_selectors_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[2])
    ctx.mark_non_differentiable(output[0], output[1])


def _hard_flags_selectors_backward(ctx, grad_hard, grad_direction, grad_tau):
    (log_tau,) = ctx.saved_tensors
    return (
        None,
        None,
        grad_tau * torch.sigmoid(log_tau),
        None,
        None,
        None,
    )


sample_hard_flags_selectors.register_autograd(
    _hard_flags_selectors_backward,
    setup_context=_hard_flags_selectors_setup,
)


class DismLM(nn.Module):
    aligned_segment_ends = True
    exact_optimizer_hyperparameters = True
    dynamic_compile = True
    document_balanced_loss = True
    count_normalized_loss = True
    loss_sum_reduction = "document_sum"
    loss_normalization = "global_valid_document_mean_v1"
    eval_hard = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = FlashConfig(
            hidden_size=config.n_embd,
            num_hidden_layers=config.n_layer,
            num_heads=config.n_head,
            head_dim=config.head_dim,
            value_dim=config.value_dim,
            readout_dim=config.readout_dim,
            qk_vocab_size=config.qk_vocab_size,
            intermediate_size=config.intermediate_size,
            attention_type="hybrid_gdn",
            vocab_size=config.vocab_size,
            bos_token_id=1,
            eos_token_id=2,
            value_residual=True,
            vocab_temperature=config.vocab_temperature,
            post_norm=True,
            mxfp8=config.mxfp8,
            mxfp8_projections=config.mxfp8_projections,
        )
        core_dtype = torch.bfloat16 if config.mxfp8 else None
        self.embedding = nn.Embedding(
            config.vocab_size, config.n_embd, dtype=core_dtype
        )
        blocks = []
        for i in range(config.n_layer):
            layer_config = c
            if i < config.n_layer // 2:
                import copy
                layer_config = copy.copy(c)
                layer_config.attention_type = "pure_gdn"
            blocks.append(DismBlock(layer_config, i))
        self.layers = nn.ModuleList(blocks)
        if not getattr(self.layers[0].attn, "is_pure_mixer", False):
            raise ValueError("value residual requires a pure GDN first layer")
        _replace_convs_with_linear_fla(
            self,
            mxfp8=config.mxfp8 and config.mxfp8_projections,
        )
        self._mark_no_decay()
        self.norm = nn.RMSNorm(config.n_embd, eps=1e-6)
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False, dtype=core_dtype
        )
        self.register_buffer("training_step", torch.zeros((), dtype=torch.int64))
        self.register_buffer("rng_counter", torch.zeros((), dtype=torch.int64))
        self.register_buffer("hard_probability", torch.zeros((), dtype=torch.float32))
        self.vocab_temperature = float(config.vocab_temperature)

    def _mark_no_decay(self):
        for layer in self.layers:
            attn = layer.attn
            if getattr(attn, "is_pure_mixer", False):
                continue
            if hasattr(attn, "tie_branch_projections"):
                attn.tie_branch_projections()
            attn.q_vocab._no_weight_decay = True
            attn.k_vocab._no_weight_decay = True

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._mark_no_decay()
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._mark_no_decay()
        if self.config.mxfp8 and self.get_device().type == "cuda":
            from flash_mxfp8 import refresh_fp8_modules
            refresh_fp8_modules(self)
        return result

    @torch.no_grad()
    def init_weights(self):
        initialized_weights = set()
        for module in self.modules():
            if isinstance(module, ShortConvolution):
                module.reset_parameters()
            if isinstance(
                module,
                (
                    nn.Linear,
                    nn.Embedding,
                    CausalShortConv1d,
                    FusedLinearRMSNormRoPE,
                    LinearShortConvSiLU,
                ),
            ):
                if id(module.weight) not in initialized_weights:
                    nn.init.normal_(module.weight, std=0.02)
                    initialized_weights.add(id(module.weight))
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, CausalShortConv1d):
                bound = 1 / math.sqrt(module.kernel_size)
                nn.init.uniform_(module.conv_weight, -bound, bound)
            if isinstance(module, LinearShortConvSiLU):
                bound = 1 / math.sqrt(module.kernel_size)
                nn.init.uniform_(module.conv.weight, -bound, bound)
            if isinstance(module, FusedLinearRMSNormRoPE):
                nn.init.ones_(module.rms_weight)
            if isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

        for layer in self.layers:
            attn = layer.attn
            if getattr(attn, "is_pure_mixer", False):
                attn.norm.weight.fill_(1)
                if hasattr(attn, "init_gdn_parameters"):
                    attn.init_gdn_parameters()
                continue
            for table in (attn.q_vocab, attn.k_vocab):
                draw = F.silu(
                    torch.randn(
                        (attn.heads, table.shape[-2], attn.head_dim),
                        device=table.device,
                        dtype=table.dtype,
                    )
                )
                table.copy_(draw[: table.shape[0]])
            attn.log_sel_tau.uniform_(-4, 4)
            attn.norm.weight.fill_(1)
            attn.init_gdn_parameters()

        self.training_step.zero_()
        self.rng_counter.zero_()
        self.hard_probability.zero_()
        self.vocab_temperature = float(self.config.vocab_temperature)
        for layer in self.layers:
            if hasattr(layer.attn, "vocab_temperature"):
                layer.attn.vocab_temperature = self.vocab_temperature
        gate_bias = float(self.config.value_residual_gate_bias)
        for module in self.modules():
            if getattr(module, "_value_residual_gate", False):
                nn.init.constant_(module.bias, gate_bias)
        if self.config.mxfp8:
            from flash_mxfp8 import MXFP8Linear, MXFP8SwiGLU, refresh_fp8_modules
            from flash_mxfp8.tk_mxfp8_down import MXFP8DownProjection

            initialized_mxfp8 = set(initialized_weights)
            for module in self.modules():
                if isinstance(module, MXFP8SwiGLU):
                    tensors = (
                        module.gate_weight,
                        module.up_weight,
                        module.down_proj.weight,
                    )
                elif isinstance(module, (MXFP8Linear, MXFP8DownProjection)):
                    tensors = (module.weight,)
                else:
                    continue
                for parameter in tensors:
                    if id(parameter) not in initialized_mxfp8:
                        nn.init.normal_(parameter, std=0.02)
                        initialized_mxfp8.add(id(parameter))
            if self.get_device().type == "cuda":
                refresh_fp8_modules(self)

    @torch.no_grad()
    def set_training_step(self, step):
        self.training_step.fill_(step)
        progress = min(max(step / max(1, self.config.anneal_steps - 1), 0.0), 1.0)
        self.hard_probability.fill_(self.config.hard_prob_max * progress)
        self.vocab_temperature = float(self.config.vocab_temperature)

        frac = self.config.tau_schedule_fraction
        floor = self.config.tau_schedule_floor
        tau_scale = (
            floor + (1.0 - floor) * min(1.0, progress / frac)
            if frac > 0.0
            else 1.0
        )
        if tau_scale != 1.0 or getattr(self, "_tau_scale_last", None) != 1.0:
            for layer in getattr(self, "layers", ()):
                attn = getattr(layer, "attn", None)
                if attn is not None and hasattr(attn, "tau_scale"):
                    attn.tau_scale = tau_scale
        self._tau_scale_last = tau_scale

    @torch.no_grad()
    def set_vocab_temperature(self, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("vocab_temperature must be finite and positive")
        self.vocab_temperature = value

    def get_device(self):
        return self.embedding.weight.device

    def num_scaling_params(self):
        embedding = self.embedding.weight.numel()
        head = self.lm_head.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        return dict(
            embedding=embedding,
            lm_head=head,
            layers=total - embedding - head,
            total=total,
            matched_total=total,
        )

    def scaling_parameter_count(self):
        return self.num_scaling_params()["total"] - self.embedding.weight.numel()

    def estimate_flops(self):
        return (
            6 * self.scaling_parameter_count()
            + 12
            * self.config.n_layer
            * self.config.n_head
            * 64
            * self.config.sequence_len
        )

    def setup_pretraining_optimizer(self, config):
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            exempt = (
                parameter.ndim < 2
                or name.endswith((".bias", "q_vocab", "k_vocab", "log_sel_tau", "rms_weight"))
                or getattr(parameter, "_no_weight_decay", False)
            )
            (no_decay if exempt else decay).append(parameter)
        groups = [
            dict(params=decay, weight_decay=config.weight_decay, kind="adamw"),
            dict(params=no_decay, weight_decay=0.0, kind="adamw"),
        ]
        if self.config.mxfp8_optimizer:
            from flash_mxfp8 import FusedAdamWMXFP8, bind_fp8_optimizer

            optimizer = FusedAdamWMXFP8(
                groups,
                lr=config.matrix_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
            bind_fp8_optimizer(self, optimizer)
        else:
            from flash_mxfp8 import bind_fp8_optimizer

            optimizer = torch.optim.AdamW(
                groups,
                lr=config.matrix_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=self.get_device().type == "cuda",
            )
            if self.config.mxfp8:
                bind_fp8_optimizer(self, optimizer)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(
        self,
        input_ids,
        targets=None,
        *,
        cu_seqlens=None,
        segment_ids=None,
        loss_reduction="mean",
        graph_layout=None,
        gdn_chunk_indices=None,
    ):
        del segment_ids
        batch, length = input_ids.shape
        if length > self.config.sequence_len or cu_seqlens is None:
            raise ValueError(
                "DISM nanochat adapter requires packed varlen metadata and bounded context"
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # FLA caches chunk metadata by boundary identity. Each microbatch
            # owns a fresh identity so a reused loader buffer cannot alias it.
            cu_seqlens = cu_seqlens.to(device=input_ids.device).clone()
            x = self.embedding(input_ids.reshape(1, -1))
            conv_seq_idx = packed_conv_sequence_ids(
                cu_seqlens, batch * length, x.device
            )
            if self.training:
                counter = self.rng_counter.clone()
                self.rng_counter.add_(1)

            v_first = None
            for i, layer in enumerate(self.layers):
                if getattr(layer.attn, "is_pure_mixer", False):
                    if v_first is None:
                        v_first = _mixer_value_first(
                            layer.attn,
                            x,
                            cu_seqlens,
                            self.config.sequence_len,
                            conv_seq_idx,
                        )
                    x = layer(
                        x,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=self.config.sequence_len,
                        use_cache=False,
                        v_first=v_first,
                        conv_seq_idx=conv_seq_idx,
                        gdn_chunk_indices=gdn_chunk_indices,
                    )
                    continue

                if self.training:
                    hard, direction, tau = sample_hard_flags_selectors(
                        counter,
                        self.hard_probability,
                        layer.attn.log_sel_tau,
                        batch,
                        self.config.n_head,
                        self.config.rng_seed + i * 104729,
                    )
                else:
                    hard = torch.full(
                        (batch, self.config.n_head),
                        self.eval_hard,
                        device=x.device,
                        dtype=torch.bool,
                    )
                    direction = torch.ones(
                        (1, self.config.n_head),
                        device=x.device,
                        dtype=torch.bool,
                    )
                    tau = None
                x = layer(
                    x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=self.config.sequence_len,
                    hard=hard,
                    direction=direction,
                    tau_override=tau,
                    use_cache=False,
                    v_first=v_first,
                    vocab_temperature=self.vocab_temperature,
                    conv_seq_idx=conv_seq_idx,
                    direction_true=True,
                    _share_branch_boundaries=True,
                    graph_layout=graph_layout,
                    gdn_chunk_indices=gdn_chunk_indices,
                )

            x = self.norm(x).to(torch.bfloat16)
            if targets is not None and loss_reduction in ("mean", "document_sum"):
                with torch.autocast("cuda", enabled=False):
                    return packed_document_ce(
                        x,
                        self.lm_head.weight,
                        targets,
                        cu_seqlens,
                        self.config.sequence_len,
                        self.config.softcap,
                        loss_reduction,
                    )
            if (
                targets is not None
                and loss_reduction == "none"
                and not torch.is_grad_enabled()
            ):
                with torch.autocast("cuda", enabled=False):
                    return fused_cross_entropy_unreduced(
                        x,
                        self.lm_head.weight.to(torch.bfloat16),
                        targets.reshape(1, -1),
                        ignore_index=-1,
                        softcap=self.config.softcap,
                    ).flatten()

            logits = self.lm_head(x).float()
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
            if targets is None:
                return logits.reshape(batch, length, -1)
            raise ValueError("unsupported differentiable loss_reduction")
