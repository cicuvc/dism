"""PreNorm DISM/hybrid decoder and fused-CE language-model head (SBA-style API)."""
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_dism import DismConfig
from .module import DismAttention
from .hybrid import DismSwaAttention
from .module_v4 import DismV4Attention, DismV4SwaAttention
from .cache import DismCache
from .kernels.conv1d import CausalShortConv1d
from .kernels.linear_rmsnorm_rope import FusedLinearRMSNormRoPE


class DismMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DismBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        attention = {'hybrid': DismSwaAttention, 'dism': DismAttention,
                     'hybrid_v4': DismV4SwaAttention, 'dism_v4': DismV4Attention}[config.attention_type]
        options = dict(head_dim=config.head_dim, value_dim=config.value_dim,
                       readout_dim=config.readout_dim, vocab_size=config.qk_vocab_size,
                       conv_size=config.conv_size, rope_theta=config.rope_theta,
                       qknorm_eps=config.qknorm_eps, layer_idx=layer_idx,
                       readout_l2_norm=config.readout_l2_norm,
                       soft_k_l2_norm=config.soft_k_l2_norm)
        if config.attention_type in ('hybrid', 'hybrid_v4'):
            options['window_size'] = config.window_size
        self.attn = attention(config.hidden_size, config.num_heads, **options)
        self.mlp = DismMLP(config)

    def forward(self, hidden_states, **kwargs):
        branch, _, _ = self.attn(self.attn_norm(hidden_states), **kwargs)
        hidden_states = hidden_states + branch
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class DismPreTrainedModel(PreTrainedModel):
    config_class = DismConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['DismBlock']
    _supports_cache_class = True

    def _init_weights(self, module):
        # Do not reset attention modules wholesale: preserve SiLU-Gaussian
        # codebooks, tau initialization and convolution coefficient scales.
        if isinstance(module, (nn.Linear, nn.Embedding, CausalShortConv1d, FusedLinearRMSNormRoPE)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        if isinstance(module, FusedLinearRMSNormRoPE):
            nn.init.ones_(module.rms_weight)
        if isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def optimizer_param_groups(self, weight_decay):
        """Honor DISM no-decay markers, all biases and normalization scales."""
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                # HF checkpoint loading may replace Parameter objects and lose
                # Python-only markers, so retain the semantic exclusions by name.
                marked_name = name.rsplit('.', 1)[-1] in ('q_vocab', 'k_vocab', 'log_sel_tau', 'rms_weight')
                exclude = parameter.ndim < 2 or name.endswith('.bias') or marked_name or getattr(parameter, '_no_weight_decay', False)
                (no_decay if exclude else decay).append(parameter)
        return [{'params': decay, 'weight_decay': weight_decay},
                {'params': no_decay, 'weight_decay': 0.}]


class DismModel(DismPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(DismBlock(config, i) for i in range(config.num_hidden_layers))
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('provide exactly one of input_ids and inputs_embeds')
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        if output_attentions:
            warnings.warn('DISM does not materialize attention weights', stacklevel=2)
        use_cache = (self.config.use_cache and not self.training) if use_cache is None else use_cache
        if self.training and (use_cache or past_key_values is not None):
            raise ValueError('training does not support a decoding cache')
        if use_cache and past_key_values is None:
            past_key_values = DismCache()
        if past_key_values is not None and not (hasattr(past_key_values, 'update') and hasattr(past_key_values, 'get_seq_length')):
            raise TypeError('past_key_values must implement the FLA Cache protocol')
        hidden = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        states = () if output_hidden_states else None
        options = dict(kwargs, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache)
        checkpointing = self.gradient_checkpointing and self.training
        if checkpointing and options.get('hard') is None and options.get('hard_seed') is None:
            options['hard_seed'] = torch.randint(0, torch.iinfo(torch.int64).max, (), device=hidden.device,
                                                dtype=torch.int64, generator=options.get('generator'))
        for layer in self.layers:
            if states is not None:
                states += (hidden,)
            if checkpointing:
                replay = dict(options)
                if replay.get('direction') is None:
                    replay['direction'] = torch.randint(2, (), device=hidden.device,
                        generator=options.get('generator')).bool().expand(hidden.shape[0], self.config.num_heads).contiguous()
                # Bind layer/options now: backward must not capture the loop's last layer.
                def run(value, current=layer, arguments=replay):
                    return current(value, **arguments)
                hidden = checkpoint(run, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden, **options)
        hidden = self.norm(hidden)
        if states is not None:
            states += (hidden,)
        result = BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values,
                                         hidden_states=states, attentions=None)
        return result if return_dict else result.to_tuple()


def next_token_labels(labels, attention_mask=None, cu_seqlens=None, ignore_index=-100):
    """Shift once and exclude document transitions/padding from the objective."""
    shifted = torch.full_like(labels, ignore_index)
    shifted[:, :-1] = labels[:, 1:]
    if attention_mask is not None:
        mask = attention_mask[:, -labels.shape[1]:].to(device=labels.device, dtype=torch.bool)
        valid = torch.zeros_like(mask)
        valid[:, :-1] = mask[:, :-1] & mask[:, 1:]
        shifted.masked_fill_(~valid, ignore_index)
    if cu_seqlens is not None:
        if labels.shape[0] != 1:
            raise ValueError('packed labels require batch=1')
        ends = cu_seqlens[1:].to(device=labels.device, dtype=torch.long)
        shifted[0, ends[ends > 0]-1] = ignore_index
    return shifted.contiguous()


class DismForCausalLM(DismPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = DismModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, value):
        self.model = value

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        if generation_config.use_cache and model_kwargs.get('past_key_values') is None:
            if generation_config.cache_implementation is not None:
                raise ValueError('select the default DISM/FLA cache, not a Transformers KV cache implementation')
            model_kwargs['past_key_values'] = DismCache()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                      inputs_embeds=None, use_cache=True, logits_to_keep=1, **kwargs):
        seen = 0 if past_key_values is None else past_key_values.get_seq_length()
        if seen:
            input_ids = input_ids[:, seen:] if input_ids.shape[1] > seen else input_ids[:, -1:]
        result = dict(past_key_values=past_key_values, attention_mask=attention_mask,
                      use_cache=use_cache, logits_to_keep=logits_to_keep)
        if inputs_embeds is not None and not seen:
            result['inputs_embeds'] = inputs_embeds
        else:
            result['input_ids'] = input_ids.contiguous()
        for key in ('hard_prob', 'hard_seed', 'direction', 'generator'):
            if key in kwargs:
                result[key] = kwargs[key]
        return result

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, logits_to_keep=0,
                return_logits=None, **kwargs):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
            raise ValueError('logits_to_keep must be a nonnegative integer')
        if labels is not None and logits_to_keep:
            raise ValueError('labels require full-sequence logits_to_keep=0')
        outputs = self.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=True, **kwargs)
        hidden = outputs.last_hidden_state
        fused = labels is not None and self.config.fuse_cross_entropy and hidden.is_cuda
        logits = None
        if return_logits is None:
            return_logits = not fused
        if return_logits or (labels is not None and not fused):
            logits = self.lm_head(hidden[:, -logits_to_keep:])
            if self.config.ce_softcap is not None:
                cap = self.config.ce_softcap
                logits = cap * torch.tanh(logits / cap)
        loss = None
        if labels is not None:
            labels = labels.to(hidden.device)
            if labels.shape != hidden.shape[:2] or labels.dtype not in (torch.int32, torch.int64):
                raise ValueError('labels must be integer [B,N] matching hidden states')
            targets = next_token_labels(labels, attention_mask, kwargs.get('cu_seqlens'), self.config.ignore_index)
            if fused:
                from .kernels.fused_cross_entropy import fused_cross_entropy
                # Token-weighted mean even for packed input. The kernel's
                # cu_seqlens mode instead averages document means equally.
                loss = fused_cross_entropy(hidden.to(torch.bfloat16).contiguous(),
                    self.lm_head.weight.to(torch.bfloat16).contiguous(), targets,
                    ignore_index=self.config.ignore_index, chunk_size=self.config.ce_chunk_size,
                    softcap=self.config.ce_softcap)
            else:
                losses = F.cross_entropy(logits.float().flatten(0, 1), targets.long().flatten(),
                                         ignore_index=self.config.ignore_index, reduction='sum')
                loss = losses / (targets != self.config.ignore_index).sum().clamp_min(1)
        if not return_logits:
            logits = None
        result = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                       hidden_states=outputs.hidden_states, attentions=None)
        return result if return_dict else ((loss,) if loss is not None else ()) + (logits,) + outputs.to_tuple()[1:]
