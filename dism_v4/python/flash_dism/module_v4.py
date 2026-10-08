"""Data-dependent v4 diagonal gates, sharing the attention hard-probability schedule."""
import torch
from torch import nn
from torch.nn import functional as F

from .module import DismAttention, _salt_hard_seed
from .hybrid import DismSwaAttention
from .kernels.random_bool import triton_rand_bool


LOG_ZERO = -1e6
# Keep gate hardening independent of vocabulary hardening with the same seed.
_GATE_SEED_SALT = 0x36A9F12B5C708D43


class _V4Gate:
    def __init__(self, width, heads, **kwargs):
        super().__init__(width, heads, **kwargs)
        self.delta_proj = nn.Linear(width, heads)
        nn.init.zeros_(self.delta_proj.bias)

    def gate_delta(self, hidden_states, hard_prob=None, *, delta_hard=None,
                   hard_seed=None, generator=None):
        """Return FP32 [B,H,N] attenuation, using the pre-logsigmoid sign.

        Soft: -logsigmoid(z), z=delta_proj(x). Hard: 0 for z>=0, 1e6
        otherwise. Only soft rows propagate gate gradients; no straight-through
        estimator. The independent gate mask uses the same annealed hard_prob
        as vocabulary hardening. Explicit delta_hard overrides mask sampling.
        Defaults match DISM: soft in train mode, hard in eval mode.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width:
            raise ValueError('hidden_states must have shape [B,N,width]')
        z = self.delta_proj(hidden_states).float().transpose(1, 2)
        probability = (0. if self.training else 1.) if hard_prob is None else float(hard_prob)
        if not 0. <= probability <= 1.:
            raise ValueError('hard_prob must be in [0,1]')
        if delta_hard is not None:
            if (delta_hard.dtype != torch.bool or delta_hard.shape != z.shape
                    or delta_hard.device != z.device):
                raise ValueError('delta_hard must be bool [B,H,N] on the input device')
        elif probability in (0., 1.):
            delta_hard = torch.full_like(z, bool(probability), dtype=torch.bool)
        else:
            seed = _salt_hard_seed(hard_seed, self.layer_idx) if hard_seed is not None else None
            if seed is not None:
                seed = seed ^ _GATE_SEED_SALT
            if z.is_cuda:
                if seed is None:
                    seed = torch.randint(0, torch.iinfo(torch.int64).max, (),
                                         device=z.device, generator=generator)
                delta_hard = triton_rand_bool(z.shape, probability, device=z.device, seed=seed)
            else:
                if seed is not None:
                    generator = torch.Generator(device=z.device).manual_seed(int(seed))
                delta_hard = torch.rand(z.shape, device=z.device, generator=generator) < probability
        soft = -F.logsigmoid(z)
        hard = torch.where(z >= 0, 0., -LOG_ZERO)
        return torch.where(delta_hard, hard, soft).contiguous()

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, *, delta_hard=None, **kwargs):
        """Same interface as v3, plus an optional explicit delta_hard row mask.

        hard_prob/hard_seed/generator also control gate hardening. Vocabulary
        `hard` and gate `delta_hard` masks are independent. An explicit
        gate_delta bypasses the learned gate for reference comparisons.
        """
        if kwargs.get('gate_delta') is None:
            kwargs['gate_delta'] = self.gate_delta(
                hidden_states, kwargs.get('hard_prob'), delta_hard=delta_hard,
                hard_seed=kwargs.get('hard_seed'), generator=kwargs.get('generator'))
        elif delta_hard is not None:
            raise ValueError('provide either gate_delta or delta_hard, not both')
        return super().forward(hidden_states, attention_mask=attention_mask,
                               past_key_values=past_key_values, use_cache=use_cache,
                               output_attentions=output_attentions, **kwargs)


class DismV4Attention(_V4Gate, DismAttention):
    """DISM v4 with a trainable per-token, per-head diagonal continuation gate."""


class DismV4SwaAttention(_V4Gate, DismSwaAttention):
    """DISM v4 + SWA; shared V and output path, independently projected SWA Q/K."""
