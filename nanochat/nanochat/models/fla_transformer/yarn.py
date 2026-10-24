"""Evaluation-only static YaRN for FLA packed RoPE, without checkpoint mutation.

Frequency ramp and attention magnitude follow the original YaRN defaults and
Transformers modeling_rope_utils: beta_fast=32, beta_slow=1, truncated bounds.
Factor depends on the declared evaluation context, never packed batch length.
"""
import math
import torch
from fla.modules.rotary import RotaryEmbedding


def yarn_parameters(dim, base, original_context, factor, *, device=None,
                    beta_fast=32., beta_slow=1.):
    if dim < 2 or dim % 2 or base <= 1 or original_context <= 0:
        raise ValueError('invalid YaRN dimensions/base/original context')
    if not math.isfinite(factor) or factor < 1 or not 0 < beta_slow < beta_fast:
        raise ValueError('invalid YaRN factor or correction range')
    pos_freqs = base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    extrapolation = 1. / pos_freqs
    if factor == 1:
        return extrapolation, 1.
    def correction(rotations):
        return dim * math.log(original_context / (rotations*2*math.pi)) / (2*math.log(base))
    low = max(math.floor(correction(beta_fast)), 0)
    high = min(math.ceil(correction(beta_slow)), dim-1)
    if low == high:
        high += .001
    ramp = ((torch.arange(dim//2,device=device,dtype=torch.float32)-low)/(high-low)).clamp(0,1)
    inv_freq = (1./(factor*pos_freqs))*ramp + extrapolation*(1-ramp)
    return inv_freq, 1.+.1*math.log(factor)


class YarnRotaryEmbedding(RotaryEmbedding):
    def __init__(self, original, original_context, factor):
        if original.scale_base is not None:
            raise ValueError('YaRN evaluation does not combine with XPos')
        super().__init__(original.dim, base=original.base, interleaved=original.interleaved,
                         device=original.inv_freq.device)
        inv_freq, self.attention_factor = yarn_parameters(
            original.dim, original.base, original_context, factor, device=original.inv_freq.device)
        self.inv_freq.copy_(inv_freq)
        self.factor = factor
        self.eval()

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if self.training:
            raise RuntimeError('YaRN wrapper is evaluation-only')
        if (self._cos_cached is None or seqlen > self._seq_len_cached
                or self._cos_cached.device != device or self._cos_cached.dtype != dtype):
            self._seq_len_cached = seqlen
            freqs = torch.outer(torch.arange(seqlen,device=device,dtype=torch.float32),self.inv_freq.float())
            self._cos_cached = (freqs.cos()*self.attention_factor).to(dtype)
            self._sin_cached = (freqs.sin()*self.attention_factor).to(dtype)


def install_yarn(model, original_context, target_context):
    if model.training or target_context < original_context:
        raise ValueError('YaRN requires an eval model and a nonshrinking target context')
    factor = target_context / original_context
    for layer in model.layers:
        layer.attn.rotary = YarnRotaryEmbedding(layer.attn.rotary, original_context, factor)
    return dict(type='yarn',factor=factor,original_context=original_context,
                target_context=target_context,beta_fast=32,beta_slow=1,
                attention_factor=1.+.1*math.log(factor),finetuned=False,
                policy='static per declared context; never per packed tensor or request length')
