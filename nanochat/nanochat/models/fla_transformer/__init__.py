"""Postnorm full-attention baseline with the DISM training recipe."""
from dataclasses import dataclass

from nanochat.models import ModelCapabilities, register_model


@dataclass
class FLATransformerConfig:
    sequence_len: int = 2048
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    intermediate_size: int = 1664
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    softcap: float = 30.0
    post_norm: bool = True
    mxfp8: bool = True
    mxfp8_optimizer: bool = True
    mxfp8_ce: bool = True

    def __post_init__(self):
        if self.sequence_len <= 0 or self.sequence_len % 256:
            raise ValueError('sequence_len must be positive and 256-aligned')
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd % self.n_head:
            raise ValueError('invalid layer/head/width configuration')
        if self.n_embd // self.n_head not in (32, 64, 128):
            raise ValueError('supported attention head dimensions: 32/64/128')
        if not self.post_norm or not self.mxfp8:
            raise ValueError('this baseline requires postnorm and MXFP8 FFN')
        if self.n_embd % 64 or self.intermediate_size % 64:
            raise ValueError('MXFP8 FFN dimensions must be 64-aligned')
        if self.mxfp8_ce and (self.n_embd % 128 or self.vocab_size % 128):
            raise ValueError('MXFP8 CE dimensions must be 128-aligned')


def build(config):
    from .model import FLATransformer
    return FLATransformer(config)


register_model('fla_transformer', 1, FLATransformerConfig, build,
               ModelCapabilities(varlen=True, generation=False, sliding_window=False, compile=False),
               scaling_reference_spec=lambda spec: spec)
