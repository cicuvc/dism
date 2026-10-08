"""DISM hybrid pretraining registration; optional dependencies load lazily."""
from dataclasses import dataclass
from nanochat.models import register_model, ModelCapabilities


@dataclass
class DismConfig:
    sequence_len: int = 1024
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    intermediate_size: int = 2152
    head_dim: int = 64
    value_dim: int = 64
    readout_dim: int = 32
    qk_vocab_size: int = 512
    window_size: int = 128
    anneal_steps: int = 3000
    softcap: float = 30.0
    rng_seed: int = 1729
    soft_k_l2_norm: bool = False
    hard_prob_max: float = 1.0  # Preserve old checkpoint semantics; new runs explicitly set .95.


def _build(config):
    from .model import DismLM
    return DismLM(config)


register_model('dism_hybrid', 1, DismConfig, _build,
               ModelCapabilities(varlen=True, generation=False, sliding_window=True, compile=True),
               scaling_reference_spec=lambda spec: spec)


def _build_swa(config):
    from .swa import SwaLM
    return SwaLM(config)


register_model('swa_control', 1, DismConfig, _build_swa,
               ModelCapabilities(varlen=True, generation=False, sliding_window=True, compile=True),
               scaling_reference_spec=lambda spec: spec)


def _build_v4(config):
    from .model import DismLM
    return DismLM(config, attention_type='hybrid_v4')


register_model('dism_v4_hybrid', 1, DismConfig, _build_v4,
               ModelCapabilities(varlen=True, generation=False, sliding_window=True, compile=True),
               scaling_reference_spec=lambda spec: spec)
