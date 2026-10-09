"""DISM hybrid pretraining registration; optional dependencies load lazily."""
from dataclasses import dataclass
from nanochat.models import register_model, ModelCapabilities


@dataclass
class DismConfig:
    direction_policy: str = "random"
    direction_freeze_step: int = 3000
    direction_antithetic: bool = False
    hard_granularity: str = "token"
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
    post_norm: bool = False
    alternating_gdn: bool = False
    rear_half_dism: bool = False
    window_size: int = 128
    anneal_steps: int = 3000
    hard_start_fraction: float = 0.0
    softcap: float = 30.0
    rng_seed: int = 1729
    hard_end_fraction: float = 1.0
    readout_mode: str = "default"
    vocab_share_qk: bool = False
    vocab_share_heads: bool = False
    vocab_share_layers: bool = False
    soft_qk_rope: bool = False
    shared_readout: bool = False
    soft_k_l2_norm: bool = False
    split_branch_output: bool = False
    hard_prob_max: float = 1.0  # Preserve old checkpoint semantics; new runs explicitly set .95.
    value_residual: bool = False
    value_residual_gate_bias: float = 2.0
    conv_impl: str = "fused"
    conv_backend: str = "cuda"
    vocab_transvq: bool = False
    vocab_transvq_lite: bool = False


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


def _build_gdn(config):
    from .model import DismLM
    if config.split_branch_output:
        raise ValueError("Initial GDN hybrid uses one shared post-sum output path")
    return DismLM(config, attention_type='hybrid_gdn')

register_model('dism_gdn_hybrid', 1, DismConfig, _build_gdn,
               ModelCapabilities(varlen=True, generation=False, sliding_window=False, compile=True),
               scaling_reference_spec=lambda spec: spec)
