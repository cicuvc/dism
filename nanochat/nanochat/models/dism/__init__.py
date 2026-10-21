"""Selected DISM-GDN pretraining architecture."""
from dataclasses import dataclass

from nanochat.models import ModelCapabilities, register_model


@dataclass
class DismConfig:
    """Scale-clean configuration for the selected valres architecture.

    Layout and routing are deliberately not configurable: post-norm,
    first-half pure GDN, rear-half GDN+DISM, fixed true direction and
    sequence-level hard sampling are the deployed model contract.
    """

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
    anneal_steps: int = 3000
    hard_prob_max: float = 0.95
    softcap: float = 30.0
    rng_seed: int = 1729
    vocab_temperature: float = 1.0
    tau_schedule_fraction: float = 0.0
    tau_schedule_floor: float = 0.0
    post_norm: bool = True
    value_residual: bool = True
    value_residual_gate_bias: float = 2.0

    def __post_init__(self):
        self.softcap = float(self.softcap)
        self.hard_prob_max = float(self.hard_prob_max)
        self.vocab_temperature = float(self.vocab_temperature)
        self.tau_schedule_fraction = float(self.tau_schedule_fraction)
        self.tau_schedule_floor = float(self.tau_schedule_floor)
        if self.sequence_len <= 0 or self.sequence_len % 256:
            raise ValueError("sequence_len must be a positive multiple of 256")
        if self.n_layer <= 0 or self.n_layer % 2:
            raise ValueError("n_layer must be positive and even")
        if not self.post_norm:
            raise ValueError("the selected DISM architecture requires post_norm=True")
        if not self.value_residual:
            raise ValueError("the selected DISM architecture requires value_residual=True")
        if not 0.0 <= self.hard_prob_max <= 1.0:
            raise ValueError("hard_prob_max must be in [0,1]")
        if not 0.0 <= self.tau_schedule_fraction <= 1.0:
            raise ValueError("tau_schedule_fraction must be in [0,1]")
        if not 0.0 <= self.tau_schedule_floor <= 1.0:
            raise ValueError("tau_schedule_floor must be in [0,1]")


def _build_gdn(config):
    from .model import DismLM
    return DismLM(config)


register_model(
    "dism_gdn_hybrid",
    1,
    DismConfig,
    _build_gdn,
    ModelCapabilities(
        varlen=True,
        generation=False,
        sliding_window=False,
        compile=True,
    ),
    scaling_reference_spec=lambda spec: spec,
)
