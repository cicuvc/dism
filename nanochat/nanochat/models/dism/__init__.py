"""Selected DISM-GDN pretraining architecture."""
from dataclasses import dataclass

from nanochat.models import ModelCapabilities, register_model


@dataclass
class DismConfig:
    """Scale-clean configuration for the selected valres architecture.

    Post-norm, fixed true direction and sequence-row hard sampling are fixed.
    layer_layout reorders the equal numbers of pure GDN (g) and GDN+DISM (d)
    layers. Omission preserves the original first-half GDN layout.
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
    mxfp8: bool = False
    mxfp8_projections: bool = False
    mxfp8_optimizer: bool = False
    mxfp8_ce: bool = False
    layer_layout: str | None = None

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
        if self.layer_layout is not None:
            if (not isinstance(self.layer_layout, str)
                    or len(self.layer_layout) != self.n_layer
                    or set(self.layer_layout) != {'g', 'd'}
                    or not self.layer_layout.startswith('g')
                    or self.layer_layout.count('d') != self.n_layer // 2):
                raise ValueError('layer_layout must start with g and contain equal numbers of g/d, one per layer')
        if not self.post_norm:
            raise ValueError("the selected DISM architecture requires post_norm=True")
        if not self.value_residual:
            raise ValueError("the selected DISM architecture requires value_residual=True")
        if self.mxfp8 and self.intermediate_size % 64:
            raise ValueError("mxfp8 requires intermediate_size divisible by 64")
        if self.mxfp8 and (
            self.n_embd % 64
            or (self.n_head * self.head_dim) % 64
            or (self.n_head * self.value_dim) % 64
            or (self.n_head * self.readout_dim) % 64
        ):
            raise ValueError(
                "mxfp8 requires model, Q/K/V and readout projection widths "
                "divisible by 64"
            )
        if self.mxfp8_optimizer and not self.mxfp8:
            raise ValueError("mxfp8_optimizer requires mxfp8=True")
        if type(self.mxfp8_ce) is not bool:
            raise ValueError("mxfp8_ce must be boolean")
        if self.mxfp8_ce and (self.n_embd % 128 or self.vocab_size % 128):
            raise ValueError("mxfp8_ce requires model/vocabulary widths divisible by 128")
        if self.mxfp8_projections and not self.mxfp8:
            raise ValueError("mxfp8_projections requires mxfp8=True")
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
