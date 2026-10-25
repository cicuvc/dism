"""Opt-in quadratic DISM v4 prototype; distinct checkpoint architecture."""
from dataclasses import dataclass

from nanochat.models import ModelCapabilities, register_model
from nanochat.models.dism import DismConfig


@dataclass
class DismV4Config(DismConfig):
    block_size: int = 64
    readout_precision: str = "fp32"
    soft_readout: str = "arithmetic"
    hybrid_fusion: str = "sum"

    def __post_init__(self):
        super().__post_init__()
        if self.block_size not in (32, 64, 128):
            raise ValueError("v4 block_size must be 32, 64 or 128")
        if self.readout_precision not in ("fp32", "bf16"):
            raise ValueError("v4 readout_precision must be fp32 or bf16")
        if self.soft_readout not in ("arithmetic", "jensen", "rmsnorm_jensen"):
            raise ValueError("v4 soft_readout must be arithmetic, jensen or rmsnorm_jensen")
        if self.soft_readout == "rmsnorm_jensen" and self.readout_precision != "fp32":
            raise ValueError("rmsnorm_jensen currently requires fp32 readout")
        if self.hybrid_fusion not in ("sum", "separate_rmsnorm"):
            raise ValueError("v4 hybrid_fusion must be sum or separate_rmsnorm")
        if self.hybrid_fusion == "separate_rmsnorm":
            if self.soft_readout != "rmsnorm_jensen" or self.mixer_output != "rmsnorm_silu":
                raise ValueError("separate_rmsnorm requires rmsnorm_jensen and rmsnorm_silu gate")


def _build(config):
    from .model import DismV4LM
    return DismV4LM(config)


register_model(
    "dism_v4_gdn_hybrid", 1, DismV4Config, _build,
    ModelCapabilities(varlen=True, generation=False, sliding_window=False, compile=False),
    scaling_reference_spec=lambda spec: spec,
)
