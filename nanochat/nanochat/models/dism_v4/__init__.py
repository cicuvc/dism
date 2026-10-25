"""Opt-in quadratic DISM v4 prototype; distinct checkpoint architecture."""
from dataclasses import dataclass

from nanochat.models import ModelCapabilities, register_model
from nanochat.models.dism import DismConfig


@dataclass
class DismV4Config(DismConfig):
    block_size: int = 64
    readout_precision: str = "fp32"

    def __post_init__(self):
        super().__post_init__()
        if self.block_size not in (32, 64, 128):
            raise ValueError("v4 block_size must be 32, 64 or 128")
        if self.readout_precision not in ("fp32", "bf16"):
            raise ValueError("v4 readout_precision must be fp32 or bf16")


def _build(config):
    from .model import DismV4LM
    return DismV4LM(config)


register_model(
    "dism_v4_gdn_hybrid", 1, DismV4Config, _build,
    ModelCapabilities(varlen=True, generation=False, sliding_window=False, compile=False),
    scaling_reference_spec=lambda spec: spec,
)
