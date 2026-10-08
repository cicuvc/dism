"""Llama 2 architecture registration."""

from nanochat.models.llama2.model import Llama2, Llama2Config
from nanochat.models.protocol import ModelCapabilities
from nanochat.models.registry import register_model
from nanochat.models.spec import ModelSpec


ARCHITECTURE = "llama2"
ARCHITECTURE_VERSION = 1


def _scaling_reference_spec(spec):
    config = dict(spec.config)
    source_depth = config["n_layer"]
    source_width = config["n_embd"]
    source_intermediate = config["intermediate_size"]
    head_dim = source_width // config["n_head"]
    reference_depth = 12
    reference_width = round(reference_depth * source_width / source_depth / head_dim) * head_dim
    reference_heads = reference_width // head_dim
    kv_ratio = config["n_kv_head"] / config["n_head"]
    reference_kv_heads = max(1, round(reference_heads * kv_ratio))
    if reference_heads % reference_kv_heads != 0:
        raise ValueError("Cannot preserve Llama 2 GQA ratio in the d12 scaling reference")
    config.update(
        n_layer=reference_depth,
        n_embd=reference_width,
        n_head=reference_heads,
        n_kv_head=reference_kv_heads,
        intermediate_size=round(source_intermediate * reference_width / source_width),
    )
    return ModelSpec(ARCHITECTURE, ARCHITECTURE_VERSION, config)


register_model(
    ARCHITECTURE,
    ARCHITECTURE_VERSION,
    Llama2Config,
    Llama2,
    ModelCapabilities(
        varlen=True,
        generation=False,
        sliding_window=False,
        compile=True,
    ),
    scaling_reference_spec=_scaling_reference_spec,
)


__all__ = ["Llama2", "Llama2Config"]
