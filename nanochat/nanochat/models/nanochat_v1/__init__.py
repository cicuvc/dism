"""Registration of the original nanochat GPT architecture."""

import math
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.models.protocol import ModelCapabilities
from nanochat.models.registry import register_model


ARCHITECTURE = "nanochat_v1"
ARCHITECTURE_VERSION = 1


def _migrate_legacy_config(config):
    config = dict(config)
    # Checkpoints from before sliding-window support used full context.
    config.setdefault("window_pattern", "L")
    return config


def _migrate_state_dict(model_data, model_config):
    # Keep compatibility with checkpoints predating learnable residual mixing.
    state_device = next(iter(model_data.values())).device
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(model_config.n_layer, device=state_device)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(model_config.n_layer, device=state_device)
    return model_data


def _scaling_reference_spec(spec):
    """Construct the historical d12 reference while preserving width/head ratios."""
    from nanochat.models.spec import ModelSpec

    config = dict(spec.config)
    source_depth = config["n_layer"]
    source_heads = config["n_head"]
    source_kv_heads = config["n_kv_head"]
    head_dim = config["n_embd"] // source_heads
    aspect_ratio = config["n_embd"] / source_depth
    reference_depth = 12
    base_dim = round(reference_depth * aspect_ratio)
    reference_dim = math.ceil(base_dim / head_dim) * head_dim
    reference_heads = reference_dim // head_dim
    if source_kv_heads == source_heads:
        reference_kv_heads = reference_heads
    else:
        reference_kv_heads = max(1, reference_heads * source_kv_heads // source_heads)
        if reference_heads % reference_kv_heads != 0:
            raise ValueError(
                "Cannot preserve the GQA ratio in the nanochat_v1 d12 scaling reference"
            )
    config.update(
        n_layer=reference_depth,
        n_embd=reference_dim,
        n_head=reference_heads,
        n_kv_head=reference_kv_heads,
    )
    return ModelSpec(ARCHITECTURE, ARCHITECTURE_VERSION, config)


register_model(
    ARCHITECTURE,
    ARCHITECTURE_VERSION,
    GPTConfig,
    GPT,
    ModelCapabilities(
        varlen=True,
        generation=True,
        sliding_window=True,
        compile=True,
    ),
    scaling_reference_spec=_scaling_reference_spec,
    migrate_legacy_config=_migrate_legacy_config,
    migrate_state_dict=_migrate_state_dict,
)
