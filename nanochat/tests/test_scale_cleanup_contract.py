"""Selected valres contract after removal of experiment-only branches."""
import json
from pathlib import Path

import pytest
import torch

from nanochat.models import ModelSpec, PretrainOptimizerConfig, build_model, optimizer_schema
from nanochat.models.dism import DismConfig
from flash_dism.hybrid_gdn import DismGdnAttention
from flash_dism.pure_gdn import PureGdnAttention


def valres_spec():
    path = Path(__file__).parents[1] / "configs/sequence_true_72m_valres.json"
    return ModelSpec.from_dict(json.loads(path.read_text()))


def test_selected_architecture_shape_and_optimizer_groups():
    spec = valres_spec()
    with torch.device("meta"):
        model = build_model(spec)
    assert sum(p.numel() for p in model.parameters()) == 72_035_808
    assert all(isinstance(layer.attn, PureGdnAttention) for layer in model.layers[:6])
    assert all(isinstance(layer.attn, DismGdnAttention) for layer in model.layers[6:])
    assert all(layer.post_norm for layer in model.layers)

    with torch.device("meta"):
        optimizer = model.setup_pretraining_optimizer(
            PretrainOptimizerConfig(.0015, .0015, .0015, .0015, .01)
        )
    schema = optimizer_schema(model, optimizer)
    assert len(schema["groups"]) == 2


def test_historical_config_is_rejected():
    with pytest.raises(TypeError, match="hard_start_fraction"):
        DismConfig(hard_start_fraction=0.0)
