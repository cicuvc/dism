"""Pure-GDN valres control with exact parameter matching."""

import json
from pathlib import Path

import torch

from nanochat.models import ModelSpec, build_model


def test_gdnonly_config_matches_valres_parameter_count():
    path = Path(__file__).parents[1] / "configs/sequence_true_72m_valres_gdnonly.json"
    spec = ModelSpec.from_dict(json.loads(path.read_text()))
    with torch.device("meta"):
        model = build_model(spec)
    assert {type(layer.attn).__name__ for layer in model.layers} == {"PureGdnAttention"}
    assert sum(p.numel() for p in model.parameters()) == 72_035_808
    assert [layer.mlp.gate_proj.weight.shape[0] for layer in model.layers] == [
        2878, 2878, 2878, 2878, 2878, 2878, 2878, 2878,
        2877, 2877, 2877, 2877,
    ]
    assert all(layer.attn.gdn_b_proj.bias.shape == (6,) for layer in model.layers)
