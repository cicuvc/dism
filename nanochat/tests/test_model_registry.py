import copy

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.models import (
    ModelSpec,
    PretrainOptimizerConfig,
    build_model,
    get_scaling_reference_spec,
    migrate_model_state_dict,
    model_spec_from_metadata,
    optimizer_schema,
    optimizer_schema_fingerprint,
)


def tiny_config(**overrides):
    config = {
        "sequence_len": 32,
        "vocab_size": 64,
        "n_layer": 2,
        "n_head": 2,
        "n_kv_head": 1,
        "n_embd": 32,
        "window_pattern": "SL",
    }
    config.update(overrides)
    return config


def test_model_spec_fingerprint_is_canonical_and_detects_semantic_changes():
    left = ModelSpec("nanochat_v1", 1, tiny_config())
    right = ModelSpec("nanochat_v1", 1, dict(reversed(list(tiny_config().items()))))
    changed = ModelSpec("nanochat_v1", 1, tiny_config(window_pattern="L"))

    assert left.fingerprint() == right.fingerprint()
    assert left.fingerprint() != changed.fingerprint()


def test_registry_build_preserves_original_model_and_state_dict():
    config = tiny_config()
    spec = ModelSpec("nanochat_v1", 1, config)

    torch.manual_seed(123)
    direct = GPT(GPTConfig(**config))
    direct.init_weights()
    torch.manual_seed(123)
    registered = build_model(spec)
    registered.init_weights()

    assert isinstance(registered, GPT)
    assert registered.model_spec == spec
    assert registered.capabilities.varlen
    assert direct.state_dict().keys() == registered.state_dict().keys()
    for name, value in direct.state_dict().items():
        torch.testing.assert_close(value, registered.state_dict()[name], rtol=0, atol=0)


def test_pretraining_optimizer_hook_covers_every_parameter_once():
    model = build_model(ModelSpec("nanochat_v1", 1, tiny_config()))
    model.init_weights()
    optimizer = model.setup_pretraining_optimizer(PretrainOptimizerConfig(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        scalar_lr=0.5,
        weight_decay=0.1,
    ))

    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(grouped) == len(list(model.parameters()))
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert all("initial_lr" in group for group in optimizer.param_groups)
    schema = optimizer_schema(model, optimizer)
    assert schema["optimizer"].endswith("MuonAdamW")
    changed_schema = copy.deepcopy(schema)
    changed_schema["groups"][0]["kind"] = "different"
    assert optimizer_schema_fingerprint(schema) != optimizer_schema_fingerprint(changed_schema)


def test_nanochat_scaling_reference_preserves_historical_d12_shape():
    spec = ModelSpec("nanochat_v1", 1, tiny_config())

    reference = get_scaling_reference_spec(spec)

    assert reference.config["n_layer"] == 12
    assert reference.config["n_embd"] == 192
    assert reference.config["n_head"] == 12
    assert reference.config["n_kv_head"] == 6
    assert reference.config["sequence_len"] == spec.config["sequence_len"]


def test_legacy_checkpoint_metadata_upgrades_to_nanochat_v1():
    legacy = {"model_config": tiny_config()}
    del legacy["model_config"]["window_pattern"]

    spec = model_spec_from_metadata(legacy)

    assert spec.architecture == "nanochat_v1"
    assert spec.architecture_version == 1
    assert spec.config["window_pattern"] == "L"


def test_checkpoint_model_fingerprint_is_validated():
    spec = ModelSpec("nanochat_v1", 1, tiny_config())
    metadata = {
        "model_spec": spec.to_dict(),
        "model_fingerprint": "corrupt",
    }

    with pytest.raises(ValueError, match="fingerprint"):
        model_spec_from_metadata(metadata)


def test_legacy_state_migration_does_not_mutate_complete_state():
    spec = ModelSpec("nanochat_v1", 1, tiny_config())
    model = build_model(spec)
    model.init_weights()
    complete = copy.deepcopy(model.state_dict())

    migrated = migrate_model_state_dict(spec, complete, model.config)

    assert migrated.keys() == model.state_dict().keys()
    torch.testing.assert_close(migrated["resid_lambdas"], model.resid_lambdas)
    torch.testing.assert_close(migrated["x0_lambdas"], model.x0_lambdas)
