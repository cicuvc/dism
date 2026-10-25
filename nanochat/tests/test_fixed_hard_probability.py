import pytest
import torch
from nanochat.models import ModelSpec, build_model
from nanochat.models.dism import DismConfig
import test_cuda_graph as graph_tests


def spec(**extra):
    return ModelSpec('dism_gdn_hybrid', 1, dict(
        sequence_len=256, vocab_size=128, n_layer=2, n_embd=128, n_head=4,
        head_dim=32, value_dim=32, readout_dim=16, intermediate_size=256,
        qk_vocab_size=32, anneal_steps=11, **extra))


def test_fixed_probability_initialization_schedule_and_state():
    model = build_model(spec(hard_prob_fixed=.5))
    assert model.hard_probability.item() == .5
    model.init_weights()
    pointer = model.hard_probability.data_ptr()
    model.rng_counter.fill_(17)
    for step in [0, 1, 5, 10, 22888, 100000]:
        model.set_training_step(step)
        assert model.hard_probability.item() == .5
        assert model.hard_probability.data_ptr() == pointer
        assert model.training_step.item() == step
        assert model.rng_counter.item() == 17
    restored = build_model(spec(hard_prob_fixed=.5))
    restored.load_state_dict(model.state_dict())
    assert restored.hard_probability.item() == .5
    assert restored.training_step.item() == 100000
    assert restored.rng_counter.item() == 17
    baseline = build_model(spec())
    for step, expected in [(0, 0.), (5, .475), (10, .95), (100000, .95)]:
        baseline.set_training_step(step)
        assert baseline.hard_probability.item() == pytest.approx(expected)
    assert list(restored.state_dict()) == list(baseline.state_dict())
    assert spec().fingerprint() != spec(hard_prob_fixed=.5).fingerprint()


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan'), float('inf')])
def test_reject_invalid_fixed_probability(value):
    with pytest.raises(ValueError, match='hard_prob_fixed'):
        DismConfig(hard_prob_fixed=value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_fixed_probability_planner_graph_and_resume(monkeypatch):
    original = graph_tests.model_pair
    def fixed_pair(*args, **kwargs):
        models = original(*args, **kwargs)
        for model in models:
            model.config.hard_prob_fixed = .5
            model.set_training_step(0)
        return models
    monkeypatch.setattr(graph_tests, 'model_pair', fixed_pair)
    graph_tests.test_packed_replay_matches_training(monkeypatch, True, False, True, 'layernorm_sigmoid')
    graph_tests.test_capture_preserves_resumed_state_and_existing_gradients(monkeypatch, True, False, 'layernorm_sigmoid')
