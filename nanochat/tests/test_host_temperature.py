import math
import torch
import pytest
from nanochat.models import ModelSpec, build_model


def test_temperature_stays_host_and_resumes_schedule():
    spec = ModelSpec('dism_gdn_hybrid', 1, dict(
        sequence_len=256, vocab_size=128, n_layer=2, n_head=2, n_embd=64,
        intermediate_size=128, head_dim=32, value_dim=32, readout_dim=16,
        qk_vocab_size=32, rear_half_dism=True, anneal_steps=101,
        vocab_temperature_start=.5, vocab_temperature_end=2.))
    model = build_model(spec).cuda()
    model.init_weights()
    assert isinstance(model.vocab_temperature, float)
    assert isinstance(model.layers[1].attn.vocab_temperature, float)
    assert not any('vocab_temperature' in k for k, _ in model.named_buffers())
    model.set_training_step(37)
    expected = .5 + .75*(1-math.cos(math.pi*.37))
    assert model.vocab_temperature == expected
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model.set_vocab_temperature(3.)
    model.load_state_dict(state)
    model.set_training_step(int(model.training_step))
    assert model.vocab_temperature == expected
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    for invalid in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            model.set_vocab_temperature(invalid)
