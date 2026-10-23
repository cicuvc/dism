import copy
import pytest
import torch
from nanochat.models import build_model, ModelSpec, PretrainOptimizerConfig, optimizer_schema
from nanochat.dataloader import build_varlen_metadata


def test_tail_padding_alignment():
    ids = torch.full((2, 1024), -1)
    ids[:, :197] = 0
    ids[:, 256:821] = 1
    cu, segments = build_varlen_metadata(ids, 18, aligned_segment_ends=True)
    assert (cu % 256 == 0).all()
    assert cu[:4].tolist() == [0, 256, 1024, 1280]
    assert segments[0, 196] == segments[0, 255]
    assert segments[0, 255] != segments[0, 256]


def test_capped_hard_schedule_and_state():
    from types import SimpleNamespace
    from nanochat.models.dism import DismConfig
    from nanochat.models.dism.model import DismLM
    assert DismConfig().hard_prob_max == 0.95
    holder = SimpleNamespace(config=DismConfig(anneal_steps=101, hard_prob_max=.95),
                             training_step=torch.zeros((), dtype=torch.int64),
                             hard_probability=torch.zeros(()),
                             vocab_temperature=1.0)
    for step, expected in [(0,0.), (50,.475), (100,.95), (200,.95)]:
        DismLM.set_training_step(holder, step)
        assert holder.training_step.item() == step
        assert holder.hard_probability.item() == pytest.approx(expected)


@pytest.mark.parametrize('layout',['', 'gggddddddgg', 'dddggggggddd', 'gggggggggggg', 'gggddddddggx'])
def test_invalid_layer_layout(layout):
    from nanochat.models.dism import DismConfig
    with pytest.raises(ValueError,match='layer_layout'):
        DismConfig(layer_layout=layout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
def test_dism_compiled_resume(tmp_path):
    pytest.importorskip('flash_dism')
    from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
    from torch.fx.experimental import _config as shape_config
    config = dict(sequence_len=256, vocab_size=128, n_layer=2, n_head=2, n_embd=64,
                  intermediate_size=128, head_dim=32, value_dim=32, readout_dim=16,
                  qk_vocab_size=32, anneal_steps=3000, hard_prob_max=.95)
    spec = ModelSpec('dism_gdn_hybrid', 1, config)
    model = build_model(spec).cuda().train()
    model.init_weights()
    optconf = PretrainOptimizerConfig(.001, .001, .001, .001, .01)
    optim = model.setup_pretraining_optimizer(optconf)
    schema = optimizer_schema(model, optim)
    assert len(schema['groups']) == 2
    no_decay = {id(p) for p in optim.param_groups[1]['params']}
    for name, p in model.named_parameters():
        if any(tag in name for tag in ('q_vocab', 'k_vocab', 'log_sel_tau', 'rms_weight')) or p.ndim < 2:
            assert id(p) in no_decay, name
    x = torch.randint(128, (2, 256), device='cuda')
    y = torch.randint(128, (2, 256), device='cuda')
    cu = torch.tensor([0, 256, 512, 512], device='cuda', dtype=torch.int32)
    with shape_config.patch(use_duck_shape=False), torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True), \
            torch._inductor.config.patch(compile_threads=1):
        compiled = torch.compile(model, dynamic=True)
        model.set_training_step(123)
        loss = compiled(x, y, cu_seqlens=cu)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        save_checkpoint(str(tmp_path), 124, model.state_dict(), optim.state_dict(), {'step': 124})
        model.set_training_step(124)
        reference = compiled(x, y, cu_seqlens=cu)
        reference.backward()
        expected_grads = [p.grad.clone() for p in model.parameters()]
        optim.step()
        restored = build_model(spec).cuda().train()
        restored.init_weights()
        state, optimizer_state, _ = load_checkpoint(str(tmp_path), 124, 'cuda', load_optimizer=True)
        restored.load_state_dict(state, assign=True)
        restored_optim = restored.setup_pretraining_optimizer(optconf)
        restored_optim.load_state_dict(optimizer_state)
        assert optimizer_schema(restored, restored_optim) == schema
        assert restored.rng_counter.item() == 1
        restored.set_training_step(124)
        restored_compiled = torch.compile(restored, dynamic=True)
        replay = restored_compiled(x, y, cu_seqlens=cu)
        replay.backward()
        torch.testing.assert_close(replay, reference, atol=1e-5, rtol=1e-5)
        for p, g in zip(restored.parameters(), expected_grads):
            torch.testing.assert_close(p.grad, g, atol=2e-4, rtol=.02)
        restored_optim.step()
        for p, r in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(p, r, atol=2e-5, rtol=.001)
        assert model.rng_counter.item() == restored.rng_counter.item() == 2


def test_mxfp8_ce_preserves_legacy_identity_and_parameter_schema():
    from nanochat.models import ModelSpec, build_model
    from nanochat.models.dism import DismConfig
    config = dict(n_layer=2, n_embd=128, n_head=4, head_dim=32,
                  value_dim=32, readout_dim=16, vocab_size=128, intermediate_size=256)
    legacy = ModelSpec('dism_gdn_hybrid', 1, config)
    selected = ModelSpec('dism_gdn_hybrid', 1, {**config, 'mxfp8_ce': True})
    with torch.device('meta'):
        old, new = build_model(legacy), build_model(selected)
    assert old.model_spec.fingerprint() == legacy.fingerprint()
    assert selected.fingerprint() != legacy.fingerprint()
    assert old.mxfp8_ce is None and not new.mxfp8_ce.state_dict()
    assert set(old.state_dict()) == set(new.state_dict())
    assert [(n, p.shape, p.dtype, getattr(p, '_no_weight_decay', False))
            for n, p in old.named_parameters()] == [
                (n, p.shape, p.dtype, getattr(p, '_no_weight_decay', False))
                for n, p in new.named_parameters()]
    assert not DismConfig(**config).mxfp8_ce
    for invalid in ({'vocab_size': 129}, {'n_embd': 64}, {'mxfp8_ce': 'true'}):
        with pytest.raises(ValueError, match='mxfp8_ce'):
            DismConfig(**{**config, 'mxfp8_ce': True, **invalid})
