"""Regression contracts of the selected remote single-forward baseline."""
import json
from pathlib import Path

import pytest
import torch

from nanochat.models import ModelSpec, build_model, PretrainOptimizerConfig
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint


def baseline_spec():
    path = Path(__file__).parents[1] / 'configs' / 'sequence_true.json'
    return ModelSpec.from_dict(json.loads(path.read_text()))


def test_baseline_structure_and_state():
    model = build_model(baseline_spec())
    assert sum(p.numel() for p in model.parameters()) == 40_467_834
    assert [getattr(layer.attn, 'is_pure_gdn', False) for layer in model.layers] == [True]*3 + [False]*3
    assert all(layer.post_norm for layer in model.layers)
    assert model.embedding.weight is not model.lm_head.weight
    for layer in model.layers[3:]:
        assert layer.attn.gdn_q_conv.weight is layer.attn.q_conv.weight
        assert layer.attn.gdn_k_conv.weight is layer.attn.k_conv.weight
    model.set_training_step(5999)
    assert model.hard_probability.item() == pytest.approx(.95)
    assert {'rng_counter', 'hard_probability', 'training_step'} <= model.state_dict().keys()
    optimizer = model.setup_pretraining_optimizer(PretrainOptimizerConfig(.0015, .0015, .0015, .0015, .01))
    no_decay = {id(p) for p in optimizer.param_groups[1]['params']}
    for name, p in model.named_parameters():
        if p.ndim < 2 or any(tag in name for tag in ('q_vocab', 'k_vocab', 'gdn_A_log', 'gdn_dt_bias')):
            assert id(p) in no_decay, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gdn_hybrid_compile_and_resume(tmp_path):
    spec = baseline_spec()
    config = dict(spec.config)
    config.update(sequence_len=256, vocab_size=128, n_layer=2, n_head=2,
                  n_embd=64, intermediate_size=128, qk_vocab_size=32)
    spec = ModelSpec(spec.architecture, spec.architecture_version, config)
    model = build_model(spec).cuda().train()
    model.init_weights()
    model.set_training_step(3000)
    x = torch.randint(128, (2, 256), device='cuda')
    y = torch.randint(128, (2, 256), device='cuda')
    cu = torch.tensor([0, 256, 512], device='cuda', dtype=torch.int32)
    opt_config = PretrainOptimizerConfig(.0015, .0015, .0015, .0015, .01)
    optimizer = model.setup_pretraining_optimizer(opt_config)
    # The remote training loop allows graph breaks around the FLA GDN operator.
    with torch._inductor.config.patch(compile_threads=1), torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(model, dynamic=True)
        loss = compiled(x, y, cu_seqlens=cu)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert model.rng_counter.item() == 1  # not dual-path training
        save_checkpoint(str(tmp_path), 3001, model.state_dict(), optimizer.state_dict(), {'step': 3001})
        expected = compiled(x, y, cu_seqlens=cu).detach()
        state, opt_state, _ = load_checkpoint(str(tmp_path), 3001, 'cuda', load_optimizer=True)
        restored = build_model(spec).cuda().train()
        restored.load_state_dict(state, assign=True)
        restored_optimizer = restored.setup_pretraining_optimizer(opt_config)
        restored_optimizer.load_state_dict(opt_state)
        assert restored.rng_counter.item() == 1
        assert restored.layers[1].attn.gdn_q_conv.weight is restored.layers[1].attn.q_conv.weight
        replay = torch.compile(restored, dynamic=True)(x, y, cu_seqlens=cu).detach()
        torch.testing.assert_close(replay, expected, atol=1e-5, rtol=1e-5)
        assert restored.rng_counter.item() == 2
