"""Exercise diagnostic traversal, which checkpoint replay benchmarks skip."""
import pytest
import torch

from nanochat.metrics import probe_qk_vocab_usage
from nanochat.models import ModelSpec, build_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def make_model():
    torch.manual_seed(51)
    config = dict(sequence_len=512, vocab_size=128, n_layer=4, n_head=2,
                  n_embd=64, intermediate_size=128, head_dim=32, value_dim=32,
                  readout_dim=16, qk_vocab_size=32,
                  post_norm=True, value_residual=True)
    model = build_model(ModelSpec('dism_gdn_hybrid', 1, config)).cuda()
    model.init_weights()
    ids = torch.randint(128, (2, 512), device='cuda')
    # Three documents across two packed rows: hard rows must not follow cu_seqlens.
    cu = torch.tensor([0, 256, 512, 1024], device='cuda', dtype=torch.int32)
    return model, ids, cu


@pytest.mark.parametrize('training', [False, True])
def test_probe_matches_all_hard_eval_and_preserves_state(training):
    model, ids, cu = make_model()
    model.eval()
    captures = []
    def capture(layer, args, kwargs, output):
        captures.append((args[0].detach().clone(), output.detach().clone(), kwargs))
    handles = [layer.register_forward_hook(capture, with_kwargs=True) for layer in model.layers]
    try:
        with torch.no_grad():
            model(ids, cu_seqlens=cu)
        expected = captures[:]
        captures.clear()
        model.train(training)
        # The diagnostic is always all-hard, even if the user's eval mode is soft.
        model.eval_hard = False
        state = {k: v.clone() for k, v in model.state_dict().items()}
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        param = next(model.parameters())
        param.grad = torch.ones_like(param)
        stats = probe_qk_vocab_usage(model, ids, cu, 512, chunk_size=256)
        assert model.training == training
        assert model.eval_hard is False
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        assert torch.all(param.grad == 1)
        assert all(p.grad is None for p in list(model.parameters())[1:])
        for key, value in model.state_dict().items():
            assert torch.equal(value, state[key]), key
        assert len(captures) == len(expected) == 4
        for index, (actual, reference) in enumerate(zip(captures, expected)):
            torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
            if index >= 2:
                assert actual[2]['hard'].shape == (2, 2)
                assert actual[2]['hard'].all()
            if index > 0:
                torch.testing.assert_close(actual[2]['v_first'], reference[2]['v_first'],
                                           rtol=0, atol=0)
        assert set(stats) == {2, 3}
        # Independently recompute selection counts on the real model's layer inputs.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for index, tags in stats.items():
                attn = model.layers[index].attn
                for tag in ('q', 'k'):
                    vec = getattr(attn, tag + '_conv')(expected[index][0],
                        cu_seqlens=cu, max_seqlen=512).reshape(1, 1024, 2, 32).float()
                    table = getattr(attn, tag + '_vocab').float()
                    indices = torch.einsum('bnhd,hvd->bnhv', vec, table).argmax(-1).reshape(-1, 2)
                    counts = torch.stack([torch.bincount(indices[:, h], minlength=32)
                                          for h in range(2)]).double()
                    assert tags[tag]['counts'] == counts.mean(0).cpu().tolist()
                    assert sum(tags[tag]['counts']) == ids.numel()
    finally:
        for handle in handles:
            handle.remove()


def test_probe_restores_training_mode_on_error():
    model, ids, cu = make_model()
    model.train()
    counter = model.rng_counter.clone()
    def fail(*args):
        raise RuntimeError('injected diagnostic failure')
    handle = model.layers[2].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match='injected diagnostic failure'):
            probe_qk_vocab_usage(model, ids, cu, 512)
        assert all(module.training for module in model.modules())
        assert torch.equal(model.rng_counter, counter)
    finally:
        handle.remove()
