from dataclasses import replace
import io
import pytest
import torch
from dism_v2.lm_model import LMConfig, DecoderLM, parameter_count, parameter_groups


@pytest.mark.parametrize('tied', [False, True])
def test_single_owner_and_checkpoint(tied):
    c = LMConfig(layers=2, width=64, heads=1, ffn_hidden=64, softcap=30.,
                 dism_output_gla=True, dism_share_vocab_layers=True, dism_tie_qk_vocab=tied)
    model = DecoderLM(c).double()
    assert all(b.dism.shared_vocab is model.shared_vocab for b in model.blocks)
    names = [name for name in model.state_dict() if 'voc' in name]
    assert sorted(names) == (['shared_vocab.q'] if tied else ['shared_vocab.k', 'shared_vocab.q'])
    groups = parameter_groups(model, .01)
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids))
    assert all(any(p is x for x in groups[1]['params']) for p in model.shared_vocab.values())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = DecoderLM(c).double()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    for block in restored.blocks:
        q, k = block.dism.expanded_vocabularies(torch.float64)
        torch.testing.assert_close(q, restored.shared_vocab['q'])
        torch.testing.assert_close(k, restored.shared_vocab['q' if tied else 'k'])
    assert parameter_count(LMConfig(dism_output_gla=True, dism_share_vocab_layers=True)) == 46_024_340


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('prob', [0., .5, 1.])
def test_shared_gradient_matches_sum(prob):
    torch.manual_seed(777)
    c = LMConfig(layers=2, width=64, heads=1, ffn_hidden=64, context=65,
                 softcap=30., dism_output_gla=True, dism_share_vocab_layers=True)
    shared = DecoderLM(c).cuda()
    reference = DecoderLM(replace(c, dism_share_vocab_layers=False)).cuda()
    state = shared.state_dict()
    ref_state = {}
    for name in reference.state_dict():
        if name.endswith('q_voc'):
            ref_state[name] = state['shared_vocab.q']
        elif name.endswith('k_voc'):
            ref_state[name] = state['shared_vocab.k']
        else:
            ref_state[name] = state[name]
    reference.load_state_dict(ref_state)
    x = torch.randint(c.vocab_size, (2,65), device='cuda')
    losses = []
    for model in (shared, reference):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, x.roll(-1, 1), prob, torch.Generator(device='cuda').manual_seed(778))
        assert torch.isfinite(loss)
        loss.backward()
        losses.append(loss.detach())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.testing.assert_close(*losses, atol=1e-6, rtol=1e-6)
    for side in ('q','k'):
        expected = sum(getattr(b.dism, side+'_voc').grad for b in reference.blocks)
        torch.testing.assert_close(shared.shared_vocab[side].grad, expected, atol=1e-6, rtol=1e-5)
        if prob < 1:
            assert expected.abs().sum() > 0
    optimizer = torch.optim.AdamW(parameter_groups(shared,.01), lr=.001)
    optimizer.step()
    assert all(b.dism.shared_vocab is shared.shared_vocab for b in shared.blocks)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        assert torch.isfinite(shared.forward_features(x,1.)).all()
