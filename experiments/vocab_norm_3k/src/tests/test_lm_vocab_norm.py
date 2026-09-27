import pytest
import torch
from dism_v2.lm_model import LMConfig, DismAttention, DecoderLM, parameter_count


def test_initialization_and_norm():
    torch.manual_seed(777)
    base = DismAttention(LMConfig())
    torch.manual_seed(777)
    norm = DismAttention(LMConfig(dism_vocab_norm=True))
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, norm.state_dict()[name], atol=0, rtol=0)
    assert parameter_count(LMConfig(dism_vocab_norm=True)) == 49_678_876
    for e in norm.expanded_vocabularies(torch.float32):
        torch.testing.assert_close(e.square().mean(-1), torch.ones_like(e[..., 0]), atol=3e-6, rtol=0)


@pytest.mark.parametrize('tied', [False, True])
def test_grouped_normalization_gradient(tied):
    m = DismAttention(LMConfig(dism_vocab_norm=True, dism_vocab_groups=2, dism_tie_qk_vocab=tied))
    q, k = m.expanded_vocabularies(torch.float32)
    parameters = [m.q_voc] if tied else [m.q_voc, m.k_voc]
    weights = [torch.randn_like(q), torch.randn_like(k)]
    grads = torch.autograd.grad((q*weights[0]).sum() + (k*weights[1]).sum(), parameters)
    raw = [p.detach().clone().requires_grad_() for p in parameters]
    eq = torch.nn.functional.rms_norm(raw[0], (64,), eps=1e-6).repeat_interleave(2, 0)
    ek = torch.nn.functional.rms_norm(raw[0] if tied else raw[1], (64,), eps=1e-6).repeat_interleave(2, 0)
    expected = torch.autograd.grad((eq*weights[0]).sum() + (ek*weights[1]).sum(), raw)
    for actual, reference in zip(grads, expected):
        torch.testing.assert_close(actual, reference, atol=3e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('prob', [0., .5, 1.])
def test_training_smoke(prob):
    torch.manual_seed(777)
    c = LMConfig(layers=1, heads=1, width=64, ffn_hidden=64, context=65,
                 softcap=30., dism_vocab_norm=True)
    m = DecoderLM(c).cuda()
    tokens = torch.randint(c.vocab_size, (2, 65), device='cuda')
    optimizer = torch.optim.AdamW(m.parameters(), lr=.001)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = m(tokens, tokens.roll(-1, 1), prob, torch.Generator(device='cuda').manual_seed(778))
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    if prob < 1:
        assert m.blocks[0].dism.q_voc.grad.abs().sum() > 0
        assert m.blocks[0].dism.k_voc.grad.abs().sum() > 0
    optimizer.step()
    m.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        features = m.forward_features(tokens, 1.)
    assert torch.isfinite(features).all()
