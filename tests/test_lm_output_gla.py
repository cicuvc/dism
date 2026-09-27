import pytest
import torch
from dism_v2.lm_model import LMConfig, DecoderLM, parameter_count, parameter_groups


def test_configuration_and_parameters():
    assert parameter_count(LMConfig(dism_output_gla=True)) == 49_694_356
    c = LMConfig(layers=1, heads=1, width=64, ffn_hidden=64,
                 softcap=30., dism_output_gla=True)
    model = DecoderLM(c)
    groups = parameter_groups(model, .01)
    no_decay = {id(p) for p in groups[1]['params']}
    assert id(model.blocks[0].dism.A_log) in no_decay
    assert id(model.blocks[0].dism.dt_bias) in no_decay
    with pytest.raises(ValueError, match='requires architecture'):
        DecoderLM(LMConfig(architecture='swa_only', dism_output_gla=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gla_against_recurrence():
    from fla.ops.simple_gla import chunk_simple_gla
    torch.manual_seed(21)
    q, k, v = [(torch.randn(1, 65, 1, 64, device='cuda') * .15)
               .bfloat16().requires_grad_() for _ in range(3)]
    g = torch.full((1, 65, 1), -.1, device='cuda', requires_grad=True)
    out, _ = chunk_simple_gla(q, k, v, g, scale=64**-.5)
    state = torch.zeros(1, 1, 64, 64, device='cuda')
    expected = []
    for t in range(65):
        state = g[:, t].exp()[..., None, None] * state + k[:, t].float()[..., :, None] * v[:, t].float()[..., None, :]
        expected.append(torch.einsum('bhk,bhkv->bhv', q[:, t].float(), state) / 8)
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(out.float(), expected, atol=2e-4, rtol=.02)
    weight = torch.randn_like(out)
    actual_grads = torch.autograd.grad((out * weight).sum(), (q, k, v, g), retain_graph=True)
    reference_grads = torch.autograd.grad((expected * weight.float()).sum(), (q, k, v, g))
    for actual, reference in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual.float(), reference.float(), atol=.002, rtol=.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('prob', [0., .5, 1.])
def test_model_training_smoke(prob):
    torch.manual_seed(777)
    c = LMConfig(layers=1, heads=1, width=64, ffn_hidden=64, context=129,
                 softcap=30., dism_output_gla=True)
    model = DecoderLM(c).cuda()
    optimizer = torch.optim.AdamW(parameter_groups(model, .01), lr=.001)
    tokens = torch.randint(c.vocab_size, (1, 129), device='cuda')
    gen = torch.Generator(device='cuda').manual_seed(778)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(tokens, tokens.roll(-1, 1), prob, gen)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for name in ('a_proj.weight', 'A_log', 'dt_bias'):
        assert dict(model.blocks[0].dism.named_parameters())[name].grad.abs().sum() > 0
    optimizer.step()
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        full = model.forward_features(tokens, 1.)
        prefix = model.forward_features(tokens[:, :64], 1.)
    assert torch.isfinite(full).all()
    torch.testing.assert_close(full[:, :64], prefix, atol=.04, rtol=.04)
