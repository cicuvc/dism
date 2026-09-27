from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from dism_v2.lm_model import LMConfig, DismAttention, DecoderLM, parameter_count


@pytest.mark.parametrize('mode', ['baseline', 'vocab_silu', 'no_qk_silu'])
def test_same_initial_weights_and_count(mode):
    torch.manual_seed(777)
    base = DismAttention(LMConfig())
    torch.manual_seed(777)
    variant = DismAttention(LMConfig(dism_activation=mode))
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, variant.state_dict()[name], atol=0, rtol=0)
    assert parameter_count(LMConfig(dism_activation=mode)) == 49_678_876
    assert variant.qd_conv.activation == (None if mode == 'no_qk_silu' else 'swish')
    assert variant.v_dism.activation == 'silu'


def test_silu_codebook_chain_rule():
    module = DismAttention(LMConfig(dism_activation='vocab_silu'))
    q, k = module.expanded_vocabularies(torch.bfloat16)
    q.retain_grad(); k.retain_grad()
    (q.float().square().sum()+k.float().square().sum()).backward()
    for master, effective in ((module.q_voc, q), (module.k_voc, k)):
        clone = master.detach().requires_grad_()
        expected = torch.autograd.grad(F.silu(clone), clone, effective.grad.float())[0]
        torch.testing.assert_close(master.grad, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('mode', ['baseline', 'vocab_silu', 'no_qk_silu'])
@pytest.mark.parametrize('probability', [0., .5, 1.])
def test_model_gradients(mode, probability):
    torch.manual_seed(777)
    cfg = LMConfig(layers=1, vocab_size=128, ffn_hidden=128, context=128,
                   softcap=30., dism_activation=mode)
    model = DecoderLM(cfg).cuda()
    ids = torch.randint(128, (2, 128), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        features = model.forward_features(ids, probability, torch.Generator(device='cuda').manual_seed(778))
        logits = model.lm_head(features).float()
    F.cross_entropy((30*(logits/30).tanh()).flatten(0, 1), ids.flatten()).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    grad = model.blocks[0].dism.q_voc.grad
    assert grad.abs().sum() > 0 if probability < 1 else grad.count_nonzero() == 0
