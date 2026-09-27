import pytest
import torch
from dism_v2.lm_model import LMConfig, DecoderLM, parameter_count, parameter_groups


def test_size_and_parameters():
    c = LMConfig(qk_vocab=128)
    assert parameter_count(c) == 46_729_756
    with torch.device('meta'):
        model = DecoderLM(c)
    for block in model.blocks:
        assert block.dism.q_voc.shape == (4, 128, 64)
        assert block.dism.k_voc.shape == (4, 128, 64)
    assert not c.dism_vocab_norm and not c.dism_output_gla


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('prob', [0., .5, 1.])
def test_training_smoke(prob):
    torch.manual_seed(777)
    c = LMConfig(layers=1, heads=1, width=64, ffn_hidden=64, context=65,
                 softcap=30., qk_vocab=128)
    model = DecoderLM(c).cuda()
    optimizer = torch.optim.AdamW(parameter_groups(model, .01), lr=.001)
    tokens = torch.randint(c.vocab_size, (2,65), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(tokens, tokens.roll(-1, 1), prob, torch.Generator(device='cuda').manual_seed(778))
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    if prob < 1:
        assert model.blocks[0].dism.q_voc.grad.abs().sum() > 0
        assert model.blocks[0].dism.k_voc.grad.abs().sum() > 0
    optimizer.step()
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        assert torch.isfinite(model.forward_features(tokens, 1.)).all()
