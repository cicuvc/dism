import pytest
import torch
from dism_v2.lm_model import LMConfig, DecoderLM, parameter_count, parameter_groups


def config(**overrides):
    args = dict(width=384, heads=6, layers=12, ffn_hidden=1024, softcap=30.)
    args.update(overrides)
    return LMConfig(**args)


def test_parameters_and_shapes():
    assert parameter_count(config()) == 72_188_040
    with torch.device('meta'):
        model = DecoderLM(config())
    assert model.embedding.weight is not model.lm_head.weight
    assert model.embedding.weight.shape == (50257,384)
    assert len(model.blocks) == 12
    for block in model.blocks:
        assert block.dism.q_voc.shape == (6,512,64)
        assert block.dism.k_voc.shape == (6,512,64)
        assert block.up.weight.shape == (2048,384)


def train_once(c, batch, probability):
    torch.manual_seed(777)
    model = DecoderLM(c).cuda()
    optimizer = torch.optim.AdamW(parameter_groups(model,.01), lr=.001, fused=True)
    x = torch.randint(50257, (batch,c.context), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(x, x.roll(-1,1), probability, torch.Generator(device='cuda').manual_seed(778))
    assert torch.isfinite(loss)
    loss.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    print(dict(loss=loss.item(), grad_norm=norm.item(), batch=batch, layers=c.layers,
               context=c.context, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),flush=True)
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        assert torch.isfinite(model.forward_features(x[:1,:65],1.)).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('probability',[0.,.5,1.])
def test_six_head_smoke(probability):
    train_once(config(layers=1,context=65),2,probability)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_full_training_microbatch():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_once(config(),8,.5)
