import pytest
import torch
from dism_v2.lm_model import LMConfig, DecoderLM, parameter_count
from test_lm_width384 import train_once


def config(**overrides):
    args = dict(width=384, heads=6, layers=12, ffn_hidden=1024,
                softcap=30., gdn_dism=True)
    args.update(overrides)
    return LMConfig(**args)


def test_layout_and_parameters():
    from fla.layers.gated_deltanet import GatedDeltaNet
    assert parameter_count(config()) == 66_593_550
    with torch.device('meta'):
        model = DecoderLM(config())
    assert model.embedding.weight is not model.lm_head.weight
    for i,b in enumerate(model.blocks):
        assert b.swa is None
        if i%4 == 3:
            assert b.dism is not None and b.gdn is None
            assert b.dism.q_voc.shape == (6,512,64)
        else:
            assert isinstance(b.gdn,GatedDeltaNet) and b.dism is None
            assert b.gdn.head_k_dim == 64 and b.gdn.head_v_dim == 128


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('prob',[0.,.5,1.])
def test_group_smoke(prob):
    train_once(config(layers=4,context=65),2,prob)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_full_microbatch():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_once(config(),8,.5)
