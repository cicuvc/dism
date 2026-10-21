import pytest
import torch
from nanochat.models import ModelSpec, build_model


@pytest.mark.parametrize('training,expected', [(True, True), (False, True)])
def test_static_direction_dispatch(training, expected):
    spec = ModelSpec('dism_gdn_hybrid',1,dict(
        sequence_len=256,vocab_size=128,n_layer=2,n_head=2,n_embd=64,
        intermediate_size=128,head_dim=32,value_dim=32,readout_dim=16,
        qk_vocab_size=32))
    model = build_model(spec).cuda()
    model.init_weights()
    model.train(training)
    captured = []
    handle = model.layers[1].attn.register_forward_pre_hook(
        lambda module,args,kwargs: captured.append(kwargs['direction_true']),with_kwargs=True)
    try:
        with torch.no_grad():
            model(torch.zeros((1,256),device='cuda',dtype=torch.long),
                  cu_seqlens=torch.tensor([0,256],device='cuda',dtype=torch.int32))
    finally:
        handle.remove()
    assert captured == [expected]
    assert model.rng_counter.item() == int(training)
