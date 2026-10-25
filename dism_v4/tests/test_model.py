import io

import pytest
import torch

from nanochat.models import ModelSpec, PretrainOptimizerConfig, build_model, optimizer_schema

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA model')


def _spec(mxfp8=False):
    return ModelSpec('dism_v4_gdn_hybrid', 1, dict(
        sequence_len=2048, vocab_size=128, n_layer=2, layer_layout='gd',
        n_head=4 if mxfp8 else 2, n_embd=128 if mxfp8 else 64,
        intermediate_size=256 if mxfp8 else 128, head_dim=32, value_dim=32,
        readout_dim=32, qk_vocab_size=32, anneal_steps=101,
        mxfp8=mxfp8, mxfp8_optimizer=mxfp8))


@pytest.mark.parametrize('mxfp8',[False,True])
def test_2k_training_eval_resume_and_packed_rows(mxfp8):
    torch.manual_seed(103)
    model = build_model(_spec(mxfp8)).cuda()
    model.init_weights()
    attn = model.layers[1].attn
    assert not hasattr(attn, 'sq_proj') and not hasattr(attn, 'sk_proj')
    assert attn.q_conv.weight is attn.gdn_q_conv.weight
    config = PretrainOptimizerConfig(.001, .001, .001, .001, .01)
    opt = model.setup_pretraining_optimizer(config)
    schema = optimizer_schema(model, opt)
    no_decay = {id(p) for p in opt.param_groups[1]['params']}
    for name, p in model.named_parameters():
        if any(s in name for s in ('q_vocab', 'k_vocab', 'log_sel_tau')):
            assert id(p) in no_decay
    x = torch.randint(128, (2,2048), device='cuda')
    y = torch.randint(128, x.shape, device='cuda')
    # Multiple documents in a row; the row is still the hard sampling unit.
    cu = torch.tensor([0,256,2048,3072,4096], device='cuda', dtype=torch.int32)
    before = attn.q_vocab.detach().clone()
    model.train()
    model.set_training_step(0)
    loss = model(x,y,cu_seqlens=cu)
    loss.backward()
    for p in (attn.q_vocab, attn.k_vocab, attn.log_sel_tau):
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and p.grad.norm()>0
    opt.step(); opt.zero_grad(set_to_none=True)
    assert not torch.equal(before, attn.q_vocab)
    model.set_training_step(50)
    stream = io.BytesIO()
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), stream)
    stream.seek(0)
    snapshot = torch.load(stream, weights_only=True)
    expected_counter = model.rng_counter.clone()
    expected_probability = model.hard_probability.clone()
    loss = model(x,y,cu_seqlens=cu)
    loss.backward()
    grads = {name: None if p.grad is None else p.grad.clone() for name,p in model.named_parameters()}
    opt.step(); opt.zero_grad(set_to_none=True)
    restored = build_model(_spec(mxfp8)).cuda()
    restored.load_state_dict(snapshot['model'])
    opt2 = restored.setup_pretraining_optimizer(config)
    opt2.load_state_dict(snapshot['optimizer'])
    assert optimizer_schema(restored, opt2) == schema
    assert torch.equal(restored.rng_counter, expected_counter)
    assert torch.equal(restored.hard_probability, expected_probability)
    assert int(restored.training_step) == 50
    assert restored.layers[1].attn.q_conv.weight is restored.layers[1].attn.gdn_q_conv.weight
    replay = restored(x,y,cu_seqlens=cu)
    replay.backward()
    torch.testing.assert_close(replay,loss,atol=1e-5,rtol=1e-5)
    for name,p in restored.named_parameters():
        if grads[name] is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad,grads[name],atol=2e-4,rtol=.02)
            # Isolate optimizer serialization from BF16/atomic replay noise.
            # Adam can amplify near-zero gradient differences into visibly
            # different updates even when the forward/backward replay agrees.
            p.grad.copy_(grads[name])
    opt2.step(); opt2.zero_grad(set_to_none=True)
    for (name,p),(_,r) in zip(model.named_parameters(),restored.named_parameters()):
        torch.testing.assert_close(p,r,atol=0,rtol=0,msg=lambda m:name+'\n'+m)
    assert int(restored.rng_counter) == int(model.rng_counter) == 2

    restored.eval()
    counter = restored.rng_counter.clone()
    with torch.no_grad():
        for hard in (False,True):
            restored.eval_hard = hard
            losses = restored(x,y,cu_seqlens=cu,loss_reduction='none')
            assert losses.shape == (4096,) and bool(torch.isfinite(losses).all())
    assert torch.equal(restored.rng_counter,counter)
    with pytest.raises(ValueError,match='planner/CUDA graph'):
        restored(x,y,cu_seqlens=cu,graph_layout=object())


def test_model_flags_stay_per_row(monkeypatch):
    from flash_dism_v4 import module
    original = module.dism_v4_blocked_from_embeddings
    calls=[]

    def observe(q,k,*args,**kwargs):
        calls.append((q.shape, kwargs['hard'].shape, kwargs['segment_ids'].clone()))
        return original(q,k,*args,**kwargs)

    monkeypatch.setattr(module,'dism_v4_blocked_from_embeddings',observe)
    model = build_model(_spec()).cuda()
    model.init_weights()
    x = torch.randint(128,(2,256),device='cuda')
    cu = torch.tensor([0,256,512],device='cuda',dtype=torch.int32)
    with torch.no_grad():
        model(x,x,cu_seqlens=cu)
    shape, flags, segments = calls[0]
    assert shape == (2,256,2,32) and flags == (2,2)
    assert segments[0].unique().numel() == segments[1].unique().numel() == 1
    assert segments[0,0] != segments[1,0]
