"""QB direction guards, disabled compatibility and exact synthetic resume."""
import copy
import pytest
import torch
from nanochat.models import ModelSpec, build_model, PretrainOptimizerConfig


def spec(**options):
    config=dict(sequence_len=256,vocab_size=128,n_layer=2,n_head=2,n_embd=64,
                intermediate_size=128,head_dim=32,value_dim=32,readout_dim=16,
                qk_vocab_size=32,rear_half_dism=True,direction_policy='true',post_norm=True,
                qb_q_strength=.25,qb_k_strength=.5,qb_samples=64,qb_iterations=4,
                qb_update_interval=1,qb_update_rate=.2)
    config.update(options)
    return ModelSpec('dism_gdn_hybrid',1,config)


@pytest.mark.parametrize('options', [dict(direction_policy='random'),dict(direction_policy='false'),
                                    dict(direction_antithetic=True)])
def test_unsupported_direction_rejected_at_construction(options):
    with pytest.raises(ValueError,match='QB requires'):
        build_model(spec(**options))


def test_disabled_has_no_new_state_or_parameters():
    baseline=spec(qb_q_strength=0.,qb_k_strength=0.)
    old_config={k:v for k,v in baseline.config.items() if not k.startswith('qb_')}
    a=build_model(baseline)
    b=build_model(ModelSpec(baseline.architecture,baseline.architecture_version,old_config))
    assert a.state_dict().keys()==b.state_dict().keys()
    assert not any('.qb.' in name for name in a.state_dict())
    assert [(n,p.shape) for n,p in a.named_parameters()]==[(n,p.shape) for n,p in b.named_parameters()]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_qb_training_resume_eval_and_direction_flip():
    torch.manual_seed(51)
    model=build_model(spec()).cuda().train()
    model.init_weights()
    model.set_training_step(1000)
    x=torch.randint(128,(1,256),device='cuda')
    y=torch.randint(128,(1,256),device='cuda')
    cu=torch.tensor([0,256],device='cuda',dtype=torch.int32)
    opt_config=PretrainOptimizerConfig(.0015,.0015,.0015,.0015,.01)
    optimizer=model.setup_pretraining_optimizer(opt_config)
    loss=model(x,y,cu_seqlens=cu)
    loss.backward()
    optimizer.step()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    qb=model.layers[1].attn.qb
    assert int(qb.forward_count)==1 and int(qb.update_count)==1
    assert qb.q_price.norm()>0 and qb.k_price.norm()>0
    state=copy.deepcopy(model.state_dict())
    clone=build_model(spec()).cuda().train()
    clone.load_state_dict(state,strict=True)
    restored_optimizer=clone.setup_pretraining_optimizer(opt_config)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert [len(g['params']) for g in optimizer.param_groups]==[len(g['params']) for g in restored_optimizer.param_groups]
    model.zero_grad(set_to_none=True)
    loss_a=model(x,y,cu_seqlens=cu)
    loss_b=clone(x,y,cu_seqlens=cu)
    torch.testing.assert_close(loss_a,loss_b,atol=0,rtol=0)
    loss_a.backward();loss_b.backward()
    for (_,a),(_,b) in zip(model.named_parameters(),clone.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=1e-5,rtol=1e-4)
    optimizer.step();restored_optimizer.step()
    for a,b in zip(model.parameters(),clone.parameters()):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    for a,b in zip(model.buffers(),clone.buffers()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    model.eval()
    frozen=copy.deepcopy(model.state_dict())
    with torch.no_grad(): model(x,y,cu_seqlens=cu)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,frozen[name],atol=0,rtol=0)
    with pytest.raises(ValueError,match='direction_flip'):
        model(x,y,cu_seqlens=cu,direction_flip=True)
    assert torch.equal(model.rng_counter,frozen['rng_counter'])
    with pytest.raises(ValueError,match='fixed direction=true'):
        model.layers[1].attn.qb(None,None,None,None,None,None,None,None,1.,torch.ones(1,dtype=torch.bool),None)
    # Usage probes must follow the prices and must not update them.
    from nanochat.metrics import probe_qk_vocab_usage
    stats=probe_qk_vocab_usage(model,x,cu,256)
    assert 1 in stats and 'coverage_per_head' in stats[1]['q']
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,frozen[name],atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('backend', ['eager','inductor'])
def test_qb_compile_graph_break_keeps_one_update(backend):
    model=build_model(spec()).cuda().train()
    model.init_weights()
    x=torch.zeros((1,256),device='cuda',dtype=torch.long)
    cu=torch.tensor([0,256],device='cuda',dtype=torch.int32)
    with torch._inductor.config.patch(compile_threads=1), torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled=torch.compile(model,backend=backend,fullgraph=False)
        for step in range(2):
            loss=compiled(x,x,cu_seqlens=cu)
            loss.backward()
            assert int(model.layers[1].attn.qb.forward_count)==step+1
            assert int(model.rng_counter)==step+1
            model.zero_grad(set_to_none=True)
