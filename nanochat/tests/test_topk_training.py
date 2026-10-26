"""Opt-in identity and unchanged native soft-path regression."""
import pytest
import torch
from nanochat.models import ModelSpec, build_model
from nanochat.models.dism import DismConfig


def test_top4_identity_preserves_parameter_schema():
    config=dict(n_layer=2,n_embd=128,n_head=4,head_dim=32,value_dim=32,
                readout_dim=16,vocab_size=128,intermediate_size=256)
    legacy=ModelSpec('dism_gdn_hybrid',1,config)
    selected=ModelSpec('dism_gdn_hybrid',1,{**config,'hard_query_topk':4})
    with torch.device('meta'):
        a,b=build_model(legacy),build_model(selected)
    assert a.model_spec.fingerprint()==legacy.fingerprint()
    assert b.model_spec.fingerprint()!=a.model_spec.fingerprint()
    assert set(a.state_dict())==set(b.state_dict())
    assert [(n,p.shape,p.dtype,getattr(p,'_no_weight_decay',False)) for n,p in a.named_parameters()]==[
        (n,p.shape,p.dtype,getattr(p,'_no_weight_decay',False)) for n,p in b.named_parameters()]
    assert b.layers[1].attn.q_conv.weight is b.layers[1].attn.gdn_q_conv.weight
    for value in (0,3,True,4.0):
        with pytest.raises(ValueError,match='hard_query_topk'):DismConfig(**config,hard_query_topk=value)


@pytest.mark.parametrize('sampled',[False,True])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_top4_all_soft_preserves_native(monkeypatch,sampled):
    monkeypatch.setenv('DISM_HARD_PLANNER','0')
    from flash_dism.compiler import voc_forward
    from flash_dism.topk_training import vocabulary
    torch.manual_seed(76)
    b,n,h,d,r,dv,c=2,256,2,32,16,32,32
    q=torch.randn(b,n,h,d,device='cuda',dtype=torch.bfloat16)*.2
    k=torch.randn_like(q)*.2
    sq=torch.randn(b,n,h,r,device='cuda',dtype=torch.bfloat16)*.2;sk=torch.randn_like(sq)*.2
    v=torch.randn(b,n,h,dv,device='cuda',dtype=torch.bfloat16)
    eq=torch.randn(h,c,d,device='cuda')*.2;ek=torch.randn_like(eq)*.2
    tau=torch.tensor([0.,.1],device='cuda')
    args=[t.requires_grad_() for t in (q,k,sq,sk,v,eq,ek,tau)]
    direction=torch.ones(b,h,device='cuda',dtype=torch.bool);hard=torch.zeros_like(direction)
    segments=torch.zeros(b,n,device='cuda',dtype=torch.int32)
    a=voc_forward(*args,1.,direction,hard,None,None,direction_true=True)[0]
    z=vocabulary(*args,direction=direction,hard=hard,segments=segments,sampled=sampled,
                 counter=torch.tensor(17,device='cuda',dtype=torch.int64),seed=1729 if sampled else -1)
    torch.testing.assert_close(z,a,atol=0,rtol=0)
    g=torch.randn_like(a)
    ga=torch.autograd.grad(a,args,g);gz=torch.autograd.grad(z,args,g)
    for x,y in zip(ga,gz):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-4)


def test_sampled_schema_and_config():
    config=dict(n_layer=2,n_embd=128,n_head=4,head_dim=32,value_dim=32,
                readout_dim=16,vocab_size=128,intermediate_size=256,hard_query_topk=4)
    with torch.device('meta'):
        a=build_model(ModelSpec('dism_gdn_hybrid',1,config))
        b=build_model(ModelSpec('dism_gdn_hybrid',1,{**config,'hard_query_sampled':True}))
    assert set(a.state_dict())==set(b.state_dict())
    assert [(n,p.shape,getattr(p,'_no_weight_decay',False)) for n,p in a.named_parameters()]==[
        (n,p.shape,getattr(p,'_no_weight_decay',False)) for n,p in b.named_parameters()]
    assert a.model_spec.fingerprint()!=b.model_spec.fingerprint()
    for kw in ({'hard_query_sampled':1},{'hard_query_topk':1,'hard_query_sampled':True}):
        with pytest.raises(ValueError,match='hard_query_sampled'):DismConfig(**{**config,**kw})


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_sampled_rng_replay_and_distribution():
    from flash_dism.kernels.sampled_hard import sample
    p=torch.tensor([.1,.2,.3,.4],device='cuda').expand(1,1,65536,4).contiguous()
    counter=torch.tensor(37,device='cuda',dtype=torch.int64)
    a=sample(p,counter,1729);b=sample(p,counter,1729)
    assert torch.equal(a,b)
    assert not torch.equal(a,sample(p,counter,1730))
    counts=torch.bincount(a.flatten().long(),minlength=4)/a.numel()
    torch.testing.assert_close(counts,p[0,0,0],atol=.008,rtol=0)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):sample(p,counter,1729)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        snap=counter.clone();counter.add_(1);out=sample(p,snap,1729)
    counter.fill_(37)
    graph.replay();torch.testing.assert_close(out,a,rtol=0,atol=0)
    expected=sample(p,counter,1729)
    graph.replay();torch.testing.assert_close(out,expected,rtol=0,atol=0)
    assert counter.item()==39 and not torch.equal(out,a)


@pytest.mark.parametrize('topk',[2,4])
@pytest.mark.parametrize('mxfp8',[False,True])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_topk_planner_packed_graph(monkeypatch,topk,mxfp8):
    from test_cuda_graph import test_packed_replay_matches_training
    test_packed_replay_matches_training(monkeypatch,mxfp8,False,True,hard_query_topk=topk)


@pytest.mark.parametrize('topk',[2,4])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_topk_eager_event_autograd(monkeypatch,topk):
    from test_cuda_graph import model_pair
    from flash_dism.topk_planner import shutdown_topk_planners
    from flash_dism.hard_training import take_route_log
    shutdown_topk_planners()
    monkeypatch.setenv('DISM_HARD_PLANNER','1')
    monkeypatch.setenv('DISM_HARD_BACKWARD','event')
    monkeypatch.setenv('DISM_PLANNER_WORKERS','2')
    model,reference=model_pair(False,True,hard_query_topk=topk)
    model.set_training_step(9);reference.set_training_step(9)
    ids=torch.randint(128,(2,512),device='cuda');targets=torch.randint(128,ids.shape,device='cuda')
    bounds=torch.tensor([0,256,512,1024],dtype=torch.int32,device='cuda')
    try:
        take_route_log()
        loss=model(ids,targets,cu_seqlens=bounds);loss.backward()
        assert take_route_log()==['e']
        monkeypatch.setenv('DISM_HARD_PLANNER','0')
        want=reference(ids,targets,cu_seqlens=bounds);want.backward()
        torch.testing.assert_close(loss,want,atol=0,rtol=0)
        for (name,p),(_,q) in zip(model.named_parameters(),reference.named_parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=.001,rtol=.03,msg=name)
        assert model.rng_counter.item()==reference.rng_counter.item()==1
    finally:shutdown_topk_planners()


@pytest.mark.parametrize('topk',[2,4])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_topk_no_grad_does_not_submit_training_plan(monkeypatch,topk):
    from test_cuda_graph import model_pair
    import flash_dism.topk_planner as planner
    monkeypatch.setenv('DISM_HARD_PLANNER','1')
    def unexpected(*args,**kwargs):raise AssertionError('no-grad evaluation submitted a backward plan')
    monkeypatch.setattr(planner,'default_topk_planner',unexpected)
    model,_=model_pair(False,True,hard_query_topk=topk)
    model.eval()
    ids=torch.randint(128,(1,256),device='cuda')
    bounds=torch.tensor([0,256],device='cuda',dtype=torch.int32)
    with torch.no_grad():loss=model(ids,ids,cu_seqlens=bounds)
    assert torch.isfinite(loss) and not loss.requires_grad
