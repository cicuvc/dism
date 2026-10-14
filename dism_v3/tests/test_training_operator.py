"""Default module/operator dispatch and storage-owned asynchronous AOT state."""
import gc
import weakref
import pytest
import torch
import cu_flash_dism as cu
from torch.utils.checkpoint import checkpoint

from flash_dism import DismAttention, voc_dism, dism_core
from flash_dism.compiler import voc_forward
from flash_dism.hard_training import shutdown_hard_planners
from test_hard_training import inputs


def test_handle_storage_ownership():
    class Payload:
        pass
    p = Payload()
    ref = weakref.ref(p)
    count = cu.training_plan_handle_count()
    token = cu.hold_training_plan(p)
    view = token.detach().view(1)
    copied = token.clone()
    del token, p
    gc.collect()
    assert cu.get_training_plan(view) is ref() and ref() is not None
    with pytest.raises(RuntimeError, match='must not be cloned'):
        cu.get_training_plan(copied)
    del view
    gc.collect()
    assert ref() is None
    assert cu.training_plan_handle_count() == count


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('backend', ['aot_eager', 'inductor'])
def test_fullgraph_vocab_forward_backward(packed, backend):
    xs, layout = inputs(packed, 'mixed')
    eq, ek = [torch.randn(2, 32, 32, device='cuda').requires_grad_() for _ in range(2)]
    args = [*xs[:5], eq, ek, xs[-1]]
    boundaries = torch.tensor([0,256,512,512,1024], dtype=torch.int32) if packed else None
    def call(*a):
        return voc_forward(*a, torch.ones((), device='cuda'), xs[9], xs[10], boundaries)[0]
    expected = voc_dism(*args, direction=xs[9], hard=xs[10], layout=layout)
    do = torch.randn_like(expected)
    eg = torch.autograd.grad(expected, args, do)
    count = cu.training_plan_handle_count()
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True), \
         torch._inductor.config.patch(compile_threads=1), \
         torch._functorch.config.patch(donated_buffer=False):
        compiled = torch.compile(call, fullgraph=True, backend=backend)
        actual = compiled(*args)
        ag = torch.autograd.grad(actual, args, do, retain_graph=True)
        again = torch.autograd.grad(actual, args, do)
        torch.testing.assert_close(actual, expected, atol=.001, rtol=.03)
        for a, b, e in zip(ag, again, eg):
            torch.testing.assert_close(a, e, atol=.001, rtol=.03)
            torch.testing.assert_close(a, b, atol=.001, rtol=.03)
        del actual, ag, again
        gc.collect()
        assert cu.training_plan_handle_count() == count
        # A discarded compiled graph must release its plan without backward.
        abandoned = compiled(*args)
        del abandoned
        gc.collect()
        assert cu.training_plan_handle_count() == count


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('mode', ['soft', 'hard', 'mixed'])
def test_module_plug_and_play(packed, mode):
    torch.manual_seed(729)
    model = DismAttention(64,2,head_dim=32,value_dim=32,readout_dim=16,vocab_size=32).cuda()
    x = torch.randn(1,512,64,device='cuda',requires_grad=True)
    hard = torch.tensor([[mode != 'soft', mode == 'hard']], device='cuda')
    direction = torch.tensor([[True,False]], device='cuda')
    bounds = torch.tensor([0,256,512], dtype=torch.int32) if packed else None
    kwargs = dict(hard=hard,direction=direction,cu_seqlens=bounds,use_cache=False)
    if packed:
        kwargs['max_seqlen'] = 512
    keys = tuple(model.state_dict())
    with torch.autocast('cuda',dtype=torch.bfloat16):
        eager = model(x,**kwargs)[0]
    eg = torch.autograd.grad(eager.float().square().mean(), [x,*model.parameters()])
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True), \
         torch._inductor.config.patch(compile_threads=1):
        compiled = torch.compile(model,fullgraph=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            actual = compiled(x,**kwargs)[0]
        ag = torch.autograd.grad(actual.float().square().mean(), [x,*model.parameters()])
    torch.testing.assert_close(actual,eager,atol=.005,rtol=.02)
    for a,e in zip(ag,eg):
        torch.testing.assert_close(a,e,atol=.001,rtol=.03)
    assert tuple(model.state_dict()) == keys
    if mode == 'hard':
        grads = dict(zip(dict(model.named_parameters()),ag[1:]))
        assert torch.count_nonzero(grads['q_vocab']) == 0
        assert torch.count_nonzero(grads['k_vocab']) == 0
        assert grads['sq_proj.weight'].abs().sum() > 0


def test_no_grad_does_not_create_runtime(monkeypatch):
    import flash_dism.hard_training as ht
    shutdown_hard_planners()
    def forbidden(*a,**kw):
        raise AssertionError('inference must not create a planner')
    monkeypatch.setattr(ht,'default_hard_planner',forbidden)
    xs,_=inputs(mode='hard')
    eq,ek=[torch.randn(2,32,32,device='cuda') for _ in range(2)]
    args=[*xs[:5],eq,ek,xs[-1]]
    with torch.no_grad():
        a=voc_dism(*args,direction=xs[9],hard=xs[10])
        b=voc_forward(*args,torch.ones((),device='cuda'),xs[9],xs[10],None)[0]
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert a.abs().sum()>0


@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('vocabulary',[False,True])
def test_public_operator_fullgraph(packed,vocabulary):
    xs,layout=inputs(packed,'mixed')
    bounds=torch.tensor([0,256,512,512,1024],dtype=torch.int32) if packed else None
    eq,ek=[torch.randn(2,32,32,device='cuda').requires_grad_() for _ in range(2)]
    if vocabulary:
        args=[*xs[:5],eq,ek,xs[-1]]
        def call(*a):
            return voc_dism(*a,direction=xs[9],hard=xs[10],cu_seqlens=bounds)
    else:
        args=list(xs)
        def call(*a):
            return dism_core(*a,cu_seqlens=bounds)
    active=[a for a in args if a.requires_grad]
    expected=call(*args)
    do=torch.randn_like(expected)
    eg=torch.autograd.grad(expected,active,do)
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True), \
         torch._inductor.config.patch(compile_threads=1):
        actual=torch.compile(call,fullgraph=True)(*args)
        ag=torch.autograd.grad(actual,active,do)
    torch.testing.assert_close(actual,expected,atol=.001,rtol=.03)
    for a,e in zip(ag,eg):
        torch.testing.assert_close(a,e,atol=.001,rtol=.03)


@pytest.mark.parametrize('compiled',[False,True])
def test_backward_frees_plan_even_when_output_retained(compiled):
    xs,_=inputs(mode='hard')
    baseline=cu.training_plan_handle_count()
    def call(*a):
        return dism_core(*a)
    fn=torch.compile(call,backend='aot_eager',fullgraph=True) if compiled else call
    out=fn(*xs)
    assert cu.training_plan_handle_count()==baseline+1
    out.sum().backward()
    gc.collect()
    assert cu.training_plan_handle_count()==baseline


@pytest.mark.parametrize('reentrant',[False,True])
def test_compiled_checkpoint_plan_replay(reentrant):
    xs,_=inputs(mode='hard')
    eq,ek=[torch.randn(2,32,32,device='cuda').requires_grad_() for _ in range(2)]
    args=[*xs[:5],eq,ek,xs[-1]]
    def call(*a):
        return voc_dism(*a,direction=xs[9],hard=xs[10])
    expected=call(*args)
    eg=torch.autograd.grad(expected.float().square().mean(),args)
    count=cu.training_plan_handle_count()
    with torch._inductor.config.patch(compile_threads=1):
        compiled=torch.compile(call,fullgraph=True)
        result=checkpoint(compiled,*args,use_reentrant=reentrant)
        result.float().square().mean().backward()
    torch.testing.assert_close(result,expected,atol=.001,rtol=.03)
    for a,e in zip(args,eg):
        torch.testing.assert_close(a.grad,e,atol=.001,rtol=.03)
    del result
    gc.collect()
    assert cu.training_plan_handle_count()==count


@pytest.mark.parametrize('compiled',[False,True])
def test_core_strided_vectors_and_int64_labels(compiled):
    xs,_=inputs(mode='hard')
    args=list(xs)
    # Preserve the native public preparation contract: channel-strided vectors
    # and integer argmax labels are materialized before the hard CUDA branch.
    for i in (2,3,4):
        x=xs[i]
        args[i]=x.detach().repeat_interleave(2,dim=-1)[...,::2].requires_grad_()
    args[7],args[8]=xs[7].long(),xs[8].long()
    fn=torch.compile(dism_core,fullgraph=True,backend='aot_eager') if compiled else dism_core
    expected=dism_core(*xs)
    actual=fn(*args)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    do=torch.randn_like(actual)
    ag=torch.autograd.grad(actual,[args[i] for i in (2,3,4)],do)
    eg=torch.autograd.grad(expected,[xs[i] for i in (2,3,4)],do)
    for a,e in zip(ag,eg):
        torch.testing.assert_close(a,e,atol=.001,rtol=.03)


@pytest.mark.parametrize('packed',[False,True])
def test_compiled_hard_long_stream_safety(packed):
    xs,layout=inputs(packed,'hard','repeat')
    def call(*a):
        return dism_core(*a,layout=layout)
    count=cu.training_plan_handle_count()
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        out=torch.compile(call,fullgraph=True,backend='aot_eager')(*xs)
        out.float().square().mean().backward()
    assert out.isfinite().all()
    assert xs[2].grad.abs().sum()>0
    assert torch.count_nonzero(xs[0].grad)==0
    gc.collect()
    assert cu.training_plan_handle_count()==count
