from concurrent.futures import Future
from types import SimpleNamespace
import threading

import pytest
import torch

from flash_dism import dism_core
from flash_dism.hard_training import HardPlanTicket, AsyncHardPlanner, shutdown_hard_planners
from flash_dism.hard_backward import backward_hard_recompute
from test_hard_training import inputs, oracle
from gradient_acceptance import assert_gradient


@pytest.mark.parametrize('status',['pending','uploading','ready','failed'])
def test_nonblocking_selection(status):
    future=Future()
    x=torch.ones(1,256,1,32,device='cuda',dtype=torch.bfloat16)
    sentinel=object()
    if status=='failed': future.set_exception(ValueError('planner failed'))
    elif status!='pending':
        def event_backward(*xs):
            return {name:torch.ones_like(t,dtype=torch.float32) for name,t in
                    zip(('sq_vec','sk_vec','v'),xs[:3])} | {'rtau':x.new_ones(1).float()}
        future.set_result(SimpleNamespace(h2d_events=(None,SimpleNamespace(query=lambda: status=='ready')),
                                           kernel=SimpleNamespace(backward=event_backward)))
    ticket=HardPlanTicket(future,None,'ready')
    if status=='failed':
        with pytest.raises(ValueError,match='planner failed'):
            ticket.backward(x,x,x,x,fallback=lambda: sentinel)
    else:
        got=ticket.backward(x,x,x,x,fallback=lambda: sentinel)
        assert ticket.route==('event' if status=='ready' else 'quadratic')
        assert (got is sentinel)==(status!='ready')


@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
def test_fallback_oracle(packed,mode):
    xs,layout=inputs(packed,mode,'repeat')
    do=torch.randn_like(xs[4])*.2
    got=backward_hard_recompute(*xs[2:5],do,xs[7],xs[8],xs[10],xs[11],layout=layout)
    # Oracle for the hard contribution only (soft rows have zero derivative).
    hard=xs[10]
    mask=hard[:,None,:,None] if not packed else hard[:,None,:,None].expand(-1,512,-1,1).reshape(1,1024,2,1)
    refs=[x.detach().double().requires_grad_(x.requires_grad) if x.is_floating_point() else x for x in xs]
    refs[10]=torch.ones_like(hard)
    out=oracle(refs,layout)
    grads=torch.autograd.grad(out,[refs[i] for i in (2,3,4,11)],do.double()*mask)
    for name,g in zip(('sq_vec','sk_vec','v','rtau'),grads): assert_gradient(got[name],g,name)


def test_packed_fallback_does_not_stack_documents(monkeypatch):
    xs,layout=inputs(True,'hard','repeat')
    do=torch.randn_like(xs[4])*.2

    def fail_stack(*args, **kwargs):
        raise AssertionError("packed fallback must consume the document task table")

    monkeypatch.setattr(torch, "stack", fail_stack)
    got=backward_hard_recompute(*xs[2:5],do,xs[7],xs[8],xs[10],xs[11],layout=layout)
    assert all(torch.isfinite(value).all() for value in got.values())


@pytest.mark.parametrize('packed',[False,True])
def test_delayed_planner_no_wait_and_retained_graph(packed):
    xs,layout=inputs(packed,'mixed')
    gate=threading.Event()
    with AsyncHardPlanner(workers=2,backward_policy='ready') as planner:
        original=planner._prepare
        def delayed(*args,**kwargs):
            assert gate.wait(30)
            return original(*args,**kwargs)
        planner._prepare=delayed
        try:
            out=dism_core(*xs,layout=layout,hard_planner=planner)
            g=torch.autograd.grad(out.sum(),xs[2],retain_graph=True)[0]
            # No worker has been allowed to finish: backward must not wait.
            again=torch.autograd.grad(out.sum(),xs[2])[0]
            torch.testing.assert_close(g,again,atol=.001,rtol=.03)
        finally:
            gate.set()


@pytest.mark.parametrize('packed',[False,True])
def test_compiled_fallback(packed,monkeypatch):
    shutdown_hard_planners()
    monkeypatch.setenv('DISM_HARD_BACKWARD','quadratic')
    xs,layout=inputs(packed,'mixed')
    def call(*a): return dism_core(*a,layout=layout)
    try:
        expected=call(*xs)
        eg=torch.autograd.grad(expected.sum(),xs[2])[0]
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            compiled=torch.compile(call,backend='aot_eager',fullgraph=True)
            actual=compiled(*xs)
            ag=torch.autograd.grad(actual.sum(),xs[2])[0]
        torch.testing.assert_close(actual,expected,atol=.001,rtol=.03)
        torch.testing.assert_close(ag,eg,atol=.001,rtol=.03)
    finally:
        shutdown_hard_planners()
