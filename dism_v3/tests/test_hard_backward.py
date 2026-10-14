"""Standalone hard quadratic B1 prototype; production dispatch is unchanged."""
import pytest
import torch
import numpy as np

from flash_dism.hard_backward import backward_hard
from flash_dism.hard_forward import forward_hard
from flash_dism.reference.dism_v3_ref import dism_ref_backward
from test_hard_forward import fixture


def reference(x, dout):
    sq,sk,v,iq,ik,hard,tau=x
    b,n,h,_=sq.shape
    q=torch.zeros(b,n,h,32,device=sq.device,dtype=torch.float64)
    lse=torch.zeros(b,n,h,device=sq.device,dtype=torch.float64)
    direction=torch.ones(b,h,device=sq.device,dtype=torch.bool)
    result=dism_ref_backward(q,q,sq.double(),sk.double(),lse,lse,iq,ik,direction,
                            torch.ones_like(iq,dtype=torch.bool),v.double(),tau.double(),
                            dout.double()*hard[:,None,:,None])
    return [result[name] for name in ('sq_vec','sk_vec','v','rtau')]


@pytest.mark.parametrize('pattern',['random','equal','disjoint','periodic','boundary'])
@pytest.mark.parametrize('n',[256,512])
def test_backward_oracle(n,pattern):
    x=fixture(n,pattern)
    dout=torch.randn_like(x[2])
    _,norm,state=forward_hard(*x,save_state=True,ctas=1)
    got=backward_hard(*x[:3],dout,*x[3:6],norm,state,ctas=1)
    expected=reference(x,dout)
    for actual,wanted in zip(got[:3],expected[:3]):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(),wanted,atol=.008,rtol=.025)
        error=(actual.double()-wanted).norm()/wanted.norm().clamp_min(1e-8)
        assert error<.01,error.item()
        assert not actual.masked_select(~x[5][:,None,:,None]).count_nonzero()
    torch.testing.assert_close(got[3].double(),expected[3],atol=.003,rtol=.01)


@pytest.mark.parametrize('tau',[-4.,-1e-4,-1e-6,0.,1e-6,1e-4,.1,.5,4.])
def test_tau_long_chain(tau):
    x=list(fixture(1024,'equal',b=1,h=1))
    x[-1].fill_(tau)
    dout=torch.randn_like(x[2])
    _,norm,state=forward_hard(*x,save_state=True)
    got=backward_hard(*x[:3],dout,*x[3:6],norm,state)
    expected=reference(x,dout)
    torch.testing.assert_close(got[3].double(),expected[3],atol=.003,rtol=.01)


def test_all_soft_skipped():
    x=list(fixture(256))
    x[5].zero_()
    _,norm,state=forward_hard(*x,save_state=True)
    got=backward_hard(*x[:3],torch.randn_like(x[2]),*x[3:6],norm,state)
    assert all(not t.count_nonzero() for t in got)


@pytest.mark.parametrize('r,dv',[(16,32),(16,64),(32,32),(32,64)])
def test_dimensions(r,dv):
    x=fixture(256,'boundary',r=r,dv=dv)
    dout=torch.randn_like(x[2])
    _,norm,state=forward_hard(*x,save_state=True)
    got=backward_hard(*x[:3],dout,*x[3:6],norm,state)
    expected=reference(x,dout)
    for actual,wanted in zip(got[:3],expected[:3]):
        torch.testing.assert_close(actual.double(),wanted,atol=.008,rtol=.025)
    torch.testing.assert_close(got[3].double(),expected[3],atol=.003,rtol=.01)


def test_reject_token_mask():
    x=fixture(256)
    _,norm,state=forward_hard(*x,save_state=True)
    with pytest.raises(RuntimeError,match='sequence hard flags'):
        backward_hard(*x[:3],x[2],*x[3:5],x[5][:,:,None].expand_as(x[3]).contiguous(),norm,state)


def test_graph_replay():
    x=fixture(256,'boundary')
    dout=torch.randn_like(x[2])
    _,norm,state=forward_hard(*x,save_state=True)
    def call():
        return backward_hard(*x[:3],dout,*x[3:6],norm,state)
    expected=call()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual=call()
    for _ in range(3): graph.replay()
    torch.cuda.synchronize()
    for a,e in zip(actual,expected):
        torch.testing.assert_close(a,e,atol=.0001,rtol=.001)


def test_sanitizer_smoke():
    """Isolate native kernels from the FP64 oracle's ATen reduction kernels."""
    for pattern in ('random','equal','boundary'):
        x=fixture(512,pattern)
        dout=torch.randn_like(x[2])
        _,norm,state=forward_hard(*x,save_state=True,ctas=1)
        outputs=[]
        for ctas in (1,0):
            got=backward_hard(*x[:3],dout,*x[3:6],norm,state,ctas=ctas)
            outputs.append([a.float().cpu().numpy() for a in got])
        for a,b in zip(*outputs):
            assert np.isfinite(a).all() and np.isfinite(b).all()
            np.testing.assert_allclose(a,b,atol=.0001,rtol=.001)
