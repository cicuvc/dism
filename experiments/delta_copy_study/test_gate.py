import torch
from experiments.delta_copy_study.run import Gate,make_tokens


def test_abba_targets():
    x=torch.arange(64).reshape(1,64)
    t=make_tokens(x,'abba')
    assert t.shape==(1,128)
    torch.testing.assert_close(t[:,:32],t[:,96:])
    torch.testing.assert_close(t[:,32:64],t[:,64:96])
    # Input query63 predicts first B, query95 predicts first A, query126 last A.
    target=t[:,64:]
    assert target[0,0]==32 and target[0,32]==0 and target[0,-1]==31


def test_shared_decisions_and_gradients():
    g=Gate.__new__(Gate)
    g.method='shared';g.mode='native';g.step=500
    g.b=torch.tensor([[[-2.,2.,-2.,2.]]],requires_grad=True)
    g.row_hard=torch.tensor([[[True,True,False,False]]])
    delta=g.delta()
    assert torch.isposinf(delta[0,0,0]) and delta[0,0,1]==0
    torch.testing.assert_close(delta[0,0,2:],torch.nn.functional.softplus(-g.b[0,0,2:]))
    keep=(-delta).exp();keep.sum().backward()
    assert torch.equal(g.b.grad[0,0,:2],torch.zeros(2))
    assert (g.b.grad[0,0,2:]>0).all()


def test_temperature_limit():
    g=Gate.__new__(Gate);g.method='temperature';g.mode='soft'
    g.b=torch.tensor([[[-2.,2.]]]);g.step=999
    keep=(-g.delta()).exp()
    torch.testing.assert_close(keep,torch.tensor([[[0.,1.]]]),atol=1e-7,rtol=0)
