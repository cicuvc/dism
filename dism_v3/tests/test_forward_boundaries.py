import pytest
import torch
import cu_flash_dism
from flash_dism.forward import forward_core
from test_forward import inputs_for


def dense_scan(inputs):
    q,k,_,_,_,lq,lk,iq,ik,direction,hard,tau = inputs
    score = torch.einsum('bnhd,bmhd->bhnm',q.double(),k.double())
    score -= torch.where(direction[...,None,None],
                         lq.double().transpose(1,2)[...,None],
                         lk.double().transpose(1,2)[...,None,:])
    score = torch.where(hard[...,None],
                        torch.where(iq[...,None] == ik[...,None,:],0.,-torch.inf),score)
    score += tau.double()[None,:,None,None]
    result = []
    previous = torch.full_like(score[...,0,:],-torch.inf)
    for row in score.unbind(-2):
        shifted = torch.nn.functional.pad(previous[...,:-1],(1,0),value=-torch.inf)
        previous = row + torch.nn.functional.softplus(shifted)
        result.append(previous)
    return torch.stack(result,-2) / torch.log(torch.tensor(2.,dtype=torch.float64))


@pytest.mark.parametrize('n',[1,15,16,17,31,32,33,127,128,129,257])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
def test_vertical_checkpoints(n, mode):
    inputs = inputs_for(n, mode)
    out,norm,state = forward_core(*inputs,ctas=1,save_state=True)
    ordinary,ordinary_norm = forward_core(*inputs,ctas=1)
    torch.testing.assert_close(out,ordinary,atol=0,rtol=0)
    torch.testing.assert_close(norm,ordinary_norm,atol=0,rtol=0)
    w = dense_scan(inputs)
    vertical = state['vertical']
    assert vertical.shape[2] == (n-1)//16
    for c in range((n-1)//16):
        j = 16*(c+1)-1
        actual = vertical[:,:,c,j:n].double()
        expected = w[:,:,j:,j]
        live = torch.isfinite(expected)
        assert (actual[~live] < -1e5).all()
        torch.testing.assert_close(actual[live],expected[live],atol=.05,rtol=1e-4)
        assert (vertical[:,:,c,:j] == -1e6).all()
        assert (vertical[:,:,c,n:] == -1e6).all()


@pytest.mark.parametrize('n',[1,17,33,65,129,257,513])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('direction',['query','key'])
def test_transposed_recompute(n, mode, direction):
    if cu_flash_dism.summary_key_dim() != 64:
        pytest.skip('initial backward is D64')
    inputs = inputs_for(n,mode,direction)
    _,_,state = forward_core(*inputs,ctas=1,save_state=True)
    actual = cu_flash_dism.backward_recompute_probe(state['operands'],state['vertical'],n)
    expected = dense_scan(inputs)
    causal = torch.ones(n,n,device='cuda',dtype=torch.bool).tril().expand_as(expected)
    live = causal & torch.isfinite(expected)
    assert (actual[causal & ~live] < -1e5).all()
    torch.testing.assert_close(actual[live].double(),expected[live],atol=.07,rtol=2e-4)
