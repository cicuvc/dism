import pytest
import cu_flash_dism as _precision_backend
import torch
import cu_flash_dism
from test_backward_summary import prepare,pytestmark


def qk_oracle(inputs,state,do,delta):
    q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau=inputs
    n=q.shape[1]
    w=cu_flash_dism.backward_recompute_probe(state['operands'],state['vertical'],n)
    w=w.masked_fill(~torch.ones(n,n,device='cuda',dtype=torch.bool).tril(),-1e6)
    p=torch.exp2(w-state['lse2'][...,None])
    a=torch.einsum('bnhr,bmhr->bhnm',sq.float(),sk.float())
    b=torch.einsum('bnhd,bmhd->bhnm',do.float(),v.float())
    g=(p*(a*b-delta[...,None])).double()
    alpha=torch.sigmoid(w.double()*0.6931471805599453)
    for i in range(n-2,-1,-1):
        g[:,:,i,:-1]+=alpha[:,:,i,:-1]*g[:,:,i+1,1:]
    gs=g.masked_fill(hard[...,None],0)
    rounded=gs.bfloat16().double()
    dq=torch.einsum('bhnm,bmhd->bnhd',rounded,k.double())
    dk=torch.einsum('bhnm,bnhd->bmhd',rounded,q.double())
    dlq=torch.where(direction[...,None],-gs.sum(-1),0.)
    dlk=torch.where(direction[...,None],0.,-gs.sum(-2))
    return (dq,dk,dlq,dlk,g.sum((0,2,3))),g


@pytest.mark.parametrize('n',[1,17,33,65,129,257])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.strict_gradient
@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_qk_gradients(n,mode):
    inputs,state,do,dop,delta=prepare(n,mode)
    _,_,_,a,b=cu_flash_dism.backward_summary(state['operands'],state['vertical'],dop,
                                           state['lse2'],delta,n,1,True)
    edge=cu_flash_dism.backward_chunk(a,b)
    got=cu_flash_dism.backward_qk(state['operands'],state['vertical'],dop,
                                 state['lse2'],delta,edge,n,1,True)
    torch.cuda.synchronize()
    expected,g=qk_oracle(inputs,state,do,delta)
    step=cu_flash_dism.backward_summary_k()
    for c in range((n+step-1)//step):
        torch.testing.assert_close(edge[:,:,c,:n].double(),g[:,:,:,c*step],atol=1e-4,rtol=2e-3)
    for name,x,y in zip(('dq','dk','dlq','dlk','tau'),got,expected):
        torch.testing.assert_close(x.double(),y,atol=2e-4,rtol=3e-3,msg=name)
