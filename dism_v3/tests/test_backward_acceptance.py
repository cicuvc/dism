"""Default numerical acceptance, distinct from opt-in strict diagnostics."""
import math

import pytest
import cu_flash_dism as _precision_backend
import torch
import torch.nn.functional as F
import cu_flash_dism as cu

from embedding_bhnd_reference import EmbInterpFunction
from flash_dism.forward import forward_core
from flash_dism.backward import backward_core
from flash_dism.reference.dism_v3_ref import dism_ref_backward
from flash_dism.voc import voc_dism
from gradient_acceptance import assert_gradient
from test_backward_summary import prepare, local_oracle, pytestmark
from test_backward_qk import qk_oracle
from test_voc_backward import torch_voc


@pytest.mark.parametrize('n', [1, 17, 65, 257, 513])
@pytest.mark.parametrize('mode', ['soft', 'mixed', 'hard'])
@pytest.mark.parametrize('tau_value', [2., math.log(64)])
@pytest.mark.parametrize('seed', [0, 7])
def test_core_gradient_acceptance(n, mode, tau_value, seed):
    # Match the real-interpolation bias audit distribution, including its worst
    # observed hard/ln64 seed. Synthetic independent-LSE stress remains strict.
    torch.manual_seed(6701+seed)
    b, h = 1, 4
    q, k = [(torch.randn(b,n,h,64,device='cuda')*.2).bfloat16() for _ in range(2)]
    sq, sk = [F.silu(torch.randn(b,n,h,32,device='cuda')*.5).bfloat16() for _ in range(2)]
    v = torch.randn(b,n,h,64,device='cuda').bfloat16()
    eq, ek = [(torch.randn(h,512,64,device='cuda')*.2).bfloat16() for _ in range(2)]
    qfk,kfq,lk,lq,_,_,ik,iq = EmbInterpFunction.apply(
        q.transpose(1,2).contiguous(), k.transpose(1,2).contiguous(), eq, ek, 1.)
    direction = torch.tensor([[False,True,False,True]],device='cuda')
    hard = torch.rand(b,h,n,device='cuda') < dict(soft=0.,mixed=.5,hard=1.)[mode]
    tau = torch.full((h,),tau_value,device='cuda')
    qv = torch.where(direction[:,None,:,None],q,kfq.transpose(1,2))
    kv = torch.where(direction[:,None,:,None],qfk.transpose(1,2),k)
    lq,lk = lq.transpose(1,2),lk.transpose(1,2)
    output,_,state = forward_core(qv,kv,sq,sk,v,lq,lk,iq,ik,direction,hard,tau,save_state=True)
    dout = torch.randn_like(output).bfloat16()
    actual = backward_core(state,dout)  # Exercise production BF16 final stores.
    expected = dism_ref_backward(qv.double(),kv.double(),sq.double(),sk.double(),
        lq.double(),lk.double(),iq,ik,direction,hard,v.double(),tau.double(),dout.double())
    for name,value in actual.items():
        assert_gradient(value,expected[name],name)


@pytest.mark.parametrize('n', [1,17,33,65,129,257])
@pytest.mark.parametrize('mode', ['soft','mixed','hard'])
@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_summary_component_acceptance(n,mode):
    inputs,state,do,dop,delta = prepare(n,mode)
    got = cu.backward_summary(state['operands'],state['vertical'],dop,state['lse2'],delta,n,1,True)
    expected = local_oracle(inputs,state,do,delta)
    for name,x,y in zip(('dv','dsq','dsk','a','b'),got,expected):
        if name in ('a','b'):
            end = (n+31)//32*32
            # Affine summary accuracy is not relaxed.
            torch.testing.assert_close(x[...,:end].double(),y[...,:end],atol=2e-4,rtol=3e-3,msg=name)
        else:
            assert_gradient(x,y,name)
            torch.testing.assert_close(x.double(),y,atol=.003,rtol=.02,msg=name)


@pytest.mark.parametrize('n', [1,17,33,65,129,257])
@pytest.mark.parametrize('mode', ['soft','mixed','hard'])
@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_qk_component_acceptance(n,mode):
    inputs,state,do,dop,delta = prepare(n,mode)
    common = (state['operands'],state['vertical'],dop,state['lse2'],delta)
    *_,a,b = cu.backward_summary(*common,n,1,True)
    edge = cu.backward_chunk(a,b)
    got = cu.backward_qk(*common,edge,n,1,True)
    expected,g = qk_oracle(inputs,state,do,delta)
    step=cu.backward_summary_k()
    for c in range((n+step-1)//step):
        torch.testing.assert_close(edge[:,:,c,:n].double(),g[:,:,:,c*step],atol=1e-4,rtol=2e-3)
    for name,x,y in zip(('dq','dk','dlq','dlk','tau'),got,expected):
        if name in ('dq','dk'):
            assert_gradient(x,y,name)
            torch.testing.assert_close(x.double(),y,atol=.003,rtol=.02,msg=name)
        else:
            # Same-state scalar reductions have no ideal-delta discrepancy.
            torch.testing.assert_close(x.double(),y,atol=2e-4,rtol=3e-3,msg=name)


@pytest.mark.parametrize('mode', ['soft','mixed','hard'])
@pytest.mark.parametrize('direction_value', [False,True])
def test_embedding_autograd_acceptance(mode,direction_value):
    # Deliberately identical fixture to the original strict end-to-end test.
    torch.manual_seed(273)
    b,n,h = 1,65,2
    q,k = [(torch.randn(b,n,h,64,device='cuda')*.2).bfloat16() for _ in range(2)]
    sq,sk = [F.silu(torch.randn(b,n,h,32,device='cuda')*.5).bfloat16() for _ in range(2)]
    v = torch.randn(b,n,h,64,device='cuda').bfloat16()
    eq,ek = [torch.randn(h,512,64,device='cuda')*.2 for _ in range(2)]
    tau = torch.full((h,),2.,device='cuda')
    direction = torch.full((b,h),direction_value,device='cuda',dtype=torch.bool)
    hard = torch.rand(b,h,n,device='cuda') < dict(soft=0.,mixed=.5,hard=1.)[mode]
    args = [x.requires_grad_() for x in (q,k,sq,sk,v,eq,ek,tau)]
    do = torch.randn_like(v)
    actual = voc_dism(*args,direction=direction,hard=hard,ctas=1)
    ga = torch.autograd.grad(actual,args,do)
    expected = torch_voc(*args,direction,hard)
    ge = torch.autograd.grad(expected,args,do.float())
    torch.testing.assert_close(actual.float(),expected,atol=.008,rtol=.025)
    for name,x,y in zip(('q','k','sq','sk','v','eq','ek','tau'),ga,ge):
        assert_gradient(x,y,name)
