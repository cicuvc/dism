import pytest
import cu_flash_dism as _precision_backend
import torch
import cu_flash_dism
from flash_dism.forward import forward_core
from flash_dism.backward import backward_core,dism_core
from flash_dism.reference.dism_v3_ref import dism_ref,dism_ref_backward
from test_backward_summary import prepare,pytestmark
from test_forward import inputs_for


@pytest.mark.parametrize('n',[1,17,65,129,257,513])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('direction',['query','key'])
@pytest.mark.strict_gradient
@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_backward_reference(n,mode,direction):
    inputs,state,do,_,_=prepare(n,mode,direction)
    g=backward_core(state,do,ctas=1,fp32_output=True)
    q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau=inputs
    oracle=dism_ref_backward(q.double(),k.double(),sq.double(),sk.double(),
        lq.double(),lk.double(),iq,ik,direction,hard,v.double(),tau.double(),do.double())
    for name,actual in g.items():
        expected=oracle[name]
        x,y=actual.double().flatten(),expected.flatten()
        assert torch.isfinite(x).all(),name
        if y.norm()<1e-10:
            assert x.norm()<1e-6,name
        else:
            relative=(x-y).norm()/y.norm()
            cosine=torch.nn.functional.cosine_similarity(x,y,dim=0)
            assert relative<.02,(name,relative.item())
            assert cosine>.9998,(name,cosine.item())
            torch.testing.assert_close(x,y,atol=.015,rtol=.04,msg=name)


@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_output_precision_and_autograd():
    inputs=inputs_for(129)
    variables=[0,1,2,3,4,5,6,11]
    for index in variables:
        inputs[index].requires_grad_()
    output=dism_core(*inputs,ctas=1)
    do=torch.randn_like(output).bfloat16()
    grads=torch.autograd.grad(output,[inputs[i] for i in variables],do)
    _,_,state=forward_core(*inputs,ctas=1,save_state=True)
    bf=backward_core(state,do,ctas=1)
    fp=backward_core(state,do,ctas=1,fp32_output=True)
    names=('q_vec','k_vec','sq_vec','sk_vec','v','q_lse','k_lse','rtau')
    for name in ('v','k_vec','sk_vec','k_lse'):
        assert bf[name].dtype==torch.bfloat16
        assert fp[name].dtype==torch.float32
        torch.testing.assert_close(bf[name],fp[name].bfloat16(),atol=0,rtol=0)
    for name in ('q_vec','sq_vec','q_lse','rtau'):
        assert bf[name].dtype==torch.float32
    for name,index,gradient in zip(names,variables,grads):
        torch.testing.assert_close(gradient,bf[name].to(inputs[index].dtype),atol=1e-5,rtol=.008)


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_delta_fp32(dtype):
    o=torch.randn(2,256,3,64,device='cuda').to(dtype)
    do=torch.randn_like(o).bfloat16()
    got=cu_flash_dism.backward_delta(o,do)
    expected=(o.double()*do.double()).sum(-1).transpose(1,2)
    assert got.dtype==torch.float32
    torch.testing.assert_close(got.double(),expected,atol=3e-6,rtol=3e-6)


@pytest.mark.parametrize('n',[17,65,129])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('direction',['query','key'])
@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_oracle_delta_diagnostic(n,mode,direction):
    inputs,state,do,dop,_=prepare(n,mode,direction)
    q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau=inputs
    args=(q.double(),k.double(),sq.double(),sk.double(),lq.double(),lk.double(),
          iq,ik,direction,hard,v.double(),tau.double())
    out=dism_ref(*args)
    delta=(out*do.double()).sum(-1).transpose(1,2).float().contiguous()
    common=(state['operands'],state['vertical'],dop,state['lse2'],delta)
    dv,dsq,dsk,a,b=cu_flash_dism.backward_summary(*common,n,1,True)
    edge=cu_flash_dism.backward_chunk(a,b)
    dq,dk,dlq,dlk,dtau=cu_flash_dism.backward_qk(*common,edge,n,1,True)
    got=dict(q_vec=dq,k_vec=dk,sq_vec=dsq,sk_vec=dsk,v=dv,q_lse=dlq.transpose(1,2),
             k_lse=dlk.transpose(1,2),rtau=dtau)
    oracle=dism_ref_backward(*args,do.double())
    for name,x in got.items():
        y=oracle[name]
        torch.testing.assert_close(x.double(),y,atol=.015,rtol=.04,msg=name)
        if y.norm()>1e-10:
            assert (x-y).norm()/y.norm()<.02,name
