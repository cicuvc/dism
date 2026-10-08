"""All shapes coexist: no rebuild, environment switch, or global backend mutation."""
from itertools import product

import pytest
import torch
import cu_flash_dism as cu

from flash_dism.backend import supported_configs, backend_for
from flash_dism.forward import forward_core
from flash_dism.backward import backward_core, dism_core
from flash_dism.varlen import VarlenLayout, forward_varlen, backward_varlen, dism_core_varlen
from flash_dism.reference.dism_v3_ref import dism_ref, dism_ref_backward
from gradient_acceptance import assert_gradient

CONFIGS = list(product((16,32),(32,64),(32,64)))


def inputs(config, n, mode):
    r,d,dv = config
    torch.manual_seed(1327 + r + d + dv + n)
    def vector(c):
        return (.2*torch.randn(1,n,2,c,device='cuda')).bfloat16()
    q,k,sq,sk,v = vector(d),vector(d),vector(r),vector(r),vector(dv)
    lq,lk = [torch.full((1,n,2),3.,device='cuda') for _ in range(2)]
    iq,ik = [torch.randint(16,(1,2,n),device='cuda',dtype=torch.int32) for _ in range(2)]
    direction = torch.tensor([[False,True]],device='cuda')
    hard = torch.rand(1,2,n,device='cuda') < dict(soft=0.,mixed=.5,hard=1.)[mode]
    tau = torch.full((2,),.5,device='cuda')
    return q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau


def oracle_args(x):
    q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau=x
    return (q.double(),k.double(),sq.double(),sk.double(),lq.double(),lk.double(),
            iq,ik,direction,hard,v.double(),tau.double())


def test_registry():
    assert supported_configs() == tuple(CONFIGS)
    assert not cu.fp32_enabled()
    for r,d,dv in CONFIGS:
        backend=cu.get_config(r,d,dv)
        assert (backend.forward_readout_dim(),backend.summary_key_dim(),backend.forward_head_dim()) == (r,d,dv)


@pytest.mark.parametrize('config',CONFIGS)
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('packed',[False,True])
def test_core_oracle(config,mode,packed):
    n=256
    x=inputs(config,n,mode)
    if packed:
        layout=VarlenLayout.from_cu_seqlens(torch.tensor([0,n],dtype=torch.int32),n)
        output,norm,state=forward_varlen(*x,layout=layout,save_state=True)
    else:
        output,norm,state=forward_core(*x,save_state=True,ctas=1)
    assert output.dtype==torch.bfloat16
    expected=dism_ref(*oracle_args(x))
    torch.testing.assert_close(output.double(),expected,atol=.008,rtol=.025)
    dout=torch.randn_like(output)
    actual=(backward_varlen(state,dout) if packed else backward_core(state,dout,ctas=1))
    reference=dism_ref_backward(*oracle_args(x),dout.double())
    for name,value in actual.items():
        assert_gradient(value,reference[name],name)
    for name in ('v','sk_vec','k_vec','k_lse'):
        assert actual[name].dtype==torch.bfloat16
    for name in ('sq_vec','q_vec','q_lse','rtau'):
        assert actual[name].dtype==torch.float32


@pytest.mark.parametrize('config',CONFIGS)
def test_ragged_replay(config):
    from test_varlen_forward import slice_inputs
    x=inputs(config,768,'mixed')
    layout=VarlenLayout.from_cu_seqlens(torch.tensor([0,256,256,768],dtype=torch.int32),768)
    output,norm,state=forward_varlen(*x,layout=layout,save_state=True)
    dout=torch.randn_like(output)
    actual=backward_varlen(state,dout)
    tau=torch.zeros_like(x[-1])
    for start,n in ((0,256),(256,512)):
        local=slice_inputs(x,start,n)
        expected,ref_norm,saved=forward_core(*local,save_state=True,ctas=1)
        torch.testing.assert_close(output[:,start:start+n],expected,atol=0,rtol=0)
        torch.testing.assert_close(norm[:,:,start:start+n],ref_norm,atol=2e-5,rtol=2e-6)
        ref=backward_core(saved,dout[:,start:start+n].contiguous())
        for name,value in ref.items():
            if name=='rtau': tau+=value
            else: torch.testing.assert_close(actual[name][:,start:start+n],value,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(actual['rtau'],tau,atol=2e-4,rtol=2e-5)


@pytest.mark.parametrize('config',CONFIGS)
@pytest.mark.parametrize('packed',[False,True])
def test_autograd(config,packed):
    x=list(inputs(config,256,'mixed'))
    positions=(0,1,2,3,4,5,6,11)
    for index in positions: x[index].requires_grad_()
    if packed:
        output=dism_core_varlen(*x,torch.tensor([0,256],dtype=torch.int32))
    else: output=dism_core(*x)
    output.float().square().sum().backward()
    for index in positions:
        assert x[index].grad is not None
        assert torch.isfinite(x[index].grad).all()


def test_invalid_config():
    with pytest.raises(ValueError): cu.get_config(64,64,64)
    with pytest.raises(ValueError): cu.get_config(32,128,64)


def test_disabled_fp32():
    x=inputs((32,64,64),256,'soft')
    with pytest.raises(RuntimeError,match='FP32'):
        forward_core(*x,fp32_output=True)


@pytest.mark.parametrize('config',CONFIGS)
@pytest.mark.parametrize('packed',[False,True])
def test_vocabulary_wrapper(config,packed):
    from flash_dism import voc_dism
    from test_voc_backward import torch_voc
    x=inputs(config,256,'mixed')
    q,k,sq,sk,v,_,_,_,_,direction,hard,tau=x
    eq,ek=[(.2*torch.randn(2,512,config[1],device='cuda')) for _ in range(2)]
    args=[t.requires_grad_() for t in (q,k,sq,sk,v,eq,ek,tau)]
    options={'cu_seqlens':torch.tensor([0,256],dtype=torch.int32)} if packed else {}
    output=voc_dism(*args,direction=direction,hard=hard,**options)
    reference=torch_voc(*args,direction,hard)
    torch.testing.assert_close(output.float(),reference,atol=.008,rtol=.025)
    do=torch.randn_like(output)
    actual=torch.autograd.grad(output,args,do)
    expected=torch.autograd.grad(reference,args,do.float())
    for name,a,b in zip(('q','k','sq','sk','v','eq','ek','tau'),actual,expected):
        assert_gradient(a,b,name)


def test_mixed_configuration_graph():
    from flash_dism import dism_core as dispatch
    left=list(inputs((16,32,64),256,'mixed'))
    right=list(inputs((32,64,32),256,'mixed'))
    for x in (left,right):
        for index in (0,1,2,3,4,5,6,11): x[index].requires_grad_()
    a=dispatch(*left)
    b=dispatch(*right,cu_seqlens=torch.tensor([0,256],dtype=torch.int32))
    (a.float().square().sum()+b.float().square().sum()).backward()
    for x in (left,right):
        for index in (0,1,2,3,4,5,6,11):
            assert x[index].grad is not None and torch.isfinite(x[index].grad).all()
