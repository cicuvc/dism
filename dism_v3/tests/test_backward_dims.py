"""Dimension sweep: real interpolation, FP64 oracle, private store and autograd."""
import json
import math
import os
from pathlib import Path

import pytest
import cu_flash_dism as _precision_backend
import torch
import torch.nn.functional as F
import cu_flash_dism as cu

from embedding_bhnd_reference import EmbInterpFunction
from flash_dism.forward import forward_core
from flash_dism.backward import backward_core
from flash_dism.voc import voc_dism
from flash_dism.reference.dism_v3_ref import dism_ref, dism_ref_backward
from gradient_acceptance import assert_gradient
from test_voc_backward import torch_voc

METRICS=[]


@pytest.fixture(scope='module',autouse=True)
def save_metrics():
    yield
    if path:=os.environ.get('DISM_DIM_METRICS'):
        Path(path).write_text(json.dumps(METRICS,indent=2))


def inputs(n,mode,tau_max=False):
    torch.manual_seed(6701)
    d,r,dv=cu.summary_key_dim(),cu.forward_readout_dim(),cu.forward_head_dim()
    b,h=1,4
    q,k=[(torch.randn(b,n,h,d,device='cuda')*.2).bfloat16() for _ in range(2)]
    sq,sk=[(F.silu(torch.randn(b,n,h,r,device='cuda'))*.5).bfloat16() for _ in range(2)]
    v=torch.randn(b,n,h,dv,device='cuda').bfloat16()
    eq,ek=[torch.randn(h,512,d,device='cuda')*.2 for _ in range(2)]
    tau=torch.full((h,),math.log(d) if tau_max else 2.,device='cuda')
    direction=torch.tensor([[False,True,False,True]],device='cuda')
    hard=torch.rand(b,h,n,device='cuda')<dict(soft=0.,mixed=.5,hard=1.)[mode]
    return (q,k,sq,sk,v,eq,ek,tau,direction,hard)


@pytest.mark.parametrize('n',[1,17,33,65,129,257])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('tau_max',[False,True])
@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_dimension_core(n,mode,tau_max):
    q,k,sq,sk,v,eq,ek,tau,direction,hard=inputs(n,mode,tau_max)
    qfk,kfq,lk,lq,_,_,ik,iq=EmbInterpFunction.apply(
        q.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),
        eq.bfloat16(),ek.bfloat16(),1.)
    qv=torch.where(direction[:,None,:,None],q,kfq.transpose(1,2)).contiguous()
    kv=torch.where(direction[:,None,:,None],qfk.transpose(1,2),k).contiguous()
    args=(qv,kv,sq,sk,lq.transpose(1,2),lk.transpose(1,2),iq,ik,direction,hard,v,tau)
    out,_,state=forward_core(qv,kv,sq,sk,v,args[4],args[5],iq,ik,direction,hard,tau,
                             ctas=1,save_state=True)
    oracle_args=tuple(x.double() if x.is_floating_point() else x for x in args)
    expected_out=dism_ref(*oracle_args)
    torch.testing.assert_close(out.double(),expected_out,atol=.008,rtol=.025)
    do=torch.randn_like(out).bfloat16()
    actual=backward_core(state,do,ctas=1)
    fp32=backward_core(state,do,ctas=1,fp32_output=True)
    for name in ('v','sk_vec','k_vec','k_lse'):
        torch.testing.assert_close(actual[name],fp32[name].bfloat16(),atol=0,rtol=0)
    delta=cu.backward_delta(out,do)
    torch.testing.assert_close(delta,(out.float()*do.float()).sum(-1).transpose(1,2),
                               atol=2e-5,rtol=2e-5)
    expected=dism_ref_backward(*oracle_args,do.double())
    for name,x in actual.items():
        y=expected[name]
        xf,yf=x.double().flatten(),y.double().flatten()
        xn,yn=xf.norm().item(),yf.norm().item()
        METRICS.append(dict(n=n,mode=mode,tau_max=tau_max,gradient=name,
            cosine=float(torch.dot(xf,yf)/(xn*yn)) if min(xn,yn)>1e-12 else None,
            norm_ratio=xn/yn if yn>1e-12 else None,
            relative_l2=float((xf-yf).norm()/yn) if yn>1e-12 else None))
        assert_gradient(x,y,name)


@pytest.mark.parametrize('mode',['soft','mixed','hard'])
def test_dimension_embedding_autograd(mode):
    q,k,sq,sk,v,eq,ek,tau,direction,hard=inputs(256,mode)
    args=[x.requires_grad_() for x in (q,k,sq,sk,v,eq,ek,tau)]
    do=torch.randn_like(v)
    out=voc_dism(*args,direction=direction,hard=hard,ctas=1)
    actual=torch.autograd.grad(out,args,do)
    expected_out=torch_voc(*args,direction,hard)
    expected=torch.autograd.grad(expected_out,args,do.float())
    torch.testing.assert_close(out.float(),expected_out,atol=.008,rtol=.025)
    for name,x,y in zip(('q','k','sq','sk','v','eq','ek','tau'),actual,expected):
        assert_gradient(x,y,name)
