"""Independent FP64 oracle and signed gradient-error metrics for packed v3."""
import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from embedding_bhnd_reference import EmbInterpFunction
from flash_dism.varlen import forward_varlen,backward_varlen
from flash_dism.reference.dism_v3_ref import dism_ref,dism_ref_backward
from gradient_acceptance import assert_gradient
from test_varlen_layout import make_layout

METRICS=[]


@pytest.fixture(scope='module',autouse=True)
def save_metrics():
    yield
    if path:=os.environ.get('DISM_VARLEN_METRICS'):
        Path(path).write_text(json.dumps(METRICS,indent=2))


def real_case(seed,mode,tau_max):
    torch.manual_seed(6701+seed)
    layout=make_layout([256,0,512],'cuda')
    t,h=layout.tokens,4
    q,k=[(torch.randn(1,t,h,64,device='cuda')*.2).bfloat16() for _ in range(2)]
    sq,sk=[F.silu(torch.randn(1,t,h,32,device='cuda')*.5).bfloat16() for _ in range(2)]
    v=torch.randn_like(q)
    eq,ek=[(torch.randn(h,512,64,device='cuda')*.2).bfloat16() for _ in range(2)]
    qfk,kfq,lk,lq,_,_,ik,iq=EmbInterpFunction.apply(
        q.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),eq,ek,1.)
    direction=torch.tensor([[False,True,False,True]],device='cuda')
    hard=torch.rand(1,h,t,device='cuda')<dict(soft=0.,mixed=.5,hard=1.)[mode]
    tau=torch.full((h,),math.log(64) if tau_max else 2.,device='cuda')
    qv=torch.where(direction[:,None,:,None],q,kfq.transpose(1,2)).contiguous()
    kv=torch.where(direction[:,None,:,None],qfk.transpose(1,2),k).contiguous()
    lq,lk=lq.transpose(1,2),lk.transpose(1,2)
    return layout,(qv,kv,sq,sk,v,lq,lk,iq,ik,direction,hard,tau)


def packed_oracle(layout,inputs,dout):
    qv,kv,sq,sk,v,lq,lk,iq,ik,direction,hard,tau=inputs
    expected={key:[] for key in ('q_vec','k_vec','sq_vec','sk_vec','v','rtau','q_lse','k_lse')}
    ref_output=[]
    for start,n,*_ in layout.table.tolist():
        if not n:
            continue
        local=[x[:,start:start+n].double() for x in (qv,kv,sq,sk,lq,lk)]
        args=(*local,iq[:,:,start:start+n],ik[:,:,start:start+n],direction,
              hard[:,:,start:start+n],v[:,start:start+n].double(),tau.double())
        ref_output.append(dism_ref(*args))
        gradients=dism_ref_backward(*args,dout[:,start:start+n].double())
        for key in expected:
            expected[key].append(gradients[key])
    return torch.cat(ref_output,dim=1),{
        name:sum(values) if name=='rtau' else torch.cat(values,dim=1)
        for name,values in expected.items()}


@pytest.mark.parametrize('seed',[0,1,7,19,23,31,47,63])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('tau_max',[False,True])
def test_packed_oracle(seed,mode,tau_max):
    layout,inputs=real_case(seed,mode,tau_max)
    output,_,state=forward_varlen(*inputs,layout=layout,save_state=True)
    dout=torch.randn_like(output).bfloat16()
    actual=backward_varlen(state,dout)
    reference_output,expected=packed_oracle(layout,inputs,dout)
    torch.testing.assert_close(output.double(),reference_output,atol=.008,rtol=.025)
    for name,ref in expected.items():
        x,y=actual[name].double().flatten(),ref.flatten()
        xn,yn=x.norm().item(),y.norm().item()
        METRICS.append(dict(seed=seed,mode=mode,tau_max=tau_max,gradient=name,
            reference_norm=yn,error_norm=(x-y).norm().item(),
            cosine=torch.dot(x,y).item()/(xn*yn) if min(xn,yn)>1e-12 else None,
            norm_ratio=xn/yn if yn>1e-12 else None,
            projection_bias=torch.dot(x-y,y).item()/yn**2 if yn>1e-12 else None,
            mean_error=(x-y).mean().item(),
            sign_flips=int(((x*y)<0).sum().item()) if name=='rtau' else None))
    for name,ref in expected.items():
        assert_gradient(actual[name],ref,name)
