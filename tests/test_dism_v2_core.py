"""Fixed-direction CUDA core tests with explicit BF16 interpolation oracle."""
import itertools
import math
from dataclasses import replace
import pytest
import torch
from dism_v2.core import forward
from dism_v2.dism_ref import interpolation_ref, voc_dism_ref

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")

@pytest.mark.parametrize("d,dv",itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("hard",(False,True))
@torch.no_grad()
def test_dimensions(d,dv,direction,hard):
    check_case(d,dv,139,direction,hard)

@pytest.mark.parametrize("n",(1,17,31,32,33,63,64,65,127,128,129,257,513))
@torch.no_grad()
def test_tails(n):
    check_case(64,64,n,"q_from_k",False)

def check_case(d,dv,n,direction,hard):
    generator=torch.Generator(device="cuda").manual_seed(41+n+d+dv)
    def rand(shape): return torch.randn(shape,device="cuda",dtype=torch.bfloat16,generator=generator)
    batch,heads=2,2
    q,k,v=rand((batch,heads,n,d)),rand((batch,heads,n,d)),rand((batch,heads,n,dv))
    qvoc,kvoc=rand((heads,11,d)),rand((heads,11,d))
    tau=torch.tensor([-0.5,0.5],device="cuda")
    scale=d**-0.5
    interp=interpolation_ref(q,k,qvoc,kvoc,scale)
    interp=replace(interp,q_from_k=interp.q_from_k.bfloat16(),k_from_q=interp.k_from_q.bfloat16())
    a,b,lse=(q,interp.q_from_k,interp.q_lse) if direction=="q_from_k" else (interp.k_from_q,k,interp.k_lse)
    actual,l2,summary,boundary=forward(a.contiguous(),b.contiguous(),v,lse.contiguous(),tau,
        interp.q_index.contiguous(),interp.k_index.contiguous(),sm_scale=scale,direction=direction,
        hard_prob=float(hard),return_debug=True)
    expected,aux=voc_dism_ref(q,k,v,tau,qvoc,kvoc,hard_prob=float(hard),direction=direction,
        sm_scale=scale,interpolation=interp,hard_mask=torch.tensor(hard,device="cuda"),return_aux=True)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(),expected.float(),atol=0.008,rtol=0.012)
    ln2=math.log(2.0)
    expected_l=torch.logaddexp(torch.logsumexp(aux["scores"],dim=-1),torch.zeros_like(l2))/ln2
    torch.testing.assert_close(l2,expected_l,atol=2e-5,rtol=2e-5)
    # Independent double recurrence checks both local summary components and
    # all resolved checkpoint values, including identity padding after N.
    np=boundary.shape[-1]; rows=summary.shape[2]*32
    logs=aux["log_m"].double()/ln2
    previous=torch.full((batch,heads,np),-torch.inf,device="cuda",dtype=torch.float64)
    local_a=torch.zeros_like(previous); local_b=torch.full_like(previous,-torch.inf)
    for i in range(rows):
        shifted=torch.nn.functional.pad(previous[...,:-1],(1,0),value=-torch.inf)
        if i%32==0:
            local_a.zero_(); local_b.fill_(-torch.inf)
        sa=torch.nn.functional.pad(local_a[...,:-1],(1,0),value=0)
        sb=torch.nn.functional.pad(local_b[...,:-1],(1,0),value=-torch.inf)
        m=torch.zeros_like(previous); valid=torch.zeros_like(previous,dtype=torch.bool)
        if i<n:
            m[...,:n]=logs[...,i,:]
            m[...,:n].masked_fill_(torch.arange(n,device="cuda")>i,-torch.inf)
            valid[...,:n]=True
        offset=torch.where(valid,m,torch.full_like(m,-torch.inf))
        previous=torch.logaddexp((shifted+m)*ln2,offset*ln2)/ln2
        local_a=sa+m
        local_b=torch.logaddexp((sb+m)*ln2,offset*ln2)/ln2
        if i%32==31:
            torch.testing.assert_close(boundary[...,i//32,:].double(),previous,atol=3e-5,rtol=3e-5)
            torch.testing.assert_close(summary[...,i//32,:,0].double(),local_a,atol=3e-5,rtol=3e-5)
            torch.testing.assert_close(summary[...,i//32,:,1].double(),local_b,atol=3e-5,rtol=3e-5)
