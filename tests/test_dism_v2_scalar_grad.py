"""Long chain oracle without an N x N attention/gradient allocation."""
import pytest
import torch
from dism_v2.core import forward
from dism_v2.backward import delta,value_gradient,operand_gradient

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")

@pytest.mark.parametrize("ws",(False,True))
@pytest.mark.parametrize("direction",("q_from_k","k_from_q"))
@pytest.mark.parametrize("probability",(0.,1.))
def test_long_scalar_chain(ws,direction,probability,record_property):
    if torch.cuda.get_device_capability()!=(12,0): pytest.skip("sm120a only")
    # More than 256 partials for BOTH B3 paths; multiple batches/heads.
    batch,heads,n,d=2,2,8193,32
    gen=torch.Generator(device="cuda").manual_seed(952)
    a=torch.zeros((batch,heads,n,d),device="cuda",dtype=torch.bfloat16)
    v=torch.randn(a.shape,device="cuda",dtype=torch.bfloat16,generator=gen)
    dout=torch.randn(a.shape,device="cuda",dtype=torch.bfloat16,generator=gen)
    lse=torch.zeros((batch,heads,n),device="cuda")
    tau=torch.zeros(heads,device="cuda")
    labels=torch.zeros_like(lse,dtype=torch.int64)
    out,norm,edges,state=forward(a,a,v,lse,tau,labels,labels,sm_scale=d**-.5,
        direction=direction,hard_prob=probability,save_boundaries=True,return_rng_state=True)
    dd=delta(dout,out)
    kw=dict(sm_scale=d**-.5,rng_state=state)
    _,_,g32=value_gradient(a,a,dout,lse,tau,labels,labels,norm,edges,
        **kw,v=v,delta=dd,warp_specialized=True)
    before=torch.cuda.get_rng_state()
    da,db,dlse,dtau=operand_gradient(a,a,v,dout,lse,tau,labels,labels,norm,dd,edges,g32,
        **kw,warp_specialized=ws)
    assert torch.equal(before,torch.cuda.get_rng_state())
    assert torch.count_nonzero(da)==0 and torch.count_nonzero(db)==0
    # logM=0 => exp(W[i,j])=j+1; d exp(W)/d tau=(j+1)(j+2)/2.
    # Use saved normalizer/delta to isolate scalar reduction from BF16 O error.
    j=torch.arange(1,n+1,device="cuda",dtype=torch.float64)
    ds=j*(j+1)/2
    numerator=(v.double()*ds[None,None,:,None]).cumsum(-2)
    expected=((numerator*dout.double()).sum(-1)-dd.double()*ds.cumsum(0))
    expected=(expected*torch.exp2(-norm.double())).sum((0,2))
    error=(dtau.double()-expected).abs().max().item()
    record_property("dtau_closed_form_max_abs",error)
    record_property("dtau_closed_form_relative_l2",
        ((dtau.double()-expected).norm()/expected.norm()).item())
    torch.testing.assert_close(dtau.double(),expected,atol=.02,rtol=.002)
    if probability==1.:
        assert torch.count_nonzero(dlse)==0
    else:
        torch.testing.assert_close(-dlse.double().sum((0,2)),dtau.double(),atol=.02,rtol=2e-4)
