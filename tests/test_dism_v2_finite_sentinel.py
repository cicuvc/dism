"""Experimental bounded finite-log-zero checks; exact-infinity suites stay unchanged."""
import math
import pytest
import torch
from dism_v2.kernel_config import TILE_LSE
from dism_v2.core import forward

pytestmark=pytest.mark.skipif(TILE_LSE!='tanh_finite' or not torch.cuda.is_available(),
                            reason='run with DISM_TILE_LSE=tanh_finite')


@pytest.mark.parametrize('d',(32,64,128))
@pytest.mark.parametrize('direction',('q_from_k','k_from_q'))
@torch.no_grad()
def test_all_unmatched_finite_zero(d,direction):
    n=1025
    x=torch.zeros((1,2,n,d),device='cuda',dtype=torch.bfloat16)
    v=torch.randn((1,2,n,32),device='cuda',dtype=torch.bfloat16)
    lse=torch.zeros((1,2,n),device='cuda')
    labels=torch.zeros((1,2,n),device='cuda',dtype=torch.int64)
    tau=torch.full((2,),math.log(d),device='cuda')
    o,l,s,b,edges=forward(x,x,v,lse,tau,labels,labels+1,sm_scale=1.,
        direction=direction,hard_prob=1.,return_debug=True,save_boundaries=True)
    assert torch.count_nonzero(o)==0 and torch.count_nonzero(l)==0
    for value in (s,b,edges.vertical,edges.horizontal):
        assert torch.isfinite(value).all()
    assert b.max() < -900000


@pytest.mark.parametrize('n',(1025,8193))
@pytest.mark.parametrize('tau_value',(-1.,0.,math.log(32)))
@torch.no_grad()
def test_finite_long_chain(n,tau_value):
    # Same analytic oracle/tolerances as the existing strict-infinity test.
    from test_dism_v2_precision import constant_chain_reference
    torch.manual_seed(901)
    q=torch.zeros((1,1,n,32),device='cuda',dtype=torch.bfloat16)
    v=torch.randn((1,1,n,64),device='cuda',dtype=torch.bfloat16)
    labels=torch.zeros((1,1,n),device='cuda',dtype=torch.int64)
    lse=torch.zeros((1,1,n),device='cuda')
    tau=torch.tensor([tau_value],device='cuda')
    o,l,s,b=forward(q,q,v,lse,tau,labels,labels,sm_scale=1.,
        direction='q_from_k',hard_prob=1.,return_debug=True)
    eo,el,w=constant_chain_reference(v,float(tau.item()))
    torch.testing.assert_close(o.double(),eo,atol=.008,rtol=.012)
    torch.testing.assert_close(l.double(),el,atol=2e-5,rtol=2e-5)
    assert torch.isfinite(b).all()
    # Compare reachable boundaries only; unreachable boundaries intentionally
    # changed representation and are checked separately, not mapped back to -inf.
    rows=torch.arange(n//32,device='cuda')*32+31
    cols=torch.arange(b.shape[-1],device='cuda')
    valid=cols[None,:]<=rows[:,None]
    expected=w[cols.clamp_max(n-1)][None,:].expand(len(rows),-1)
    torch.testing.assert_close(b[0,0,:n//32].double()[valid],expected[valid],atol=3e-5,rtol=3e-5)


def test_bounded_log_zero_margin():
    # Conditional bound: each log2 score <=7, total chain length <=65536.
    # W <= N*7 + log2(N); a path containing a -1e6 edge remains far below zero.
    bound=-1e6+65536*7+math.log2(65536)
    assert bound < -500000
    assert torch.exp2(torch.tensor(bound,device='cuda',dtype=torch.float32))==0


@pytest.mark.parametrize('d',(32,64,128))
@pytest.mark.parametrize('direction',('q_from_k','k_from_q'))
@pytest.mark.parametrize('n',(65,257))
@torch.no_grad()
def test_finite_persistent_reuse(d,direction,n):
    batch=torch.cuda.get_device_properties(0).multi_processor_count//2+3
    q=torch.zeros((batch,2,n,d),device='cuda',dtype=torch.bfloat16)
    v=torch.randn((batch,2,n,32),device='cuda',dtype=torch.bfloat16)
    lse=torch.zeros((batch,2,n),device='cuda')
    labels=torch.arange(n,device='cuda').expand(batch,2,n).contiguous()
    tau=torch.full((2,),math.log(d),device='cuda')
    kw=dict(sm_scale=1.,direction=direction,hard_prob=.37,
        return_rng_state=True,save_boundaries=True)
    o,l,edges,state=forward(q,q,v,lse,tau,labels,labels,**kw)
    again=forward(q,q,v,lse,tau,labels,labels,rng_state=state,**kw)
    for x,y in zip((o,l,edges.vertical,edges.horizontal),
                   (again[0],again[1],again[2].vertical,again[2].horizontal)):
        assert torch.isfinite(x).all()
        torch.testing.assert_close(x,y,atol=0,rtol=0)
