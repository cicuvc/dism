"""Separate scan correctness from discontinuous BF16 coefficient conversion.

Dense debug buffers exist only in these tests; no training call allocates them.
Raw coefficients/G are first verified independently, THEN used as a fixed GEMM
input for checking downstream layout and accumulation. This does not replace
the stricter ideal-reference tests or hide their known precision failures.
"""
import pytest
import cu_flash_dism as _precision_backend
import torch
import cu_flash_dism as cu
from test_backward_summary import prepare,pytestmark
from test_backward_qk import qk_oracle


@pytest.mark.parametrize('n',[1,17,33,65,129,257,513])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.skipif(not getattr(cu,'backward_debug_enabled',lambda:False)(),reason='requires DISM_BACKWARD_DEBUG=1')
@pytest.mark.skipif(not _precision_backend.fp32_enabled(),
                    reason='requires optional FP32 validation instances')
def test_component_gradients(n,mode):
    x,s,do,dop,delta=prepare(n,mode)
    q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau=x
    common=(s['operands'],s['vertical'],dop,s['lse2'],delta)
    dv,dsq,dsk,a,b,ca,cb=cu.backward_summary_debug(*common,n)
    edge=cu.backward_chunk(a,b)
    dq,dk,dlq,dlk,dtau,g=cu.backward_qk_debug(*common,edge,n)
    w=cu.backward_recompute_probe(s['operands'],s['vertical'],n).double()
    causal=torch.ones(n,n,device='cuda',dtype=torch.bool).tril()
    p=torch.exp2(w-s['lse2'].double()[...,None]).masked_fill(~causal,0)
    expected_ca=p*torch.einsum('bnhr,bmhr->bhnm',sq.double(),sk.double())
    expected_cb=p*torch.einsum('bnhd,bmhd->bhnm',do.double(),v.double())
    torch.testing.assert_close(ca.double(),expected_ca,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(cb.double(),expected_cb,atol=4e-6,rtol=2e-5)
    _,expected_g=qk_oracle(x,s,do,delta)
    torch.testing.assert_close(g.double(),expected_g,atol=1e-4,rtol=2e-3)
    # No tolerance changes around a BF16 midpoint: fix the measured FP32
    # coefficients only after independently validating the unquantized values.
    ca,cb=ca.bfloat16().double(),cb.bfloat16().double()
    gs=g.masked_fill(hard[...,None],0)
    rounded=gs.bfloat16().double()
    expected=(torch.einsum('bhnm,bnhd->bmhd',ca,do.double()),
              torch.einsum('bhnm,bmhr->bnhr',cb,sk.double()),
              torch.einsum('bhnm,bnhr->bmhr',cb,sq.double()),
              torch.einsum('bhnm,bmhd->bnhd',rounded,k.double()),
              torch.einsum('bhnm,bnhd->bmhd',rounded,q.double()))
    for actual,oracle in zip((dv,dsq,dsk,dq,dk),expected):
        torch.testing.assert_close(actual.double(),oracle,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(dlq.double(),torch.where(direction[...,None],-gs.double().sum(-1),0.),
                               atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(dlk.double(),torch.where(direction[...,None],0.,-gs.double().sum(-2)),
                               atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(dtau.double(),g.double().sum((0,2,3)),atol=5e-5,rtol=2e-5)
    # Diagnostic stores must not change the production calculation.
    plain=cu.backward_summary(*common,n,1,True)
    for actual,oracle in zip(plain,(dv,dsq,dsk,a,b)):
        torch.testing.assert_close(actual,oracle,atol=2e-5,rtol=2e-5)
    plain=cu.backward_qk(*common,edge,n,1,True)
    for actual,oracle in zip(plain,(dq,dk,dlq,dlk,dtau)):
        torch.testing.assert_close(actual,oracle,atol=2e-5,rtol=2e-5)
