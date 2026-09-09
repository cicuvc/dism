"""Packed row decisions must preserve Philox identity, replay and gradients."""
import itertools
import math
import pytest
import torch
from dism_v2.autograd import voc_dism
from dism_v2.core import RowRNGState
from dism_v2.embedding import forward as embedding
from test_dism_v2_precision import row_mask

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('d',(32,64,128))
@pytest.mark.parametrize('n',(1,17,31,32,33,63,64,65,129,257,1025))
@torch.no_grad()
def test_embedding_bitset(d,n):
    q=torch.randn((2,2,n,d),device='cuda',dtype=torch.bfloat16)
    e=torch.randn((2,33,d),device='cuda',dtype=torch.bfloat16)
    state=RowRNGState(2**63+12345,2**34+12,tuple(q.shape[:3]),'q_from_k',.37)
    raw=embedding(q,q,e,e,d**-.5,row_rng=(state.seed,state.offset,state.hard_prob))
    ordinary=embedding(q,q,e,e,d**-.5)
    for x,y in zip(raw[:8],ordinary): torch.testing.assert_close(x,y,atol=0,rtol=0)
    bits=raw[8].long()&0xffffffff
    unpacked=((bits[...,None]>>torch.arange(32,device='cuda'))&1).flatten(-2)
    assert torch.equal(unpacked[...,:n].bool(),row_mask(state).squeeze(-1))
    assert not unpacked[...,n:].any()

@pytest.mark.parametrize('d,dv',itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize('direction',('q_from_k','k_from_q','random'))
def test_packed_autograd(d,dv,direction,monkeypatch):
    torch.manual_seed(431+d+dv)
    n=139
    def rand(shape): return torch.randn(shape,device='cuda',dtype=torch.bfloat16).requires_grad_()
    inputs=[rand((1,2,n,d)),rand((1,2,n,d)),rand((1,2,n,dv)),
        torch.full((2,),math.log(d),device='cuda',requires_grad=True),rand((2,65,d)),rand((2,65,d))]
    dout=torch.randn_like(inputs[2])
    records=[]
    for mode in ('0','1'):
        monkeypatch.setenv('DISM_ROW_BITSET',mode)
        gen=torch.Generator(device='cuda').manual_seed(923)
        gen.set_offset(2**34+12)
        out,state=voc_dism(*inputs,sm_scale=d**-.5,direction=direction,hard_prob=.37,
            generator=gen,embedding_backend='cuda',embedding_backward_backend='cuda',return_rng_state=True)
        grad=torch.autograd.grad(out,inputs,dout)
        records.append((out,state,grad,gen.get_offset()))
    x,y=records
    assert x[1]==y[1] and x[3]==y[3]==2**34+12+4+(4 if direction=='random' else 0)
    torch.testing.assert_close(x[0],y[0],atol=0,rtol=0)
    for a,b in zip(x[2],y[2]): torch.testing.assert_close(a,b,atol=.002,rtol=.008)
    replay,state=voc_dism(*inputs,sm_scale=d**-.5,direction=direction,hard_prob=.37,
        rng_state=y[1],embedding_backend='cuda',return_rng_state=True)
    assert state==y[1]
    torch.testing.assert_close(replay,y[0],atol=0,rtol=0)

@pytest.mark.parametrize('p',(0.,1.))
def test_endpoints_no_bitset(p,monkeypatch):
    import dism_v2.embedding as module
    original=module.forward
    def checked(*args,**kwargs):
        assert kwargs.get('row_rng') is None
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'forward',checked)
    monkeypatch.setenv('DISM_ROW_BITSET','1')
    q=torch.randn((1,1,17,32),device='cuda',dtype=torch.bfloat16)
    e=q[0,:,:7].contiguous()
    tau=torch.zeros(1,device='cuda')
    gen=torch.Generator(device='cuda').manual_seed(12)
    voc_dism(q,q,q,tau,e,e,direction='q_from_k',hard_prob=p,generator=gen,embedding_backend='cuda')
    assert gen.get_offset()==0
