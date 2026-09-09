"""Int32 metadata and endpoint-specialized summaries preserve the int64 oracle path."""
import itertools
import math
import pytest
import torch
from dism_v2.core import forward
from dism_v2.embedding import forward as embedding

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('d,dv',itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize('direction',('q_from_k','k_from_q'))
@pytest.mark.parametrize('prob',(0.,.37,1.))
@torch.no_grad()
def test_label_width(d,dv,direction,prob):
    torch.manual_seed(413+d+dv)
    def rand(shape):return torch.randn(shape,device='cuda',dtype=torch.bfloat16)
    n=139
    q,k=rand((1,2,n,d)),rand((1,2,n,d))
    v=rand((1,2,n,dv));e,f=rand((2,65,d)),rand((2,65,d))
    raw=embedding(q,k,e,f,d**-.5)
    a,b,lse=(q,raw[0],raw[3]) if direction=='q_from_k' else (raw[1],k,raw[2])
    tau=torch.full((2,),math.log(d),device='cuda')
    results=[]
    for dtype in (torch.int64,torch.int32):
        result=forward(a,b,v,lse,tau,raw[7].to(dtype),raw[6].to(dtype),
            sm_scale=d**-.5,direction=direction,hard_prob=prob,return_debug=True,
            generator=torch.Generator(device='cuda').manual_seed(792),save_boundaries=True)
        results.append((*result[:4],result[4].vertical,result[4].horizontal))
    for x,y in zip(*results):torch.testing.assert_close(x,y,atol=0,rtol=0)

@torch.no_grad()
def test_int64_not_truncated():
    q=torch.zeros((1,1,65,32),device='cuda',dtype=torch.bfloat16)
    v=torch.ones_like(q);lse=torch.zeros(q.shape[:3],device='cuda');tau=torch.zeros(1,device='cuda')
    labels=torch.zeros(q.shape[:3],device='cuda',dtype=torch.int64)
    o,_=forward(q,q,v,lse,tau,labels,labels+2**32,sm_scale=1.,direction='q_from_k',hard_prob=1.)
    assert torch.count_nonzero(o)==0
