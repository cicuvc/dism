import pytest
import torch
import cu_flash_dism as cu

from flash_dism.varlen import stage_operands
from flash_dism.summary import summarize
from test_varlen_layout import make_layout


def packed_inputs(lengths, mode='mixed', heads=3):
    torch.manual_seed(3409)
    layout=make_layout(lengths,'cuda')
    t,h=layout.tokens,heads
    q,k=[(.2*torch.randn(1,t,h,64,device='cuda')).bfloat16() for _ in range(2)]
    sq,sk=[(.2*torch.randn(1,t,h,32,device='cuda')).bfloat16() for _ in range(2)]
    v=torch.randn_like(q)
    lq,lk=[torch.full((1,t,h),6.,device='cuda') for _ in range(2)]
    iq,ik=[torch.randint(8,(1,h,t),device='cuda') for _ in range(2)]
    hard=torch.rand(1,h,t,device='cuda')<dict(soft=0.,mixed=.5,hard=1.)[mode]
    direction=(torch.arange(h,device='cuda')%2==0)[None].contiguous()
    tau=torch.full((h,),2.,device='cuda')
    return layout,(q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau)


@pytest.mark.parametrize('lengths',[[0,0],[256,512,0,256],[768,1024]])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
def test_ragged_summary(lengths,mode):
    layout,x=packed_inputs(lengths,mode)
    operands=stage_operands(layout,*x)
    a,b=cu.varlen_summary(operands,layout.table)
    assert a.numel()==b.numel()==layout.forward_elements*3
    q,k,_,_,_,lq,lk,iq,ik,direction,hard,tau=x
    for start,n,_,p,offset,*_ in layout.table.tolist():
        if n<=32:
            continue
        rows=(n-1)//32
        oracle=summarize(q[:,start:start+n].contiguous(),k[:,start:start+n].contiguous(),
            lq[:,start:start+n]-tau,lk[:,start:start+n]-tau,
            iq[:,:,start:start+n].contiguous(),ik[:,:,start:start+n].contiguous(),
            direction,hard[:,:,start:start+n].contiguous(),tau,ctas=1)
        valid=torch.arange(p,device='cuda')[None,:] < (torch.arange(rows,device='cuda')[:,None]+1)*32
        for result,expected in zip((a,b),oracle):
            actual=result[offset*3:(offset+rows*p)*3].view(1,3,rows,p)
            torch.testing.assert_close(actual[...,valid],expected[...,valid],atol=0,rtol=0)
