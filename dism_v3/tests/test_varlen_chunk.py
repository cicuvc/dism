import pytest
import torch
import cu_flash_dism as cu

from test_varlen_layout import make_layout


@pytest.mark.parametrize('lengths',[[0,0],[256,512,256],[768,0,1024,2048]])
@pytest.mark.parametrize('reverse',[False,True])
def test_ragged_chunk(lengths,reverse):
    layout=make_layout(lengths,'cuda')
    h=3
    size=(layout.backward_elements if reverse else layout.forward_elements)*h
    torch.manual_seed(632)
    a=torch.rand(size,device='cuda')
    b=torch.randn_like(a)
    call=cu.varlen_backward_chunk if reverse else cu.varlen_chunk
    result=call(a,b,layout.table,h)
    expected=[]
    for _,n,_,p,f,_,r in layout.table.tolist():
        count=(n+31)//32 if reverse else max(0,(n-1)//32)
        if not count:
            continue
        offset=(r if reverse else f)*h
        aa=a[offset:offset+h*count*p].view(1,h,count,p)
        bb=b[offset:offset+h*count*p].view_as(aa)
        independent=cu.backward_chunk(aa,bb) if reverse else cu.chunk_scan(aa,bb,n)
        expected.append(independent.flatten())
    ref=torch.cat(expected) if expected else torch.empty_like(result)
    torch.testing.assert_close(result,ref,atol=0,rtol=0)
