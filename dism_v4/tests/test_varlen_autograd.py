import pytest
import torch

from flash_dism.backward import dism_core
from flash_dism.voc import voc_dism
from flash_dism.varlen import dism_core_varlen,voc_dism_varlen
from test_varlen_summary import packed_inputs
from test_varlen_forward import slice_inputs
from gradient_acceptance import assert_gradient


@pytest.mark.parametrize('mode',['soft','mixed','hard'])
def test_core_autograd(mode):
    layout,values=packed_inputs([256,512,0,256],mode)
    values=list(values)
    indices=(0,1,2,3,4,5,6,11)
    for i in indices:
        values[i].requires_grad_()
    actual=dism_core_varlen(*values,layout=layout)
    upstream=torch.randn_like(actual)
    ga=torch.autograd.grad(actual,[values[i] for i in indices],upstream)
    expected=torch.cat([dism_core(*slice_inputs(values,start,n))
        for start,n,*_ in layout.table.tolist() if n],dim=1)
    ge=torch.autograd.grad(expected,[values[i] for i in indices],upstream)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    for a,e in zip(ga,ge):
        torch.testing.assert_close(a,e,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('empty',[False,True])
def test_embedding_autograd(mode,empty):
    layout,x=packed_inputs([0,0] if empty else [256,512,0,256],mode)
    q,k,sq,sk,v,*_=x
    h=q.shape[2]
    eq,ek=[(torch.randn(h,512,64,device='cuda')*.2).requires_grad_() for _ in range(2)]
    tau=x[-1].requires_grad_()
    args=[tensor.requires_grad_() for tensor in (q,k,sq,sk,v)]+[eq,ek,tau]
    out=voc_dism_varlen(*args,direction=x[9],hard=x[10],layout=layout)
    derivative=torch.randn_like(out)
    gradients=torch.autograd.grad(out,args,derivative)
    if empty:
        for value in gradients:
            assert torch.count_nonzero(value)==0
        return
    outputs=[]
    for start,n,*_ in layout.table.tolist():
        if n:
            local=[a[:,start:start+n].contiguous() for a in args[:5]]+args[5:]
            outputs.append(voc_dism(*local,direction=x[9],hard=x[10][:,:,start:start+n].contiguous()))
    expected=torch.cat(outputs,dim=1)
    reference=torch.autograd.grad(expected,args,derivative)
    torch.testing.assert_close(out,expected,atol=.002,rtol=.015)
    # Triton interpolation uses different token tile sizes at document ends;
    # compare with established numerical gates, not an exact accumulation order.
    for name,a,b in zip(('q','k','sq','sk','v','eq','ek','tau'),gradients,reference):
        assert_gradient(a,b,name)
