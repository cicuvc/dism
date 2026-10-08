import pytest
import torch

from flash_dism.forward import forward_core
from flash_dism.varlen import forward_varlen
from test_varlen_summary import packed_inputs


def slice_inputs(inputs,start,n):
    q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau=inputs
    matrices=[x[:,start:start+n].contiguous() for x in (q,k,sq,sk,v,lq,lk)]
    return (*matrices,iq[:,:,start:start+n].contiguous(),ik[:,:,start:start+n].contiguous(),
            direction,hard[:,:,start:start+n].contiguous(),tau)


@pytest.mark.parametrize('lengths',[[0,0],[256,512,256,0],
                                   [768,1024,256,512]])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('fp32_output',[False,True])
def test_packed_forward(lengths,mode,fp32_output):
    layout,inputs=packed_inputs(lengths,mode)
    actual,norm,state=forward_varlen(*inputs,layout=layout,save_state=True,fp32_output=fp32_output)
    assert actual.dtype == (torch.float32 if fp32_output else torch.bfloat16)
    assert actual.shape==inputs[4].shape
    assert state['vertical'].numel()==layout.vertical_elements*3
    for start,n,_,p,f,v,_ in layout.table.tolist():
        if not n:
            continue
        expected,ref_norm,ref_state=forward_core(*slice_inputs(inputs,start,n),save_state=True,
                                               fp32_output=fp32_output)
        torch.testing.assert_close(actual[:,start:start+n],expected,atol=0,rtol=0)
        torch.testing.assert_close(norm[:,:,start:start+n],ref_norm,atol=2e-5,rtol=2e-6)
        count=(n-1)//16
        saved=state['vertical'][v*3:(v+count*p)*3].view(1,3,count,p)
        torch.testing.assert_close(saved,ref_state['vertical'],atol=2e-5,rtol=2e-6)
    inference,_=forward_varlen(*inputs,layout=layout,fp32_output=fp32_output)
    torch.testing.assert_close(actual,inference,atol=0,rtol=0)


def test_document_isolation():
    layout,inputs=packed_inputs([256,512,256,768])
    before,_=forward_varlen(*inputs,layout=layout)
    changed=list(inputs)
    changed[4]=changed[4].clone()
    changed[4][:,256:768].add_(2)
    after,_=forward_varlen(*changed,layout=layout)
    torch.testing.assert_close(before[:,:256],after[:,:256],atol=0,rtol=0)
    torch.testing.assert_close(before[:,768:],after[:,768:],atol=0,rtol=0)
    assert not torch.equal(before[:,256:768],after[:,256:768])
