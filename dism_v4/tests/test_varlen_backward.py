import pytest
import torch
import cu_flash_dism as cu

from flash_dism.forward import forward_core
from flash_dism.backward import backward_core
from flash_dism.varlen import forward_varlen,backward_varlen
from test_varlen_summary import packed_inputs
from test_varlen_forward import slice_inputs


@pytest.mark.parametrize('lengths',[[0,0],[256,512,256],[768,0,256,512]])
@pytest.mark.parametrize('mode',['soft','mixed','hard'])
@pytest.mark.parametrize('fp32_output',[False,True])
def test_packed_backward(lengths,mode,fp32_output):
    layout,x=packed_inputs(lengths,mode)
    out,_,state=forward_varlen(*x,layout=layout,save_state=True)
    dout=torch.randn_like(out).bfloat16()
    gradients,diagnostics=backward_varlen(state,dout,fp32_output=fp32_output,return_diagnostics=True)
    tau=torch.zeros_like(x[-1])
    for start,n,pstart,_,*rest in layout.table.tolist():
        if not n:
            continue
        _,_,reference=forward_core(*slice_inputs(x,start,n),save_state=True)
        derivative=dout[:,start:start+n].contiguous()
        expected=backward_core(reference,derivative,fp32_output=fp32_output)
        for name,actual in gradients.items():
            if name=='rtau':
                continue
            torch.testing.assert_close(actual[:,start:start+n],expected[name],atol=2e-6,rtol=2e-5)
        tau+=expected['rtau']
        delta=cu.backward_delta(reference['output'],derivative)
        actual_delta=diagnostics['delta'].view(1,3,layout.tokens)[:,:,start:start+n]
        torch.testing.assert_close(actual_delta,delta,atol=0,rtol=0)
    torch.testing.assert_close(gradients['rtau'],tau,atol=2e-6,rtol=2e-5)
    for name in ('v','sk_vec','k_vec','k_lse'):
        assert gradients[name].dtype==(torch.float32 if fp32_output else torch.bfloat16)
    for name in ('q_vec','sq_vec','q_lse','rtau'):
        assert gradients[name].dtype==torch.float32


def test_backward_document_isolation():
    layout,x=packed_inputs([256,512,256,768])
    out,_,state=forward_varlen(*x,layout=layout,save_state=True)
    dout=torch.zeros_like(out).bfloat16()
    dout[:,256:768].normal_()
    gradients=backward_varlen(state,dout)
    for name,value in gradients.items():
        if name=='rtau':
            continue
        assert torch.count_nonzero(value[:,:256])==0,name
        assert torch.count_nonzero(value[:,768:])==0,name
