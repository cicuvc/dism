import pytest
import torch
import cu_flash_dism


@pytest.mark.parametrize('chunks',[1,2,3,5,17])
@pytest.mark.parametrize('padded',[16,17,32,48,256,544])
def test_reverse_chunk(chunks,padded):
    torch.manual_seed(919)
    a=torch.rand(2,3,chunks,padded,device='cuda')
    b=torch.randn_like(a)
    got=cu_flash_dism.backward_chunk(a,b)
    expected=torch.zeros_like(a,dtype=torch.float64)
    state=torch.zeros(2,3,padded,device='cuda',dtype=torch.float64)
    step=cu_flash_dism.backward_summary_k()
    for c in range(chunks-1,-1,-1):
        following=torch.nn.functional.pad(state[...,min(step,padded):],(0,min(step,padded)))
        state=a[:,:,c].double()*following+b[:,:,c].double()
        expected[:,:,c]=state
    torch.testing.assert_close(got.double(),expected,atol=1e-6,rtol=2e-6)
