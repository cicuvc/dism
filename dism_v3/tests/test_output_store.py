import pytest
from pathlib import Path
import subprocess
import torch
import cu_flash_dism as ext


@pytest.mark.parametrize('n',[1,17,33,65,128,129,257,500,769])
@pytest.mark.parametrize('fp32_output',[False,True])
def test_warp_shared_store(n,fp32_output):
    actual = ext.output_store_probe(torch.empty(0,device='cuda'),n,fp32_output)
    assert actual.dtype == (torch.float32 if fp32_output else torch.bfloat16)
    dv = ext.forward_head_dim()
    expected = torch.arange(2*3*n*dv,device='cuda',dtype=torch.float32).reshape(2,3,n,dv)
    expected = (expected.remainder(251) - 125).to(actual.dtype)
    torch.testing.assert_close(actual,expected.transpose(1,2),atol=0,rtol=0)


def test_output_store_codegen():
    root = Path(__file__).resolve().parents[1]
    sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','-sass',
        str(root/f'build/object/r32_d64_v64/forward.cu.dev.sm120a.o')],text=True)
    assert 'CALL' not in sass
    assert 'UTMASTG' not in sass
    assert 'BAR.SYNC.DEFER_BLOCKING 0x4, 0x100' not in sass
    assert 'ACQBULK' not in sass
    assert 'UTMAPF' not in sass
    if ext.forward_output_shared():
        assert 'STG.E.128' in sass
