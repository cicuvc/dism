"""Guard against silent setmaxnreg removal or TMA extern-call lowering."""
import re
import subprocess
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME
from dism_v2.core import _extension


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_sm120a_native_tma_and_register_reallocation():
    if torch.cuda.get_device_capability() != (12,0):
        pytest.skip("sm120a code-generation guard")
    tool=Path(CUDA_HOME)/"bin/cuobjdump"
    binary=_extension().__file__
    sass=subprocess.check_output([str(tool),"--dump-sass",binary],text=True)
    # Nine output and three summary specializations; passing does not reallocate.
    assert sass.count("USETMAXREG.DEALLOC.CTAPOOL")==12
    assert sass.count("USETMAXREG.TRY_ALLOC.CTAPOOL")==12
    assert "UTMALDG.5D" in sass
    assert not re.search(r"\bCALL(?:\.|\s)",sass)
    assert not re.search(r"\b(?:LDL|STL)(?:\.|\s)",sass)
    resources=subprocess.check_output([str(tool),"--dump-resource-usage",binary],text=True)
    local=re.findall(r"LOCAL:(\d+)",resources)
    assert local and all(int(size)==0 for size in local)
    stack=re.findall(r"STACK:(\d+)",resources)
    assert stack and all(int(size)==0 for size in stack)
