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
    # Nine output; summary D x direction x (soft + hard/mixed x label width).
    assert sass.count("USETMAXREG.DEALLOC.CTAPOOL")==39
    assert sass.count("USETMAXREG.TRY_ALLOC.CTAPOOL")==39
    assert "UTMALDG.5D" in sass
    functions=re.split(r"Function : (\S+)",sass)
    summaries={functions[i]:functions[i+1] for i in range(1,len(functions),2)
               if "summary_persistent" in functions[i]}
    assert len(summaries)==30
    # One Q transfer per workload and one full-width K transfer per key tile,
    # including D128; neither warp-row nor swizzle-panel emission loops remain.
    assert all(body.count("UTMALDG.5D")==2 for body in summaries.values())
    # CTA initialization plus an independent 128-thread exit barrier per WG.
    # Removing all exit synchronization has stalled persistent N257 workloads.
    for name,body in summaries.items():
        barriers=[line for line in body.splitlines() if "BAR.SYNC" in line]
        assert len(barriers)==2,name
        assert sum(bool(re.search(r"BAR\.SYNC[^;]*, 0x80\s*;",line))
                   for line in barriers)==1,name
    # Two query-label and two distributed key-label loads; no per-score LDG64.
    for name,body in summaries.items():
        wide='Li1ExE' in name or 'Li2ExE' in name
        assert body.count("LDG.E.64")== (4 if wide else 0),name
    assert not re.search(r"\bCALL(?:\.|\s)",sass)
    assert not re.search(r"\b(?:LDL|STL)(?:\.|\s)",sass)
    resources=subprocess.check_output([str(tool),"--dump-resource-usage",binary],text=True)
    local=re.findall(r"LOCAL:(\d+)",resources)
    assert local and all(int(size)==0 for size in local)
    stack=re.findall(r"STACK:(\d+)",resources)
    assert stack and all(int(size)==0 for size in stack)
