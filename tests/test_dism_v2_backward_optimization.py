"""WS optimization codegen gates, reporting (not rejecting) allowed spills."""
import json
import os
import re
import subprocess
from pathlib import Path
import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME
from dism_v2.backward import _extension

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_ws_codegen_with_spill_report(record_property):
    if torch.cuda.get_device_capability()!=(12,0):pytest.skip('sm120a')
    binary=_extension().__file__
    tool=str(Path(CUDA_HOME)/'bin/cuobjdump')
    sass=subprocess.check_output([tool,'--dump-sass',binary],text=True)
    resources=subprocess.check_output([tool,'--dump-resource-usage',binary],text=True)
    functions=re.split(r'Function : (\S+)',sass)
    usages=dict(re.findall(r'Function (\S+):\n([^\n]+)',resources))
    selected={functions[i]:functions[i+1] for i in range(1,len(functions),2)
              if 'ws14value_backwardI' in functions[i] or 'ab_ws3runI' in functions[i]}
    assert len(selected)==18
    report={}
    for name,body in selected.items():
        assert not re.search(r'\bCALL(?:\.|\s)',body),name
        assert body.count('USETMAXREG.DEALLOC.CTAPOOL')==1,name
        assert body.count('USETMAXREG.TRY_ALLOC.CTAPOOL')==1,name
        assert 'UTMALDG.5D' in body,name
        if int(os.environ.get('DISM_BWD_OPT','0'))>=3:
            assert body.count('UTMALDG.5D')==4,name
            assert body.count('BAR.SYNC')==2,name
        if 'ab_ws3runI' in name:assert 'UTMAREDG.3D.ADD' in body,name
        report[name]=usages[name].strip()
    record_property('ws_resources',json.dumps(report))
