"""WS optimization codegen gates, reporting (not rejecting) allowed spills."""
import json
import os
import re
import subprocess
from pathlib import Path
import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME
from dism_v2.backward import _extension,DEFAULT_OPTIMIZATION

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_ws_codegen_with_spill_report(record_property):
    if torch.cuda.get_device_capability()!=(12,0):pytest.skip('sm120a')
    binary=_extension().__file__
    tool=str(Path(CUDA_HOME)/'bin/cuobjdump')
    sass=subprocess.check_output([tool,'--dump-sass',binary],text=True)
    assert not re.search(r'\bCALL(?:\.|\s)',sass), 'Uninlined call anywhere in backward extension'
    resources=subprocess.check_output([tool,'--dump-resource-usage',binary],text=True)
    functions=re.split(r'Function : (\S+)',sass)
    usages=dict(re.findall(r'Function (\S+):\n([^\n]+)',resources))
    selected={functions[i]:functions[i+1] for i in range(1,len(functions),2)
              if 'ws14value_backwardI' in functions[i] or 'ab_ws3runI' in functions[i]}
    assert len(selected)==(180 if int(os.environ.get('DISM_BWD_OPT',DEFAULT_OPTIMIZATION))>=10 else 18)
    report={}
    for name,body in selected.items():
        assert not re.search(r'\bCALL(?:\.|\s)',body),name
        assert body.count('USETMAXREG.DEALLOC.CTAPOOL')==1,name
        assert body.count('USETMAXREG.TRY_ALLOC.CTAPOOL')==1,name
        assert 'UTMALDG.5D' in body,name
        if int(os.environ.get('DISM_BWD_OPT',DEFAULT_OPTIMIZATION))>=3:
            expected_tma=4
            if int(os.environ.get('DISM_BWD_OPT',DEFAULT_OPTIMIZATION))==6:
                d,dv=map(int,re.search(r'ILi(\d+)ELi(\d+)E',name).groups())
                if 'ab_ws3runI' in name:
                    prefetch=d+dv<=128 # First A/dO aliases16-KiB scratch.
                else:
                    requested=int(os.environ.get('DISM_BWD_STAGES','2'))
                    input_bytes=128*(d+dv)
                    candidate=max(requested,2)*input_bytes+4*requested*512+512
                    slots=requested if candidate<=63*1024 else 2
                    prefetch=(max(slots,2)+1)*input_bytes+4*slots*512+512<=63*1024
                if prefetch:expected_tma=6
            assert body.count('UTMALDG.5D')==expected_tma,name
            assert body.count('BAR.SYNC')==2,name
        if 'ab_ws3runI' in name:
            hard_only=False
            if int(os.environ.get('DISM_BWD_OPT',DEFAULT_OPTIMIZATION))>=11:
                spec=int(re.search(r'ILi\d+ELi\d+ELi(\d+)E',name).group(1))
                hard_only=spec%5 in (1,2)
            assert ('UTMAREDG.3D.ADD' in body)==(not hard_only),name
            if hard_only:
                assert not re.search(r'\b(?:ATOM|RED)\.',body),name
            if int(os.environ.get('DISM_BWD_OPT',DEFAULT_OPTIMIZATION))==9:
                d=int(re.search(r'ILi(\d+)ELi(\d+)E',name).group(1))
                assert body.count('STS.128')>=2*4*(d//16),name
        report[name]=usages[name].strip()
    record_property('ws_resources',json.dumps(report))
