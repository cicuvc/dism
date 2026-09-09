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


@pytest.mark.skipif(int(os.environ.get('DISM_BWD_OPT',DEFAULT_OPTIMIZATION))<12,
                   reason='single-FFMA experiment required')
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_single_ffma_score_codegen(tmp_path):
    """Real B1/B3 D64 mixed specializations:32 score FFMAs/warp, no score LDG/BRA.

    Source annotations may include scheduler-moved integer predicates, so only
    forbid extra floating arithmetic, memory loads and branches at the score
    expression. The whole-kernel noCALL/resource gates remain separate.
    """
    if torch.cuda.get_device_capability()!=(12,0):pytest.skip('sm120a')
    build=Path(_extension().__file__).parent
    source=Path(__file__).parents[1]/'dism_v2/csrc/backward_metadata.cuh'
    line=next(i for i,s in enumerate(source.read_text().splitlines(),1)
              if 'result=fmaf(dot,key.scale2,bias);' in s)
    marker=f'backward_metadata.cuh", line {line} '
    for stem in ('core_dv_ws','core_ab_ws'):
        dest=tmp_path/stem;dest.mkdir()
        subprocess.check_call([str(Path(CUDA_HOME)/'bin/cuobjdump'),'-xelf','all',
                               str(build/(stem+'.cuda.o'))],cwd=dest)
        cubins=list(dest.glob('*.cubin'));assert len(cubins)==1
        symbols=subprocess.check_output(['readelf','-sW',str(cubins[0])],text=True)
        indices=[s.split(':',1)[0].strip() for s in symbols.splitlines()
                 if ' FUNC ' in s and re.search(r'ILi64ELi64ELi[38]EE',s)]
        assert len(indices)==2
        for index in indices:
            sass=subprocess.check_output([str(Path(CUDA_HOME)/'bin/nvdisasm'),
                '-c','-gi','-fun',index,str(cubins[0])],text=True)
            annotations=[];active=False;ops=[]
            for s in sass.splitlines():
                if '//##' in s:annotations.append(s)
                elif re.search(r'/\*[0-9a-f]+\*/',s):
                    if annotations:
                        active=any(marker in a for a in annotations);annotations=[]
                    if active:
                        ops.append(re.sub(r'^.*?\*/\s*(?:@!?P\d+\s*)?','',s).split()[0])
            assert ops.count('FFMA')==32,(stem,index,ops)
            assert not any(op.startswith(('FADD','FMUL','LDG','BRA','CALL')) for op in ops),ops
            if stem=='core_ab_ws' and int(os.environ.get('DISM_BWD_OPT',DEFAULT_OPTIMIZATION))>=14:
                src=(source.parent/'core_ab_ws.cu').read_text().splitlines()
                begin=next(i for i,s in enumerate(src,1) if 'if(warp==9 && leader)' in s)
                end=next(i for i,s in enumerate(src,1) if 'if(warp==8)' in s)
                # Every output issue/commit/wait must belong to the elected
                # writer branch, never to the consumer's computation region.
                annotations=[];writer=False;counts={}
                for s in sass.splitlines():
                    if '//##' in s:annotations.append(s)
                    elif re.search(r'/\*[0-9a-f]+\*/',s):
                        if annotations:
                            lines=[int(m.group(1)) for a in annotations
                                   if (m:=re.search(r'core_ab_ws.cu", line (\d+)\b',a))]
                            writer=any(begin<=line<end for line in lines)
                            annotations=[]
                        if any(op in s for op in ('UTMAREDG.3D.ADD','UTMACMDFLUSH','DEPBAR')):
                            assert writer,(index,s)
                            for op in ('UTMAREDG.3D.ADD','UTMACMDFLUSH','DEPBAR'):
                                if op in s:counts[op]=counts.get(op,0)+1
                assert counts['UTMAREDG.3D.ADD']==128,counts
                assert counts['UTMACMDFLUSH']==32,counts
                assert counts['DEPBAR']>=32,counts
