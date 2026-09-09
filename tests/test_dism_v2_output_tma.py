"""Compare KV1-aliased TMA O against direct BF16 stores, including task tails."""
import os
import pytest
import torch
from dism_v2 import core

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.fixture(scope='module')
def output_modules():
    if os.environ.get('DISM_OUTPUT_Q_ALIAS','kv')!='kv':
        pytest.skip('TMA O experiment requires conservative kv')
    prior=os.environ.get('DISM_OUTPUT_TMA')
    try:
        modules=[]
        for mode in ('0','1'):
            os.environ['DISM_OUTPUT_TMA']=mode
            core._extension.cache_clear()
            modules.append(core._extension())
        yield modules
    finally:
        if prior is None: os.environ.pop('DISM_OUTPUT_TMA',None)
        else: os.environ['DISM_OUTPUT_TMA']=prior
        core._extension.cache_clear()

@pytest.mark.parametrize('n',(1,31,64,65,127,128,129,257,385,1024))
@pytest.mark.parametrize('direction',('q_from_k','k_from_q'))
@pytest.mark.parametrize('probability',(0.,.37,1.))
@torch.no_grad()
def test_output_store_replay(output_modules,monkeypatch,n,direction,probability):
    batch=torch.cuda.get_device_properties(0).multi_processor_count+3
    torch.manual_seed(721)
    a,b,v=[torch.randn(batch,1,n,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    lse=torch.full((batch,1,n),4.,device='cuda')
    tau=torch.zeros(1,device='cuda')
    ql,kl=[torch.randint(0,13,(batch,1,n),device='cuda',dtype=torch.int32) for _ in range(2)]
    inputs=(a,b,v,lse,tau,ql,kl)
    options=dict(sm_scale=.125,direction=direction,hard_prob=probability,
                 return_rng_state=True,save_boundaries=True)
    with monkeypatch.context() as m:
        m.setattr(core,'_extension',lambda:output_modules[0])
        expected=core.forward(*inputs,**options)
    with monkeypatch.context() as m:
        m.setattr(core,'_extension',lambda:output_modules[1])
        for _ in range(2):
            actual=core.forward(*inputs,**options,rng_state=expected[-1])
            for a,b in zip(actual[:2],expected[:2]):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            for field in ('vertical','horizontal'):
                torch.testing.assert_close(getattr(actual[2],field),getattr(expected[2],field),rtol=0,atol=0)
