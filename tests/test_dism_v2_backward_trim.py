"""WS candidate vs reference, including dense summaries and padded identity."""
import itertools
import os
import pytest
import torch
from dism_v2 import core,backward

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.fixture(scope='module')
def modules():
    prior=os.environ.get('DISM_BWD_OPT')
    prior_stages=os.environ.get('DISM_BWD_STAGES')
    target=prior or backward.DEFAULT_OPTIMIZATION
    reference=os.environ.get('DISM_BWD_REFERENCE_OPT','1')
    if int(target)<2:pytest.skip('trim/persistent candidate required')
    try:
        out=[]
        for mode in (reference,target):
            os.environ['DISM_BWD_OPT']=mode
            os.environ['DISM_BWD_STAGES']='2' if mode==reference else (prior_stages or '2')
            backward._extension.cache_clear()
            out.append(backward._extension())
        yield out
    finally:
        if prior is None:os.environ.pop('DISM_BWD_OPT',None)
        else:os.environ['DISM_BWD_OPT']=prior
        if prior_stages is None:os.environ.pop('DISM_BWD_STAGES',None)
        else:os.environ['DISM_BWD_STAGES']=prior_stages
        backward._extension.cache_clear()

@pytest.mark.parametrize('d,dv',itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize('direction',('q_from_k','k_from_q'))
@pytest.mark.parametrize('probability',(0.,.37,1.))
@pytest.mark.parametrize('n',(129,257))
@torch.no_grad()
def test_trim_equivalence(modules,monkeypatch,d,dv,direction,probability,n,batch=2,
                          label_dtype=torch.int32,high_bits=False):
    torch.manual_seed(861)
    def rand(dim):return torch.randn(batch,1,n,dim,device='cuda',dtype=torch.bfloat16)
    a,b=rand(d),rand(d)
    v,do=rand(dv),rand(dv)
    tau=torch.zeros(1,device='cuda')
    lse=torch.full((batch,1,n),4.,device='cuda')
    ql,kl=[torch.randint(0,11,(batch,1,n),device='cuda',dtype=label_dtype) for _ in range(2)]
    if high_bits:
        assert label_dtype==torch.int64
        for label in (ql,kl):
            label.add_(torch.randint(-1,2,label.shape,device='cuda')*(1<<40))
    out,norm,edges,state=core.forward(a,b,v,lse,tau,ql,kl,sm_scale=.125,
        direction=direction,hard_prob=probability,save_boundaries=True,return_rng_state=True)
    def run(module):
        with monkeypatch.context() as m:
            m.setattr(backward,'_extension',lambda:module)
            delta=backward.delta(do,out)
            dv,summary,boundary=backward.value_gradient(a,b,do,lse,tau,ql,kl,norm,edges,
                sm_scale=.125,rng_state=state,v=v,delta=delta,warp_specialized=True)
            grads=backward.operand_gradient(a,b,v,do,lse,tau,ql,kl,norm,delta,edges,boundary,
                sm_scale=.125,rng_state=state,warp_specialized=True)
            return dv,summary,boundary,grads
    baseline=run(modules[0]);actual=run(modules[1])
    for x,y in zip(actual[:3],baseline[:3]):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    for x,y in zip(actual[3],baseline[3]):
        torch.testing.assert_close(x,y,rtol=3e-5,atol=3e-5)
    if probability==1. and int(os.environ.get('DISM_BWD_OPT',backward.DEFAULT_OPTIMIZATION))>=11:
        for grad in actual[3][:3]:
            assert torch.count_nonzero(grad)==0 # dA/dB/dLSE; tau remains checked above.
    summary=actual[1]
    for chunk in range(4,summary.shape[-3]):
        end=(chunk//4)*128
        omitted=summary[...,chunk,:end,:]
        assert torch.count_nonzero(omitted[...,1])==0
        assert torch.all(omitted[...,0]==float(chunk*32>=n))

@pytest.mark.skipif(int(os.environ.get('DISM_BWD_OPT',backward.DEFAULT_OPTIMIZATION))<3,reason='persistent candidate required')
@pytest.mark.parametrize('d,dv',((32,32),(64,64),(128,128)))
@pytest.mark.parametrize('direction',('q_from_k','k_from_q'))
@pytest.mark.parametrize('n',(1,65,129,257,385))
def test_persistent_epochs(modules,monkeypatch,d,dv,direction,n):
    test_trim_equivalence(modules,monkeypatch,d,dv,direction,.37,n,
        batch=torch.cuda.get_device_properties(0).multi_processor_count+3)

@pytest.mark.skipif(int(os.environ.get('DISM_BWD_OPT',backward.DEFAULT_OPTIMIZATION))<10,reason='specialized candidate required')
@pytest.mark.parametrize('d,dv',itertools.product((32,64,128),repeat=2))
@pytest.mark.parametrize('direction',('q_from_k','k_from_q'))
@pytest.mark.parametrize('probability',(0.,.37,1.))
def test_specialized_int64(modules,monkeypatch,d,dv,direction,probability):
    # Labels include +/-2**40: matching low32 bits must not imply a hard match.
    test_trim_equivalence(modules,monkeypatch,d,dv,direction,probability,129,
                          label_dtype=torch.int64,high_bits=True)
