"""Opt-in prototype numerical and persistent-queue regression tests."""
import os
import pytest
import torch

pytestmark=pytest.mark.skipif(os.environ.get('DISM_BUILD_PROBES')!='1',reason='opt-in summary prototype')
from flash_dism.inference.build import load_summary_probe
from flash_dism.inference import GpuPlannerCache
from test_sam_inference import inputs,oracle


@pytest.fixture(autouse=True)
def owner_stream():
    if os.environ.get('DISM_BUILD_PROBES')!='1':
        yield
        return
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield
    stream.synchronize()


@pytest.mark.parametrize('pattern',['random','repeat','mismatch'])
@pytest.mark.parametrize('workers',[1,8,64])
def test_summary_program_decode(pattern,workers):
    iq,ik,sq,sk,v,tau=inputs(pattern);b,n,h,r=sk.shape;start=73
    cache=GpuPlannerCache(b*h,r,32,n,tau*b,rebuild_interval=32)
    labels=torch.stack((ik.permute(2,0,1),iq.permute(2,0,1),torch.zeros_like(iq).permute(2,0,1)),1).reshape(n,3,b*h).contiguous()
    keys=sk.permute(0,2,1,3).reshape(b*h,n,r).contiguous()
    vals=v.permute(0,2,1,3).reshape(b*h,n,32).contiguous()
    cache.prime(labels[:start],keys[:,:start].contiguous(),vals[:,:start].contiguous())
    program=load_summary_probe().SummaryProgram(ik[...,:start].reshape(b*h,start).cpu().contiguous(),tau*b,r,32,8,64)
    program.prepare(keys)
    baseline=program.baseline(keys,vals).clone()
    # Compare the probe baseline's packing/launches to the production cache.
    torch.testing.assert_close(baseline,cache.snapshot[2],rtol=0,atol=0)
    ref=program.reference(keys[:,:start].contiguous().float().cpu(),vals[:,:start].contiguous().float().cpu())
    for _ in range(3):actual=program.run(keys,vals,workers).clone()
    torch.testing.assert_close(actual.cpu().double(),ref,rtol=2e-4,atol=2e-5)
    snap=list(cache.snapshot);snap[2]=actual
    planner=cache.extension.GpuPlanner(snap,start)
    outs=[]
    for i in range(start,n):
        outs.append(planner.step(labels[i],keys[:,i].contiguous(),sq[:,i].reshape(b*h,r).contiguous(),vals[:,i].contiguous()).clone().reshape(b,h,32))
    expected=oracle(iq,ik,sq,sk,v,tau)[:,start:]
    torch.testing.assert_close(torch.stack(outs,1).cpu().double(),expected,rtol=2e-3,atol=2e-5)


@pytest.mark.parametrize('n,block,tile',[(1,1,32),(79,9,64),(257,17,128),(1025,32,128)])
@pytest.mark.parametrize('tau',[0.,.35,1.5])
def test_summary_program_reference(n,block,tile,tau):
    torch.manual_seed(n)
    labels=torch.stack((torch.zeros(n,dtype=torch.int32),torch.arange(n,dtype=torch.int32)%7,torch.randint(0,4,(n,),dtype=torch.int32)))
    p=load_summary_probe().SummaryProgram(labels,[tau]*3,16,32,block,tile)
    k=torch.randn(3,n,16,device='cuda',dtype=torch.bfloat16)*.2
    v=torch.randn(3,n,32,device='cuda',dtype=torch.bfloat16)*.2
    p.prepare(k)
    ref=p.reference(k.float().cpu(),v.float().cpu())
    for workers in (1,32):
        out=p.run(k,v,workers).clone()
        torch.testing.assert_close(out.cpu().double(),ref,rtol=2e-4,atol=2e-5)
    if not p.stats()['active_heads']:return
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):out=p.run(k,v,8)
    for _ in range(3):graph.replay()
    torch.testing.assert_close(out.cpu().double(),ref,rtol=2e-4,atol=2e-5)


@pytest.mark.parametrize('pattern',['repeat','periodic','branching'])
def test_long_zero_tau_boundary_reference(pattern):
    # At tau=0 a wrong attachment to a segment's tail cannot be hidden by decay.
    n=8192;torch.manual_seed(90)
    labels=(torch.zeros(1,n,dtype=torch.int32) if pattern=='repeat' else
            torch.arange(n,dtype=torch.int32)[None,:]%7 if pattern=='periodic' else
            torch.randint(0,4,(1,n),dtype=torch.int32))
    p=load_summary_probe().SummaryProgram(labels,[0.],32,64,128,128)
    k=torch.randn(1,n,32,device='cuda',dtype=torch.bfloat16)*.2
    v=torch.randn(1,n,64,device='cuda',dtype=torch.bfloat16)*.2
    p.prepare(k);out=p.run(k,v,256).cpu().double()
    ref=p.reference(k.float().cpu(),v.float().cpu())
    # Aggregate summary criterion; decode tests above retain the existing
    # elementwise FP64 output tolerances rather than substituting this metric.
    assert ((out-ref).norm()/ref.norm().clamp_min(1e-30)).item()<2e-6
