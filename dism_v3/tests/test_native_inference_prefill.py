"""Exact native packing, routing and ownership against the old inference path."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch

from flash_dism.inference import HardDismPrefill
from flash_dism.inference.prefill_native import NativePrefillPlanner
from flash_dism.inference.prefill_parallel import ParallelPrefillPlanner
from flash_dism.inference.prefill_batch import _merge_chunks
from flash_dism.inference.prefill_triton import TritonPrefillPlan


@pytest.mark.parametrize('pattern',['random','repeat','mismatch'])
@pytest.mark.parametrize('width',[16,32,64])
@pytest.mark.parametrize('threshold',[0,1,64])
def test_exact_packing_and_routing(pattern,width,threshold):
    rng=np.random.default_rng(91)
    planner=NativePrefillPlanner(4)
    with ParallelPrefillPlanner(2) as old:
        for n,with_reset in [(1,False),(79,True),(1025,False),(17,True)]:
            labels=rng.integers(-5,5,(3,2,3,n),dtype=np.int32)
            labels[2]=(rng.random((2,3,n))<.2) if with_reset else 0
            if pattern=='repeat':labels[:2].fill(np.iinfo(np.int32).min)
            if pattern=='mismatch':labels[0].fill(0);labels[1].fill(1)
            tau=np.array([[0.,.5,-.3],[.2,0.,-.1]])
            host=planner.plan(labels if with_reset else labels[:2],tau,
                              chunk_size=width,dispatch_threshold=threshold)
            expected=_merge_chunks(old.plan(labels[0],labels[1],tau,
                reset=labels[2].astype(bool),chunk_size=width),2,3,n)
            for key,value in expected.items():
                np.testing.assert_array_equal(host.data[key].numpy().view(np.uint8),value.view(np.uint8))
            counts=np.diff(expected['offsets']);longs=np.flatnonzero(counts>threshold)
            long_chunks=[np.arange(expected['offsets'][s],expected['offsets'][s+1]) for s in longs]
            routing=dict(stream_ids=np.arange(len(counts)),chunk_ids=np.arange(len(expected['lengths'])),
                         local_offsets=expected['offsets'][:-1],short_stream_ids=np.flatnonzero(counts<=threshold),
                         long_stream_ids=longs,long_chunk_ids=np.concatenate(long_chunks) if len(longs) else [],
                         long_local_offsets=np.r_[0,np.cumsum(counts[longs])[:-1]] if len(longs) else [])
            for key,value in routing.items():np.testing.assert_array_equal(host.data[key].numpy(),value)
            assert host.storage.is_pinned()
            assert all(x.data_ptr()%8==0 for x in host.data.values() if x.numel())
            assert not any('dlog' in k for k in host.data)
    planner.close()


def test_concurrent_calls_and_closed_planner():
    planner=NativePrefillPlanner(4)
    rng=np.random.default_rng(83)
    labels=[rng.integers(0,8,(2,2,3,n),dtype=np.int32) for n in (79,129,257,1025)]
    def run(x):return planner.plan(x,np.full(3,.5))
    reference=[run(x) for x in labels]
    with ThreadPoolExecutor(4) as pool:actual=list(pool.map(run,labels))
    planner.close()
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a.storage,b.storage,rtol=0,atol=0)
    with pytest.raises(RuntimeError,match='closed'):run(labels[0])


def test_validation_and_empty_streams():
    planner=NativePrefillPlanner(2)
    labels=np.zeros((3,1,2,17),dtype=np.int32)
    labels[1]=1
    result=planner.plan(labels,[0.,.5])
    assert result.statistics['streams']==0 and result.data['offsets'].tolist()==[0]
    for bad in (labels.astype(np.int64),labels[...,::2]):
        with pytest.raises(ValueError):planner.plan(bad,[0.,.5])
    for tau in ([.5],[float('nan'),0.]):
        with pytest.raises(ValueError):planner.plan(labels,tau)
    labels[2,0,0,3]=2
    with pytest.raises(ValueError,match='reset'):planner.plan(labels,[0.,.5])
    labels[2]=0
    with pytest.raises(ValueError):planner.plan(labels,[0.,.5],chunk_size=17)
    # Failed submissions leave the pool usable.
    assert planner.plan(labels,[0.,.5]).statistics['streams']==0
    planner.close()


def test_upload_ownership_stream_and_graph():
    torch.manual_seed(93)
    q=torch.randint(0,4,(1,2,257),device='cuda',dtype=torch.int32)
    k=torch.randint(0,4,q.shape,device='cuda',dtype=torch.int32)
    sq=torch.randn(1,257,2,32,device='cuda',dtype=torch.bfloat16)*.2
    sk=torch.randn_like(sq)*.2
    value=torch.randn_like(sq)*.2
    reset=torch.zeros_like(q,dtype=torch.bool);reset[...,31::33]=True
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    engine=HardDismPrefill(workers=4,dispatch_threshold=1)
    with torch.cuda.stream(stream):
        prepared=engine.prepare(q,k,[0.,.5],reset=reset)
        # A second prepare must not overwrite the first pinned or GPU blob.
        other=engine.prepare(q+1,k,[.3,.7])
        engine.close()
        for _ in range(2):prepared.execute_dispatch(sq,sk,value)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):out=prepared.execute_dispatch(sq,sk,value)
        graph.replay()
        expected=prepared.execute_dispatch(sq,sk,value)
        torch.testing.assert_close(out,expected,rtol=2e-4,atol=2e-6)
        assert other.execute(sq,sk,value).isfinite().all().item()
    torch.cuda.current_stream().wait_stream(stream)
    with pytest.raises(ValueError,match='upload stream'):prepared.execute(sq,sk,value)
    # Reference upload produces identical packed metadata, including indices.
    with ParallelPrefillPlanner(2) as old:
        packed=_merge_chunks(old.plan(q.cpu().numpy(),k.cpu().numpy(),[0.,.5],
            reset=reset.cpu().numpy()),1,2,257)
    baseline=TritonPrefillPlan.from_packed(packed,514,dispatch_threshold=1)
    for name,tensor in vars(baseline).items():
        if isinstance(tensor,torch.Tensor):
            torch.testing.assert_close(getattr(prepared.core,name),tensor,rtol=0,atol=0)


def test_async_upload_releases_host_safely():
    """Pinned source must not recycle while an earlier stream delays H2D."""
    rng=np.random.default_rng(551)
    planner=NativePrefillPlanner(2)
    host=planner.plan(rng.integers(0,4,(2,1,2,257),dtype=np.int32),[0.,.5])
    expected={name:tensor.clone() for name,tensor in host.data.items()}
    size=host.storage.numel()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(150_000_000)
        core=host.upload(device='cuda',mma_precision='bf16')
    del host
    planner.close()
    # Immediately allocate/fill similarly sized pinned memory while the copy is
    # queued. The allocator must defer reuse of the source until copy completion.
    for _ in range(4):
        trash=torch.empty(size,dtype=torch.uint8,pin_memory=True).fill_(0xcd)
        del trash
    stream.synchronize()
    for name,tensor in expected.items():
        torch.testing.assert_close(getattr(core,name).cpu(),tensor,rtol=0,atol=0)
