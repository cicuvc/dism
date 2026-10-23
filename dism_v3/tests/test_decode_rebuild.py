"""Incremental GPU history/band import and pinned arena reuse boundaries."""
import pytest
import torch

from flash_dism.inference import GpuPlannerCache
from flash_dism.inference.runtime import NativeDecodeCache
from test_sam_inference import oracle


@pytest.mark.parametrize('interval',[11,128,512,1024])
@pytest.mark.parametrize('workers',[1,4])
def test_incremental_snapshot_matches_fresh_prime(interval,workers):
    torch.manual_seed(819+interval)
    h,n=2,73;count=2*interval+3;total=n+count
    labels=torch.zeros(total,3,h,dtype=torch.int32,device='cuda')
    labels[:,0,1]=torch.arange(total,device='cuda')%7
    labels[:,1,1]=(torch.arange(total,device='cuda')+2)%7
    labels[17::31,2,1]=1
    sk=torch.randn(h,total,16,device='cuda',dtype=torch.bfloat16)*.2
    value=torch.randn(h,total,32,device='cuda',dtype=torch.bfloat16)*.2
    sq=torch.randn(count,h,16,device='cuda',dtype=torch.bfloat16)*.2
    cache=GpuPlannerCache(h,16,32,total,[0.,.35],rebuild_interval=interval,rebuild_workers=workers)
    cache.prime(labels[:n],sk[:,:n].contiguous(),value[:,:n].contiguous())
    history_ptr=cache.snapshot[7].data_ptr()
    for start in range(0,count,interval):
        stop=min(start+interval,count)
        for i in range(start,stop):
            cache.step(labels[n+i],sk[:,n+i].contiguous(),sq[i],value[:,n+i].contiguous())
        pos=n+stop
        cache.rebuild()
        stats=cache.cpu.memory_stats()
        assert stats['last_refresh_tokens']==stop-start
        assert stats['last_sam_extended_keys']==h*(stop-start)
        assert stats['last_refresh_band_reused']==1
        assert cache.snapshot[7].data_ptr()==history_ptr
        # Fresh initialize reconstructs the complete band by diagonal matching
        # and builds the SAM from zero, independently of GPU band import/reuse.
        fresh=NativeDecodeCache(h,16,32,total,[0.,.35],rebuild_interval=interval,rebuild_workers=workers)
        fresh.prime(labels[:pos],sk[:,:pos].contiguous(),value[:,:pos].contiguous())
        expected=fresh.native.gpu_snapshot()
        for j in (2,3,4,5,6,8,9):torch.testing.assert_close(cache.snapshot[j],expected[j],rtol=0,atol=0)
        torch.testing.assert_close(cache.snapshot[7][:pos],labels[:pos],rtol=0,atol=0)
    assert cache.check_status()==total


def test_cpu_labels_pending_upload_and_snapshot_ownership():
    torch.manual_seed(77)
    h,n,prime=2,57,17
    iq=torch.zeros(1,h,n,dtype=torch.int32,device='cuda');ik=iq.clone()
    iq[:,1]=torch.arange(n,device='cuda')%3;ik[:,1]=torch.arange(n,device='cuda')%3
    sq,sk,v=[torch.randn(1,n,h,width,device='cuda',dtype=torch.bfloat16)*.2 for width in (16,16,32)]
    expected=oracle(iq,ik,sq,sk,v,[0.,.35])
    labels=torch.stack((ik[0].T,iq[0].T,torch.zeros(n,h,device='cuda',dtype=torch.int32)),dim=1).cpu()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache=NativeDecodeCache(h,16,32,n,[0.,.35],rebuild_interval=2,sample_interval=5,
                                materialize_threshold=32,rebuild_workers=4)
        torch.cuda._sleep(100_000_000)
        cache.prime(labels[:prime],sk[0,:prime].transpose(0,1).contiguous(),v[0,:prime].transpose(0,1).contiguous())
        first=cache.native.gpu_snapshot()
        old_topology=first[3].clone()
        second=cache.native.gpu_snapshot()
        outputs=[]
        for i in range(prime,n):
            outputs.append(cache.step(labels[i],sk[0,i].contiguous(),sq[0,i].contiguous(),v[0,i].contiguous()))
        actual=torch.stack(outputs).cpu()
        torch.testing.assert_close(first[3],old_topology,rtol=0,atol=0)
        assert first[3].data_ptr()!=second[3].data_ptr()
    stream.synchronize()
    torch.testing.assert_close(actual.double(),expected[0,prime:],rtol=.002,atol=2e-5)


def test_legacy_refresh_and_validation():
    cache=GpuPlannerCache(1,16,32,32,[.5],rebuild_interval=8)
    labels=torch.zeros(3,1,device='cuda',dtype=torch.int32)
    sk=torch.ones(1,16,device='cuda',dtype=torch.bfloat16)
    v=torch.ones(1,32,device='cuda',dtype=torch.bfloat16)
    for _ in range(5):cache.step(labels,sk,sk,v)
    cache.cpu.native.refresh_gpu(cache.snapshot[7],5) # Legacy full-history replacement.
    cache._export(5)
    assert cache.check_status()==5
    assert cache.cpu.memory_stats()['last_refresh_band_reused']==0
    for bad in (cache.snapshot[6].float(),cache.snapshot[6][...,:3]):
        with pytest.raises(RuntimeError):cache.cpu.native.refresh_gpu(cache.snapshot[7],5,bad)
    for workers in (0,65):
        with pytest.raises(RuntimeError,match='rebuild_workers'):
            NativeDecodeCache(1,16,32,32,[.5],rebuild_workers=workers)
    cache.rebuild()
    assert cache.check_status()==5
