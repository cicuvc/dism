"""Persistent summaries: exact-history decoding, scratch reuse and ownership."""
import pytest
import torch
from flash_dism.inference.summary_runtime import PersistentGpuPlannerCache, SummaryWorkspace
from test_sam_inference import oracle


@pytest.mark.parametrize('pattern',['repeat','periodic','random','empty'])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
@torch.inference_mode()
def test_rebuilds_match_dense(pattern,dtype):
    torch.manual_seed(911)
    h,n,prime=3,257,137
    ik=torch.randint(0,4,(1,h,n),device='cuda',dtype=torch.int32)
    iq=torch.randint(0,4,ik.shape,device='cuda',dtype=torch.int32)
    if pattern=='repeat':ik.zero_();iq.zero_()
    elif pattern=='periodic':ik.copy_(torch.arange(n,device='cuda')[None,None,:]%3);iq.copy_(ik)
    elif pattern=='empty':ik.copy_(torch.arange(n,device='cuda')[None,None,:]);iq.copy_(ik)
    reset=torch.zeros_like(ik,dtype=torch.bool);reset[...,19::37]=True
    sq,sk,v=[(torch.randn(1,n,h,d,device='cuda')*.2).to(dtype) for d in (16,16,32)]
    tau=[0.,.35,1.5]
    expected=oracle(iq,ik,sq,sk,v,tau,reset)
    labels=torch.stack((ik[0].T,iq[0].T,reset[0].T.int()),1).contiguous()
    k,val=sk[0].transpose(0,1).contiguous(),v[0].transpose(0,1).contiguous()
    c=PersistentGpuPlannerCache(h,16,32,n,tau,rebuild_interval=16,cache_dtype=dtype,summary_workers=8,summary_slots=3,summary_block=16,summary_tile=64)
    native_owner=c.cpu
    c.prime(labels[:prime],k[:,:prime].contiguous(),val[:,:prime].contiguous())
    pointers=[c.snapshot[i].data_ptr() for i in (0,1,6,7)]
    outputs=[]
    for i in range(prime,n):
        if (i-prime)%16==0:
            old_output=c.snapshot[2];old_copy=old_output.clone()
            c.rebuild()
            assert c.last_rebuild['sam_extended_keys']==h*c.last_rebuild['delta_labels']
            torch.testing.assert_close(old_output,old_copy,atol=0,rtol=0)
        outputs.append(c.step(labels[i],k[:,i].contiguous(),sq[0,i].contiguous(),val[:,i].contiguous()).clone())
    assert c.check_status()==n and c.cpu is native_owner
    assert pointers==[c.snapshot[i].data_ptr() for i in (0,1,6,7)]
    torch.testing.assert_close(torch.stack(outputs)[None].cpu().double(),expected[:,prime:],atol=2e-5,rtol=2e-3)
    assert c.last_rebuild['diagnostic_baseline_scratch_bytes']==0
    if pattern!='empty':assert c.last_rebuild['reused_scratch_buffers']>0
    # Same cutoff, same requirement: no allocation and identical pool addresses.
    c.rebuild(); ptr={k:v.data_ptr() for k,v in c.workspace.buffers.items()}
    c.rebuild()
    assert c.last_rebuild['allocated_scratch_buffers']==0
    assert ptr=={k:v.data_ptr() for k,v in c.workspace.buffers.items()}


def test_workspace_wrong_stream():
    w=SummaryWorkspace('cuda')
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError,match='stream mismatch'):w.check_owner()


@pytest.mark.parametrize('tau',[0.,.35,1.5])
@torch.inference_mode()
def test_pruning_preserves_outputs_and_reuses_pool(tau):
    from flash_dism.inference.build import load_summary_runtime
    module=load_summary_runtime()
    torch.manual_seed(92)
    n=2049
    labels=torch.stack((torch.randint(0,32,(n,),dtype=torch.int32),torch.arange(n,dtype=torch.int32)%7))
    k=torch.randn(2,n,16,device='cuda',dtype=torch.bfloat16)*.2
    v=torch.randn(2,n,32,device='cuda',dtype=torch.bfloat16)*.2
    full=module.SummaryProgram(labels,[tau]*2,16,32,32,64,3,True)
    full.prepare(k,False)
    expected=full.run(k,v,32).clone()
    old_pool=full.workspace()
    pruned=module.SummaryProgram(labels,[tau]*2,16,32,32,64,3,True,True,False)
    pruned.prepare(k,False,old_pool)
    actual=pruned.run(k,v,32).clone()
    # Same Euler prefix and surviving heavy-chain arithmetic: bitwise equality.
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert pruned.stats()['total_tasks']<=full.stats()['total_tasks']
    assert pruned.stats()['allocated_scratch_buffers']==0
    assert all(t.data_ptr()==old_pool[name].data_ptr() for name,t in pruned.workspace().items())
    # Output is not scratch: later writes do not modify the old snapshot.
    copy=expected.clone()
    pruned.run(k*2,v,8)
    torch.testing.assert_close(expected,copy,rtol=0,atol=0)


@torch.inference_mode()
def test_native_pending_summary_and_borrowed_lifetime():
    from flash_dism.inference.runtime import NativeDecodeCache
    h,n,r,d=2,257,16,32
    labels=torch.zeros(n,3,h,dtype=torch.int32,device='cuda')
    keys=torch.randn(h,n,r,device='cuda',dtype=torch.bfloat16)*.2
    values=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16)*.2
    c=NativeDecodeCache(h,r,d,n,[0.,.35],rebuild_interval=16)
    c.prime(labels,keys,values)
    snapshot=c.native.gpu_snapshot()
    expected=snapshot[2].clone()
    c.native.refresh_gpu(snapshot[7],n,snapshot[6],False)
    with pytest.raises(RuntimeError,match='installed'):
        c.native.gpu_snapshot()
    p=c.native.summary_program(32,64,3)
    output=p.prepare(keys,False)
    c.native.install_summary(output)
    c.native.gpu_snapshot()
    # Program keeps planner storage alive without keeping the complete native
    # GPU cache, worker pool or diagnostic scratch alive.
    del c,snapshot
    p.prepare(keys,False,p.workspace())
    actual=p.run(keys,values,8).clone()
    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
    p.release_compiler()
    torch.testing.assert_close(p.run(keys,values,8),actual,rtol=0,atol=0)
