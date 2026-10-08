import pytest
import torch
import cu_flash_dism as cu
from flash_dism.forward import forward_core
from flash_dism.backward import backward_core
from flash_dism.varlen import VarlenLayout,forward_varlen
from test_multi_config import inputs


@pytest.mark.parametrize('n',[0,1,65,255,257,511])
def test_fixed_rejects_unaligned(n):
    x=inputs((32,64,64),n,'mixed')
    with pytest.raises(RuntimeError,match='256-token aligned'):
        forward_core(*x)
    with pytest.raises(RuntimeError,match='256-token aligned'):
        cu.core_forward(x,None,0,False,False,False)


@pytest.mark.parametrize('offsets',[[0,128,512],[0,256,257],[0,1,256]])
def test_varlen_rejects_unaligned(offsets):
    with pytest.raises(RuntimeError,match='256-token aligned'):
        VarlenLayout.from_cu_seqlens(torch.tensor(offsets,dtype=torch.int32),offsets[-1])


@pytest.mark.parametrize('packed',[False,True])
def test_no_python_dispatch_or_pad(monkeypatch,packed):
    import flash_dism.backend as backend
    def forbidden(*a,**kw): raise AssertionError('Python dispatch/pad used')
    monkeypatch.setattr(backend,'backend_for',forbidden)
    monkeypatch.setattr(torch.nn.functional,'pad',forbidden)
    x=inputs((16,32,64),512,'mixed')
    if packed:
        layout=VarlenLayout.from_cu_seqlens(torch.tensor([0,256,512],dtype=torch.int32),512)
        out,_,state=forward_varlen(*x,layout=layout,save_state=True)
        from flash_dism.varlen import backward_varlen
        gradients=backward_varlen(state,torch.randn_like(out))
    else:
        out,_,state=forward_core(*x,save_state=True)
        gradients=backward_core(state,torch.randn_like(out))
    for a,b in zip(x[:5],state['operands'][:5]):
        assert a.data_ptr()==b.data_ptr()
    assert all(torch.isfinite(g).all() for g in gradients.values())
    # Public BNH gradients must be views, retaining native contiguous BHN
    # storage for interpolation. A contiguous BNH return adds two round trips.
    for name in ('q_lse', 'k_lse'):
        grad = gradients[name]
        assert grad.shape == x[5].shape
        assert not grad.is_contiguous()
        bhn = grad.transpose(1, 2)
        assert bhn.is_contiguous()
        assert bhn.data_ptr() == grad.data_ptr()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            prepared = bhn.float().contiguous()
        events = {event.key for event in profile.key_averages()}
        assert 'aten::clone' not in events
        assert prepared.is_contiguous()
        # dK-LSE is BF16 and still needs its precision conversion. dQ-LSE is
        # already FP32, so its entire preparation must remain an alias.
        if grad.dtype == torch.float32:
            assert prepared.data_ptr() == grad.data_ptr()


def test_fused_embedding_lse_and_gradient():
    from flash_dism.emb_kernel import EmbInterpFunction
    x=inputs((32,64,64),256,'mixed')
    q,k=x[0],x[1]
    eq,ek=[torch.randn(2,512,64,device='cuda').bfloat16()*.1 for _ in range(2)]
    args=[t.detach().requires_grad_() for t in (q,k,eq,ek)]
    tau=torch.tensor([.5,1.25],device='cuda')
    raw=EmbInterpFunction.apply(*args,1.,None)
    absorbed=EmbInterpFunction.apply(*args,1.,tau.detach())
    for i in (0,1,4,5,6,7): torch.testing.assert_close(absorbed[i],raw[i],atol=0,rtol=0)
    for i in (2,3): torch.testing.assert_close(absorbed[i],raw[i]-tau[None,:,None],atol=1e-6,rtol=1e-6)
    do=[torch.randn_like(t) for t in raw[:4]]
    a=torch.autograd.grad(raw[:4],args,do)
    b=torch.autograd.grad(absorbed[:4],args,do)
    for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=.002,rtol=.005)


def test_vocab_has_no_separate_lse_subtract_or_pad():
    from flash_dism.voc import voc_dism
    x=inputs((16,32,64),256,'mixed')
    q,k,sq,sk,v,_,_,_,_,direction,hard,tau=x
    eq,ek=[torch.randn(2,512,32,device='cuda')*.1 for _ in range(2)]
    voc_dism(q,k,sq,sk,v,eq,ek,tau,direction=direction,hard=hard)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        voc_dism(q,k,sq,sk,v,eq,ek,tau,direction=direction,hard=hard)
    names={event.key for event in profile.key_averages()}
    assert 'aten::sub' not in names
    assert 'aten::constant_pad_nd' not in names
