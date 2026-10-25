import pytest
import torch

from flash_dism_v4.reference import dism_v4_ref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA prototype")


@pytest.mark.parametrize('mode', ['hard', 'soft', 'jensen'])
@pytest.mark.parametrize('block', [32, 64, 128])
@pytest.mark.parametrize('tau_value', [-.5, 0., .7])
def test_blocks_against_fp64(mode, block, tau_value):
    from flash_dism_v4.blocked import dism_v4_blocked
    torch.manual_seed(77)
    n = 73
    q = torch.randn(2,n,2,5,device='cuda',requires_grad=True)
    k = torch.randn_like(q,requires_grad=True)
    v = torch.randn(2,n,2,4,device='cuda',requires_grad=True)
    tau = torch.full((2,),tau_value,device='cuda',requires_grad=True)
    reset = torch.zeros(2,n,device='cuda',dtype=torch.bool)
    reset[:,[0,31,61]] = True
    actual = dism_v4_blocked(q,k,v,tau,key_mode=mode,block_size=block,query_reset=reset)
    expected = dism_v4_ref(q.double(),k.double(),v.double(),tau.double(),key_mode=mode,query_reset=reset)
    torch.testing.assert_close(actual.double(),expected,atol=3e-6,rtol=3e-5)
    g = torch.randn_like(actual)
    da = torch.autograd.grad((actual*g).sum(),(q,k,v,tau),allow_unused=True)
    de = torch.autograd.grad((expected*g).sum(),(q,k,v,tau),allow_unused=True)
    for a,e in zip(da,de):
        if e is None:
            assert a is None
        else:
            torch.testing.assert_close(a,e,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('mode', ['hard','soft','jensen'])
def test_extreme_per_symbol_scaling(mode):
    from flash_dism_v4.blocked import dism_v4_blocked
    q = torch.tensor([[[[1000.,-1000.]],[[1000.,-1000.]],[[1000.,-1000.]],[[0.,0.]]]],device='cuda',requires_grad=True)
    k = q.detach().clone(); k[0,-1,0] = torch.tensor([-1000.,1000.],device='cuda'); k.requires_grad_()
    v = torch.randn(1,4,1,3,device='cuda',requires_grad=True)
    tau = torch.tensor([1000.],device='cuda',requires_grad=True)
    actual = dism_v4_blocked(q,k,v,tau,key_mode=mode)
    expected = dism_v4_ref(q.double(),k.double(),v.double(),tau.double(),key_mode=mode)
    if mode != 'hard':
        # Explicitly retained counterexample: unconditional GEMM loses the
        # weak symbol bucket here. The former slow protection was removed at
        # the user's request; do not claim finite extreme-range soft output.
        assert bool(torch.isfinite(expected).all())
        assert not bool(torch.isfinite(actual).all())
        return
    torch.testing.assert_close(actual.double(),expected,atol=1e-5,rtol=1e-5)
    ga = torch.autograd.grad(actual.sum(),(q,k,v,tau),allow_unused=True)
    ge = torch.autograd.grad(expected.sum(),(q,k,v,tau),allow_unused=True)
    for a,e in zip(ga,ge):
        if e is None:assert a is None
        else:torch.testing.assert_close(a,e,atol=2e-4,rtol=1e-4)


@pytest.mark.parametrize('soft_readout',['arithmetic','jensen'])
def test_mixed_rows_packing_and_autocast(soft_readout):
    from flash_dism_v4.blocked import dism_v4_blocked
    torch.manual_seed(93)
    q = torch.randn(1,71,2,5,device='cuda',requires_grad=True)
    k = torch.randn_like(q,requires_grad=True)
    v = torch.randn(1,71,2,4,device='cuda',requires_grad=True)
    tau = torch.tensor([.5,.8],device='cuda',requires_grad=True)
    cu = torch.tensor([0,30,71],device='cuda',dtype=torch.int32)
    hard = torch.tensor([[True,False]],device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actual = dism_v4_blocked(q,k,v,tau,hard=hard,cu_seqlens=cu,soft_readout=soft_readout)
    expected = torch.cat([dism_v4_ref(q[:,:,i:i+1].double(),k[:,:,i:i+1].double(),v[:,:,i:i+1].double(),tau[i:i+1].double(),key_mode=m,cu_seqlens=cu) for i,m in enumerate(('hard','jensen' if soft_readout=='jensen' else 'soft'))],2)
    torch.testing.assert_close(actual.double(),expected,atol=3e-6,rtol=3e-5)
    ga = torch.autograd.grad(actual.square().sum(),(q,k,v,tau))
    ge = torch.autograd.grad(expected.square().sum(),(q,k,v,tau))
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('mode',['soft','jensen'])
def test_saved_activations_are_not_pair_matrices(mode):
    from flash_dism_v4.blocked import dism_v4_blocked
    q = torch.randn(1,257,2,7,device='cuda',requires_grad=True)
    k = torch.randn_like(q,requires_grad=True)
    v = torch.randn(1,257,2,4,device='cuda',requires_grad=True)
    tau = torch.zeros(2,device='cuda',requires_grad=True)
    saved=[]
    def pack(t):
        saved.append(tuple(t.shape));return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        out=dism_v4_blocked(q,k,v,tau,key_mode=mode)
    assert not any(s.count(257)>1 for s in saved),saved
    out.sum().backward()


@pytest.mark.parametrize('mode',['soft','hard','jensen'])
def test_bf16_mma_explicit_error_budget(mode):
    from flash_dism_v4.blocked import dism_v4_blocked
    torch.manual_seed(52)
    q=torch.randn(1,129,2,32,device='cuda',requires_grad=True)
    k=torch.randn_like(q,requires_grad=True)
    v=torch.randn(1,129,2,32,device='cuda',requires_grad=True)
    tau=torch.tensor([.1,.8],device='cuda',requires_grad=True)
    exact=dism_v4_blocked(q,k,v,tau,key_mode=mode,precision='fp32')
    approx=dism_v4_blocked(q,k,v,tau,key_mode=mode,precision='bf16')
    g=torch.randn_like(v)
    ge=torch.autograd.grad((exact*g).sum(),(q,k,v,tau),allow_unused=True)
    ga=torch.autograd.grad((approx*g).sum(),(q,k,v,tau),allow_unused=True)
    for a,e in [(approx,exact),*zip(ga,ge)]:
        if e is None:assert a is None
        else:
            assert (a-e).norm()/e.norm().clamp_min(1e-8)<.02
            assert torch.nn.functional.cosine_similarity(a.flatten(),e.flatten(),dim=0)>.999


@pytest.mark.parametrize('mode',['soft','hard','jensen'])
def test_long_matching_chain_across_blocks(mode):
    from flash_dism_v4.blocked import dism_v4_blocked
    torch.manual_seed(13)
    q=torch.randn(1,193,3,3,device='cuda')*.1
    q[...,0]+=3
    q.requires_grad_()
    k=q.detach().clone().requires_grad_()
    v=torch.randn(1,193,3,4,device='cuda',requires_grad=True)
    tau=torch.tensor([-1e-6,1e-6,.03],device='cuda',requires_grad=True)
    actual=dism_v4_blocked(q,k,v,tau,key_mode=mode)
    expected=dism_v4_ref(q.double(),k.double(),v.double(),tau.double(),key_mode=mode)
    torch.testing.assert_close(actual.double(),expected,atol=3e-6,rtol=3e-5)
    g=torch.randn_like(v)
    ga=torch.autograd.grad((actual*g).sum(),(q,k,v,tau),allow_unused=True)
    ge=torch.autograd.grad((expected*g).sum(),(q,k,v,tau),allow_unused=True)
    for a,e in zip(ga,ge):
        if e is None:assert a is None
        else:torch.testing.assert_close(a,e,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('mode',['soft','jensen'])
def test_sharp_logits_use_gemm_without_protection(mode,monkeypatch):
    import flash_dism_v4.blocked as blocked
    torch.manual_seed(134)
    q=torch.randn(1,71,2,7,device='cuda',requires_grad=True)
    k=torch.randn_like(q)*10
    k[...,0]+=30
    k.requires_grad_()
    assert float(k.detach().log_softmax(-1).min()) < -40
    v=torch.randn(1,71,2,5,device='cuda',requires_grad=True)
    tau=torch.tensor([.1,.7],device='cuda',requires_grad=True)
    name='_soft_weights' if mode=='soft' else '_jensen_weights'
    original=getattr(blocked,name);calls=[]
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(blocked,name,counted)
    actual=blocked.dism_v4_blocked(q,k,v,tau,key_mode=mode)
    expected=dism_v4_ref(q.double(),k.double(),v.double(),tau.double(),key_mode=mode)
    torch.testing.assert_close(actual.double(),expected,atol=3e-6,rtol=3e-5)
    g=torch.randn_like(v)
    ga=torch.autograd.grad((actual*g).sum(),(q,k,v,tau))
    ge=torch.autograd.grad((expected*g).sum(),(q,k,v,tau))
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,atol=2e-5,rtol=2e-4)
    assert len(calls)==4  # two blocks each in forward and recomputation


def test_jensen_coefficient_bound_and_onehot_hard_endpoint():
    from flash_dism_v4.blocked import dism_v4_blocked
    torch.manual_seed(135)
    q=torch.randn(1,23,1,5,device='cuda')
    k=torch.randn_like(q)
    basis=torch.eye(23,device='cuda')[None,:,None,:]
    tau=torch.tensor([.6],device='cuda')
    exact=dism_v4_blocked(q,k,basis,tau,key_mode='soft')
    jensen=dism_v4_blocked(q,k,basis,tau,key_mode='jensen')
    assert bool((jensen<=exact+2e-6).all())
    assert bool((jensen.sum(-1)<=1+2e-6).all())
    k=(torch.nn.functional.one_hot(k.argmax(-1),5).float()*2-1)*1000
    hard=dism_v4_blocked(q,k,basis,tau,key_mode='hard')
    jensen=dism_v4_blocked(q,k,basis,tau,key_mode='jensen')
    torch.testing.assert_close(jensen,hard,atol=3e-6,rtol=3e-5)
