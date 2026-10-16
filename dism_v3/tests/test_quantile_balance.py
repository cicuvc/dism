"""Independent probability oracle, QB state, and native additive-bias gradients."""
import copy
import pytest
import torch
from flash_dism.quantile_balance import QuantileBalance, PricedInterpolation, fit_prices, validate_qb


def explicit(q, k, eq, ek, aq, ak, beta):
    zq = beta*torch.einsum('bnhd,hvd->bhnv', q, eq)-aq[:, None]
    zk = beta*torch.einsum('bnhd,hvd->bhnv', k, ek)-ak[:, None]
    pk = zk.softmax(-1)
    return (torch.einsum('bhnv,hvd->bnhd', pk, eq), zq.logsumexp(-1).transpose(1, 2),
            (pk*aq[:, None]).sum(-1).transpose(1, 2), zq, zk)


@pytest.mark.parametrize('strengths', [(0., 1.), (1., 0.), (.3, .8)])
def test_interpolation_lower_bound_and_gradients(strengths):
    torch.manual_seed(12)
    q, k = [torch.randn(2, 9, 2, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    eq, ek = [torch.randn(2, 7, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    aq, ak = [torch.randn(2, 7, dtype=torch.float64)*s for s in strengths]
    actual = PricedInterpolation.apply(q, k, eq, ek, aq, ak, 1.7)
    ref = explicit(q, k, eq, ek, aq, ak, 1.7)
    for a, r in zip(actual[:3], ref[:3]):
        torch.testing.assert_close(a, r, atol=1e-12, rtol=1e-12)
    g = [torch.randn_like(x) for x in actual[:3]]
    ga = torch.autograd.grad(actual[:3], (q,k,eq,ek), g)
    gr = torch.autograd.grad(ref[:3], (q,k,eq,ek), g, retain_graph=True)
    for a, r in zip(ga, gr):
        torch.testing.assert_close(a, r, atol=1e-11, rtol=1e-11)
    lower = 1.7*torch.einsum('bihd,bjhd->bhij',q,actual[0])-actual[1].transpose(1,2)[...,None]-actual[2].transpose(1,2)[:,:,None,:]
    cross_entropy = torch.einsum('bhic,bhjc->bhij',ref[3].log_softmax(-1),ref[4].softmax(-1))
    match = torch.einsum('bhic,bhjc->bhij',ref[3].softmax(-1),ref[4].softmax(-1)).log()
    torch.testing.assert_close(lower, cross_entropy, atol=1e-12, rtol=1e-12)
    assert (lower <= match+1e-12).all()
    assert torch.equal(actual[3], ref[3].argmax(-1))
    assert torch.equal(actual[4], ref[4].argmax(-1))


def test_binary_quantile_solution_and_fractional_capacity():
    scores = torch.tensor([[[6.,0.],[5.,0.],[4.,0.],[1.,0.]]],dtype=torch.float64)
    price = fit_prices(scores, torch.zeros(1,2,dtype=torch.float64), 64)
    assert torch.equal((scores-price[:,None]).argmax(-1),torch.tensor([[0,0,1,1]]))
    # N<V and nonintegral capacity remain finite and cannot increase the dual.
    scores = torch.randn(2,3,8,dtype=torch.float64)
    p = fit_prices(scores,torch.zeros(2,8,dtype=torch.float64),16)
    assert torch.isfinite(p).all()
    assert ((scores-p[:,None]).amax(-1).mean(-1) <= scores.amax(-1).mean(-1)+1e-12).all()


@pytest.mark.parametrize('q,k', [(-1.,0.),(0.,1.1),(float('nan'),0.)])
def test_strength_validation(q,k):
    with pytest.raises(ValueError):
        validate_qb(q,k,4,16,2048,.1)


def test_price_state_replay_no_rng_or_parameters():
    torch.manual_seed(8)
    module = QuantileBalance(2,8,q_strength=.5,k_strength=.7,samples=32)
    q,k = [torch.randn(1,32,2,4) for _ in range(2)]
    eq,ek = [torch.randn(2,8,4) for _ in range(2)]
    state = torch.get_rng_state().clone()
    module._update(q,k,eq,ek,1.)
    assert torch.equal(torch.get_rng_state(),state)
    clone = QuantileBalance(2,8,q_strength=.5,k_strength=.7,samples=32)
    clone.load_state_dict(copy.deepcopy(module.state_dict()),strict=True)
    module._update(q,k,eq,ek,1.)
    clone._update(q,k,eq,ek,1.)
    for a,b in zip(module.buffers(),clone.buffers()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert not list(module.parameters())
    with pytest.raises(RuntimeError):
        clone.load_state_dict({},strict=True)


@pytest.mark.parametrize('packed', [False,True])
@pytest.mark.parametrize('mixed', [False,True])
@pytest.mark.parametrize('config', [(16,32,32),(32,64,64)])
@pytest.mark.parametrize('strict', [False, pytest.param(True, marks=pytest.mark.strict_gradient)])
def test_cuda_bias_core_oracle(packed,mixed,config,strict):
    if not torch.cuda.is_available(): pytest.skip('CUDA required')
    from flash_dism.compiler import core_forward
    from flash_dism.reference.dism_v3_ref import dism_ref
    from flash_dism.varlen import VarlenLayout
    torch.manual_seed(99)
    n=512 if packed else 256
    r,d,dv=config
    q,k,sq,sk,v=[(torch.randn(1,n,2,d,device='cuda')*.15).bfloat16().requires_grad_()
                 for d in (d,d,r,r,dv)]
    lq=(torch.randn(1,n,2,device='cuda')*.1+2).requires_grad_()
    m=(torch.randn(1,n,2,device='cuda')*.2).requires_grad_()
    tau=torch.tensor([.5,.7],device='cuda',requires_grad=True)
    iq,ik=[torch.randint(8,(1,2,n),device='cuda',dtype=torch.int32) for _ in range(2)]
    direction=torch.ones((1,2),device='cuda',dtype=torch.bool)
    hard=torch.tensor([[False,mixed],[mixed,False]] if packed else [[False,mixed]],device='cuda')
    table=VarlenLayout.from_cu_seqlens(torch.tensor([0,256,512],dtype=torch.int32),n).table if packed else None
    out=core_forward((q,k,sq,sk,v,lq-tau.detach(),m,iq,ik,direction,hard,tau),table=table,absorbed=True,qb_key_bias=True)
    args=(q,k,sq,sk,v,lq,m,tau)
    # Augmented vectors encode both biases for the unchanged dense oracle.
    ref=[]
    for row,start in enumerate(range(0,n,256)):
        sl=slice(start,start+256)
        qa=torch.cat((q[:,sl].float(),torch.ones_like(q[:,sl,:,:1]).float()),-1)
        ka=torch.cat((k[:,sl].float(),-m[:,sl].unsqueeze(-1)),-1)
        ref.append(dism_ref(qa,ka,sq[:,sl],sk[:,sl],lq[:,sl],torch.zeros_like(lq[:,sl]),
                   iq[:,:,sl],ik[:,:,sl],direction,hard[row:row+1,:,None].expand(1,2,256),v[:,sl],tau))
    expected=torch.cat(ref,1)
    torch.testing.assert_close(out.float(),expected,atol=2e-4,rtol=.02)
    do=torch.randn_like(out)
    actual_grad=torch.autograd.grad(out,args,do)
    expected_grad=torch.autograd.grad(expected,args,do.float())
    from gradient_acceptance import assert_gradient
    names=('q_vec','k_vec','sq_vec','sk_vec','v','q_lse','k_lse','rtau')
    for name,a,b in zip(names,actual_grad,expected_grad):
        assert_gradient(a,b,name)
        if strict:
            rel=(a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-9)
            assert rel < .04, (name,a.shape,float(rel))
    # Both native reductions are tested against the oracle above. Their signed
    # totals need not coincide after BF16 key-gradient stores (cancellation).
    # Check the mathematical conservation identity on the FP32 oracle instead.
    assert actual_grad[5].norm()>0 and actual_grad[6].norm()>0
    torch.testing.assert_close(expected_grad[5].sum((0,1)), expected_grad[6].sum((0,1)),atol=1e-6,rtol=1e-5)


def _distributed_worker(rank, rendezvous):
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(32)
        all_q=torch.randn(2,16,2,4)
        eq=torch.randn(2,8,4)
        module=QuantileBalance(2,8,q_strength=.8,samples=16,iterations=8,update_rate=1.)
        module._update(all_q[rank:rank+1],all_q[rank:rank+1],eq,eq,1.)
        scores=all_q.reshape(32,2,4).transpose(0,1) @ eq.transpose(-1,-2)
        expected=fit_prices(scores,torch.zeros(2,8),8)
        torch.testing.assert_close(module.q_price,expected,atol=1e-6,rtol=1e-6)
        assert torch.count_nonzero(module.k_price)==0
    finally:
        dist.destroy_process_group()


def test_distributed_prices_fit_global_samples(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)


def test_price_update_excludes_padding_and_empty_batch():
    torch.manual_seed(81)
    a=QuantileBalance(2,8,q_strength=1.,k_strength=.5,samples=16,update_rate=1.)
    b=copy.deepcopy(a)
    q,k=[torch.randn(1,32,2,4) for _ in range(2)]
    eq,ek=[torch.randn(2,8,4) for _ in range(2)]
    mask=torch.arange(32)%2==0
    a._update(q,k,eq,ek,1.,mask)
    b._update(q[:,mask],k[:,mask],eq,ek,1.)
    for x,y in zip(a.buffers(),b.buffers()):
        torch.testing.assert_close(x,y,atol=0,rtol=0)
    saved=copy.deepcopy(a.state_dict())
    a._update(q,k,eq,ek,1.,torch.zeros(32,dtype=torch.bool))
    for name,value in a.state_dict().items():
        torch.testing.assert_close(value,saved[name],atol=0,rtol=0)
