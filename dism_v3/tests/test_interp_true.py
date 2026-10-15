import pytest
import torch
from flash_dism.emb_kernel import emb_fwd_wrapper
from flash_dism.kernels.interp_true import emb_fwd_true
from flash_dism.kernels.interp_bwd_split import emb_bwd_split
from gradient_acceptance import assert_gradient


def test_true_ties_and_aliased_qk():
    torch.manual_seed(705)
    q = torch.zeros((1,256,2,32),device='cuda',dtype=torch.bfloat16)
    eq,ek = [(torch.randn(2,37,32,device='cuda')*.2).bfloat16() for _ in range(2)]
    old = emb_fwd_wrapper(q,q,eq,ek)
    new = emb_fwd_true(q,q,eq,ek)
    assert torch.count_nonzero(new[6]) == 0
    assert torch.count_nonzero(new[7]) == 0
    doq = torch.randn_like(q)
    dlq = torch.zeros((1,2,256),device='cuda')
    dlk = torch.randn_like(dlq)
    got = emb_bwd_split(q,q,eq,ek,*new[:4],doq,q,dlq,dlk,
                        direction_true=True,autotune=False)
    expected = emb_bwd_split(q,q,eq,ek,*old[:4],doq,q,dlq,dlk,autotune=False)
    for a,e in zip(got,expected):
        torch.testing.assert_close(a,e,atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('d,vocab,strided', [(32,37,False), (64,512,False), (64,37,True)])
@pytest.mark.parametrize('beta', [1., .7, 1.75])
@pytest.mark.parametrize('autotune', [False, True], ids=['fixed','tuned'])
def test_true_interpolation(d, vocab, strided, beta,autotune):
    torch.manual_seed(701)
    b, n, h = 2, 256, 2
    def rand(shape):
        return (torch.randn(shape, device='cuda')*.25).bfloat16()
    q, k = [rand((b,n*2,h,d))[:,::2] if strided else rand((b,n,h,d)) for _ in range(2)]
    eq, ek = [rand((h,vocab,d)) for _ in range(2)]
    tau = torch.tensor([.5, 1.], device='cuda')
    old = emb_fwd_wrapper(q,k,eq,ek,beta,tau)
    new = emb_fwd_true(q,k,eq,ek,beta,tau)
    assert new[1].numel() == 0
    for i in (0,2,3,4,5,6,7):
        if i in (0,6,7):
            torch.testing.assert_close(new[i],old[i],atol=0,rtol=0)
        else:
            torch.testing.assert_close(new[i],old[i])
    doq, dok = rand((b,n,h,d)), torch.zeros((b,n,h,d),device='cuda',dtype=torch.bfloat16)
    dlq = torch.zeros((b,h,n),device='cuda')
    dlk = torch.randn((b,h,n),device='cuda')
    args = (q,k,eq,ek,*new[:4],doq,dok,dlq,dlk,beta,tau)
    got = emb_bwd_split(*args,direction_true=True,autotune=autotune)
    baseline = emb_bwd_split(q,k,eq,ek,*old[:4],doq,dok,dlq,dlk,beta,tau,autotune=autotune)
    xq,xk,qe,ke = [x.double().detach().requires_grad_() for x in (q,k,eq,ek)]
    sq = torch.einsum('bnhd,hvd->bhnv',xq,qe)*beta
    sk = torch.einsum('bnhd,hvd->bhnv',xk,ke)*beta
    output = torch.einsum('bhnv,hvd->bnhd',sk.softmax(-1),qe)
    loss = (output*doq.double()).sum() + (sq.logsumexp(-1)*dlk.double()).sum()
    oracle = torch.autograd.grad(loss,(xq,xk,qe,ke))
    for name,a,e,r in zip(('q','k','eq','ek'),got,baseline,oracle):
        assert_gradient(a,r,name)
        assert (a-e).norm()/e.norm().clamp_min(1e-12) < 1e-4


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('mode', ['soft','mixed','hard'])
@pytest.mark.parametrize('r,d,dv', [(16,32,32),(32,64,64),(16,64,32),(32,32,64)])
def test_true_operator(packed,mode,r,d,dv):
    from flash_dism.compiler import voc_forward
    torch.manual_seed(702)
    inputs = [(torch.randn(1,256,2,c,device='cuda')*.2).bfloat16().requires_grad_()
              for c in (d,d,r,r,dv)]
    inputs += [(torch.randn(2,64,d,device='cuda')*.2).requires_grad_() for _ in range(2)]
    inputs += [torch.ones(2,device='cuda',requires_grad=True)]
    hard = torch.tensor([[mode != 'soft',mode == 'hard']],device='cuda')
    direction = torch.ones(1,2,device='cuda',dtype=torch.bool)
    cu = torch.tensor([0,0,256],device='cuda',dtype=torch.int32) if packed else None
    expected = voc_forward(*inputs,1.75,direction,hard,cu)[0]
    result = voc_forward(*inputs,1.75,direction,hard,cu,direction_true=True)
    actual = result[0]
    assert result[2].numel() == 0
    dout = torch.randn_like(actual)
    eg = torch.autograd.grad(expected,inputs,dout)
    ag = torch.autograd.grad(actual,inputs,dout)
    torch.testing.assert_close(actual,expected,atol=.002,rtol=.03)
    for a,e in zip(ag,eg):
        torch.testing.assert_close(a,e,atol=.001,rtol=.03)


@pytest.mark.parametrize('packed', [False, True])
def test_true_inductor(packed):
    from flash_dism.compiler import voc_forward
    from flash_dism.varlen import VarlenLayout
    torch.manual_seed(703)
    inputs = [(torch.randn(1,512,2,c,device='cuda')*.2).bfloat16().requires_grad_()
              for c in (64,64,32,32,64)]
    inputs += [(torch.randn(2,64,64,device='cuda')*.2).requires_grad_() for _ in range(2)]
    inputs += [torch.ones(2,device='cuda',requires_grad=True)]
    direction = torch.ones(1,2,device='cuda',dtype=torch.bool)
    hard = torch.tensor([[True,False]],device='cuda')
    cu = torch.tensor([0,256,512],device='cuda',dtype=torch.int32) if packed else None
    layout = VarlenLayout.from_cu_seqlens(cu,512).table.flatten() if packed else None
    def run(*args):
        return voc_forward(*args,.7,direction,hard,cu,layout,direction_true=True)[0]
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True), \
         torch._inductor.config.patch(compile_threads=1):
        compiled = torch.compile(run,dynamic=True,fullgraph=True)
        expected = run(*inputs)
        actual = compiled(*inputs)
        dout = torch.randn_like(actual)
        eg = torch.autograd.grad(expected,inputs,dout)
        ag = torch.autograd.grad(actual,inputs,dout)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        for a,e in zip(ag,eg):
            torch.testing.assert_close(a,e,atol=.001,rtol=.03)
        with torch._dynamo.config.patch(error_on_recompile=True):
            actual = compiled(*inputs)
            torch.autograd.grad(actual,inputs,dout)
