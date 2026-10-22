"""TMA dQ+dKV fusion: unchanged oracle gates, partial vocab/strides and replay."""
import pytest
import torch
from flash_dism.emb_kernel import emb_fwd_wrapper
from flash_dism.kernels.interp_bwd_split import emb_bwd_split
from gradient_acceptance import assert_gradient

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or
    torch.cuda.get_device_capability()[0] < 9, reason='TMA requires SM90+')


def inputs(d=64, vocab=37, strided=False, beta=.7, tau_value=.5, direction_true=True):
    torch.manual_seed(823)
    b,n,h = 2,256,3
    def rand(shape):
        return (torch.randn(shape,device='cuda')*.25).bfloat16()
    q,k = [rand((b,n*2,h,d))[:,::2] if strided else rand((b,n,h,d)) for _ in range(2)]
    eq,ek = [rand((h,vocab*2,d))[:,::2] if strided else rand((h,vocab,d)) for _ in range(2)]
    tau = None if tau_value is None else torch.full((h,),tau_value,device='cuda')
    saved = emb_fwd_wrapper(q,k,eq,ek,beta,tau)
    doq,dok = [rand((b,n,h,d)) for _ in range(2)]
    if direction_true:
        dok.zero_()
    dlq,dlk = [torch.randn(b,h,n,device='cuda') for _ in range(2)]
    return q,k,eq,ek,*saved[:4],doq,dok,dlq,dlk,beta,tau


@pytest.mark.parametrize('case', [(32,37,False,1.,None), (64,512,False,1.,.5),
                                  (64,37,True,.7,.5), (32,512,True,1.7,2.)])
@pytest.mark.parametrize('direction_true', [False,True])
@pytest.mark.parametrize('autotune', [False,True])
def test_oracle(case,direction_true,autotune):
    args = inputs(*case,direction_true)
    baseline = emb_bwd_split(*args,direction_true=direction_true)
    actual = emb_bwd_split(*args,direction_true=direction_true,fuse_dq_tma=True,
                          autotune=autotune,dkv_m=64,dkv_v=64,splits=3,num_stages=2)
    q,k,eq,ek = [t.double().detach().requires_grad_() for t in args[:4]]
    doq,dok,dlq,dlk,beta = args[8:13]
    sq = torch.einsum('bnhd,hvd->bhnv',q,eq)*beta
    sk = torch.einsum('bnhd,hvd->bhnv',k,ek)*beta
    oq = torch.einsum('bhnv,hvd->bnhd',sk.softmax(-1),eq)
    ok = torch.einsum('bhnv,hvd->bnhd',sq.softmax(-1),ek)
    loss = (oq*doq.double()).sum()+(ok*dok.double()).sum()
    loss += (sk.logsumexp(-1)*dlq.double()).sum()+(sq.logsumexp(-1)*dlk.double()).sum()
    expected = torch.autograd.grad(loss,(q,k,eq,ek))
    for name,a,b,e in zip(('q','k','eq','ek'),actual,baseline,expected):
        assert_gradient(a,e,name)
        relative = (a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-12)
        assert relative < 1e-4, (name,relative.item())


def test_graph_replay_rezeros_token_and_vocab_gradients():
    args = inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            emb_bwd_split(*args,direction_true=True,fuse_dq_tma=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = emb_bwd_split(*args,direction_true=True,fuse_dq_tma=True)
    for factor in (.5,2.,0.):
        for i in (8,9,10,11):
            args[i].mul_(factor)
        graph.replay()
        baseline = emb_bwd_split(*args,direction_true=True)
        for a,b in zip(result,baseline):
            assert (a-b).norm()/b.norm().clamp_min(1e-12) < 1e-4
            if factor==0:
                assert torch.count_nonzero(a).item()==0
