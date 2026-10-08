"""Compare BNHD interpolation with the unchanged v2 BHND implementation."""
import pytest
import torch

from flash_dism.emb_kernel import emb_fwd_wrapper, emb_bwd_wrapper
from embedding_bhnd_reference import emb_fwd_wrapper as old_forward
from embedding_bhnd_reference import emb_bwd_wrapper as old_backward


@pytest.mark.parametrize('d', [32, 64])
@pytest.mark.parametrize('vocab', [37, 512])
@pytest.mark.parametrize('strided', [False, True])
def test_embedding_bnhd(d, vocab, strided):
    torch.manual_seed(271)
    b, n, h = 2, 256, 3
    def rand(shape):
        return torch.randn(shape, device='cuda', dtype=torch.bfloat16) * .25
    if strided:
        q = rand((b, h, n, d)).transpose(1, 2)
        k = rand((b, n*2, h, d))[:, ::2]
    else:
        q, k = rand((b,n,h,d)), rand((b,n,h,d))
    eq, ek = rand((h,vocab,d)), rand((h,vocab,d))
    old_q, old_k = (x.transpose(1,2).contiguous() for x in (q,k))
    actual = emb_fwd_wrapper(q,k,eq,ek)
    expected = old_forward(old_q,old_k,eq,ek)
    for i, (a,e) in enumerate(zip(actual,expected)):
        if i < 2:
            e = e.transpose(1,2)
            assert a.is_contiguous() and a.shape == q.shape
        torch.testing.assert_close(a,e,atol=0,rtol=0)
    doq = rand((b,h,n,d)).transpose(1,2)
    dok = rand((b,n*2,h,d))[:,::2]
    dlq, dlk = (torch.randn((b,h,n), device='cuda') for _ in range(2))
    grads = emb_bwd_wrapper(q,k,eq,ek,*actual[:4],doq,dok,dlq,dlk)
    old_grads = old_backward(old_q,old_k,eq,ek,*expected[:4],
                             doq.transpose(1,2).contiguous(),
                             dok.transpose(1,2).contiguous(),dlq,dlk)
    for i,(a,e) in enumerate(zip(grads,old_grads)):
        if i < 2:
            e = e.transpose(1,2)
            assert a.is_contiguous() and a.shape == q.shape
        torch.testing.assert_close(a,e,atol=0,rtol=0)
