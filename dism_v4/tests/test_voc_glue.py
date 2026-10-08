import pytest
import torch
from flash_dism.kernels.voc_glue import (prepare_embedding, select_operands, split_interpolation_gradients,
                                        merge_token_gradients, cast_vocabulary_gradients)


@pytest.mark.parametrize('r,d', [(16,32), (32,32), (16,64), (32,64)])
@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('tokens', [0, 256])
def test_glue_exact_rounding_and_layout(r, d, shared, tokens):
    torch.manual_seed(128)
    shape = (2, tokens, 3, d)
    head_shape = shape
    rand = lambda s, dtype: torch.randn(s, device='cuda', dtype=dtype)
    q, k = (rand(shape, torch.bfloat16) for _ in range(2))
    qfk, kfq = (rand(head_shape, torch.bfloat16) for _ in range(2))
    direction = torch.tensor([[True, False, True], [False, True, False]], device='cuda')
    choose = direction[:, None, :, None]
    a, b = select_operands(q, k, qfk, kfq, direction)
    torch.testing.assert_close(a, torch.where(choose, q, kfq), atol=0, rtol=0)
    torch.testing.assert_close(b, torch.where(choose, qfk, k), atol=0, rtol=0)
    da, db = rand(shape, torch.float32), rand(shape, torch.bfloat16)
    doq, dok = split_interpolation_gradients(da, db, direction)
    torch.testing.assert_close(doq, torch.where(choose, db, 0), atol=0, rtol=0)
    torch.testing.assert_close(dok, torch.where(choose, 0, da.bfloat16()), atol=0, rtol=0)
    iq, ik = (rand(head_shape, torch.float32) for _ in range(2))
    dsq = rand((2,tokens,3,r), torch.float32)
    dq, dk, sq = merge_token_gradients(da, db, iq, ik, dsq, direction)
    torch.testing.assert_close(dq, iq.bfloat16() + torch.where(choose, da.bfloat16(), 0), atol=0, rtol=0)
    torch.testing.assert_close(dk, ik.bfloat16() + torch.where(choose, 0, db), atol=0, rtol=0)
    torch.testing.assert_close(sq, dsq.bfloat16(), atol=0, rtol=0)
    gq, gk = (rand((3,37,d), torch.float32) for _ in range(2))
    eq, ek = (rand((37,d) if shared else (3,37,d), torch.float32) for _ in range(2))
    eq, ek = (t.transpose(-1,-2).contiguous().transpose(-1,-2) for t in (eq,ek))
    query, key, qe, ke = prepare_embedding(q, k, eq, ek)
    assert query is q and key is k
    torch.testing.assert_close(query, q, atol=0, rtol=0)
    torch.testing.assert_close(key, k, atol=0, rtol=0)
    for out, original in ((qe,eq), (ke,ek)):
        expected = original.bfloat16()
        if shared:
            expected = expected.unsqueeze(0).expand(3,-1,-1)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
    actual = cast_vocabulary_gradients(gq, gk, eq, ek)
    for out, grad in zip(actual, (gq,gk)):
        expected = grad.bfloat16().float()
        if shared:
            expected = expected.sum(0)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
