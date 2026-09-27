import torch
from dism_v2.visualize_lm_attention import hard_attention
from dism_v2.dism_ref import dism_recurrence


def test_dense_diagnostic_recurrence():
    rng = torch.Generator().manual_seed(123)
    qi, ki = [torch.randint(0, 7, (4, 65), generator=rng) for _ in range(2)]
    tau = torch.tensor([-3., 0., 1., 4.])
    w, p, fallback, z = hard_attention(qi, ki, tau)
    logm = tau[:, None, None].expand(4, 65, 65).masked_fill(qi[:, :, None]!=ki[:, None, :], -torch.inf)
    oracle = dism_recurrence(logm[None])[0]
    torch.testing.assert_close(w, oracle)
    torch.testing.assert_close(p.sum(-1)+fallback, torch.ones_like(fallback))
    assert (p.triu(1)==0).all()
    assert torch.isfinite(z).all()


def test_unmatched_fallback():
    _, p, fallback, z = hard_attention(torch.zeros(2, 8).long(), torch.ones(2, 8).long(), torch.ones(2))
    assert not p.any()
    assert (fallback==1).all()
    assert (z==0).all()
