import torch

from nanochat.models.dism.model import (
    sample_hard_flags,
    sample_hard_flags_selectors,
)


def test_selector_fusion_preserves_hard_rng_and_tau_gradient():
    n, heads = 8, 3
    counter = torch.tensor(17, device="cuda", dtype=torch.int64)
    probability = torch.tensor(0.63, device="cuda")
    log_tau = torch.tensor([-2.0, 0.0, 2.0], device="cuda", requires_grad=True)

    expected = sample_hard_flags(
        counter, probability, n, heads, 123
    ).squeeze(0).T.contiguous()
    hard, direction, tau = sample_hard_flags_selectors(
        counter, probability, log_tau, n, heads, 123
    )

    assert torch.equal(hard, expected)
    assert direction.shape == (1, heads) and direction.all()
    torch.testing.assert_close(tau, torch.nn.functional.softplus(log_tau))
    tau.sum().backward()
    torch.testing.assert_close(log_tau.grad, torch.sigmoid(log_tau.detach()))
