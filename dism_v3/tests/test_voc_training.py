"""Optimization smoke, not a substitute for the strict derivative tests."""
import pytest
import torch
import torch.nn.functional as F

from flash_dism.voc import voc_dism
from test_backward_summary import pytestmark


@pytest.mark.parametrize('n', [256,512])
@pytest.mark.parametrize('probability', [0., .5, 1.])
def test_voc_optimizer_steps(n, probability):
    torch.manual_seed(902)
    b, h = 1, 2
    q, k = [torch.nn.Parameter(torch.randn(b, n, h, 64, device='cuda') * .2)
            for _ in range(2)]
    sq, sk = [torch.nn.Parameter(torch.randn(b, n, h, 32, device='cuda') * .5)
              for _ in range(2)]
    v = torch.nn.Parameter(torch.randn_like(q))
    eq, ek = [torch.nn.Parameter(torch.randn(h, 512, 64, device='cuda') * .2)
              for _ in range(2)]
    tau = torch.nn.Parameter(torch.full((h,), 2., device='cuda'))
    parameters = [q, k, sq, sk, v, eq, ek, tau]
    initial_parameters = [x.detach().clone() for x in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=.003, weight_decay=0.)
    hard = torch.rand(b, h, n, device='cuda') < probability
    direction = torch.tensor([[False, True]], device='cuda')

    def forward():
        # FP32 owners, caller-owned activation, then BF16 kernel operands.
        return voc_dism(q.bfloat16(), k.bfloat16(), F.silu(sq).bfloat16(),
                        F.silu(sk).bfloat16(), v.bfloat16(), eq, ek, tau,
                        hard=hard, direction=direction).float()

    with torch.no_grad():
        target = .5 * forward()
    losses = []
    # Mixed-mode argmax labels can change during optimization, so individual
    # steps need not descend. Check a short fit, not per-step monotonicity.
    for _ in range(64):
        optimizer.zero_grad(set_to_none=True)
        loss = (forward() - target).square().mean()
        assert torch.isfinite(loss)
        losses.append(loss.item())
        loss.backward()
        for parameter in parameters:
            assert parameter.grad is not None
            assert parameter.grad.dtype == torch.float32
            assert torch.isfinite(parameter.grad).all()
        optimizer.step()
    with torch.no_grad():
        final_loss = (forward() - target).square().mean().item()
    print(f'n={n} hard_prob={probability} initial={losses[0]:.8g} final={final_loss:.8g}')
    assert final_loss < losses[0], (losses, final_loss)
    assert not torch.equal(v, initial_parameters[4])
    if probability == 1.:
        # Hard labels are nondifferentiable; readout/tau still train.
        for index in (0, 1, 5, 6):
            torch.testing.assert_close(parameters[index], initial_parameters[index],
                                       atol=0, rtol=0)
