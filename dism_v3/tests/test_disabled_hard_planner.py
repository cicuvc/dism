"""`DISM_HARD_PLANNER=0` must bypass planning entirely, not just discard it.

`DISM_HARD_BACKWARD=quadratic` still submits every ticket and pays the native
planner's CPU. The disabled planner hands out a future-less ticket instead, so
the fallback runs with no worker pool, no coordinator and no D2H/H2D, and the
observable backward result must be the same.
"""
import os

import pytest
import torch

from flash_dism import dism_core
from flash_dism.hard_training import (AsyncHardPlanner, DisabledHardPlanner,
                                      HardPlanTicket, default_hard_planner,
                                      shutdown_hard_planners, take_route_log)
from test_hard_training import inputs, oracle
from gradient_acceptance import assert_gradient


def test_disabled_planner_has_no_worker_pool_or_coordinator(monkeypatch):
    monkeypatch.setenv('DISM_HARD_PLANNER', '0')
    shutdown_hard_planners()
    try:
        planner = default_hard_planner('cuda')
        assert isinstance(planner, DisabledHardPlanner)
        assert not hasattr(planner, 'native') and not hasattr(planner, '_executor')
        ticket = planner.submit(torch.zeros((1, 2, 256), dtype=torch.int32, device='cuda'),
                                torch.zeros((1, 2, 256), dtype=torch.int32, device='cuda'),
                                torch.ones((1, 2), dtype=torch.bool, device='cuda'),
                                torch.ones(2, device='cuda'), layout=None)
        assert isinstance(ticket, HardPlanTicket) and ticket.future is None
    finally:
        shutdown_hard_planners()


@pytest.mark.parametrize('mode', ['hard', 'mixed'])
def test_disabled_planner_matches_the_quadratic_policy(monkeypatch, mode):
    """Same forward, same fallback backward, same gradients."""
    try:
        xs, layout = inputs(packed=False, mode=mode, pattern='random')
        torch.manual_seed(3)
        with AsyncHardPlanner(backward_policy='quadratic', workers=2) as reference_planner:
            reference = dism_core(*xs, layout=layout, hard_planner=reference_planner)
        do = (torch.randn_like(reference) * .2).contiguous()
        want = torch.autograd.grad(reference, [xs[2], xs[3], xs[4], xs[11]], do,
                                   allow_unused=True)

        monkeypatch.setenv('DISM_HARD_PLANNER', '0')
        shutdown_hard_planners()
        got_out = dism_core(*xs, layout=layout)
        got = torch.autograd.grad(got_out, [xs[2], xs[3], xs[4], xs[11]], do,
                                  allow_unused=True)

        torch.testing.assert_close(got_out, reference, atol=0, rtol=0)
        for name, a, b in zip(('sq_vec', 'sk_vec', 'v', 'rtau'), got, want):
            assert_gradient(a, b.double(), name)
        # Every dispatch decision took the fallback and nothing was planned.
        assert set(take_route_log()) == {'q'}
    finally:
        shutdown_hard_planners()
