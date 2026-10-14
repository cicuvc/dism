"""Default compact/ready integration and explicit classic/event control."""
import pytest
import torch

from flash_dism import dism_core
import flash_dism.hard_training as ht
from flash_dism.kernels.hard_backward_fused import CompactFusedHardPrefillBackward
from test_hard_training import inputs


def test_default_runtime_is_compact_and_ready(monkeypatch):
    ht.shutdown_hard_planners()
    monkeypatch.delenv('DISM_HARD_BACKWARD', raising=False)
    monkeypatch.delenv('DISM_PLANNER_COMPACT', raising=False)
    planner = ht.default_hard_planner('cuda')
    try:
        assert planner.backward_policy == 'ready'
        assert planner.compact is True
        xs, _ = inputs(mode='hard')
        ticket = planner.submit(xs[7], xs[8], xs[10], xs[11])
        prepared = ticket.future.result(timeout=30)
        assert isinstance(prepared.kernel, CompactFusedHardPrefillBackward)
        assert prepared.kernel.single_count + prepared.kernel.multi_count == prepared.kernel.streams
        prepared.h2d_events[1].synchronize()
        ticket.backward(xs[2], xs[3], xs[4], torch.ones_like(xs[4]))
        assert ticket.route == 'event'
    finally:
        ht.shutdown_hard_planners()


@pytest.mark.parametrize('packed', [False, True])
def test_compact_and_classic_event_gradients(packed):
    xs, layout = inputs(packed, 'mixed', 'repeat')
    upstream = torch.randn_like(xs[4])*.2
    results = []
    for compact in (False, True):
        with ht.AsyncHardPlanner(workers=2, compact=compact, backward_policy='event') as planner:
            out = dism_core(*xs, layout=layout, hard_planner=planner)
            results.append(torch.autograd.grad(out, [xs[i] for i in (2, 3, 4, 11)], upstream))
    for actual, expected in zip(results[0], results[1]):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
