"""Diagnostic hard-route census: records the dispatch decision, changes nothing.

CPU-only. The census exists so training can log how many hard backwards took the
prepared event kernel and how many fell back to the quadratic recompute; these
tests pin the recording contract that base_train drains.
"""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from flash_dism.hard_training import HardPlanTicket, take_route_log


def ready_ticket(policy='ready'):
    """A ticket whose plan is fully prepared, so the event kernel is used."""
    def event_backward(*xs):
        gradients = {name: torch.ones_like(t) for name, t in
                     zip(('sq_vec', 'sk_vec', 'v'), xs[:3])}
        gradients['rtau'] = torch.ones(1)
        return gradients

    future = Future()
    future.set_result(SimpleNamespace(
        h2d_events=(None, SimpleNamespace(query=lambda: True)),
        kernel=SimpleNamespace(backward=event_backward)))
    return HardPlanTicket(future, None, policy)


def pending_ticket(policy='ready'):
    """A ticket whose plan has not landed yet, so the quadratic fallback runs."""
    return HardPlanTicket(Future(), None, policy)


def test_records_both_routes_in_replay_order():
    x = torch.ones(1, 8, 1, 32)
    assert take_route_log() == []
    pending_ticket().backward(x, x, x, x, fallback=lambda: 'quadratic')
    ready_ticket().backward(x, x, x, x, fallback=lambda: 'quadratic')
    assert take_route_log() == ['q', 'e']
    assert take_route_log() == []            # draining clears


def test_forced_policy_is_recorded_without_a_decision():
    x = torch.ones(1, 8, 1, 32)
    take_route_log()
    event = ready_ticket('event').backward(x, x, x, x, fallback=lambda: 'x')
    assert 'sq_vec' in event and 'rtau' in event
    assert pending_ticket('quadratic').backward(x, x, x, x, fallback=lambda: 'x') == 'x'
    assert take_route_log() == ['e', 'q']


def test_census_is_bounded_and_the_route_decision_does_not_read_it():
    x = torch.ones(1, 8, 1, 32)
    original = HardPlanTicket._ROUTE_LOG_LIMIT
    HardPlanTicket._ROUTE_LOG_LIMIT = 3
    try:
        take_route_log()
        for _ in range(5):
            pending_ticket().backward(x, x, x, x, fallback=lambda: 'quadratic')
        assert take_route_log() == ['q', 'q', 'q']
        # Routing is unaffected by a saturated census.
        ticket = ready_ticket()
        ticket.backward(x, x, x, x, fallback=lambda: 'quadratic')
        assert ticket.route == 'event'
        assert take_route_log() == ['e']
    finally:
        HardPlanTicket._ROUTE_LOG_LIMIT = original


def test_missing_fallback_still_raises_rather_than_recording():
    x = torch.ones(1, 8, 1, 32)
    take_route_log()
    with pytest.raises(RuntimeError, match='requires a fallback'):
        pending_ticket().backward(x, x, x, x)
    assert take_route_log() == []
