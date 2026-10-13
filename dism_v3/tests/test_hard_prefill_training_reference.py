"""Mathematical hard-training prototype; deliberately not production dispatch."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_hard_prefill_training import check_case, check_gpu_case, fixture
from hard_prefill_grad_reference import SymbolicPlanner, dense_oracle


@pytest.mark.parametrize('pattern', ['random', 'repeat', 'mismatch', 'periodic'])
@pytest.mark.parametrize('tau', [-2., 0., 1e-8, .5, 2., 16.])
def test_event_backward(pattern, tau):
    torch.set_num_threads(1)
    check_case(pattern, 65, tau)


def test_document_isolation_shared_tau_and_direction():
    # Two documents from the same hard training row: separate plans, one tau.
    torch.set_num_threads(1)
    q, k, sq, sk, v, do = fixture('repeat', 34)
    q[17:] = k[17:] = 0  # Same labels still MUST NOT match across the boundary.
    total_tau, reference_tau = 0., 0.
    outputs = []
    for start, stop in ((0, 17), (17, 34)):
        parts = [x[start:stop] for x in (q, k, sq, sk, v, do)]
        a, b, qs, ks, val, grad = parts
        out, grads, _ = SymbolicPlanner(a, b).compile(.5).evaluate(qs, ks, val, grad)
        for direction in (False, True):
            ref, rg = dense_oracle(a, b, qs, ks, val, .5, grad, direction)
            np.testing.assert_allclose(out, ref, rtol=2e-8, atol=2e-10)
            for actual, expected in zip(grads, rg):
                np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-10)
        total_tau += grads[3]
        reference_tau += rg[3]
        outputs.append(out)
    np.testing.assert_allclose(total_tau, reference_tau, rtol=2e-8, atol=2e-10)
    joined, _, _ = SymbolicPlanner(q, k).compile(.5).evaluate(sq, sk, v, do)
    assert np.linalg.norm(joined-np.concatenate(outputs)) > 1e-4


def test_vector_directional_finite_difference():
    from flash_dism.inference import HardPrefillPlan
    q, k, sq, sk, v, do = fixture('periodic', 33)
    plan = HardPrefillPlan(q, k, .7)
    _, grads, _ = SymbolicPlanner(q, k).compile(.7).evaluate(sq, sk, v, do)
    rng = np.random.default_rng(931)
    values = [sq, sk, v]
    step = 1e-5
    for index in range(3):
        delta = rng.normal(size=values[index].shape)
        plus, minus = list(values), list(values)
        plus[index] = values[index] + step*delta
        minus[index] = values[index] - step*delta
        fd = np.sum((plan.execute(*plus, dtype=np.float64)-plan.execute(*minus, dtype=np.float64))*do)/(2*step)
        np.testing.assert_allclose(np.sum(grads[index]*delta), fd, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize('pattern', ['random', 'repeat', 'mismatch', 'periodic'])
def test_existing_gpu_prefill_dsq(pattern):
    torch.set_num_threads(1)
    result = check_gpu_case(pattern, .5)
    assert not result['native_core_acceptance_failures']


def test_tau_precision_discrepancies_remain_visible():
    torch.set_num_threads(1)
    threshold = check_case('periodic', 256, .5)
    assert threshold['exact_fp64']['tau']['max_abs'] < 1e-10
    assert isinstance(threshold['maintained_oracle_strict_failures'], dict)
    cancellation = check_case('periodic', 256, 2.)
    assert isinstance(cancellation['fp32_strict_failures'], dict)
    assert not cancellation['fp32_production_failures']
