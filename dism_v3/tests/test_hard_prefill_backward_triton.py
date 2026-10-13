"""Opt-in event backward; no changes to production gradient tolerances."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_hard_prefill_backward_triton import check_case
from check_hard_prefill_training import fixture
from hard_prefill_backward_triton import HardPrefillBackward, pack_program
from hard_prefill_grad_reference import Program


@pytest.mark.parametrize('precision', ['tf32x3', 'bf16'])
@pytest.mark.parametrize('pattern', ['random', 'repeat', 'mismatch', 'periodic'])
@pytest.mark.parametrize('n,tau', [(1, 0.), (17, -2.), (65, .5), (256, 2.)])
def test_event_backward_vectors_and_tau_diagnostics(precision, pattern, n, tau):
    torch.set_num_threads(1)
    result = check_case(pattern, n, tau, precision=precision)
    if precision == 'tf32x3':
        assert not result['tau_material_sign_flip'], result
    if pattern == 'mismatch':
        assert result['tau_actual'] == 0
        assert not result['production_acceptance_failures']


@pytest.mark.parametrize('r,dv,c', [(16, 32, 16), (16, 64, 32), (32, 32, 64)])
@pytest.mark.parametrize('precision', ['tf32x3', 'bf16'])
def test_dimensions_chunks_fp32_input(r, dv, c, precision):
    torch.set_num_threads(1)
    result = check_case('repeat', 65, .5, r=r, dv=dv, chunk_size=c,
                        precision=precision, dtype=torch.float32)
    assert not result['tau_material_sign_flip'], result


def test_chunk_reset_and_underflow_convention():
    # Reset halfway through an otherwise full chunk must start a new chunk.
    p = Program(np.array([0, 5]), np.array([0, -1, 1, -2, -3]),
                np.array([0., 1., 0., 1., 1.]), np.ones(5), np.zeros(5),
                np.zeros(3), np.zeros(3))
    packed = pack_program(p, 16)
    np.testing.assert_array_equal(packed['offsets'], [0, 2])
    np.testing.assert_array_equal(packed['lengths'], [2, 3])
    np.testing.assert_array_equal(packed['resets'], [1, 1])
    assert np.isfinite(packed['prefixes']).all()
    # Exercise the reset in BOTH GPU scans, not just the CPU packing helper.
    plan = HardPrefillBackward([0, 0, 0], [1, 1, 1], 0.)
    plan.program, plan.packed, plan.streams = p, packed, 1
    plan.data = {k: torch.as_tensor(v, device='cuda') for k, v in packed.items()}
    rng = np.random.default_rng(123)
    xs = [torch.tensor(rng.normal(size=(3, width)), device='cuda').bfloat16()
          for width in (16, 16, 32, 32)]
    actual = plan.backward(*xs)
    _, expected, _ = p.evaluate(*[x.float().cpu().double().numpy() for x in xs])
    for name, ref in zip(('sq_vec', 'sk_vec', 'v', 'rtau'), expected):
        torch.testing.assert_close(actual[name].cpu().double(), torch.as_tensor(ref), rtol=2e-5, atol=2e-6)


def test_independent_documents_shared_tau_and_graph_replay():
    q, k, *xs = fixture('repeat', 34, r=16, dv=32)
    xs = [torch.tensor(x, device='cuda').bfloat16() for x in xs]
    plans = [HardPrefillBackward(q[a:b], k[a:b], .5) for a, b in ((0, 17), (17, 34))]
    slices = [[x[a:b].contiguous() for x in xs] for a, b in ((0, 17), (17, 34))]
    expected = [plan.backward(*x) for plan, x in zip(plans, slices)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = [plan.backward(*x) for plan, x in zip(plans, slices)]
        tau = actual[0]['rtau'] + actual[1]['rtau']
    for _ in range(2):
        graph.replay()
        for a, b in zip(actual, expected):
            for name in ('sq_vec', 'sk_vec', 'v', 'rtau'):
                torch.testing.assert_close(a[name], b[name], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(tau, expected[0]['rtau']+expected[1]['rtau'], rtol=2e-5, atol=2e-6)


def test_reject_bad_shapes_and_layout():
    q, k, *xs = fixture('repeat', 17, r=16, dv=32)
    plan = HardPrefillBackward(q, k, .5)
    xs = [torch.tensor(x, device='cuda').bfloat16() for x in xs]
    with pytest.raises(ValueError, match='expected'):
        plan.backward(*xs[:3], xs[3][:-1])
    with pytest.raises(ValueError, match='contiguous'):
        plan.backward(xs[0].cpu(), *xs[1:])
    with pytest.raises(ValueError, match='contiguous'):
        plan.backward(xs[0].T.contiguous().T, *xs[1:])
