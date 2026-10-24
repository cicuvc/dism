from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from flash_dism.inference.build import load_prefill
from flash_dism.inference.training_planner import NativeTrainingPlanner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_hard_prefill_training import fixture
from hard_prefill_grad_reference import SymbolicPlanner, Program, assert_equivalent_programs
from hard_prefill_backward_triton import HardPrefillBackward, pack_program


@pytest.mark.parametrize('pattern', ['random', 'repeat', 'mismatch', 'periodic'])
@pytest.mark.parametrize('tau', [-2., -1e-8, 0., 1e-8, .5, 2., 16.])
@pytest.mark.parametrize('n', [1, 65, 513])
def test_native_derivative_program(pattern, tau, n):
    q, k, *_ = fixture(pattern, n)
    p = SymbolicPlanner(q, k).compile(tau)
    native = load_prefill().plan_training(q.tolist(), k.tolist(), tau)
    arrays = native.arrays()
    program = Program(**arrays)
    assert_equivalent_programs(program, p)
    # Independently pack the native topology; the symbolic topology can differ.
    expected = pack_program(program, 16)
    actual = native.chunks(16)
    for name in expected:
        if name == 'dlog' and not expected[name].size:
            continue  # Empty inference-style Program has no event derivative array.
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize('tau', [-2., 0., .5, 16.])
def test_root_forest_causal_boundaries(tau):
    # Branch 4 has queries before its first key only, branch 7 has keys without
    # any queries, and branch 9 has queries without keys. Branch 3 includes
    # equal-time pairs and a key after its final query; only valid pairs survive.
    q = np.array([4, 9, 3, 3, 9, 9, 9, 9], np.int32)
    k = np.array([7, 7, 3, 3, 4, 3, 7, 4], np.int32)
    native = load_prefill().plan_training(q.tolist(), k.tolist(), tau)
    arrays = native.arrays()
    assert_equivalent_programs(Program(**arrays), SymbolicPlanner(q, k).compile(tau))
    rows = arrays['rows']
    assert set(rows[rows >= 0]).issubset({2, 3})
    assert set(-rows[rows < 0]-1) == {2, 3}
    np.testing.assert_array_equal(arrays['logden'][[0, 1, 4, 5, 6, 7]], 0.)


def batch_fixture():
    rng = np.random.default_rng(712)
    labels = rng.integers(0, 3, (2, 2, 3, 65), dtype=np.int32)
    labels[:, 0, 0] = 0
    labels[:, 1, 1] = 0
    documents = np.array([[0, 0, 17], [0, 17, 65], [1, 0, 33], [1, 33, 65]], dtype=np.int64)
    hard = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    return labels, np.array([.5, 0., 2.]), documents, hard


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('chunk_size', [16, 32, 64])
def test_batch_matches_independent_document_plans(workers, chunk_size):
    labels, tau, documents, hard = batch_fixture()
    planner = NativeTrainingPlanner(workers)
    result = planner.prepare(labels, tau, documents=documents, hard=hard, chunk_size=chunk_size)
    expected = {k: [] for k in ('rows', 'prefixes', 'weights', 'dlog', 'lengths', 'resets', 'stream_heads')}
    offsets, cb = [0], 0
    for b, start, stop in documents:
        for h in range(3):
            if not hard[b, h]:
                continue
            arrays = load_prefill().plan_training(
                *[x.tolist() for x in labels[:, b, h, start:stop]], tau[h]).arrays()
            p = pack_program(Program(**arrays), chunk_size)
            row = p['rows'].astype(np.int64)
            absolute = (np.where(row >= 0, row, -row-1)+b*65+start)*3+h
            expected['rows'].extend(np.where(row >= 0, absolute, -absolute-1))
            for name in ('prefixes', 'weights', 'dlog', 'lengths', 'resets'):
                expected[name].extend(p[name])
            expected['stream_heads'].extend([h]*(len(p['offsets'])-1))
            offsets.extend(p['offsets'][1:]+cb)
            cb += len(p['lengths'])
    expected['offsets'] = offsets
    for name in expected:
        np.testing.assert_allclose(result.data[name].numpy(), expected[name], rtol=2e-6, atol=2e-7)
    assert result.statistics['jobs'] == 8
    assert result.storage.is_pinned()
    reused = planner.prepare(labels, tau, documents=documents, hard=hard, chunk_size=chunk_size, storage=result.storage)
    assert reused.storage.data_ptr() == result.storage.data_ptr()


@pytest.mark.parametrize('per_row_tau', [False, True])
def test_gpu_batch_backward_and_cross_stream_upload(per_row_tau):
    labels, tau, documents, hard = batch_fixture()
    if per_row_tau:
        tau = np.stack((tau, tau+.2))
    host = NativeTrainingPlanner(4).prepare(labels, tau, documents=documents, hard=hard)
    uploaded = host.upload(stream=torch.cuda.Stream())
    plan = HardPrefillBackward.from_native(uploaded)
    rng = np.random.default_rng(91)
    xs = [torch.tensor(rng.normal(size=(2, 65, 3, width))*.2, device='cuda').bfloat16()
          for width in (32, 32, 64, 64)]
    out = plan.backward(*[x.reshape(-1, x.shape[-1]) for x in xs])
    actual = [out[name].cpu().numpy().reshape(x.shape) for name, x in zip(('sq_vec', 'sk_vec', 'v'), xs)]
    expected = [np.zeros(x.shape) for x in xs[:3]]
    dtau = np.zeros(tau.shape)
    for b, start, stop in documents:
        for h in range(3):
            if not hard[b, h]:
                continue
            t = tau[b, h] if per_row_tau else tau[h]
            p = SymbolicPlanner(*labels[:, b, h, start:stop]).compile(t)
            values = [x[b, start:stop, h].float().cpu().double().numpy() for x in xs]
            _, grads, _ = p.evaluate(*values)
            for expected_array, grad in zip(expected, grads):
                expected_array[b, start:stop, h] = grad
            if per_row_tau:
                dtau[b, h] += grads[3]
            else:
                dtau[h] += grads[3]
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-6)
    np.testing.assert_allclose(out['rtau'].cpu().numpy(), dtau.ravel(), rtol=2e-4, atol=2e-5)


def test_empty_hard_batch_and_worker_reuse_after_error():
    labels, tau, documents, hard = batch_fixture()
    planner = NativeTrainingPlanner(2)
    with pytest.raises(ValueError, match='finite'):
        planner.prepare(labels, np.full(3, np.nan))
    result = planner.prepare(labels, tau, documents=documents, hard=np.zeros_like(hard))
    assert result.statistics['jobs'] == result.statistics['streams'] == 0
    np.testing.assert_array_equal(result.data['offsets'], [0])
    plan = HardPrefillBackward.from_native(result.upload())
    x, v = torch.ones((390, 32), device='cuda'), torch.ones((390, 64), device='cuda')
    out = plan.backward(x, x, v, v)
    assert all(torch.count_nonzero(out[k]) == 0 for k in ('sq_vec', 'sk_vec', 'v', 'rtau'))
    # Overflow occurs inside a worker; every queued task must drain safely.
    with pytest.raises(ValueError, match='range'):
        planner.prepare(labels, np.array([1e308, .5, .5]))
    assert planner.prepare(labels, tau).statistics['jobs'] == 6


def test_concurrent_calls_and_validation():
    labels, tau, documents, hard = batch_fixture()
    planner = NativeTrainingPlanner(2)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: planner.prepare(labels, tau, documents=documents, hard=hard), range(4)))
    for result in results[1:]:
        torch.testing.assert_close(result.storage, results[0].storage, rtol=0, atol=0)
    with pytest.raises(ValueError, match='documents'):
        planner.prepare(labels, tau, documents=np.array([[0, 0, 64], [1, 0, 65]], np.int64))
    with pytest.raises(ValueError, match='hard'):
        planner.prepare(labels, tau, hard=np.full((2, 3), 2, np.uint8))
    with pytest.raises(ValueError, match='contiguous'):
        planner.prepare(labels[..., ::-1], tau)
    native = planner.native.plan(labels, tau, documents, hard, 16)
    buffers = {k: v.numpy() for k, v in results[0].data.items()}
    buffers['dlog'] = buffers['weights']
    with pytest.raises(ValueError, match='overlap'):
        native.pack_into(buffers)
    with pytest.raises(ValueError, match='workers'):
        NativeTrainingPlanner(0)
