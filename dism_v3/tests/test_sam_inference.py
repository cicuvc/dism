"""SAM migration regression: independent FP64 recurrence, prefill and caches."""
import numpy as np
import pytest
import torch

from flash_dism.inference import HardDismDecoder, HardDismPrefill, HardPrefillPlan, GpuPlannerCache


def oracle(iq, ik, sq, sk, value, tau, reset=None):
    """Dense causal hard DISM, no SAM or vocabulary interpolation involved."""
    iq, ik = iq.cpu(), ik.cpu()
    sq, sk, value = [x.cpu().double() for x in (sq, sk, value)]
    b, n, h, _ = sq.shape
    previous = torch.full((b, h, n), -torch.inf, dtype=torch.float64)
    out = torch.empty((b, n, h, value.shape[-1]), dtype=torch.float64)
    tau = torch.as_tensor(tau, dtype=torch.float64)[None, :, None]
    reset = torch.zeros_like(iq, dtype=torch.bool) if reset is None else reset.cpu()
    for i in range(n):
        predecessor = torch.cat((torch.full_like(previous[..., :1], -torch.inf),
                                 previous[..., :-1]), dim=-1)
        predecessor = predecessor.masked_fill(reset[..., i, None], -torch.inf)
        score = torch.nn.functional.softplus(predecessor) + tau
        score = score.masked_fill(iq[..., i, None] != ik, -torch.inf)
        score[..., i+1:] = -torch.inf
        probability = torch.softmax(torch.cat((torch.zeros_like(score[..., :1]), score), -1), -1)[..., 1:]
        weight = torch.einsum('bhr,bnhr->bhn', sq[:, i], sk)
        out[:, i] = torch.einsum('bhn,bnhd->bhd', probability * weight, value)
        previous = score
    return out


def inputs(pattern, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(83)
    b, h, n, r, dv = 2, 2, 79, 16, 32
    iq = torch.randint(0, 4, (b, h, n), generator=g, dtype=torch.int32)
    ik = torch.randint(0, 4, (b, h, n), generator=g, dtype=torch.int32)
    if pattern == 'repeat':
        iq.zero_()
        ik.zero_()
    if pattern == 'mismatch':
        iq.zero_()
        ik.fill_(1)
    vectors = [(torch.randn(b, n, h, c, generator=g) * .2).to(dtype).cuda()
               for c in (r, r, dv)]
    return iq.cuda(), ik.cuda(), *vectors, [0., .35]


@pytest.mark.parametrize('pattern', ['random', 'repeat', 'mismatch'])
@pytest.mark.parametrize('precision', ['bf16', 'tf32x3'])
def test_prefill(pattern, precision):
    iq, ik, sq, sk, v, tau = inputs(pattern)
    expected = oracle(iq, ik, sq, sk, v, tau)
    for workers in (1, 4):
        with HardDismPrefill(workers=workers, mma_precision=precision) as engine:
            actual = engine(iq, ik, sq, sk, v, tau)
        torch.testing.assert_close(actual.cpu().double(), expected,
                                   atol=3e-4 if precision == 'bf16' else 2e-6,
                                   rtol=.025 if precision == 'bf16' else 2e-4)
    # CPU event execution is an additional high precision oracle.
    plan = HardPrefillPlan(iq[0, 1].cpu().numpy(), ik[0, 1].cpu().numpy(), tau[1])
    result = plan.execute(*(x[0, :, 1].float().cpu().numpy() for x in (sq, sk, v)), dtype=np.float64)
    np.testing.assert_allclose(result, expected[0, :, 1].numpy(), atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize('planner', ['cpu', 'gpu'])
@pytest.mark.parametrize('pattern', ['random', 'repeat', 'mismatch'])
@pytest.mark.parametrize('prime', [False, True])
def test_decode(planner, pattern, prime):
    iq, ik, sq, sk, v, tau = inputs(pattern)
    expected = oracle(iq, ik, sq, sk, v, tau)
    decoder = HardDismDecoder(2, 2, 16, 32, 79, tau, planner_backend=planner,
                              rebuild_interval=11, sample_interval=5,
                              materialize_threshold=20)
    start = 31 if prime else 0
    if prime:
        decoder.prime(iq[..., :start], ik[..., :start], sk[:, :start], v[:, :start])
    actual = decoder.append(iq[..., start:], ik[..., start:],
                            sq[:, start:], sk[:, start:], v[:, start:])
    torch.testing.assert_close(actual.cpu().double(), expected[:, start:], atol=2e-5, rtol=2e-3)
    assert decoder.position == 79


def test_binary_query_reset():
    iq, ik, sq, sk, v, tau = inputs('repeat')
    reset = torch.zeros_like(iq, dtype=torch.bool)
    reset[..., 17::19] = True
    expected = oracle(iq, ik, sq, sk, v, tau, reset)
    with HardDismPrefill(mma_precision='tf32x3') as engine:
        actual = engine(iq, ik, sq, sk, v, tau, reset=reset)
    torch.testing.assert_close(actual.cpu().double(), expected, atol=2e-6, rtol=2e-4)
    decoder = HardDismDecoder(2, 2, 16, 32, 79, tau, planner_backend='gpu', rebuild_interval=11)
    actual = decoder.append(iq, ik, sq, sk, v, reset=reset)
    torch.testing.assert_close(actual.cpu().double(), expected, atol=2e-5, rtol=2e-3)


def test_gpu_graph_rebuild_and_capacity():
    iq, ik, sq, sk, v, tau = inputs('repeat')
    expected = oracle(iq, ik, sq, sk, v, tau)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = GpuPlannerCache(4, 16, 32, 79, tau * 2, rebuild_interval=11,
                                sample_interval=5, materialize_threshold=20)
        labels = torch.zeros(3, 4, dtype=torch.int32, device='cuda')
        ssq, ssk = [torch.empty(4, 16, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        vv = torch.empty(4, 32, device='cuda', dtype=torch.bfloat16)
        graph = None
        for i in range(79):
            if i and i % 11 == 0:
                cache.rebuild()
                graph = None
            ssq.copy_(sq[:, i].reshape(4, 16))
            ssk.copy_(sk[:, i].reshape(4, 16))
            vv.copy_(v[:, i].reshape(4, 32))
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = cache.step(labels, ssk, ssq, vv)
            graph.replay()
            torch.testing.assert_close(out.cpu().double().reshape(2, 2, 32),
                                       expected[:, i], atol=2e-5, rtol=2e-3)
        assert cache.check_status() == 79
        assert cache.step(labels, ssk, ssq, vv).isnan().all()
        with pytest.raises(RuntimeError, match='capacity/rebuild horizon/reset'):
            cache.check_status()
    stream.synchronize()


@pytest.mark.parametrize('invalid_reset', [False, True])
def test_gpu_horizon_guard(invalid_reset):
    cache = GpuPlannerCache(1, 16, 32, 32, [.5], rebuild_interval=2)
    labels = torch.zeros(3, 1, dtype=torch.int32, device='cuda')
    sk = torch.ones(1, 16, dtype=torch.bfloat16, device='cuda')
    v = torch.ones(1, 32, dtype=torch.bfloat16, device='cuda')
    if invalid_reset:
        labels[2] = 2
    else:
        for _ in range(2):
            cache.step(labels, sk, sk, v)
    assert cache.step(labels, sk, sk, v).isnan().all()
    with pytest.raises(RuntimeError, match='capacity/rebuild horizon/reset'):
        cache.check_status()


@pytest.mark.parametrize('pattern', ['random', 'repeat', 'mismatch', 'alternating'])
@pytest.mark.parametrize('reset', [False, True])
@pytest.mark.parametrize('tau', [0.0, 0.35, -0.2, 1.5])
def test_prefill_lca_matches_binary_lifting(pattern, reset, tau):
    """The O(component) in-component LCA must equal the original oracle.

    `plan_reference` keeps the pre-optimization binary-lifting path; both
    planners share every other stage, so identical Program arrays prove the
    replacement is exact (not just numerically close).
    """
    from flash_dism.inference.build import load_prefill
    n = 257
    rng = np.random.default_rng(20240)
    iq = rng.integers(0, 4, n).astype(np.int32)
    ik = rng.integers(0, 4, n).astype(np.int32)
    if pattern == 'repeat':
        iq = np.zeros(n, np.int32)
        ik = np.zeros(n, np.int32)
    elif pattern == 'mismatch':
        iq = np.zeros(n, np.int32)
        ik = np.ones(n, np.int32)
    elif pattern == 'alternating':
        iq = (np.arange(n) % 2).astype(np.int32)
        ik = iq.copy()
    resets = (np.arange(n) % 13 == 0).astype(np.int32) if reset else np.zeros(n, np.int32)
    module = load_prefill()
    fast = module.plan(iq.tolist(), ik.tolist(), resets.tolist(), float(tau)).arrays()
    slow = module.plan_reference(iq.tolist(), ik.tolist(), resets.tolist(), float(tau)).arrays()
    for name in ('offsets', 'rows', 'decay', 'weight', 'logden'):
        np.testing.assert_array_equal(fast[name], slow[name],
                                      err_msg=f'{pattern} reset={reset} tau={tau} {name}')


@pytest.mark.parametrize('precision', ['bf16', 'tf32x3'])
@pytest.mark.parametrize('pattern', ['random', 'repeat', 'mismatch'])
@pytest.mark.parametrize('reset', [False, True])
@pytest.mark.parametrize('chunk_size', [16, 32])
def test_prefill_parallel_state(precision, pattern, reset, chunk_size):
    """Chunk-parallel summary/passing path vs serial stream carry and oracle."""
    iq, ik, sq, sk, v, tau = inputs(pattern)
    rst = None
    if reset:
        rst = torch.zeros_like(iq, dtype=torch.bool)
        rst[..., 11::17] = True
    expected = oracle(iq, ik, sq, sk, v, tau, rst)
    with HardDismPrefill(mma_precision=precision, chunk_size=chunk_size) as engine:
        prepared = engine.prepare(iq, ik, tau, reset=rst)
        serial = prepared.execute(sq, sk, v)
        b, n, h, r = sq.shape
        dv = v.shape[-1]
        parallel = prepared.core.execute_parallel(
            sq.reshape(-1, r), sk.reshape(-1, r), v.reshape(-1, dv)).view(b, n, h, dv)
    atol = 3e-4 if precision == 'bf16' else 2e-6
    rtol = .025 if precision == 'bf16' else 2e-4
    torch.testing.assert_close(parallel.cpu().double(), serial.cpu().double(),
                               atol=atol, rtol=rtol)
    torch.testing.assert_close(parallel.cpu().double(), expected[:, :, :, :],
                               atol=atol, rtol=rtol)


@pytest.mark.parametrize('precision', ['bf16', 'tf32x3'])
@pytest.mark.parametrize('pattern', ['random', 'repeat'])
@pytest.mark.parametrize('reset', [False, True])
@pytest.mark.parametrize('threshold', [0, 1, 64])
def test_prefill_dispatch(precision, pattern, reset, threshold):
    """Serial/short and chunk-parallel/long routing must match the oracle.

    threshold=0 sends every stream to the parallel phases, threshold=64 keeps
    everything serial, threshold=1 gives a genuine mix on these patterns.
    """
    iq, ik, sq, sk, v, tau = inputs(pattern)
    rst = None
    if reset:
        rst = torch.zeros_like(iq, dtype=torch.bool)
        rst[..., 11::17] = True
    expected = oracle(iq, ik, sq, sk, v, tau, rst)
    with HardDismPrefill(mma_precision=precision, dispatch_threshold=threshold) as engine:
        prepared = engine.prepare(iq, ik, tau, reset=rst)
        serial = prepared.execute(sq, sk, v)
        dispatched = prepared.execute_dispatch(sq, sk, v)
    atol = 3e-4 if precision == 'bf16' else 2e-6
    rtol = .025 if precision == 'bf16' else 2e-4
    torch.testing.assert_close(dispatched.cpu().double(), serial.cpu().double(),
                               atol=atol, rtol=rtol)
    torch.testing.assert_close(dispatched.cpu().double(), expected,
                               atol=atol, rtol=rtol)
