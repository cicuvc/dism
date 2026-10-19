"""Chunk-parallel dSQ/dSK/dV and analytic tau, with unchanged numerical gates."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_hard_prefill_backward_triton import check_case
from hard_prefill_backward_fused import FusedHardPrefillBackward
from hard_prefill_backward_triton import HardPrefillBackward, pack_program
from hard_prefill_grad_reference import Program
from flash_dism.inference.training_planner import NativeTrainingPlanner


@pytest.mark.parametrize('precision', ['bf16', 'tf32x3'])
@pytest.mark.parametrize('pattern', ['random', 'repeat', 'periodic', 'mismatch'])
@pytest.mark.parametrize('n,tau', [(1, 0.), (17, -2.), (65, .5), (256, 2.), (256, 16.)])
def test_fused_reference(precision, pattern, n, tau):
    result = check_case(pattern, n, tau, precision=precision, fused=True)
    if precision == 'tf32x3':
        assert not result['tau_material_sign_flip'], result
    if pattern == 'mismatch':
        assert result['tau_actual'] == 0


@pytest.mark.parametrize('r,dv,c', [(16, 32, 16), (16, 64, 32), (32, 32, 64)])
@pytest.mark.parametrize('precision', ['bf16', 'tf32x3'])
def test_fused_dimensions(r, dv, c, precision):
    result = check_case('repeat', 65, .5, r=r, dv=dv, chunk_size=c,
                        precision=precision, dtype=torch.float32, fused=True)
    assert not result['tau_material_sign_flip'], result


@pytest.mark.parametrize('all_soft', [False, True])
@pytest.mark.parametrize('chunk_size', [16, 64])
def test_native_mixed_buckets_upload_and_graph(all_soft, chunk_size):
    rng = np.random.default_rng(712)
    labels = rng.integers(0, 3, (2, 2, 3, 65), dtype=np.int32)
    labels[:, 0, 0] = 0
    hard = np.zeros((2, 3), np.uint8) if all_soft else np.array([[1, 0, 1], [0, 1, 1]], np.uint8)
    host = NativeTrainingPlanner(2).prepare(labels, np.array([.5, 0., 2.]), hard=hard, chunk_size=chunk_size)
    device = host.upload(stream=torch.cuda.Stream())
    source = HardPrefillBackward.from_native(device)
    plan = FusedHardPrefillBackward.from_plan(source, host.data['offsets'].numpy(),
                                            host_lengths=host.data['lengths'].numpy())
    xs = [torch.randn((390, c), device='cuda').bfloat16()*.2 for c in (32, 32, 64, 64)]
    expected = source.backward(*xs)
    actual = plan.backward(*xs)
    for name in ('sq_vec', 'sk_vec', 'v', 'rtau'):
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-4, atol=2e-5)
    # Missing CPU lengths is a correct (unbucketed), readback-free fallback.
    fallback = FusedHardPrefillBackward.from_plan(source, host.data['offsets'].numpy())
    other = fallback.backward(*xs)
    for name in ('sq_vec', 'sk_vec', 'v', 'rtau'):
        torch.testing.assert_close(other[name], expected[name], rtol=2e-4, atol=2e-5)
    # Upload wait must be outside capture; retain device ownership throughout.
    device.ready.synchronize()
    del plan.device_plan
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = plan.backward(*xs)
    graph.replay()
    for name in ('sq_vec', 'sk_vec', 'v', 'rtau'):
        torch.testing.assert_close(replayed[name], actual[name], rtol=2e-5, atol=2e-6)
    # In-place summary -> boundary storage must be rebuilt on every replay.
    xs[0].mul_(.5)
    expected = source.backward(*xs)
    graph.replay()
    for name in ('sq_vec', 'sk_vec', 'v', 'rtau'):
        torch.testing.assert_close(replayed[name], expected[name], rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize('chunk_size', [16, 64])
def test_reset_between_chunks_and_nonidentity_stream_ids(chunk_size):
    p = Program(np.array([0, 2, 7, 9]),
                np.array([0, -1, 0, -1, 1, -2, -3, 2, -3]),
                np.array([0., 1., 0., 1., 0., 1., 1., 0., 1.]),
                np.ones(9), np.linspace(-.5, .5, 9), np.zeros(3), np.zeros(3))
    packed = pack_program(p, chunk_size)
    source = HardPrefillBackward([0, 0, 0], [1, 1, 1], 0., chunk_size=chunk_size)
    source.program, source.packed, source.streams = p, packed, 3
    source.data = {k: torch.as_tensor(v, device='cuda') for k, v in packed.items()}
    plan = FusedHardPrefillBackward.from_plan(source, packed['offsets'])
    assert (plan.single_count, plan.multi_count) == (2, 1)
    rng = np.random.default_rng(123)
    xs = [torch.tensor(rng.normal(size=(3, c)), device='cuda').bfloat16()
          for c in (16, 16, 32, 32)]
    actual = plan.backward(*xs)
    _, expected, _ = p.evaluate(*[x.float().cpu().double().numpy() for x in xs])
    for name, ref in zip(('sq_vec', 'sk_vec', 'v', 'rtau'), expected):
        torch.testing.assert_close(actual[name].cpu().double(), torch.as_tensor(ref), rtol=2e-5, atol=2e-6)
    with pytest.raises(ValueError, match='cover'):
        FusedHardPrefillBackward.from_plan(source, [0, 1, 2, 3])
    with pytest.raises(ValueError, match='lengths'):
        FusedHardPrefillBackward.from_plan(source, packed['offsets'], host_lengths=[0])


@pytest.mark.parametrize('events', [15, 16, 17, 32, 33, 64])
def test_bf16_tau_cancellation_at_chunk_boundaries(events):
    # A single stream whose physical pair derivative is 256 - 255.5 = .5.
    # This isolates the carried-state contractions from SAM/normalizer algebra.
    # Width counts events, not tokens: include exact and partial final chunks.
    c, n = 16, (events+1)//2
    e = np.arange(events)
    rows = np.where(e % 2 == 0, e//2, -(e//2)-1)
    decay = np.ones(events)
    decay[0] = 0.
    p = Program(np.array([0, events]), rows, decay, np.full(events, 1/n),
                np.where(e % 2 == 0, 256., -255.5), np.zeros(n), np.zeros(n))
    packed = pack_program(p, c)
    source = HardPrefillBackward([0]*n, [1]*n, 0., chunk_size=c,
                                mma_precision='bf16')
    source.program, source.packed, source.streams = p, packed, 1
    source.data = {k: torch.as_tensor(v, device='cuda') for k, v in packed.items()}
    plan = FusedHardPrefillBackward.from_plan(source, packed['offsets'])
    assert len(packed['lengths']) == (events+c-1)//c
    rng = np.random.default_rng(123)
    xs = [torch.tensor(rng.normal(size=(n, d)), device='cuda').bfloat16()
          for d in (32, 32, 64, 64)]
    actual = plan.backward(*xs)
    _, expected, _ = p.evaluate(*[x.float().cpu().double().numpy() for x in xs])
    torch.testing.assert_close(actual['rtau'].cpu().double(),
                               torch.as_tensor(expected[3]), rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize('precision', ['bf16', 'tf32x3'])
@pytest.mark.parametrize('tau', [-2., 0., .5, 2., 16.])
def test_c64_training_dimensions(precision, tau):
    result = check_case('repeat', 256, tau, chunk_size=64, precision=precision, fused=True)
    if precision == 'tf32x3':
        assert not result['tau_material_sign_flip'], result


@pytest.mark.parametrize('chunk_size', [16, 64])
def test_long_stream_matches_scan(chunk_size):
    n = 4096
    source = HardPrefillBackward([0]*n, [0]*n, .5, chunk_size=chunk_size)
    plan = FusedHardPrefillBackward.from_plan(source, source.packed['offsets'])
    assert np.diff(source.packed['offsets']).max() >= 64
    torch.manual_seed(714)
    xs = [torch.randn((n, c), device='cuda').bfloat16()*.2 for c in (32, 32, 64, 64)]
    actual, expected = plan.backward(*xs), source.backward(*xs)
    for name in ('sq_vec', 'sk_vec', 'v'):
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-4, atol=2e-5)
    # Compare cancellation-sensitive shared tau by scale; do not replace the
    # production acceptance gates exercised against FP64 in check_case.
    torch.testing.assert_close(actual['rtau'], expected['rtau'], rtol=2e-3, atol=2e-4)
