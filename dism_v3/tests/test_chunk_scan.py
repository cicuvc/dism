"""Independent row-wise oracle; CUDA implementation follows diagonals instead."""
import math
from pathlib import Path
import subprocess

import pytest
import torch
import torch.nn.functional as F
import cu_flash_dism

from flash_dism.summary import chunk_scan, summarize
from test_summary import make_inputs, summary_oracle


def passing_oracle(a, b):
    state = torch.full_like(a[..., 0, :], -1e6, dtype=torch.float64)
    result = torch.empty_like(a, dtype=torch.float64)
    for checkpoint in range(a.shape[-2]):
        incoming = F.pad(state[..., :-32], (32, 0), value=-1e6)
        state = torch.logaddexp2(incoming + a[..., checkpoint, :].double(),
                                b[..., checkpoint, :].double())
        state[..., (checkpoint + 1) * 32:] = -1e6
        result[..., checkpoint, :] = state
    return result.float()


@pytest.mark.parametrize('n', [256,512,768,1280,4352])
@pytest.mark.parametrize('kind', ['random', 'zero', 'growth'])
def test_chunk_scan_synthetic(n, kind):
    torch.manual_seed(59)
    shape = (2, 3, (n - 1) // 32, math.ceil(n / 256) * 256)
    a = torch.randn(shape, device='cuda') * 3
    b = torch.randn_like(a) * 3
    if kind == 'zero':
        a.fill_(0.)
        b.fill_(-1e6)
    elif kind == 'growth':
        a.fill_(32 * math.log2(128))
        b.copy_(a)
    # Poison all unspecified upper-triangle/padding data, including some
    # allocated elements that summary never writes at all.
    columns = torch.arange(shape[-1], device='cuda')
    rows = torch.arange(shape[-2], device='cuda') * 32 + 31
    invalid = columns[None, :] > rows[:, None]
    a[..., invalid] = float('nan')
    b[..., invalid] = float('nan')
    before_a, before_b = a.clone(), b.clone()
    actual = chunk_scan(a, b, n)
    expected = passing_oracle(a, b) if shape[-2] else b
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-6)
    torch.testing.assert_close(a, before_a, equal_nan=True, atol=0, rtol=0)
    torch.testing.assert_close(b, before_b, equal_nan=True, atol=0, rtol=0)
    assert actual.is_contiguous() and torch.isfinite(actual).all()


@pytest.mark.parametrize('n', [512,768,1280])
@pytest.mark.parametrize('mode', ['soft', 'mixed', 'hard'])
@pytest.mark.parametrize('direction', ['query', 'key'])
def test_summary_then_chunk_scan(n, mode, direction):
    inputs = make_inputs(n, mode, direction)
    q, k, lq, lk, *metadata = inputs
    tau = metadata[-1][None, None, :]
    a, b = summarize(q, k, lq - tau, lk - tau, *metadata, ctas=1)
    actual = chunk_scan(a, b, n)
    # Tight test of passing, separately from summary's tanh approximation.
    torch.testing.assert_close(actual, passing_oracle(a, b), atol=2e-4, rtol=2e-6)
    ideal = summary_oracle(inputs, a.shape[-1])
    ideal = ideal.masked_fill(~torch.isfinite(ideal), -1e6)
    expected = passing_oracle(ideal[..., 0], ideal[..., 1])
    reachable = expected > -1e5
    assert (actual[~reachable] < -1e5).all()
    torch.testing.assert_close(actual[reachable], expected[reachable], atol=.03, rtol=2e-5)


def test_chunk_scan_stream_and_graph():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = torch.full((1, 1, 23, 768), .2, device='cuda')
        b = torch.full_like(a, -.5)
        expected = passing_oracle(a, b)
        chunk_scan(a, b, 768)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = chunk_scan(a, b, 768)
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)


def test_chunk_scan_validation():
    a = torch.empty((1, 1, 8, 512), device='cuda')
    for b, n in [(a.double(), 257), (a[..., :256], 257), (a, 256), (a, 0)]:
        with pytest.raises(RuntimeError):
            chunk_scan(a, b, n)
    with pytest.raises(RuntimeError):
        chunk_scan(a.cpu(), a.cpu(), 257)


def test_chunk_scan_default():
    a = torch.full((1, 1, 15, 512), -.2, device='cuda')
    b = torch.full_like(a, .5)
    expected = chunk_scan(a, b, 512)
    torch.testing.assert_close(chunk_scan(a, b, 512), expected, atol=0, rtol=0)
    torch.testing.assert_close(cu_flash_dism.chunk_scan(a, b, 512), expected, atol=0, rtol=0)


def test_chunk_scan_codegen():
    binary = Path(__file__).resolve().parents[1] / 'build/object/r32_d64_v64/chunk_scan.cu.dev.sm120a.o'
    sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '-sass', str(binary)], text=True)
    assert 'CALL' not in sass
    assert 'LDL' not in sass and 'STL' not in sass
    assert 'LDGSTS' in sass and 'DEPBAR' in sass
