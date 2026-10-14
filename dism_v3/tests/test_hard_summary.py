"""Exact integer hard F1 checks; no relaxed float tolerance or oracle edits."""
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
import torch

from flash_dism.summary import summarize_hard
from flash_dism.varlen import VarlenLayout


def labels(n, pattern, b=2, h=3):
    rng = np.random.default_rng(20261014)
    q = rng.integers(-3, 4, (b, h, n), dtype=np.int32)
    k = rng.integers(-3, 4, (b, h, n), dtype=np.int32)
    if pattern == 'equal':
        q.fill(np.iinfo(np.int32).max)
        k[:] = q
    elif pattern == 'disjoint':
        q.fill(np.iinfo(np.int32).min)
        k.fill(np.iinfo(np.int32).max)
    elif pattern == 'periodic':
        q[:] = np.arange(n, dtype=np.int32) % 7
        k[:] = (np.arange(n, dtype=np.int32) + 2) % 7
    elif pattern == 'boundary':
        q.fill(1)
        k.fill(1)
        for i in (0, 30, 31, 32, 62, 63, 64, 254, 255, 256, 510, 511, 512):
            if i < n:
                q[:, ::2, i] = -100
                k[:, 1::2, i] = 100
    return torch.from_numpy(q).cuda(), torch.from_numpy(k).cuda()


def oracle(q, k, hard):
    """Independent row-major integer recurrence, reset at every 32-row strip."""
    q, k, hard = q.cpu().numpy(), k.cpu().numpy(), hard.cpu().numpy()
    b, h, n = q.shape
    out = np.full((b, h, (n-1)//32, n), -1, dtype=np.int32)
    col = np.arange(n, dtype=np.int32)
    for cp in range(out.shape[2]):
        state = np.full((b, h, n), -1, dtype=np.int32)
        for row in range(cp*32, cp*32+32):
            incoming = np.concatenate((np.full((b,h,1), -1, dtype=np.int32), state[:,:,:-1]), axis=-1)
            state = np.maximum(incoming, np.where(q[:,:,row,None] == k, -1, col))
        out[:,:,cp] = np.where(hard[:,:,None] & (col <= cp*32+31), state, -1)
    return torch.from_numpy(out)


def assert_composed_lengths(summary, q, k, hard):
    """CPU composition must recover the full document's scalar L recurrence."""
    q, k, hard = q.cpu().numpy(), k.cpu().numpy(), hard.cpu().numpy()
    summary = summary.cpu().numpy()
    b, h, n = q.shape
    length = np.zeros((b, h, n), dtype=np.int32)
    prefix = np.full_like(length, -1)
    columns = np.arange(n, dtype=np.int32)
    for row in range(n-32):
        incoming = np.concatenate((np.zeros((b,h,1), dtype=np.int32), length[:,:,:-1]), axis=-1)
        length = np.where(q[:,:,row,None] == k, incoming + 1, 0)
        if row % 32 == 31:
            previous = np.concatenate((np.full((b,h,32), -1, dtype=np.int32), prefix[:,:,:-32]), axis=-1)
            prefix = np.maximum(previous, summary[:,:,row//32])
            valid = np.broadcast_to(hard[:,:,None] & (columns <= row), length.shape)
            np.testing.assert_array_equal((columns-prefix)[valid], length[valid])


@pytest.mark.parametrize('n', [256, 512, 1024, 2048])
@pytest.mark.parametrize('pattern', ['random', 'equal', 'disjoint', 'periodic', 'boundary'])
def test_fixed_exact(n, pattern):
    q, k = labels(n, pattern)
    hard = torch.tensor([[True, False, True], [False, True, True]], device='cuda')
    expected = oracle(q, k, hard)
    for ctas in (1, 2, 0):
        actual = summarize_hard(q, k, hard, ctas=ctas)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert_composed_lengths(actual, q, k, hard)


@pytest.mark.parametrize('pattern', ['random', 'equal', 'periodic', 'boundary'])
def test_packed_documents_and_row_flags(pattern):
    q, k = labels(2048, pattern, b=1, h=2)
    hard = torch.tensor([[True, False], [False, True]], device='cuda')
    offsets = [0, 0, 256, 512, 512, 1024, 2048]
    layout = VarlenLayout.from_cu_seqlens(torch.tensor(offsets, dtype=torch.int32), 2048)
    expected = []
    for a, z in zip(offsets, offsets[1:]):
        if z > a:
            expected.append(oracle(q[:,:,a:z], k[:,:,a:z], hard[a//1024:a//1024+1]).flatten())
    expected = torch.cat(expected)
    for ctas in (1, 2, 0, 1):
        actual = summarize_hard(q, k, hard, layout=layout, ctas=ctas)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_flag_changes_and_graph_replay():
    q, k = labels(512, 'boundary')
    hard = torch.ones((2,3), device='cuda', dtype=torch.bool)
    summarize_hard(q, k, hard, ctas=1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = summarize_hard(q, k, hard, ctas=1)
    for flags in (False, True, False, True):
        hard.fill_(flags)
        graph.replay()
        torch.testing.assert_close(result.cpu(), oracle(q,k,hard), rtol=0, atol=0)


@pytest.mark.parametrize('packed', [False, True])
def test_runtime_handoffs(packed):
    """Small memcheck/synccheck/racecheck target with CPU-only reference."""
    q, k = labels(1024, 'boundary', b=1 if packed else 2, h=2)
    hard = torch.tensor([[True, False], [False, True]], device='cuda')
    layout = VarlenLayout.from_cu_seqlens(torch.tensor([0,256,512,1024], dtype=torch.int32), 1024) if packed else None
    expected = (torch.cat([oracle(q[:,:,a:z], k[:,:,a:z], hard[a//512:a//512+1]).flatten()
                          for a,z in ((0,256),(256,512),(512,1024))]) if packed else oracle(q,k,hard))
    for _ in range(4):
        actual = summarize_hard(q,k,hard,layout=layout,ctas=1)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_invalid_inputs():
    q, k = labels(512, 'random', b=1, h=2)
    hard = torch.ones((1,2), device='cuda', dtype=torch.bool)
    for bad in (q.long(), q.cpu(), q[:,:,:257], q[:,:,::2]):
        with pytest.raises(RuntimeError):
            summarize_hard(bad, k, hard)
    with pytest.raises(RuntimeError, match='token-level'):
        summarize_hard(q,k,hard[:,:,None].expand(1,2,512))
    with pytest.raises(RuntimeError, match='CTA'):
        summarize_hard(q,k,hard,ctas=-1)
    layout = VarlenLayout.from_cu_seqlens(torch.tensor([0,512], dtype=torch.int32),512)
    with pytest.raises(RuntimeError, match='cross a hard training-row'):
        summarize_hard(q,k,hard.expand(2,2).contiguous(),layout=layout)


def test_hard_summary_codegen():
    root = Path(__file__).resolve().parents[1]
    # Dispatch uses the first compiled backend. Audit every compiled instance.
    import cu_flash_dism as cu
    for r,d,v in cu.configs:
        tag = f'r{r}_d{d}_v{v}'
        obj = root / f'build/object/{tag}/hard_summary.cu.dev.sm120a.o'
        sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','-sass',str(obj)],text=True)
        assert 'hard_summary_kernel' in sass
        assert not re.search(r'\b(?:CALL|HMMA)\b|MUFU\.(?:TANH|EX2|LG2)', sass)
        # MUFU.RCP is allowed: integer task-index division uses a reciprocal.
        assert 'LDGSTS' in sass  # cp.async label staging remains present
        dep = (root / f'build/deps/{tag}/hard_summary.cu.d').read_text()
        assert '/torch/include/' not in dep
