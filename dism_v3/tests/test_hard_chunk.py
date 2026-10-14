"""Exact hard boundary composition and summary -> chunk integration."""
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from flash_dism.summary import summarize_hard, hard_chunk_scan
from flash_dism.varlen import VarlenLayout
from test_hard_summary import labels


def compose_cpu(summary):
    result = summary.cpu().numpy().copy()
    n = result.shape[-1]
    for cp in range(result.shape[-2]):
        if cp:
            result[:,:,cp,32:] = np.maximum(result[:,:,cp,32:], result[:,:,cp-1,:-32])
        result[:,:,cp,cp*32+32:n] = -1
    return torch.from_numpy(result)


def dense_boundary(q, k, hard):
    """Full scalar L recurrence, without independent summaries or max passing."""
    q, k, hard = q.cpu().numpy(), k.cpu().numpy(), hard.cpu().numpy()
    b,h,n = q.shape
    length = np.zeros((b,h,n), dtype=np.int32)
    result = np.full((b,h,(n-1)//32,n), -1, dtype=np.int32)
    columns = np.arange(n, dtype=np.int32)
    for row in range(n-32):
        previous = np.concatenate((np.zeros((b,h,1), dtype=np.int32), length[:,:,:-1]), axis=-1)
        length = np.where(q[:,:,row,None] == k, previous+1, 0)
        if row % 32 == 31:
            result[:,:,row//32] = np.where(hard[:,:,None] & (columns <= row), columns-length, -1)
    return torch.from_numpy(result)


@pytest.mark.parametrize('n', [256,512,2048,4096])
def test_random_summary_composition(n):
    rng = np.random.default_rng(63)
    source = torch.from_numpy(rng.integers(-1,n,(2,3,(n-1)//32,n),dtype=np.int32)).cuda()
    before = source.clone()
    # Poison noncausal entries: they must never propagate into causal chains.
    for cp in range(source.size(2)):
        source[:,:,cp,cp*32+32:] = np.iinfo(np.int32).max
    expected = compose_cpu(source)
    before.copy_(source)
    for _ in range(3):
        actual = hard_chunk_scan(source)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(source, before, rtol=0, atol=0)


@pytest.mark.parametrize('n', [256,1024,2048])
@pytest.mark.parametrize('pattern', ['random','equal','disjoint','periodic','boundary'])
def test_complete_boundaries(n, pattern):
    q,k = labels(n,pattern)
    hard = torch.tensor([[True,False,True],[False,True,True]],device='cuda')
    summary = summarize_hard(q,k,hard,ctas=1)
    actual = hard_chunk_scan(summary)
    torch.testing.assert_close(actual.cpu(), dense_boundary(q,k,hard), rtol=0, atol=0)


@pytest.mark.parametrize('pattern', ['random','equal','boundary'])
def test_packed_boundaries(pattern):
    q,k = labels(2048,pattern,b=1,h=2)
    hard = torch.tensor([[True,False],[False,True]],device='cuda')
    offsets = [0,0,256,512,512,1024,2048]
    layout = VarlenLayout.from_cu_seqlens(torch.tensor(offsets,dtype=torch.int32),2048)
    summary = summarize_hard(q,k,hard,layout=layout,ctas=1)
    actual = hard_chunk_scan(summary,layout=layout,heads=2)
    expected = torch.cat([dense_boundary(q[:,:,a:z],k[:,:,a:z],hard[a//1024:a//1024+1]).flatten()
                          for a,z in zip(offsets,offsets[1:]) if z>a])
    torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)


def test_graph_replay_and_all_soft():
    q,k = labels(512,'boundary')
    hard = torch.ones((2,3),device='cuda',dtype=torch.bool)
    for _ in range(2):
        hard_chunk_scan(summarize_hard(q,k,hard))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = hard_chunk_scan(summarize_hard(q,k,hard))
    for flag in (True,False,True):
        hard.fill_(flag)
        graph.replay()
        torch.testing.assert_close(output.cpu(),dense_boundary(q,k,hard),rtol=0,atol=0)


@pytest.mark.parametrize('packed', [False,True])
def test_runtime_handoffs(packed):
    q,k = labels(1024,'boundary',b=1 if packed else 2,h=2)
    hard = torch.tensor([[True,False],[False,True]],device='cuda')
    layout = VarlenLayout.from_cu_seqlens(torch.tensor([0,256,512,1024],dtype=torch.int32),1024) if packed else None
    expected = (torch.cat([dense_boundary(q[:,:,a:z],k[:,:,a:z],hard[a//512:a//512+1]).flatten()
                          for a,z in ((0,256),(256,512),(512,1024))]) if packed else dense_boundary(q,k,hard))
    for _ in range(4):
        summary = summarize_hard(q,k,hard,layout=layout,ctas=1)
        actual = hard_chunk_scan(summary,layout=layout,heads=2)
        torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)


def test_invalid_and_empty_packed():
    source = torch.full((1,2,7,256),-1,device='cuda',dtype=torch.int32)
    for bad in (source.float(),source.cpu(),source[:,:,:,:128],source[:,:,::2]):
        with pytest.raises(RuntimeError):
            hard_chunk_scan(bad)
    with pytest.raises(RuntimeError,match='head'):
        hard_chunk_scan(source,heads=3)
    layout = VarlenLayout.from_cu_seqlens(torch.tensor([0,256],dtype=torch.int32),256)
    with pytest.raises(RuntimeError,match='packed'):
        hard_chunk_scan(source.flatten(),layout=layout)
    empty = VarlenLayout.from_cu_seqlens(torch.tensor([0,0,0],dtype=torch.int32),0)
    actual = hard_chunk_scan(torch.empty(0,device='cuda',dtype=torch.int32),layout=empty,heads=2)
    assert actual.numel() == 0 and actual.dtype == torch.int32


def test_hard_chunk_codegen():
    import cu_flash_dism as cu
    root = Path(__file__).resolve().parents[1]
    for r,d,v in cu.configs:
        tag = f'r{r}_d{d}_v{v}'
        obj = root / f'build/object/{tag}/hard_chunk.cu.dev.sm120a.o'
        sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','-sass',str(obj)],text=True)
        assert 'hard_chunk_kernel' in sass and 'LDGSTS' in sass
        assert all(name not in sass for name in ('CALL','HMMA','MUFU.TANH','MUFU.EX2','MUFU.LG2'))
        assert '/torch/include/' not in (root / f'build/deps/{tag}/hard_chunk.cu.d').read_text()
