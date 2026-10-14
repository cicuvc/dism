"""Compact equivalence and cross-stream regressions using the current worktree.

Use 3 streams: metadata upload, compact construction, gradient consumption.
The delayed zero-initialized upload deterministically exposes premature reads
without feeding arbitrary uninitialized indices to the kernel.
"""
import gc
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

from flash_dism.inference.training_planner import NativeTrainingPlanner
from flash_dism.kernels.hard_backward_fused import CompactFusedHardPrefillBackward

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_compact_training_planner import check_cpu_case, check_gpu_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/pinned memory required')


@pytest.mark.parametrize('non_blocking', [False, True])
def test_upload_construction_and_backward_on_distinct_streams(non_blocking):
    planner=NativeTrainingPlanner(4)
    labels=np.random.default_rng(47).integers(0,8,(2,2,3,256),dtype=np.int32)
    host=planner.prepare(labels,np.full(3,.5),chunk_size=64,compact=True)
    device=host.upload()
    reference=CompactFusedHardPrefillBackward.from_compact(device)
    torch.manual_seed(9)
    vectors=[torch.randn(2*3*256,d,device='cuda',dtype=torch.bfloat16)*.2
             for d in (32,32,64,64)]
    expected=reference.backward(*vectors)
    torch.cuda.synchronize()
    upload,construct,consume=(torch.cuda.Stream() for _ in range(3))
    device.storage.zero_()
    torch.cuda.synchronize()
    with torch.cuda.stream(upload):
        torch.cuda._sleep(100_000_000)
        device.storage.copy_(host.storage,non_blocking=True)
        device.ready.record()
    with torch.cuda.stream(construct):
        compact=CompactFusedHardPrefillBackward.from_compact(device,non_blocking=non_blocking)
    with torch.cuda.stream(consume):
        actual=compact.backward(*vectors)
        done=torch.cuda.Event()
        done.record()
    done.synchronize()
    for key in ('sq_vec','sk_vec','v','rtau'):
        torch.testing.assert_close(actual[key],expected[key],rtol=1e-4,atol=1e-4)
    # Allocator ownership after enqueue: drop all metadata owners while the
    # consumer stream is delayed and encourage reuse on the producer stream.
    sizes=(device.storage.numel(), compact.tau_heads.numel())
    del reference
    with torch.cuda.stream(consume):
        torch.cuda._sleep(50_000_000)
        actual=compact.backward(*vectors)
        done.record()
    del compact,device
    gc.collect()
    with torch.cuda.stream(construct):
        scratch=[torch.zeros(sizes[0],device='cuda',dtype=torch.uint8),
                 torch.zeros(sizes[1],device='cuda',dtype=torch.int64)]
    done.synchronize()
    for key in ('sq_vec','sk_vec','v','rtau'):
        torch.testing.assert_close(actual[key],expected[key],rtol=1e-4,atol=1e-4)

@pytest.mark.parametrize('pattern', ['rand512', 'rand8', 'same', 'period16'])
@pytest.mark.parametrize('chunk', [16, 32, 64])
@pytest.mark.parametrize('tau', [-2., 0., 16.])
@pytest.mark.parametrize('hard', ['all', 'alt', 'none'])
def test_compact_host_exact_layout(pattern, chunk, tau, hard):
    check_cpu_case(pattern, 65, tau, chunk, 'ragged', hard)


@pytest.mark.parametrize('pattern', ['rand512', 'rand8', 'same'])
@pytest.mark.parametrize('chunk', [16, 64])
@pytest.mark.parametrize('hard', ['all', 'alt', 'none'])
def test_compact_gpu_matches_classic(pattern, chunk, hard):
    check_gpu_case(pattern, 256, chunk, 'ragged', hard, b=2, h=2)


@pytest.mark.parametrize('r,dv', [(16, 32), (32, 64)])
@pytest.mark.parametrize('precision', ['bf16', 'tf32x3'])
def test_compact_per_row_tau_and_vector_geometry(r, dv, precision):
    check_cpu_case('rand8', 65, .5, 64, 'ragged', 'alt', per_row_tau=True)
    check_gpu_case('rand8', 65, 64, 'ragged', 'alt', b=2, h=3, r=r, dv=dv,
                   precision=precision, per_row_tau=True)
