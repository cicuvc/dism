"""Explicit fused-launch and compensated BF16 boundary regression coverage."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from flash_dism.inference.training_planner import NativeTrainingPlanner
from flash_dism.kernels.hard_backward_scan import HardPrefillBackward
from flash_dism.kernels.hard_backward_fused import (
    FusedHardPrefillBackward, CompactFusedHardPrefillBackward)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_hard_prefill_training import dense_oracle, assert_gradient


@pytest.mark.parametrize('c', [16, 32, 64])
@pytest.mark.parametrize('warps', [2, 4, 8])
@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32, 'mixed'])
def test_explicit_fused_launch_matches_fp64(c, warps, compact, dtype):
    n, r, dv = 129, 32, 64
    labels = np.zeros((2, 1, 1, n), np.int32)
    host = NativeTrainingPlanner(2).prepare(labels, np.array([.5]),
                                            chunk_size=c, compact=compact)
    device = host.upload()
    # General FP32 vectors use the explicit accurate mode; mixed storage
    # exercises the BF16-mode fallback using exactly representable BF16 values.
    precision = 'tf32x3' if dtype == torch.float32 else 'bf16'
    if compact:
        plan = CompactFusedHardPrefillBackward.from_compact(
            device, mma_precision=precision, num_warps=warps)
    else:
        source = HardPrefillBackward.from_native(device, mma_precision=precision)
        plan = FusedHardPrefillBackward.from_plan(
            source, host.data['offsets'].numpy(),
            host_lengths=host.data['lengths'].numpy(), num_warps=warps)
    torch.manual_seed(321)
    storage_dtype = torch.bfloat16 if dtype == 'mixed' else dtype
    xs = [(torch.randn((n, d), device='cuda')*.2).to(storage_dtype)
          for d in (r, r, dv, dv)]
    if dtype == 'mixed':
        xs[-1] = xs[-1].float()
    actual = plan.backward(*xs)
    cpu = [x.cpu().double().numpy() for x in xs]
    _, expected = dense_oracle(labels[0, 0, 0], labels[1, 0, 0],
                               *cpu[:3], .5, cpu[3], direction=True)
    for name, reference in zip(('sq_vec', 'sk_vec', 'v', 'rtau'), expected):
        reference = torch.as_tensor(reference)
        if name == 'rtau':
            reference = reference.reshape(1)  # Native plan retains its one tau group.
        assert_gradient(actual[name].cpu(), reference, name)
    torch.testing.assert_close(actual['rtau'].cpu().double().reshape(()),
                               torch.as_tensor(expected[3]), rtol=2e-4, atol=2e-6)
