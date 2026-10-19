"""Event multiplicity, padding and fallback checks for the one-pair readout."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_hard_prefill_training import assert_gradient
from hard_prefill_backward_fused import FusedHardPrefillBackward
from hard_prefill_backward_triton import HardPrefillBackward, pack_program
from hard_prefill_grad_reference import Program


@pytest.mark.parametrize('r,dv', [(16, 32), (32, 64)])
@pytest.mark.parametrize('c', [16, 64])
@pytest.mark.parametrize('dtype', ['bf16', 'fp32', 'mixed'])
def test_single_pair_event_multiplicity_and_fallback(r, dv, c, dtype):
    # Two streams hit the same token with different event derivatives. Include
    # query-before-key and key-only streams, which must bypass the fast branch.
    p = Program(np.array([0, 2, 4, 6, 8, 9]),
                np.array([0, -1, 0, -1, -2, 1, 1, 2, 2]),
                np.array([0., 1., 0., 1., 1., .5, 0., .5, 0.]),
                np.array([.3, .7, .2, .6, .9, .4, .5, .8, .7]),
                np.array([256., -255.5, -128., 128.25, 5., -2., 3., 7., 0.]),
                np.zeros(3), np.zeros(3))
    packed = pack_program(p, c)
    source = HardPrefillBackward([0]*3, [1]*3, .5, chunk_size=c,
                                mma_precision='tf32x3' if dtype == 'fp32' else 'bf16')
    source.program, source.packed, source.streams = p, packed, 5
    source.data = {k: torch.as_tensor(v, device='cuda') for k, v in packed.items()}
    # Nonzero padding catches accidental extra pairs without adding NaNs to
    # the reference. Ordinary plans can use a physical width64 for length2.
    padded = np.arange(c)[None, :] >= packed['lengths'][:, None]
    for key in ('weights', 'dlog'):
        source.data[key].view(-1, c)[torch.as_tensor(padded, device='cuda')] = 1000.
    plan = FusedHardPrefillBackward.from_plan(source, packed['offsets'])
    torch.manual_seed(817)
    xs = [(torch.randn((3, d), device='cuda')*.2).to(
        torch.float32 if dtype == 'fp32' else torch.bfloat16) for d in (r, r, dv, dv)]
    if dtype == 'mixed':
        xs[-1] = xs[-1].float()
    expected = p.evaluate(*[x.float().cpu().double().numpy() for x in xs])[1]
    actual = plan.backward(*xs)
    for name, ref in zip(('sq_vec', 'sk_vec', 'v', 'rtau'), expected):
        assert_gradient(actual[name].cpu(), torch.as_tensor(ref), name)
    torch.testing.assert_close(actual['rtau'].cpu().double(),
                               torch.as_tensor(expected[3]), rtol=2e-4, atol=2e-6)
    assert actual['sq_vec'][1:].count_nonzero() == 0
    assert actual['sk_vec'][1:].count_nonzero() == 0
    assert actual['v'][1:].count_nonzero() == 0
