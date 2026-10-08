"""First output/readout milestone, independent of embedding interpolation."""
import math
from pathlib import Path
import subprocess

import pytest
import torch
import torch.nn.functional as F
import cu_flash_dism

from flash_dism.forward import forward_core
from flash_dism.reference.dism_v3_ref import dism_ref
from test_summary import make_inputs

pytestmark = []  # Production approximation is fixed to tanh.


def inputs_for(n, mode='mixed', direction='mixed'):
    q, k, lq, lk, iq, ik, selected, hard, tau = make_inputs(n, mode, direction)
    b, _, h, _ = q.shape
    sq, sk = [(F.silu(torch.randn(b,n,h,cu_flash_dism.forward_readout_dim(),device='cuda')) * .4).bfloat16()
              for _ in range(2)]
    v = torch.randn(b,n,h,cu_flash_dism.forward_head_dim(),device='cuda').bfloat16()
    return q,k,sq,sk,v,lq,lk,iq,ik,selected,hard,tau


def reference(inputs):
    q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau = inputs
    return dism_ref(q.double(),k.double(),sq.double(),sk.double(),lq.double(),lk.double(),
                    iq,ik,direction,hard,v.double(),tau.double()).float()


@pytest.mark.parametrize('n', [256,512])
@pytest.mark.parametrize('mode', ['soft', 'mixed', 'hard'])
@pytest.mark.parametrize('direction', ['query', 'key'])
@pytest.mark.parametrize('fp32_output', [False, True])
def test_forward_oracle(n, mode, direction, fp32_output):
    inputs = inputs_for(n, mode, direction)
    output, lse2 = forward_core(*inputs, ctas=1, fp32_output=fp32_output)
    expected = reference(inputs)
    assert output.dtype == (torch.float32 if fp32_output else torch.bfloat16)
    assert lse2.dtype == torch.float32
    output = output.float()
    assert torch.isfinite(output).all() and torch.isfinite(lse2).all()
    torch.testing.assert_close(output, expected, atol=.008, rtol=.025)
    error = (output - expected).norm() / expected.norm().clamp_min(1e-8)
    assert error < .01, error.item()


def test_forward_signed_readout_and_denominator():
    inputs = list(inputs_for(512))
    original, norm = forward_core(*inputs, ctas=1)
    inputs[2] = -inputs[2]
    opposite, opposite_norm = forward_core(*inputs, ctas=1)
    torch.testing.assert_close(opposite, -original, atol=0, rtol=0)
    torch.testing.assert_close(opposite_norm, norm, atol=0, rtol=0)
    inputs[2].zero_()
    zero, zero_norm = forward_core(*inputs, ctas=1)
    assert torch.count_nonzero(zero) == 0
    torch.testing.assert_close(zero_norm, norm, atol=0, rtol=0)


def test_forward_fallback():
    inputs = list(inputs_for(256, 'hard'))
    inputs[7].fill_(1)
    inputs[8].fill_(2)
    output, lse2 = forward_core(*inputs, ctas=1)
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(lse2) == 0


def test_forward_repeated_tasks():
    inputs = inputs_for(1024)
    expected, norm = forward_core(*inputs, ctas=1)
    for ctas in (1, 2, 0, 1):
        output, lse2 = forward_core(*inputs, ctas=ctas)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.testing.assert_close(lse2, norm, atol=0, rtol=0)


def test_forward_codegen():
    path = Path(__file__).resolve().parents[1] / f'build/object/r32_d64_v64/forward.cu.dev.sm120a.o'
    sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '-sass', str(path)], text=True)
    assert 'CALL' not in sass
    assert sass.count('USETMAXREG.DEALLOC') == (2 if cu_flash_dism.fp32_enabled() else 1)
    assert sass.count('USETMAXREG.TRY_ALLOC') == (2 if cu_flash_dism.fp32_enabled() else 1)


@pytest.mark.parametrize('prefix', [7, 32, 65, 127])
def test_forward_causal_values(prefix):
    inputs = list(inputs_for(256, 'soft'))
    expected, _ = forward_core(*inputs, ctas=1)
    inputs[4][:, prefix:] = 100
    actual, _ = forward_core(*inputs, ctas=1)
    torch.testing.assert_close(actual[:, :prefix], expected[:, :prefix], atol=0, rtol=0)


@pytest.mark.parametrize('tau', [0., .1, math.log(cu_flash_dism.summary_key_dim())])
def test_forward_long_chain(tau):
    inputs = list(inputs_for(512, 'hard'))
    inputs[7].zero_()
    inputs[8].zero_()
    inputs[11].fill_(tau)
    actual, norm = forward_core(*inputs, ctas=1)
    actual = actual.float()
    expected = reference(inputs)
    torch.testing.assert_close(actual, expected, atol=.008, rtol=.025)
    assert (actual-expected).norm() / expected.norm().clamp_min(1e-8) < .01
    # All-match hard recurrence: W[i,j]=w[j] for j<=i. Includes fallback0.
    w = torch.zeros(513, dtype=torch.float64, device='cuda')
    state = torch.full((), -torch.inf, dtype=torch.float64, device='cuda')
    for j in range(512):
        state = torch.logaddexp(state, state.new_zeros(())) + tau
        w[j+1] = state
    expected_norm = torch.logcumsumexp(w,0)[1:] / math.log(2)
    torch.testing.assert_close(norm.double(), expected_norm.expand_as(norm), atol=.03, rtol=0)
