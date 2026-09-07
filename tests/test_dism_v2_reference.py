"""Oracle readiness tests; these do NOT test a new CUDA attention kernel.

Run from the repository root with the conda blkw Python interpreter.
Explicit masks below are debugging inputs, not the production RNG design.
"""

import itertools

import pytest
import torch

from dism_v2.dism_ref import interpolation_ref, voc_dism_ref


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_inputs(d, dv, n, batch=2, heads=2):
    generator = torch.Generator(device="cuda").manual_seed(20260907)

    def randn(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    return (
        randn((batch, heads, n, d)),
        randn((batch, heads, n, d)),
        randn((batch, heads, n, dv)),
        torch.linspace(-0.25, 0.75, heads, device="cuda"),
        randn((heads, 11, d)),
        randn((heads, 11, d)),
    )


@pytest.mark.parametrize("d,dv", itertools.product((32, 64, 128), repeat=2))
@pytest.mark.parametrize("direction", ("q_from_k", "k_from_q"))
@pytest.mark.parametrize("mode", ("soft", "hard", "mixed"))
@torch.no_grad()
def test_reference_inputs_and_direction(d, dv, direction, mode):
    args = make_inputs(d, dv, 17)
    q, k, v, tau, qvoc, kvoc = args
    scale = d**-0.5
    interp = interpolation_ref(q, k, qvoc, kvoc, scale)
    rows = torch.arange(q.shape[2], device="cuda")
    mask = ((rows % 2 == 0) if mode == "mixed" else
            torch.full_like(rows, mode == "hard", dtype=torch.bool)).view(1, 1, -1, 1)
    generator = torch.Generator(device="cuda").manual_seed(19)
    before = generator.get_state().clone()
    out, aux = voc_dism_ref(
        *args, hard_prob=0.5, direction=direction, sm_scale=scale,
        interpolation=interp, hard_mask=mask, generator=generator, return_aux=True,
    )
    # Fixed direction plus explicit mask must not consume fresh random values.
    assert torch.equal(before, generator.get_state())
    assert out.shape == v.shape
    assert torch.isfinite(out).all()
    if direction == "q_from_k":
        expected = q.float() @ interp.q_from_k.transpose(-1, -2) * scale - interp.q_lse[..., None]
    else:
        expected = interp.k_from_q @ k.float().transpose(-1, -2) * scale - interp.k_lse[..., None, :]
    expected = expected + tau[None, :, None, None]
    torch.testing.assert_close(aux["soft_log_m"], expected)
    matched = interp.q_index[..., :, None] == interp.k_index[..., None, :]
    hard_score = torch.where(matched, tau[None, :, None, None], -torch.inf)
    torch.testing.assert_close(aux["log_m"], torch.where(mask, hard_score, expected))
    causal = torch.ones(17, 17, device="cuda", dtype=torch.bool).tril()
    assert torch.isneginf(aux["scores"][..., ~causal]).all()


@pytest.mark.parametrize("n", (1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129))
@pytest.mark.parametrize("direction", ("q_from_k", "k_from_q"))
@torch.no_grad()
def test_reference_tail_and_fallback(n, direction):
    args = make_inputs(32, 64, n, batch=1, heads=1)
    q, k, v, _, qvoc, kvoc = args
    interp = interpolation_ref(q, k, qvoc, kvoc, 32**-0.5)
    # Construct legal but universally different labels to isolate zero mapping.
    from dataclasses import replace

    interp = replace(interp, q_index=torch.zeros_like(interp.q_index),
                     k_index=torch.ones_like(interp.k_index))
    out = voc_dism_ref(
        *args, hard_prob=1.0, direction=direction, sm_scale=32**-0.5,
        interpolation=interp, hard_mask=torch.tensor(True, device="cuda"),
    )
    assert out.shape == v.shape
    assert torch.equal(out, torch.zeros_like(out))
