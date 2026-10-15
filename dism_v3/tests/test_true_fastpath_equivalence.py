"""The fixed-true interpolation specialization must equal the two-sided path.

`direction_policy="true"` lets the forward skip the `k_from_q` interpolation and
the key-side LSE (`kernels/interp_true.py::emb_fwd_true`); every production run
uses it. The general two-sided path (`emb_fwd_wrapper` + `select_operands`) still
exists, and both are implementations of the same soft operator.

Nothing covered the *composition* of vocabulary interpolation with the core:
`test_autograd_reference` drives `dism_core` with `lq`/`lk` as constant inputs and
raw `q`/`k`, so it never exercises the interpolation, and `test_interp_true.py`
checks the interpolation alone. These tests close that gap by pushing identical
tensors through both real paths and through the documented operator.

Convention this pins: the interpolation stores `LSE_beta - tau` (the detached tau
is folded into the LSE) and the core does not add `rtau` again on soft rows, so
undoing the fold must recover `dism_ref`. A future change to either half is
caught here.
"""
import pytest
import torch

import cu_flash_dism as cu
from flash_dism.emb_kernel import emb_fwd_wrapper
from flash_dism.kernels.interp_true import emb_fwd_true
from flash_dism.kernels.voc_glue import select_operands
from flash_dism.reference.dism_v3_ref import dism_ref


def _build(n=256, b=1, h=2, d=32, v_=16, seed=5):
    torch.manual_seed(seed)
    q, k = [(torch.randn(b, n, h, d, device='cuda') * .3).bfloat16() for _ in range(2)]
    eq, ek = [(torch.randn(h, v_, d, device='cuda') * .3).bfloat16() for _ in range(2)]
    sq, sk = [(torch.randn(b, n, h, 32, device='cuda') * .3).bfloat16() for _ in range(2)]
    v = (torch.randn(b, n, h, 64, device='cuda') * .3).bfloat16()
    return q, k, sq, sk, v, eq, ek


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize('tau_value', [0.0, 0.5, -0.5])
def test_fixed_true_matches_two_sided_and_reference(tau_value):
    beta = 1.0
    q, k, sq, sk, v, eq, ek = _build()
    b, n, h, _ = q.shape
    direction = torch.ones((b, h), dtype=torch.bool, device='cuda')
    hard = torch.zeros((b, h), dtype=torch.bool, device='cuda')   # all soft rows
    tau = torch.tensor([tau_value] * h, device='cuda')

    qfk_t, _, lk_t, lq_t, _, _, ik_t, iq_t = emb_fwd_true(
        q, k, eq, ek, beta, tau.detach())
    true_out, _, _ = cu.core_forward(
        (q, qfk_t, sq, sk, v, lq_t.transpose(1, 2).contiguous(),
         lk_t.transpose(1, 2).contiguous(), iq_t, ik_t, direction, hard, tau),
        None, 0, False, False, True, True)

    qfk_w, kfq_w, lk_w, lq_w, _, _, ik_w, iq_w = emb_fwd_wrapper(
        q, k, eq, ek, beta, tau.detach())
    qv, kv = select_operands(q, k, qfk_w, kfq_w, direction)
    general_out, _, _ = cu.core_forward(
        (qv, kv, sq, sk, v, lq_w.transpose(1, 2).contiguous(),
         lk_w.transpose(1, 2).contiguous(), iq_w, ik_w, direction, hard, tau),
        None, 0, False, False, True, False)

    ref_out = dism_ref(
        q.double(), qfk_t.double(), sq.double(), sk.double(),
        (lq_t.double() + tau_value).transpose(1, 2),   # undo the tau fold
        (lk_t.double() + tau_value).transpose(1, 2),
        iq_t, ik_t, direction, hard[:, :, None].expand_as(iq_t), v.double(),
        tau.double())

    # Labels drive the hard path, so a difference here would change hard outputs.
    assert int((iq_t != iq_w).sum()) == 0
    assert int((ik_t != ik_w).sum()) == 0
    # Same interpolant, and the LSEs agree to FP32 rounding.
    torch.testing.assert_close(qfk_t.float(), qfk_w.float(), rtol=0, atol=0)
    torch.testing.assert_close(lq_t, lq_w, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lk_t, lk_w, rtol=1e-5, atol=1e-5)
    # Both core paths agree with each other. Compare as a whole tensor: the
    # outputs are BF16, so a single element can legitimately differ by one ULP
    # (6.1e-5 near magnitude 0.014) purely from operand-materialisation order.
    assert _rel(true_out, general_out) < 1e-4
    assert _rel(true_out, ref_out) < 5e-3
    assert _rel(general_out, ref_out) < 5e-3
