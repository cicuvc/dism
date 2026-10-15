"""Three-way check of the core's direction_true path.

The existing oracle test drives `dism_core` (lq/lk as inputs, raw q/k, no
vocabulary interpolation) with the default `direction_true=False`, and compares
against `dism_ref`. `_voc_forward_impl` with a fixed-true policy instead calls
`cu.core_forward(..., direction_true=True)`. This script compares, on identical
operands and identical LSEs:

  A  core_forward(direction_true=True)
  B  core_forward(direction_true=False)  with operands selected the same way
  C  dism_ref, the documented operator

so a mismatch localises to the core's true path, the core's general path, or the
shared convention.
"""
import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / 'dism_v3' / 'python'), str(_root / 'nanochat')]

import cu_flash_dism as cu                                            # noqa: E402
from flash_dism.kernels.interp_true import emb_fwd_true               # noqa: E402
from flash_dism.emb_kernel import emb_fwd_wrapper                     # noqa: E402
from flash_dism.kernels.voc_glue import select_operands               # noqa: E402
from flash_dism.reference.dism_v3_ref import dism_ref                 # noqa: E402


def build(n=256, b=1, h=2, d=32, v_=16, seed=5):
    torch.manual_seed(seed)
    q, k = [(torch.randn(b, n, h, d, device='cuda') * .3).bfloat16() for _ in range(2)]
    eq, ek = [(torch.randn(h, v_, d, device='cuda') * .3).bfloat16() for _ in range(2)]
    sq, sk = [(torch.randn(b, n, h, 32, device='cuda') * .3).bfloat16() for _ in range(2)]
    v = (torch.randn(b, n, h, 64, device='cuda') * .3).bfloat16()
    return q, k, sq, sk, v, eq, ek


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def main():
    beta = 1.0
    q, k, sq, sk, v, eq, ek = build()
    b, n, h, _ = q.shape
    direction = torch.ones((b, h), dtype=torch.bool, device='cuda')
    hard = torch.zeros((b, h), dtype=torch.bool, device='cuda')
    for tau_value in (0.0, 0.5):
        tau = torch.tensor([tau_value] * h, device='cuda')
        qfk, _, lk_k, lq_k, _, _, ik_k, iq_k = emb_fwd_true(
            q, k, eq, ek, beta, tau.detach())
        qfk_w, kfq_w, lk_w, lq_w, _, _, ik_w, iq_w = emb_fwd_wrapper(
            q, k, eq, ek, beta, tau.detach())

        a, _, _ = cu.core_forward(
            (q, qfk, sq, sk, v, lq_k.transpose(1, 2).contiguous(),
             lk_k.transpose(1, 2).contiguous(), iq_k, ik_k, direction, hard, tau),
            None, 0, False, False, True, True)

        qv, kv = select_operands(q, k, qfk_w, kfq_w, direction)
        bb, _, _ = cu.core_forward(
            (qv, kv, sq, sk, v, lq_w.transpose(1, 2).contiguous(),
             lk_w.transpose(1, 2).contiguous(), iq_w, ik_w, direction, hard, tau),
            None, 0, False, False, True, False)

        # drop the tau fold from the LSE to get the natural-log convention
        lq_nat = lq_k.double() + tau_value
        lk_nat = lk_k.double() + tau_value
        c, _, _, _ = None, None, None, None
        c = dism_ref(q.double(), qfk.double(), sq.double(), sk.double(),
                     lq_nat.transpose(1, 2), lk_nat.transpose(1, 2), iq_k, ik_k,
                     direction, hard[:, :, None].expand_as(iq_k), v.double(),
                     tau.double())

        print(f'tau={tau_value:+.1f}   norms: A={a.double().norm():.4e} '
              f'B={bb.double().norm():.4e} C={c.norm():.4e}', flush=True)
        print(f'  A(true)  vs B(false) rel = {rel(a, bb):.3e}', flush=True)
        print(f'  A(true)  vs C(ref)   rel = {rel(a, c):.3e}', flush=True)
        print(f'  B(false) vs C(ref)   rel = {rel(bb, c):.3e}', flush=True)


if __name__ == '__main__':
    main()
