"""Compare the fixed-true interpolation specialization against the two-sided path.

`direction_policy="true"` lets the forward skip the k_from_q interpolation and
the key-side LSE entirely (`kernels/interp_true.py::emb_fwd_true`). The general
two-sided path (`emb_fwd_wrapper` + `select_operands`) still exists. Both are
production implementations of the same soft operator, so pushing identical
tensors through both and comparing is a direct check with no hand-written
reference in the loop.

Also reports the ingredients separately, so a mismatch localises to the labels
(which the hard path uses), the interpolant, or the LSE.
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


def build(n=256, b=1, h=2, d=32, v_=16, seed=5):
    torch.manual_seed(seed)
    q, k = [(torch.randn(b, n, h, d, device='cuda') * .3).bfloat16() for _ in range(2)]
    eq, ek = [(torch.randn(h, v_, d, device='cuda') * .3).bfloat16() for _ in range(2)]
    sq, sk = [(torch.randn(b, n, h, 32, device='cuda') * .3).bfloat16() for _ in range(2)]
    v = (torch.randn(b, n, h, 64, device='cuda') * .3).bfloat16()
    return q, k, sq, sk, v, eq, ek


def main():
    beta = 1.0
    q, k, sq, sk, v, eq, ek = build()
    b, n, h, _ = q.shape
    direction = torch.ones((b, h), dtype=torch.bool, device='cuda')
    hard = torch.zeros((b, h), dtype=torch.bool, device='cuda')   # all soft
    for tau_value in (0.0, 0.5, -0.5):
        tau = torch.tensor([tau_value] * h, device='cuda')

        qfk_t, _, lk_t, lq_t, _, _, ik_t, iq_t = emb_fwd_true(
            q, k, eq, ek, beta, tau.detach())
        out_t, _, _ = cu.core_forward(
            (q, qfk_t, sq, sk, v, lq_t.transpose(1, 2).contiguous(),
             lk_t.transpose(1, 2).contiguous(), iq_t, ik_t, direction, hard, tau),
            None, 0, False, False, True, True)

        qfk_w, kfq_w, lk_w, lq_w, _, _, ik_w, iq_w = emb_fwd_wrapper(
            q, k, eq, ek, beta, tau.detach())
        qsk, ksk = ((qfk_w, kfq_w) if beta == 1.0 else (qfk_w * beta, kfq_w * beta))
        qv, kv = select_operands(q, k, qsk, ksk, direction)
        out_w, _, _ = cu.core_forward(
            (qv, kv, sq, sk, v, lq_w.transpose(1, 2).contiguous(),
             lk_w.transpose(1, 2).contiguous(), iq_w, ik_w, direction, hard, tau),
            None, 0, False, False, True, False)

        def d8(a, b_):
            return (a.float() - b_.float()).abs().max().item()

        print(f'tau={tau_value:+.1f}', flush=True)
        print(f'  labels   iq mismatches={int((iq_t != iq_w).sum())}/{iq_t.numel()} '
              f'  ik mismatches={int((ik_t != ik_w).sum())}/{ik_t.numel()}', flush=True)
        print(f'  q_from_k max|true-two-sided| = {d8(qfk_t, qfk_w):.3e}', flush=True)
        print(f'  q_lse    max|true-two-sided| = {d8(lq_t, lq_w):.3e}   '
              f'true={lq_t.float().mean().item():+.4f} two-sided={lq_w.float().mean().item():+.4f}',
              flush=True)
        print(f'  k_lse    max|true-two-sided| = {d8(lk_t, lk_w):.3e}', flush=True)
        print(f'  output   rel|true-two-sided| = '
              f'{((out_t.double()-out_w.double()).norm()/out_w.double().norm()).item():.3e}  '
              f'|true|={out_t.double().norm().item():.4e} '
              f'|two-sided|={out_w.double().norm().item():.4e}', flush=True)


if __name__ == '__main__':
    main()
