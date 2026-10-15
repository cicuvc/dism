"""Measure the tau gradient tightly, per row type.

`gradient_acceptance.TAU` allows 50% relative L2 and cosine 0.95, far looser than
the 3% used for the vector gradients, and tau has already produced one defect
(the multi-chunk planner route). The two row types deliver tau by different
mechanisms:

* soft rows: `recompute` builds the score as `raw - q_lse` and never touches the
  tau pointer; the backward (`backward_qk.cu`, which skips hard rows) accounts
  for tau by accumulating one unit per pair, matching the `LSE - tau` fold the
  interpolation performed with a detached tau.
* hard rows: the core adds `rtau` to the hard score and the hard backward takes
  the derivative of the recurrence.

This script compares the real core against FP64 `dism_ref` for all-soft,
all-hard and mixed rows, reporting relative L2 and cosine for tau and, as a
control, for the vector gradients.
"""
import argparse
import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / 'dism_v3' / 'python'), str(_root / 'nanochat'),
                str(_root / 'dism_v3' / 'tests')]

from flash_dism import dism_core                                        # noqa: E402
from flash_dism.hard_training import shutdown_hard_planners             # noqa: E402
from test_hard_training import inputs                                   # noqa: E402

INDEX = (2, 3, 4, 11)
NAMES = ('sq_vec', 'sk_vec', 'v', 'rtau')


def stats(actual, reference):
    x, y = actual.double().flatten(), reference.double().flatten()
    yn = y.norm().item()
    rel = (x - y).norm().item() / max(yn, 1e-30)
    cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item() if yn > 0 else float('nan')
    return rel, cos, yn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--scale', type=float, default=1.0,
                    help='multiply q and k; sharpens the soft scores and is a '
                         'conditioning probe for the soft-path tau gradient')
    args = ap.parse_args()
    for mode in ('soft', 'hard', 'mixed'):
        rows = []
        for seed in range(args.seeds):
            xs, layout = inputs(packed=False, mode=mode, pattern='random')
            if args.scale != 1.0:
                xs = list(xs)
                xs[0] = (xs[0] * args.scale).requires_grad_()
                xs[1] = (xs[1] * args.scale).requires_grad_()
                xs = tuple(xs)
            torch.manual_seed(1000 + seed)
            out = dism_core(*xs, layout=layout)
            do = (torch.randn_like(out) * .2).contiguous()
            got = torch.autograd.grad(out, [xs[i] for i in INDEX], do,
                                      allow_unused=True)
            refs = [x.detach().double().requires_grad_() if x.requires_grad else x
                    for x in xs]
            from test_hard_training import oracle
            expected = oracle(refs, layout)
            want = torch.autograd.grad(expected, [refs[i] for i in INDEX],
                                       do.double(), allow_unused=True)
            rows.append({n: (torch.zeros_like(xs[i]) if g is None else g, w)
                         for n, i, g, w in zip(NAMES, INDEX, got, want)})
        print(f'mode={mode}  (n={len(rows)} seeds)', flush=True)
        for name in NAMES:
            rels, coss, norms = zip(*[stats(*r[name]) for r in rows])
            print(f'  {name:7s} rel_l2 mean={sum(rels)/len(rels):8.3e} max={max(rels):8.3e}   '
                  f'cosine min={min(coss):.5f}   |ref| mean={sum(norms)/len(norms):.3e}',
                  flush=True)
    shutdown_hard_planners()


if __name__ == '__main__':
    main()
