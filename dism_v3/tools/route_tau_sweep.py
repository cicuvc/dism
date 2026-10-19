"""Sweep symbol-block length and compare the two dispatch routes on rtau.

Diagnostic for the mismatch found by dism_v3/tests/test_hard_route_consistency.py:
the original BF16 event route had percent-scale tau error on long matching
streams while the quadratic fallback matched the FP64 oracle to about 1e-6.

Labels here are alternating blocks of `block` equal symbols. This changes event
topology; it does not cap the common suffix (the diagonal still matches all N
tokens). Planner chunks count events, not tokens. Everything else is fixed.
Use `detailed` for all six gradient components against both quadratic and FP64.
"""
import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[1].parent
sys.path[:0] = [str(_root / 'dism_v3' / 'python'), str(_root / 'nanochat'),
                str(_root / 'dism_v3' / 'tests')]

from flash_dism import dism_core                                      # noqa: E402
from flash_dism.hard_training import AsyncHardPlanner, shutdown_hard_planners  # noqa: E402
from test_hard_training import oracle                                 # noqa: E402

INDEX = (0, 1, 2, 3, 4, 11)
NAMES = ('q_vec', 'k_vec', 'sq_vec', 'sk_vec', 'v', 'rtau')


def make(block, n=512, b=2, h=2, tau=0.5, seed=918):
    torch.manual_seed(seed)
    q, k, sq, sk, v = [(torch.randn(b, n, h, c, device='cuda') * .2).bfloat16().requires_grad_()
                       for c in (32, 32, 32, 32, 64)]
    lq, lk = [torch.full((b, n, h), 2., device='cuda', requires_grad=True) for _ in range(2)]
    if block is None:
        iq = torch.randint(0, 2, (b, h, n), device='cuda', dtype=torch.int32)
        ik = torch.randint(0, 2, (b, h, n), device='cuda', dtype=torch.int32)
    else:
        pattern = (torch.arange(n, device='cuda') // block) % 2
        iq = pattern.to(torch.int32).expand(b, h, n).contiguous()
        ik = iq.clone()
    hard = torch.ones((b, h), dtype=torch.bool, device='cuda')
    direction = torch.zeros((b, h), dtype=torch.bool, device='cuda')
    t = torch.full((h,), tau, device='cuda', dtype=torch.float32, requires_grad=True)
    return (q, k, sq, sk, v, lq, lk, iq, ik, direction, hard, t)


def grads(xs, policy, do, chunk_size=64, precision='bf16', compact=True):
    with AsyncHardPlanner(backward_policy=policy, workers=2, chunk_size=chunk_size,
                          mma_precision=precision, compact=compact) as planner:
        out = dism_core(*xs, hard_planner=planner)
        g = torch.autograd.grad(out, [xs[i] for i in INDEX], do, allow_unused=True)
    return {n: (torch.zeros_like(xs[i]) if gi is None else gi)
            for n, i, gi in zip(NAMES, INDEX, g)}


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def metrics(actual, reference):
    """Signed bias is mean(actual-reference); undefined zero ratios stay N/A."""
    a, b = actual.detach().double().flatten(), reference.detach().double().flatten()
    an, bn = a.norm().item(), b.norm().item()
    return dict(cosine=(float(a @ b)/(an*bn) if an and bn else None),
                relative_l2=((a-b).norm().item()/bn if bn else None),
                norm_ratio=(an/bn if bn else None),
                signed_bias=(a-b).mean().item(), max_abs=(a-b).abs().max().item())


def detailed():
    """One fixed hard row, two heads, a 512-token matching suffix; <100 MiB."""
    xs = make(512, b=1)
    torch.manual_seed(77)
    do = (torch.randn_like(xs[4])*.2).contiguous()
    refs = [x.detach().double().requires_grad_() if x.requires_grad else x for x in xs]
    expected = oracle(refs, None)
    og = torch.autograd.grad(expected, [refs[i] for i in INDEX], do.double(), allow_unused=True)
    truth = {name: torch.zeros_like(xs[i]) if g is None else g
             for name, i, g in zip(NAMES, INDEX, og)}
    quadratic = grads(xs, 'quadratic', do)
    print('route component comparison cosine relative_l2 norm_ratio signed_bias max_abs')
    for precision, compact in (('bf16', True), ('bf16', False), ('tf32x3', True)):
        actual = grads(xs, 'event', do, precision=precision, compact=compact)
        for comparison, reference in (('quadratic', quadratic), ('FP64', truth)):
            for name in NAMES:
                values = metrics(actual[name], reference[name])
                formatted = ' '.join('N/A' if v is None else f'{v:.9e}' for v in values.values())
                print(f'{precision}/compact={compact} {name} {comparison} {formatted}')
    for name in NAMES:
        values = metrics(quadratic[name], truth[name])
        formatted = ' '.join('N/A' if v is None else f'{v:.9e}' for v in values.values())
        print(f'quadratic {name} FP64 {formatted}')
    with AsyncHardPlanner(backward_policy='event', workers=2) as planner:
        out = dism_core(*xs, hard_planner=planner)
        print('forward vs FP64', metrics(out, expected))
        ticket = planner.submit(xs[7], xs[8], xs[10], xs[11])
        prepared = ticket.future.result()
        offsets = prepared.host.multi['offsets']
        spans = offsets[1:]-offsets[:-1]
        print('plan statistics', prepared.host.statistics)
        print('max chunks per stream', int(spans.max()) if spans.numel() else 1)
    print('peak allocated MiB', torch.cuda.max_memory_allocated()/2**20)
    shutdown_hard_planners()


def main():
    torch.set_num_threads(1)
    if len(sys.argv) > 1 and sys.argv[1] == 'detailed':
        detailed()
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'chunk':
        print(f'{"chunk":>6} {"block":>6} {"block/chunk":>11} {"tau event-vs-quad":>18}')
        for chunk in (16, 32, 64):
            for block in (16, 32, 64, 128, 256, 512):
                xs = make(block)
                torch.manual_seed(77)
                with AsyncHardPlanner(backward_policy='event', workers=2,
                                      chunk_size=chunk) as planner:
                    shape = dism_core(*xs, hard_planner=planner)
                do = (torch.randn_like(shape) * .2).contiguous()
                ev = grads(xs, 'event', do, chunk)
                qu = grads(xs, 'quadratic', do, chunk)
                print(f'{chunk:>6} {block:>6} {block / chunk:>10.1f} '
                      f'{rel(ev["rtau"], qu["rtau"]):18.3e}')
                del xs, ev, qu
        shutdown_hard_planners()
        return
    print(f'{"block":>6} {"tau":>10} {"sq/sk/v max":>12} {"tau oracle":>11} '
          f'{"quad oracle":>12}')
    for block in (None, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        xs = make(block)
        torch.manual_seed(77)
        with AsyncHardPlanner(backward_policy='event', workers=2) as planner:
            shape = dism_core(*xs, hard_planner=planner)
        do = (torch.randn_like(shape) * .2).contiguous()
        ev, qu = grads(xs, 'event', do), grads(xs, 'quadratic', do)
        refs = [x.detach().double().requires_grad_() if x.requires_grad else x for x in xs]
        expected = oracle(refs, None)
        og = torch.autograd.grad(expected, [refs[i] for i in INDEX], do.double(),
                                 allow_unused=True)
        truth = {n: (torch.zeros_like(xs[i]) if g is None else g)
                 for n, i, g in zip(NAMES, INDEX, og)}
        vec = max(rel(ev[n], qu[n]) for n in ('sq_vec', 'sk_vec', 'v'))
        torch_ok = all(torch.isfinite(ev[n]).all() for n in NAMES)
        print(f'{str(block):>6} {rel(ev["rtau"], qu["rtau"]):10.3e} {vec:12.3e} '
              f'{rel(ev["rtau"], truth["rtau"]):11.3e} '
              f'{rel(qu["rtau"], truth["rtau"]):12.3e} finite={torch_ok}')
        del xs, ev, qu, refs, expected, og, truth
    shutdown_hard_planners()


if __name__ == '__main__':
    main()
