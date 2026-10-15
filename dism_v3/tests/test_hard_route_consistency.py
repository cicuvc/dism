"""The hard backward must not depend on which dispatch route the planner took.

Both routes share the same forward: `_voc_forward_impl` always computes the hard
output with `forward_hard`. Only the backward differs:

* `policy='event'` uses the planner's compiled event program
  (`CompactFusedHardPrefillBackward` / `FusedHardPrefillBackward`).
* `policy='quadratic'` uses `backward_hard_recompute`, which recomputes the
  forward state and calls the explicit quadratic `cu.hard_backward`.

`test_autograd_reference` pins the event route to an FP64 oracle and
`test_fallback_oracle` pins the quadratic route to the same oracle, but each only
to `gradient_acceptance` tolerances (3% relative L2 for vectors, 50% for tau).
Neither compares the two routes with each other, so a route-dependent bias inside
those tolerances passes unnoticed. In a live run the choice is timing dependent:
the last DISM layers always fall back, so such a bias would apply selectively to
exactly those layers.

This test compares the routes directly, at a tolerance far tighter than the
oracle gates, and prints the numbers so the failing side can be identified.
"""
import pytest
import torch

from flash_dism import dism_core
from flash_dism.hard_training import AsyncHardPlanner, shutdown_hard_planners
from test_hard_training import inputs, oracle


NAMES = ('sq_vec', 'sk_vec', 'v', 'rtau')
INDEX = (2, 3, 4, 11)
# Tight enough to see a route bias that the oracle gates would hide. The
# reference and both kernels accumulate in FP32/BF16, so this is not bitwise
# exact, but a shared-input mismatch above this is a real discrepancy.
REL_L2 = 2e-3
MAX_ABS = 2e-3


def _relative(a, b):
    denom = b.double().norm().clamp_min(1e-30)
    return (a.double() - b.double()).norm() / denom


def _route_grads(xs, layout, policy, do, chunk_size=64):
    with AsyncHardPlanner(backward_policy=policy, workers=2, chunk_size=chunk_size) as planner:
        out = dism_core(*xs, layout=layout, hard_planner=planner)
        grads = torch.autograd.grad(out, [xs[i] for i in INDEX], do,
                                    allow_unused=True)
    return {name: (torch.zeros_like(xs[i]) if g is None else g)
            for name, i, g in zip(NAMES, INDEX, grads)}


def _compare(packed, mode, pattern, block=None, chunk_size=64, tight=True,
             tau_value=None):
    xs, layout = inputs(packed, mode, pattern)
    if block is not None:
        # Same-length runs of one symbol, so the longest common suffix is the
        # block length and streams can span several planner chunks.
        n = xs[0].shape[1]
        pattern_ids = ((torch.arange(n, device='cuda') // block) % 2).to(torch.int32)
        xs = list(xs)
        xs[7] = pattern_ids.expand_as(xs[7]).contiguous()
        xs[8] = pattern_ids.expand_as(xs[8]).contiguous()
        xs = tuple(xs)
    if tau_value is not None:
        # The reproducer needs both heads on the same side of tau=0; the
        # default inputs() pair is (+0.5, -0.5) and does not reproduce at
        # block=512 with chunk=64.
        xs = list(xs)
        xs[11] = torch.full_like(xs[11], tau_value).requires_grad_()
        xs = tuple(xs)
    torch.manual_seed(77)
    with AsyncHardPlanner(backward_policy='event', workers=2) as planner:
        reference_shape = dism_core(*xs, layout=layout, hard_planner=planner)
    do = (torch.randn_like(reference_shape) * .2).contiguous()

    event = _route_grads(xs, layout, 'event', do, chunk_size)
    quadratic = _route_grads(xs, layout, 'quadratic', do, chunk_size)

    refs = [x.detach().double().requires_grad_() if x.requires_grad else x for x in xs]
    expected = oracle(refs, layout)
    oracle_grads = torch.autograd.grad(
        expected, [refs[i] for i in INDEX], do.double(), allow_unused=True)
    truth = {name: (torch.zeros_like(xs[i]) if g is None else g)
             for name, i, g in zip(NAMES, INDEX, oracle_grads)}

    rows = []
    for name in NAMES:
        rows.append(
            f'{name:7s} event-vs-quadratic rel_l2={_relative(event[name], quadratic[name]):9.2e} '
            f'max_abs={(event[name] - quadratic[name]).abs().max().item():9.2e} | '
            f'event-vs-oracle {_relative(event[name], truth[name]):9.2e} | '
            f'quad-vs-oracle {_relative(quadratic[name], truth[name]):9.2e}')
    print(f'\n[packed={packed} mode={mode} pattern={pattern} block={block} '
          f'chunk={chunk_size}]')
    print('\n'.join('  ' + r for r in rows), flush=True)

    if tight:
        for name in NAMES:
            torch.testing.assert_close(
                event[name].double(), quadratic[name].double(),
                rtol=REL_L2, atol=MAX_ABS, msg=f'{name} differs between routes')
    return event, quadratic, truth


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('mode', ['hard', 'mixed'])
def test_dispatch_route_does_not_change_the_hard_gradient(packed, mode):
    try:
        _compare(packed, mode, 'random')
        _compare(packed, mode, 'random', block=64)      # one planner chunk
    finally:
        shutdown_hard_planners()


@pytest.mark.parametrize('chunk_size,block', [(64, 512), (64, 256), (32, 256), (16, 512)])
@pytest.mark.xfail(strict=True, reason=(
    'Known defect: the planner event route returns a wrong tau gradient once a '
    'stream spans more than one planner chunk. Measured event-vs-quadratic '
    'relative L2 on rtau grows with the chunk count: 1 chunk 1e-7, 4 chunks '
    '4e-4..1.5e-2, 8 chunks 5e-2, 32 chunks 1.6e-1, while the quadratic '
    'fallback stays within 1.3e-5 of the FP64 oracle. sq/sk/v agree between '
    'routes to bf16 rounding at every run length, so the defect is tau-only. '
    'Long runs dominate the tau gradient (~s times their weight sensitivity), '
    'so this hits the part of dtau with the largest contribution. Verified by '
    'dism_v3/build/route_tau_sweep.py.'))
def test_multi_chunk_streams_route_invariant(chunk_size, block):
    try:
        _compare(False, 'hard', 'random', block=block, chunk_size=chunk_size,
                 tau_value=0.5)
    finally:
        shutdown_hard_planners()
