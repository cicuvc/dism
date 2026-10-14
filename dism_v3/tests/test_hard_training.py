"""End-to-end eager dispatch, async ownership and original-row packed semantics."""
import gc
import threading

import pytest
import torch
import cu_flash_dism as cu
from torch.utils.checkpoint import checkpoint

from flash_dism import dism_core, voc_dism, VarlenLayout
from flash_dism.hard_training import AsyncHardPlanner
from flash_dism.reference.dism_v3_ref import dism_ref
from gradient_acceptance import assert_gradient


def inputs(packed=False, mode='mixed', pattern='random'):
    torch.manual_seed(918)
    b, n, h = (1, 1024, 2) if packed else (2, 512, 2)
    q, k, sq, sk, v = [(torch.randn(b, n, h, c, device='cuda')*.2).bfloat16().requires_grad_()
                        for c in (32, 32, 32, 32, 64)]
    lq, lk = [torch.full((b, n, h), 2., device='cuda', requires_grad=True) for _ in range(2)]
    iq, ik = [torch.randint(0, 8, (b, h, n), device='cuda', dtype=torch.int32) for _ in range(2)]
    if pattern == 'repeat':
        iq.zero_()
        ik.zero_()
    hard = torch.tensor([[True, False], [False, True]], device='cuda')
    if mode != 'mixed':
        hard.fill_(mode == 'hard')
    direction = torch.zeros((b, h), dtype=torch.bool, device='cuda')
    tau = torch.tensor([.5, -.5], device='cuda', requires_grad=True)
    layout = VarlenLayout.from_cu_seqlens(torch.tensor([0, 256, 512, 512, 1024], dtype=torch.int32), n) if packed else None
    return (q, k, sq, sk, v, lq, lk, iq, ik, direction, hard, tau), layout


def oracle(xs, layout):
    q, k, sq, sk, v, lq, lk, iq, ik, direction, hard, tau = xs
    if layout is None:
        return dism_ref(q, k, sq, sk, lq, lk, iq, ik, direction,
                        hard[:, :, None].expand_as(iq), v, tau)
    outputs = []
    row_n = layout.tokens//hard.shape[0]
    for start, length, *_ in layout.table.tolist():
        if length:
            vectors = [x[:, start:start+length] for x in (q, k, sq, sk, lq, lk)]
            flags = hard[start//row_n][None, :, None].expand(1, iq.shape[1], length)
            outputs.append(dism_ref(*vectors, iq[:, :, start:start+length], ik[:, :, start:start+length],
                                    direction, flags, v[:, start:start+length], tau))
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('mode', ['soft', 'hard', 'mixed'])
@pytest.mark.parametrize('pattern', ['random', 'repeat'])
def test_autograd_reference(packed, mode, pattern):
    xs, layout = inputs(packed, mode, pattern)
    with AsyncHardPlanner(backward_policy='event', workers=2) as planner:
        out = dism_core(*xs, layout=layout, hard_planner=planner)
        do = (torch.randn_like(out)*.2).contiguous()
        active = [xs[i] for i in (0, 1, 2, 3, 4, 5, 6, 11)]
        actual = torch.autograd.grad(out, active, do)
        refs = [x.detach().double().requires_grad_() if x.requires_grad else x for x in xs]
        expected = oracle(refs, layout)
        reference = torch.autograd.grad(expected, [refs[i] for i in (0, 1, 2, 3, 4, 5, 6, 11)], do.double(), allow_unused=True)
        torch.testing.assert_close(out.float(), expected.float(), atol=.008, rtol=.025)
        assert (out.double()-expected).norm()/expected.norm().clamp_min(1e-8) < .01
        for name, grad, ref in zip(('q_vec', 'k_vec', 'sq_vec', 'sk_vec', 'v', 'q_lse', 'k_lse', 'rtau'), actual, reference):
            ref = torch.zeros_like(grad) if ref is None else ref
            assert_gradient(grad, ref, name)
        if mode == 'hard':
            for i in (0, 1, 5, 6):
                assert torch.count_nonzero(actual[i]) == 0


def test_submit_snapshot_bound_and_background_wait():
    xs, _ = inputs(mode='hard')
    iq, ik, hard, tau = xs[7], xs[8], xs[10], xs[11]
    with AsyncHardPlanner(backward_policy='event', workers=2, max_inflight=1) as planner:
        gate = threading.Event()
        original = planner.native.prepare
        def delayed(*args, **kwargs):
            assert gate.wait(20)
            return original(*args, **kwargs)
        planner.native.prepare = delayed
        ticket = planner.submit(iq, ik, hard, tau)
        try:
            assert not ticket.future.done()
            with pytest.raises(RuntimeError, match='max_inflight'):
                planner.submit(iq, ik, hard, tau)
            # Change all sources immediately after submit; planner uses snapshots.
            iq.fill_(100)
            ik.fill_(200)
            hard.zero_()
            with torch.no_grad():
                tau.fill_(4.)
        finally:
            gate.set()
        prepared = ticket.future.result(timeout=30)
        assert prepared.host.statistics['streams'] > 0
        gs = ticket.backward(xs[2], xs[3], xs[4], torch.ones_like(xs[4]))
        assert gs['v'].abs().sum() > 0
        del ticket, prepared
        gc.collect()
        empty = planner.submit(iq, ik, hard, tau)
        assert empty.future.result(timeout=30).host.statistics['streams'] == 0
    with pytest.raises(RuntimeError, match='closed'):
        planner.submit(iq, ik, hard, tau)


@pytest.mark.parametrize('reentrant', [True, False])
def test_checkpoint_and_retained_graph(reentrant):
    xs, layout = inputs(mode='hard')
    with AsyncHardPlanner(backward_policy='event', workers=2) as planner:
        call = lambda *args: dism_core(*args, hard_planner=planner)
        baseline = call(*xs)
        expected = torch.autograd.grad(baseline.sum(), xs[2], retain_graph=True)[0]
        again = torch.autograd.grad(baseline.sum(), xs[2])[0]
        torch.testing.assert_close(again, expected, atol=2e-5, rtol=2e-4)
        out = checkpoint(call, *xs, use_reentrant=reentrant)
        out.sum().backward()
        torch.testing.assert_close(xs[2].grad, expected, atol=2e-5, rtol=2e-4)


def test_cross_stream_backward_and_early_discard():
    xs, _ = inputs(mode='hard')
    with AsyncHardPlanner(backward_policy='event', workers=2) as planner:
        forward_stream, backward_stream = torch.cuda.Stream(), torch.cuda.Stream()
        forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(forward_stream):
            out = dism_core(*xs, hard_planner=planner)
        backward_stream.wait_stream(forward_stream)
        with torch.cuda.stream(backward_stream):
            grad = torch.autograd.grad(out.sum(), xs[2])[0]
        torch.cuda.current_stream().wait_stream(backward_stream)
        assert grad.isfinite().all()
        discarded = dism_core(*xs, hard_planner=planner)
        del discarded
        gc.collect()


def test_forward_does_not_wait_for_planner_and_failure_recovers():
    xs, _ = inputs(mode='hard')
    with AsyncHardPlanner(backward_policy='event', workers=2) as planner:
        gate = threading.Event()
        original = planner.native.prepare
        def delayed(*args, **kwargs):
            assert gate.wait(20)
            raise ValueError('injected planning failure')
        planner.native.prepare = delayed
        try:
            out = dism_core(*xs, hard_planner=planner)
            assert not cu.get_training_plan(out.grad_fn.saved_tensors[-1]).future.done()
        finally:
            gate.set()
        with pytest.raises(ValueError, match='injected'):
            out.sum().backward()
        planner.native.prepare = original
        good = dism_core(*xs, hard_planner=planner)
        good.sum().backward()
        assert xs[2].grad.isfinite().all()


def test_saved_input_mutation_is_rejected():
    xs, _ = inputs(mode='hard')
    with AsyncHardPlanner(backward_policy='event', workers=2) as planner:
        out = dism_core(*xs, hard_planner=planner)
        xs[7].fill_(5)
        with pytest.raises(RuntimeError, match='modified by an inplace operation'):
            out.sum().backward()


def test_packed_document_cannot_cross_training_row():
    xs, _ = inputs(packed=True)
    bad = VarlenLayout.from_cu_seqlens(torch.tensor([0, 768, 1024], dtype=torch.int32), 1024)
    with AsyncHardPlanner(backward_policy='event', workers=2) as planner:
        with pytest.raises((ValueError, RuntimeError), match='cross'):
            dism_core(*xs, layout=bad, hard_planner=planner)


def test_multiple_live_plans_overlap_native_allocations():
    # Exercise CPU-thread H2D allocations while main-thread native forwards
    # allocate checkpoints. Particularly important with sanitizer callbacks:
    # the native entry must release the GIL before taking allocator locks.
    xs, _ = inputs(mode='mixed')
    with AsyncHardPlanner(backward_policy='event', workers=2, max_inflight=8) as planner:
        outputs = [dism_core(*xs, hard_planner=planner) for _ in range(8)]
        sum(x.float().square().mean() for x in outputs).backward()
        assert all(xs[i].grad.isfinite().all() for i in (0, 1, 2, 3, 4, 11))


@pytest.mark.parametrize('packed', [False, True])
def test_vocabulary_hard_zero_gradients(packed):
    xs, layout = inputs(packed, 'hard')
    eq, ek = [(torch.randn(2, 32, 32, device='cuda')*.2).bfloat16().requires_grad_() for _ in range(2)]
    with AsyncHardPlanner(backward_policy='event', workers=2) as planner:
        out = voc_dism(*xs[:5], eq, ek, xs[-1], direction=xs[9], hard=xs[10],
                       layout=layout, hard_planner=planner)
        grads = torch.autograd.grad(out.sum(), [xs[0], xs[1], eq, ek, xs[2]])
        assert all(torch.count_nonzero(g) == 0 for g in grads[:4])
        assert grads[4].abs().sum() > 0


@pytest.mark.parametrize('packed', [False, True])
def test_default_compiler_custom_op_matches_complete_eager(packed):
    from flash_dism.compiler import voc_forward
    xs, layout = inputs(packed, 'mixed')
    eq, ek = [torch.randn(2, 32, 32, device='cuda').requires_grad_() for _ in range(2)]
    args = [*xs[:5], eq, ek, xs[-1]]
    expected = voc_dism(*args, direction=xs[9], hard=xs[10], layout=layout)
    bounds = torch.tensor([0, 256, 512, 512, 1024], dtype=torch.int32, device='cuda') if packed else None
    actual = voc_forward(*args, torch.ones((), device='cuda'), xs[9], xs[10], bounds)[0]
    do = torch.randn_like(actual)
    eg = torch.autograd.grad(expected, args, do)
    ag = torch.autograd.grad(actual, args, do)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, e in zip(ag, eg):
        torch.testing.assert_close(a, e, atol=.001, rtol=.03)


@pytest.mark.parametrize('packed', [False, True])
def test_async_hard_long_stream_safety(packed):
    # Small unfiltered sanitizer target: exercise all three long-stream stages
    # without spending sanitizer time on the independent dense oracle kernels.
    xs, layout = inputs(packed, 'hard', 'repeat')
    with AsyncHardPlanner(backward_policy='event', workers=2) as planner:
        out = dism_core(*xs, layout=layout, hard_planner=planner)
        out.float().square().sum().backward()
        assert out.isfinite().all()
        assert all(xs[i].grad.isfinite().all() for i in (2, 3, 4, 11))
        assert torch.count_nonzero(xs[0].grad) == 0
