"""P2/P3/P4 regression: real native plans, dynamic graphs, immutable routing."""
import threading

import numpy as np
import pytest
import torch

from flash_dism.inference.training_planner import NativeTrainingPlanner
from flash_dism.kernels.hard_backward_fused import CompactFusedHardPrefillBackward
from flash_dism.kernels.planner_graph import GraphCompactBackward
from flash_dism.planner_graph import GraphPlannerRuntime, upload_plan
from flash_dism.hard_backward import backward_hard_recompute
from test_hard_forward import fixture
from test_hard_backward import reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
KEYS = ('sq_vec', 'sk_vec', 'v', 'rtau')


@pytest.fixture
def cuda_stream():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield stream
    stream.synchronize()


def compact_reference(planner, x, do, documents=None):
    sq, sk, value, iq, ik, hard, tau = x
    host = planner.prepare(torch.stack((iq, ik)).cpu(), tau.cpu(), hard=hard.cpu(),
        documents=documents, compact=True, chunk_size=64)
    old = CompactFusedHardPrefillBackward.from_compact(host.upload())
    return host, old.backward(*[t.reshape(-1, t.shape[-1]) for t in (sq, sk, value, do)])


def compare(actual, expected):
    for key in KEYS:
        torch.testing.assert_close(actual[key].float().reshape(-1), expected[key].float().reshape(-1),
            atol=1e-4, rtol=1e-4)


def test_descriptor_changes_in_one_graph():
    planner = NativeTrainingPlanner(4)
    x = list(fixture(512, 'random', b=1, h=2))
    do = torch.randn_like(x[2])*.2
    desc = torch.zeros(64, dtype=torch.int64, device='cuda')
    choice = torch.ones(1, dtype=torch.int32, device='cuda')
    consumer = GraphCompactBackward(desc, choice, ctas=3)  # force grid-stride traversal
    def call():
        return consumer.backward(*[t.reshape(-1, t.shape[-1]) for t in (*x[:3], do)], heads=2)
    # Null descriptor/counts are valid for an empty plan, including warmup.
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = call()
    structures, pointers = set(), set()
    for i, pattern in enumerate(('random', 'equal', 'boundary', 'periodic', 'disjoint', 'random')):
        fresh = fixture(512, pattern, b=1, h=2)
        for dst, src in zip(x, fresh):
            dst.copy_(src)
        if i == 4:
            x[5].zero_()
        x[-1].fill_((i-2)*.1)
        docs = np.array([(0, 0, 256), (0, 256, 512)] if i % 2 else [(0, 0, 512)], dtype=np.int64)
        host, expected = compact_reference(planner, x, do, docs)
        stream = torch.cuda.current_stream()
        payload = upload_plan(host, device=torch.device('cuda', 0), stream=stream, budget_bytes=2**30)
        desc.copy_(payload.descriptor, non_blocking=True)
        graph.replay()
        torch.cuda.synchronize()
        compare(actual, expected)
        stats = host.statistics
        structures.add(tuple(stats['bucket_blocks'])+(stats['multi_chunks'], stats['multi_streams']))
        pointers.add(payload.device.storage.data_ptr())
        del payload
    assert len(structures) >= 4 and len(pointers) >= 2


@pytest.mark.parametrize('strict', [False, pytest.param(True, marks=pytest.mark.strict_gradient)])
@pytest.mark.parametrize('pattern', ['random', 'equal', 'boundary', 'periodic', 'disjoint'])
def test_event_matches_existing_fp64_contract(pattern, strict):
    planner = NativeTrainingPlanner(4)
    x = fixture(256, pattern, b=1, h=2)
    do = torch.randn_like(x[2])
    host, old = compact_reference(planner, x, do)
    payload = upload_plan(host, device=x[0].device, stream=torch.cuda.current_stream(), budget_bytes=2**30)
    desc = payload.descriptor.to(x[0].device)
    consumer = GraphCompactBackward(desc, torch.ones(1, dtype=torch.int32, device='cuda'), ctas=3)
    got = consumer.backward(*[t.reshape(-1, t.shape[-1]) for t in (*x[:3], do)], heads=2)
    compare(got, old)
    expected = reference(x, do)
    for key, target in zip(KEYS, expected):
        actual = got[key].view_as(target).double()
        if strict:
            # Additional cross-route diagnostic: these are quadratic's pointwise
            # tolerances, not the event consumer's existing production contract.
            # Keep the failure visible; the docs report old/new independently.
            torch.testing.assert_close(actual, target, atol=.003 if key == 'rtau' else .008,
                                       rtol=.01 if key == 'rtau' else .025)
        else:
            from gradient_acceptance import assert_gradient
            assert_gradient(actual, target, key)


@pytest.mark.parametrize('mode', ['ready', 'late', 'cancel', 'budget', 'error'])
def test_real_planner_lifecycle_and_dual_path(mode):
    runtime = GraphPlannerRuntime(workers=4, max_slots=1, max_bytes=2**29)
    slot = runtime.slot(rows=1, heads=2, row_tokens=512)
    x = list(fixture(512, 'boundary', b=1, h=2))
    do = torch.randn_like(x[2])*.2
    slot.snapshot(*x[3:])
    slot.backward(*x[:3], do, *x[3:])
    slot.finish()
    torch.cuda.synchronize()
    inputs, consumers = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(inputs):
        slot.snapshot(*x[3:])
    with torch.cuda.graph(consumers):
        actual = slot.backward(*x[:3], do, *x[3:])
        slot.finish()
    release, entered = threading.Event(), threading.Event()
    original = runtime.native.prepare
    snapshots = []
    def prepare(*args, **kwargs):
        snapshots.append(tuple(a.clone() for a in (args[0], args[1], kwargs['hard'])))
        entered.set()
        if not release.wait(30):
            raise TimeoutError('test producer gate')
        if mode == 'error':
            raise RuntimeError('injected planner failure')
        return original(*args, **kwargs)
    runtime.native.prepare = prepare
    try:
        # Multiple real generations: prior ready word remains set between rounds.
        for generation in range(3 if mode != 'error' else 1):
            fresh = fixture(512, 'equal' if generation % 2 else 'random', b=1, h=2)
            for dst, src in zip(x, fresh):
                dst.copy_(src)
            x[-1].fill_(generation*.1)
            release.clear()
            entered.clear()
            slot.arm()
            inputs.replay()
            slot.launched()
            assert entered.wait(30)
            if mode == 'budget':
                slot.budget = 1
            if mode == 'cancel':
                slot.cancel()
            if mode in ('ready', 'budget', 'cancel'):
                release.set()
                slot.future.result(timeout=30)
            consumers.replay()
            slot.finished.synchronize()
            assert slot.choice.item() == (mode == 'ready')
            release.set()
            if mode == 'error':
                with pytest.raises(RuntimeError, match='injected planner failure'):
                    slot.drain()
                with pytest.raises(RuntimeError, match='failed'):
                    slot.arm()
                continue
            slot.drain()
            if mode == 'ready':
                _, expected = compact_reference(NativeTrainingPlanner(2), x, do)
            else:
                expected = backward_hard_recompute(*x[:3], do, *x[3:])
            compare(actual, expected)
            labels, tau, hard = snapshots[-1]
            torch.testing.assert_close(labels, torch.stack(x[3:5]).cpu())
            torch.testing.assert_close(tau, x[-1].cpu())
            torch.testing.assert_close(hard, x[-2].cpu())
    finally:
        release.set()
        if mode == 'error':
            with pytest.raises(RuntimeError):
                runtime.close()
        else:
            runtime.close()


def test_full_replay_handoff_and_no_producer():
    with GraphPlannerRuntime(workers=4, max_slots=1, max_bytes=2**29) as runtime:
        slot = runtime.slot(rows=1, heads=1, row_tokens=256)
        x = fixture(256, 'boundary', b=1, h=1)
        do = torch.randn_like(x[2])*.2
        def call():
            slot.snapshot(*x[3:])
            torch.cuda._sleep(2_000_000)  # finite slack; route is NOT a timing assertion
            out = slot.backward(*x[:3], do, *x[3:])
            slot.finish()
            return out
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = call()
        for i in range(12):
            slot.arm()
            if i % 3:
                runtime.replay(graph)
                slot.drain()
            else:
                graph.replay()
                slot.finished.synchronize()
                assert slot.choice.item() == 0
                slot.drain()  # no CPU work was submitted
            if slot.choice.item():
                _, expected = compact_reference(runtime.native, x, do)
            else:
                expected = backward_hard_recompute(*x[:3], do, *x[3:])
            compare(actual, expected)


def test_capture_failure_and_cancel_before_planning():
    with GraphPlannerRuntime(workers=2, max_slots=1) as runtime:
        slot = runtime.slot(rows=1, heads=1, row_tokens=256)
        x = fixture(256, b=1, h=1)
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(ValueError, match='capture aborted'):
            with torch.cuda.graph(graph):
                slot.snapshot(*x[3:])
                raise ValueError('capture aborted')
        slot.arm()
        slot.drain()  # captured input event was never launched; must not wait
        slot.arm()
        slot.snapshot(*x[3:])
        slot.finish()
        slot.cancel()
        def forbidden(*args, **kwargs):
            raise AssertionError('cancelled plan was dispatched')
        runtime.native.prepare = forbidden
        slot.launched()
        slot.drain()
        assert slot.reason == 'cancelled' and slot.ready.item() == 0


def test_dispatch_failure_retires_every_launched_slot():
    runtime = GraphPlannerRuntime(workers=2, max_slots=2)
    slots = [runtime.slot(rows=1, heads=1, row_tokens=256) for _ in range(2)]
    x = fixture(256, b=1, h=1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for slot in slots:
            slot.snapshot(*x[3:])
            slot.finish()
    for slot in slots:
        slot.arm()
    submit = runtime.pool.submit
    calls = 0
    def failing_submit(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected dispatch failure')
        return submit(*args, **kwargs)
    runtime.pool.submit = failing_submit
    with pytest.raises(RuntimeError, match='failed'):
        runtime.replay(graph)
    assert all(slot.state == 'idle' and slot.future is None and slot.payload is None for slot in slots)
    assert all(slot.finished.query() for slot in slots)
    with pytest.raises(RuntimeError, match='failed'):
        runtime.close()


def test_close_without_replay_and_limits():
    runtime = GraphPlannerRuntime(workers=2, max_slots=1, max_bytes=2**20)
    slot = runtime.slot(rows=1, heads=1, row_tokens=256)
    with pytest.raises(RuntimeError, match='slot limit'):
        runtime.slot(rows=1, heads=1, row_tokens=256)
    slot.arm()
    slot.cancel()
    runtime.close()
    runtime.close()
    with pytest.raises(RuntimeError, match='closed'):
        slot.arm()


def test_packed_document_snapshot_and_epoch_wrap():
    import cu_flash_dism as cu
    from flash_dism.varlen import VarlenLayout
    with GraphPlannerRuntime(workers=2, max_slots=2, max_bytes=2**28) as runtime:
        slots = [runtime.slot(rows=2, heads=2, row_tokens=512) for _ in range(2)]
        x = fixture(512, 'random', b=2, h=2)
        iq, ik = [t.permute(1, 0, 2).reshape(1, 2, 1024).contiguous() for t in x[3:5]]
        table = cu.make_graph_layout(torch.tensor([0,512,1024], dtype=torch.int32), 1024, 512, 512)
        layout = VarlenLayout.for_graph(table, 1024)
        def call():
            for slot in slots:
                slot.snapshot(iq, ik, *x[5:])
                slot.sample()
                slot.finish()
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        seen = []
        original = runtime.native.prepare
        def prepare(labels, tau, **kwargs):
            seen.append((labels.clone(), kwargs['hard'].clone(), tau.clone(), kwargs['documents'].copy()))
            return original(labels, tau, **kwargs)
        runtime.native.prepare = prepare
        # Both slots dispatch independently; wrap only after all writers drain.
        slots[0].epoch = 2**32-1
        slots[0].ready.fill_(-1)
        for slot in slots:
            slot.arm(layout=layout)
        assert slots[0].epoch == 1
        # Caller mutations after arm cannot change the private document input.
        cu.update_graph_layout(table, torch.tensor([0,0,256,512,1024], dtype=torch.int32))
        graph.replay()
        for slot in slots:
            slot.launched()
        runtime.drain()
        assert len(seen) == 2
        for labels, hard, tau, docs in seen:
            torch.testing.assert_close(labels, torch.stack(x[3:5]).cpu())
            torch.testing.assert_close(hard, x[5].cpu())
            torch.testing.assert_close(tau, x[6].cpu())
            np.testing.assert_array_equal(docs, [(0,0,512), (1,0,512)])
        # uint32's high bit survives host staging and GPU comparison.
        slots[0].epoch = 2**31-1
        slots[0].arm(layout=layout)
        assert slots[0].expected.item() == -(2**31)
        slots[0].drain()


def test_late_publish_after_sampling_before_consumers():
    from flash_dism.kernels.planner_graph import sample_route
    planner = NativeTrainingPlanner(2)
    x = fixture(256, 'boundary', b=1, h=2)
    do = torch.randn_like(x[2])*.2
    with GraphPlannerRuntime(workers=2, max_slots=1) as runtime:
        slot = runtime.slot(rows=1, heads=2, row_tokens=256)
        # Capture branches separately to hold the sampling/consumer boundary.
        def call():
            event = slot.consumer.backward(*[t.reshape(-1,t.shape[-1]) for t in (*x[:3], do)], heads=2)
            quad = backward_hard_recompute(*x[:3], do, *x[3:], _graph_choice=slot.choice)
            return {key: event[key].view_as(quad[key])+quad[key].float() for key in KEYS}
        call()
        slot.sample()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = call()
        host, _ = compact_reference(planner, x, do)
        slot.expected.fill_(1)
        slot.ready.zero_()
        slot.descriptor.zero_()  # invalid/null payload must be ignored
        sample_route[(1,)](slot.ready, slot.expected, slot.choice, num_warps=1)
        torch.cuda.synchronize()
        payload = upload_plan(host, device=slot.device, stream=slot.upload, budget_bytes=2**29)
        with torch.cuda.stream(slot.upload):
            slot.descriptor.copy_(payload.descriptor, non_blocking=True)
            runtime.writer.publish(slot.upload, slot.ready, 1)
        slot.upload.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert slot.choice.item() == 0 and slot.ready.item() == 1
        compare(actual, backward_hard_recompute(*x[:3], do, *x[3:]))


def test_upload_failure_is_drained_and_reported():
    runtime = GraphPlannerRuntime(workers=2, max_slots=1)
    slot = runtime.slot(rows=1, heads=1, row_tokens=256)
    x = fixture(256, b=1, h=1)
    slot.arm()
    slot.snapshot(*x[3:])
    slot.finish()
    def fail(*args):
        raise RuntimeError('injected publication failure after H2D')
    runtime.writer.publish = fail
    slot.launched()
    with pytest.raises(RuntimeError, match='injected publication failure'):
        slot.drain()
    assert slot.ready.item() == 0 and slot.payload is None and slot.future is None
    with pytest.raises(RuntimeError, match='failed'):
        runtime.close()


def test_packed_dual_graph_changes_documents_and_original_row_masks(cuda_stream):
    import cu_flash_dism as cu
    from flash_dism.varlen import VarlenLayout
    with GraphPlannerRuntime(workers=4, max_slots=1, max_bytes=2**29) as runtime:
        slot = runtime.slot(rows=2, heads=2, row_tokens=512)
        source = fixture(512, 'boundary', b=2, h=2)
        x = [t.reshape(1,1024,2,-1) for t in source[:3]]
        x += [t.permute(1,0,2).reshape(1,2,1024).contiguous() for t in source[3:5]]
        x += list(source[5:])
        do = torch.randn_like(x[2])*.2
        table = cu.make_graph_layout(torch.tensor([0,512,1024], dtype=torch.int32),1024,512,512)
        layout = VarlenLayout.for_graph(table,1024)
        slot.snapshot(*x[3:])
        slot.backward(*x[:3],do,*x[3:],layout=layout)
        slot.finish()
        torch.cuda.synchronize()
        inputs, backward = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(inputs, stream=cuda_stream):
            slot.snapshot(*x[3:])
        with torch.cuda.graph(backward, stream=cuda_stream):
            actual = slot.backward(*x[:3],do,*x[3:],layout=layout)
            slot.finish()
        for i, bounds in enumerate(([0,512,1024], [0,256,512,768,1024], [0,0,512,1024], [0,512,1024])):
            cu.update_graph_layout(table,torch.tensor(bounds,dtype=torch.int32))
            x[5].copy_(torch.tensor([[i%2==0,False],[False,i%2!=0]],device='cuda'))
            x[6].fill_(.1*i)
            slot.arm(layout=layout)
            inputs.replay()
            slot.launched()
            if i % 2 == 0:
                slot.future.result(timeout=30)
            else:
                # Cancellation after upload is legal too; force a stale tag to
                # select fallback only AFTER all upload work has retired.
                slot.future.result(timeout=30)
                slot.ready.zero_()
            backward.replay()
            slot.finished.synchronize()
            assert slot.choice.item() == (i%2==0)
            slot.drain()
            if i%2==0:
                labels = torch.stack([t[0].reshape(2,2,512).permute(1,0,2) for t in x[3:5]])
                host = runtime.native.prepare(labels.cpu(),x[-1].cpu(),hard=x[-2].cpu(),
                    documents=slot.documents,compact=True,chunk_size=64)
                old = CompactFusedHardPrefillBackward.from_compact(host.upload())
                expected = old.backward(*[t.reshape(-1,t.shape[-1]) for t in (*x[:3],do)])
            else:
                expected = backward_hard_recompute(*x[:3],do,*x[3:],layout=layout)
            compare(actual,expected)
            for key in KEYS[:3]:
                mask = x[5][:,None,:,None].expand(2,512,2,actual[key].shape[-1]).reshape_as(actual[key])
                assert not actual[key].masked_select(~mask).count_nonzero()
