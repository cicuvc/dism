"""Opt-in concurrent snapshot lifetime, handoff, and deadline regressions."""
import os
import threading
import time

import pytest
import torch

from flash_dism.inference.concurrent_probe import ConcurrentGpuPlannerCache
from test_sam_inference import oracle

pytestmark = pytest.mark.skipif(os.environ.get('DISM_BUILD_PROBES') != '1', reason='opt-in concurrent prototype')


@pytest.fixture(autouse=True)
def owner_stream():
    if os.environ.get('DISM_BUILD_PROBES') != '1':
        yield
        return
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield
    stream.synchronize()


def workload(pattern, dtype=torch.bfloat16, n=321):
    torch.manual_seed(911)
    h, r, dv = 3, 16, 32
    ik = torch.randint(0, 4, (1, h, n), device='cuda', dtype=torch.int32)
    iq = torch.randint(0, 4, ik.shape, device='cuda', dtype=torch.int32)
    if pattern == 'repeat':
        ik.zero_(); iq.zero_()
    elif pattern == 'periodic':
        ik.copy_(torch.arange(n, device='cuda')[None, None, :] % 3); iq.copy_(ik)
    elif pattern == 'empty':
        ik.copy_(torch.arange(n, device='cuda')[None, None, :]); iq.copy_(ik)
    reset = torch.zeros_like(ik, dtype=torch.bool)
    reset[..., 19::37] = True
    sq, sk, v = [(torch.randn(1, n, h, d, device='cuda')*.2).to(dtype) for d in (r, r, dv)]
    labels = torch.stack((ik[0].T, iq[0].T, reset[0].T.int()), 1).contiguous()
    keys, queries, vals = (x[0].transpose(0, 1).contiguous() for x in (sk, sq, v))
    tau = [0., .35, 1.5]
    expected = oracle(iq, ik, sq, sk, v, tau, reset)
    return labels, keys, queries, vals, tau, expected


@pytest.mark.parametrize('pattern', ['repeat', 'periodic', 'random', 'empty'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_multiple_generations(pattern, dtype):
    labels, k, q, v, tau, expected = workload(pattern, dtype)
    outputs = []
    with ConcurrentGpuPlannerCache(3, 16, 32, 321, tau, rebuild_interval=32,
            workers=8, slots=3, block=16, tile=64, cache_dtype=dtype) as c:
        c.prime(labels[:201], k[:, :201].contiguous(), v[:, :201].contiguous())
        shared = [c.snapshot[i].data_ptr() for i in (0, 1, 6, 7)]
        for i in range(201, 321):
            outputs.append(c.step(labels[i], k[:, i].contiguous(), q[:, i].contiguous(), v[:, i].contiguous()).clone())
            if i in (207, 239, 271, 303):
                assert c.start_rebuild()
                assert not c.start_rebuild()
            if i in (216, 248, 280, 312):
                # May already have published via step's nonblocking poll.
                c.try_publish(wait=True)
        assert c.check_status() == 321
        assert shared == [c.snapshot[i].data_ptr() for i in (0, 1, 6, 7)]
        assert len(c.completed) >= 4
        assert all(row['lag_tokens'] <= 32 for row in c.stats()['jobs'])
        torch.testing.assert_close(torch.stack(outputs)[None].cpu().double(), expected[:, 201:], atol=2e-5, rtol=2e-3)


def test_delayed_handoff_cancel_and_deadline():
    labels, k, q, v, tau, expected = workload('repeat')
    outputs = []
    with ConcurrentGpuPlannerCache(3, 16, 32, 321, tau, rebuild_interval=16,
            workers=8, slots=3, block=16, tile=64) as c:
        c.prime(labels[:201], k[:, :201].contiguous(), v[:, :201].contiguous())
        original = c._build
        gate, entered = threading.Event(), threading.Event()
        def gated(*args):
            entered.set()
            if not gate.wait(10):
                raise RuntimeError('test worker gate timed out')
            return original(*args)
        c._build = gated
        try:
            for i in range(201, 217):
                outputs.append(c.step(labels[i], k[:, i].contiguous(), q[:, i].contiguous(), v[:, i].contiguous()).clone())
                if i == 204:
                    assert c.start_rebuild()
                    assert entered.wait(5)
            assert c.position == 217 and c.cutoff == 201
            assert not c.try_publish()
            timer = threading.Timer(.02, gate.set)
            timer.start()
            outputs.append(c.step(labels[217], k[:, 217].contiguous(), q[:, 217].contiguous(), v[:, 217].contiguous()).clone())
            timer.join()
            assert c.deadline_waits == 1
            assert c.cutoff == 205
            assert c.completed[-1]['lag_tokens'] == 12
            assert c.check_status() == 218
            assert c.start_rebuild()
            assert c.cancel_rebuild()
            assert not c.try_publish(wait=True)
            assert c.cutoff == 205 and c.completed[-1]['cancelled']
            c._build = original
            assert c.start_rebuild()
            assert c.try_publish(wait=True)
            assert c.cutoff == 218
            torch.testing.assert_close(torch.stack(outputs)[None].cpu().double(), expected[:, 201:218], atol=2e-5, rtol=2e-3)
        finally:
            gate.set()


def test_close_pending_and_failure():
    labels, k, q, v, tau, _ = workload('random')
    c = ConcurrentGpuPlannerCache(3, 16, 32, 321, tau, rebuild_interval=32, block=16)
    c.prime(labels[:201], k[:, :201].contiguous(), v[:, :201].contiguous())
    c.step(labels[201], k[:, 201].contiguous(), q[:, 201].contiguous(), v[:, 201].contiguous())
    assert c.start_rebuild()
    c.close(); c.close()
    with pytest.raises(RuntimeError, match='closed'):
        c.start_rebuild()
    with ConcurrentGpuPlannerCache(3, 16, 32, 321, tau, rebuild_interval=32, block=16) as c:
        c.prime(labels[:201], k[:, :201].contiguous(), v[:, :201].contiguous())
        c.step(labels[201], k[:, 201].contiguous(), q[:, 201].contiguous(), v[:, 201].contiguous())
        def fail(*args):
            raise ValueError('injected worker failure')
        c._build = fail
        c.start_rebuild()
        with pytest.raises(ValueError, match='injected'):
            c.try_publish(wait=True)
        with pytest.raises(RuntimeError, match='background rebuild failed'):
            c.step(labels[202], k[:, 202].contiguous(), q[:, 202].contiguous(), v[:, 202].contiguous())


def test_stream_and_capacity_guards():
    with ConcurrentGpuPlannerCache(1, 16, 32, 2, [.5], rebuild_interval=8, block=1) as c:
        labels = torch.zeros(3, 1, device='cuda', dtype=torch.int32)
        k = torch.ones(1, 16, device='cuda', dtype=torch.bfloat16)
        v = torch.ones(1, 32, device='cuda', dtype=torch.bfloat16)
        assert not c.start_rebuild()
        with torch.cuda.stream(torch.cuda.Stream()):
            with pytest.raises(RuntimeError, match='stream mismatch'):
                c.step(labels, k, k, v)
        c.step(labels, k, k, v); c.step(labels, k, k, v)
        with pytest.raises(RuntimeError, match='capacity'):
            c.step(labels, k, k, v)
        assert c.check_status() == 2


@pytest.mark.parametrize('pattern', ['repeat', 'periodic'])
def test_long_lag_keeps_live_band_and_old_snapshot(pattern):
    labels, k, q, v, tau, expected = workload(pattern, n=1281)
    with ConcurrentGpuPlannerCache(3, 16, 32, 1281, tau, rebuild_interval=256,
            workers=32, slots=3, block=32, tile=64) as c:
        c.prime(labels[:1025], k[:, :1025].contiguous(), v[:, :1025].contiguous())
        gate = threading.Event()
        original = c._build
        def gated(*args):
            if not gate.wait(15):
                raise RuntimeError('test worker gate timed out')
            return original(*args)
        c._build = gated
        outputs = []
        try:
            for i in range(1025, 1273):
                outputs.append(c.step(labels[i], k[:, i].contiguous(), q[:, i].contiguous(), v[:, i].contiguous()).clone())
                if i == 1032:
                    c.start_rebuild()
            old = c.snapshot
            saved_state = old[5].clone()
            saved_band = old[6].clone()
            saved_summary = old[2].clone()
            gate.set()
            assert c.try_publish(wait=True)
            assert c.completed[-1]['lag_tokens'] == 240
            assert c.snapshot[2].data_ptr() != old[2].data_ptr()
            assert c.check_status() == 1273
            torch.testing.assert_close(old[5], saved_state, rtol=0, atol=0)
            torch.testing.assert_close(old[6], saved_band, rtol=0, atol=0)
            torch.testing.assert_close(old[2], saved_summary, rtol=0, atol=0)
            for i in range(1273, 1281):
                outputs.append(c.step(labels[i], k[:, i].contiguous(), q[:, i].contiguous(), v[:, i].contiguous()).clone())
            torch.testing.assert_close(torch.stack(outputs)[None].cpu().double(), expected[:, 1025:], atol=2e-5, rtol=2e-3)
        finally:
            gate.set()


def test_unprimed_deadline_fallback():
    labels, k, q, v, tau, expected = workload('random', n=79)
    with ConcurrentGpuPlannerCache(3, 16, 32, 79, tau, rebuild_interval=16,
            workers=8, slots=3, block=8, tile=64) as c:
        outputs = [c.step(labels[i], k[:, i].contiguous(), q[:, i].contiguous(), v[:, i].contiguous()).clone()
                   for i in range(79)]
        assert c.check_status() == 79
        assert c.deadline_waits == 4
        torch.testing.assert_close(torch.stack(outputs)[None].cpu().double(), expected, atol=2e-5, rtol=2e-3)
