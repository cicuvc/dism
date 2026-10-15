"""Hard/soft training with invocation-owned asynchronous plans.

Default public operators share one lazy planner per process/device; callers may
override it for eager experiments. No global plan registry or checkpoint state.
Compiled operators transport ownership through native CPU tensor storage.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import atexit
import os
import time
import weakref

import numpy as np
import torch
import cu_flash_dism as cu

from .hard_forward import forward_hard
from .inference.training_planner import NativeTrainingPlanner
from .kernels.hard_backward_scan import HardPrefillBackward
from .kernels.hard_backward_fused import FusedHardPrefillBackward, CompactFusedHardPrefillBackward


class _Lease:
    def __init__(self, slots):
        weakref.finalize(self, slots.release)


@dataclass
class _Prepared:
    host: object
    kernel: object
    timings: dict
    d2h_events: tuple
    h2d_events: tuple


class HardPlanTicket:
    """Owns this invocation's inputs/metadata, including retained-graph replay."""

    # Diagnostic census of the dispatch decision, one entry per backward: 'e'
    # when the prepared event kernel runs, 'q' when the ticket falls back to the
    # quadratic recompute. The decision never reads this record, so it cannot
    # change routing, the objective, RNG, optimizer state or checkpoints; the
    # training loop drains it on log steps to keep it bounded. Entries are
    # appended without the lock because list.append is atomic, so a racing drain
    # can at worst attribute one entry to the neighbouring interval.
    _ROUTE_LOG: list = []
    _ROUTE_LOG_LIMIT = 1 << 20
    _ROUTE_LOCK = threading.Lock()

    def __init__(self, future, lease, policy='event', abandoned=None):
        self.future, self._lease = future, lease
        self.host_wait_ms = 0.
        self.policy = policy
        self.abandoned = abandoned or threading.Event()
        self.route = None

    @classmethod
    def _record_route(cls, decision: str) -> None:
        if len(cls._ROUTE_LOG) < cls._ROUTE_LOG_LIMIT:
            cls._ROUTE_LOG.append(decision)

    def backward(self, sq, sk, v, dout, *, fallback=None):
        prepared = None
        if self.policy != 'event':
            # result() is only inspected after done(): never hide planner errors
            # behind a fallback. A completed CPU future is not a completed H2D.
            # A disabled planner hands out tickets with no future at all; they
            # always take the fallback, which is what `not ready` already means.
            if self.future is not None and self.future.done():
                prepared = self.future.result()
            ready = prepared is not None and prepared.h2d_events[1].query()
            if self.policy == 'quadratic' or not ready:
                if fallback is None:
                    raise RuntimeError('nonblocking hard backward requires a fallback')
                self.abandoned.set()
                self.route, self.host_wait_ms = 'quadratic', 0.
                self._record_route('q')
                return fallback()
        start = time.perf_counter()
        prepared = self.future.result() if prepared is None else prepared
        self.host_wait_ms = (time.perf_counter()-start)*1000
        self.route = 'event'
        self._record_route('e')
        xs = [x.reshape(-1, x.shape[-1]).contiguous() for x in (sq, sk, v, dout)]
        gradients = prepared.kernel.backward(*xs)
        for name, x in zip(('sq_vec', 'sk_vec', 'v'), (sq, sk, v)):
            gradients[name] = gradients[name].view_as(x)
        return gradients

    def metrics(self):
        """Diagnostic only: waits copy completion, never used by training.

        h2d_ms is the bulk+bucket upload window including CPU launch gaps,
        not an isolated PCIe transfer. host_wait_ms excludes device waits.
        """
        if self.future is None:
            return dict(route=self.route, host_wait_ms=0., abandoned=True)
        p = self.future.result()
        if p is None:
            return dict(route=self.route, host_wait_ms=self.host_wait_ms, abandoned=True)
        p.h2d_events[1].synchronize()
        return dict(p.timings, route=self.route, host_wait_ms=self.host_wait_ms,
                    d2h_ms=p.d2h_events[0].elapsed_time(p.d2h_events[1]),
                    h2d_ms=p.h2d_events[0].elapsed_time(p.h2d_events[1]),
                    metadata_bytes=p.host.statistics['metadata_bytes'])


class AsyncHardPlanner:
    """One coordinator + bounded native pool; no host wait in submit().

    max_inflight bounds *live invocation tickets*, including completed plans.
    Saturation raises instead of deadlocking forward before backward can free
    tickets. Size it for live layers/microbatches/checkpoint recomputations.
    close() drains submitted CPU jobs; existing tickets remain usable.
    CUDA graph capture is intentionally unsupported.
    """
    def __init__(self, *, device='cuda', workers=16, chunk_size=64,
                 max_inflight=32, mma_precision='bf16', backward_policy='ready', compact=True):
        if backward_policy not in ('event','ready','quadratic'):
            raise ValueError('backward_policy must be event, ready or quadratic')
        self.backward_policy = backward_policy
        self.compact = compact
        if chunk_size not in (16, 32, 64) or max_inflight < 1:
            raise ValueError('chunk_size must be 16/32/64 and max_inflight positive')
        if mma_precision not in ('bf16', 'tf32x3'):
            raise ValueError('mma_precision must be bf16 or tf32x3')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA device required')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.native = NativeTrainingPlanner(workers)
        self.chunk_size, self.precision = chunk_size, mma_precision
        self.d2h_stream = torch.cuda.Stream(device=self.device)
        self.h2d_stream = torch.cuda.Stream(device=self.device)
        self._slots = threading.BoundedSemaphore(max_inflight)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dism-plan')
        self._closed = False

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @torch.no_grad()
    def submit(self, iq, ik, hard, tau, *, layout=None):
        with torch.cuda.device(self.device):
            capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            raise RuntimeError('asynchronous hard planning does not support CUDA Graph capture')
        if (iq.ndim != 3 or ik.shape != iq.shape or iq.dtype != torch.int32
                or ik.dtype != torch.int32 or hard.ndim != 2 or hard.dtype != torch.bool
                or tau.dtype != torch.float32 or tau.shape != (iq.shape[1],)
                or any(x.device != self.device for x in (iq, ik, hard, tau))):
            raise ValueError('CUDA int32 BHN labels, bool row/head flags, FP32 head tau required')
        b, h, n = iq.shape
        rows = hard.shape[0]
        if min(b, h, n, rows) < 1 or hard.shape[1] != h or n % 256:
            raise ValueError('nonempty aligned labels and matching hard heads required')
        if layout is None:
            if rows != b:
                raise ValueError('fixed hard rows must match batch')
            documents = None
            row_n = n
        else:
            if b != 1 or layout.tokens != n or n % rows or (n//rows) % 256:
                raise ValueError('packed inputs require equal aligned training rows')
            row_n = n//rows
            documents = []
            cursor = 0
            for start, length, *_ in layout.table.tolist():
                if start != cursor or length < 0 or start % 256 or length % 256:
                    raise ValueError('invalid aligned document layout')
                cursor += length
                if not length:
                    continue
                row = start//row_n
                if (start+length-1)//row_n != row:
                    raise ValueError('documents cannot cross original training rows')
                documents.append((row, start % row_n, start % row_n+length))
            if cursor != n:
                raise ValueError('documents must cover packed inputs')
            documents = np.asarray(documents, dtype=np.int64)
        with self._lock:
            if self._closed:
                raise RuntimeError('planner is closed')
            if not self._slots.acquire(blocking=False):
                raise RuntimeError('hard planner max_inflight reached; release old graphs or increase the explicit limit')
            lease = _Lease(self._slots)
            submitted = time.perf_counter()
            # Snapshot on the producer stream: subsequent in-place updates must
            # not change a queued plan. Packed BH(T) -> original row/head/token.
            def unpack(x):
                return x if layout is None else x[0].reshape(h, rows, row_n).permute(1, 0, 2)
            snapshots = (torch.stack((unpack(iq), unpack(ik))), hard.clone(), tau.clone())
            hosts = tuple(torch.empty(x.shape, dtype=x.dtype, pin_memory=True) for x in snapshots)
            events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
            self.d2h_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.d2h_stream):
                events[0].record()
                for host, src in zip(hosts, snapshots):
                    host.copy_(src, non_blocking=True)
                    src.record_stream(self.d2h_stream)
                events[1].record()
            abandoned = threading.Event()
            future = self._executor.submit(self._prepare, hosts, documents, events, submitted, lease, abandoned)
            return HardPlanTicket(future, lease, self.backward_policy, abandoned)

    def _prepare(self, hosts, documents, d2h_events, submitted, lease, abandoned=None):
        # The lease is also held by this job if its graph is discarded early.
        with torch.cuda.device(self.device), torch.no_grad():
            d2h_events[1].synchronize()  # background coordinator only
            # Keep pinned hosts alive through D2H even for abandoned tickets.
            # Do not Future.cancel(): already-issued copies still own inputs.
            if abandoned is not None and abandoned.is_set():
                return None
            start = time.perf_counter()
            labels, hard, tau = hosts
            host = self.native.prepare(labels, tau, hard=hard, documents=documents,
                                       chunk_size=self.chunk_size, compact=self.compact)
            if abandoned is not None and abandoned.is_set():
                return None
            events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
            with torch.cuda.stream(self.h2d_stream):
                events[0].record()
                device = host.upload(device=self.device, stream=self.h2d_stream)
                if self.compact:
                    kernel = CompactFusedHardPrefillBackward.from_compact(
                        device, mma_precision=self.precision, non_blocking=True)
                else:
                    base = HardPrefillBackward.from_native(device, mma_precision=self.precision)
                    kernel = FusedHardPrefillBackward.from_plan(
                        base, host.data['offsets'].numpy(), host_lengths=host.data['lengths'].numpy(),
                        non_blocking=True)
                events[1].record()
            return _Prepared(host, kernel,
                             dict(queue_and_input_ms=(start-submitted)*1000,
                                  cpu_total_ms=host.timings['total_ms'],
                                  submit_to_enqueued_ms=(time.perf_counter()-submitted)*1000),
                             d2h_events, events)


_runtimes = {}
_runtime_lock = threading.Lock()


class DisabledHardPlanner:
    """Planner substitute selected by `DISM_HARD_PLANNER=0`.

    Every hard backward takes the quadratic fallback. No native worker pool, no
    coordinator thread, no pinned staging and no D2H/H2D traffic are created, so
    this also removes the planner's CPU cost rather than discarding its output.
    The forward is unchanged: `_voc_forward_impl` still computes the hard output
    with `forward_hard`, and the fallback recomputes that same state.
    """

    def __init__(self, device):
        self.device = torch.device(device)

    def submit(self, iq, ik, hard, tau, *, layout=None):
        return HardPlanTicket(None, None, 'quadratic')

    def close(self):
        pass


def take_route_log():
    """Return and clear the recorded hard-backward dispatch decisions.

    Diagnostic only. The entries are 'e' (prepared event kernel) or 'q'
    (quadratic recompute) in the order the autograd engine replayed them. The
    planner reads none of this, so draining it cannot change training.
    """
    with HardPlanTicket._ROUTE_LOCK:
        log, HardPlanTicket._ROUTE_LOG = HardPlanTicket._ROUTE_LOG, []
    return log


def default_hard_planner(device):
    """Lazy process/device runtime, never a module buffer or checkpoint field."""
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('CUDA device required')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('DISM asynchronous training does not support CUDA Graph capture')
    key = (os.getpid(), device.index)
    with _runtime_lock:
        if key not in _runtimes:
            if os.environ.get('DISM_HARD_PLANNER', '1') == '0':
                _runtimes[key] = DisabledHardPlanner(device)
            else:
                _runtimes[key] = AsyncHardPlanner(
                    device=device, workers=int(os.environ.get('DISM_PLANNER_WORKERS', '16')),
                    max_inflight=int(os.environ.get('DISM_PLANNER_MAX_INFLIGHT', '32')),
                    backward_policy=os.environ.get('DISM_HARD_BACKWARD', 'ready'),
                    compact=bool(int(os.environ.get('DISM_PLANNER_COMPACT', '1'))))
        return _runtimes[key]


def shutdown_hard_planners():
    """Drain default coordinators; already-created graphs keep valid tickets."""
    with _runtime_lock:
        current = list(_runtimes.values())
        _runtimes.clear()
    for planner in current:
        planner.close()


atexit.register(shutdown_hard_planners)


def layout_from_table(table, tokens):
    """Reconstitute a validated host layout without any GPU metadata readback."""
    if table is None:
        return None
    from .varlen import VarlenLayout
    table = table.view(-1, 7)
    lengths = tuple(table[:, 1].tolist())
    return VarlenLayout(table, lengths, tokens, tokens,
                        sum(((n-1)//32)*n for n in lengths if n),
                        sum(((n-1)//16)*n for n in lengths if n),
                        sum(((n+31)//32)*n for n in lengths))


class _MixedCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, sq, sk, v, lq, lk, iq, ik, direction, hard, tau,
                planner, layout, ctas, absorbed):
        raw = (q, k, sq, sk, v, lq, lk, iq, ik, direction, hard, tau)
        # Native validation before a background task is launched. Retain the
        # native soft checkpoint separately: hard output must not enter B1 delta.
        cu.prepare_operands(raw, layout is not None, absorbed)
        iq_c, ik_c = iq.to(torch.int32).contiguous(), ik.to(torch.int32).contiguous()
        ticket = planner.submit(iq_c, ik_c, hard, tau, layout=layout)
        out, norm, state = cu.core_forward(raw, None if layout is None else layout.table,
                                            ctas, True, False, absorbed)
        hard_out, _ = forward_hard(sq.contiguous(), sk.contiguous(), v.contiguous(), iq_c, ik_c,
                                   hard.contiguous(), tau.contiguous(), layout=layout, ctas=ctas)
        ctx.save_for_backward(*state['operands'], state['vertical'], out,
                              state['normalizer'] if layout is not None else norm,
                              sq, sk, v, iq, ik, hard, tau,
                              *([] if layout is None else [layout.table]), cu.hold_training_plan(ticket))
        ctx.layout, ctx.ctas = layout, ctas
        return out+hard_out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        saved = ctx.saved_tensors
        dout = dout.to(torch.bfloat16).contiguous()
        # Queue soft GPU work first, providing more slack before the host future.
        g, _ = cu.core_backward(saved[:13], saved[13], saved[14], saved[15], dout,
                                None if ctx.layout is None else ctx.layout.table,
                                ctx.ctas, False, False)
        from .hard_backward import backward_hard_recompute
        hg = cu.get_training_plan(saved[-1]).backward(*saved[16:19], dout,
            fallback=lambda: backward_hard_recompute(*saved[16:19],dout,*saved[19:23],
                                                      layout=ctx.layout,ctas=ctx.ctas))
        for name in ('sq_vec', 'sk_vec', 'v', 'rtau'):
            g[name] = g[name].float()+hg[name]
        return (g['q_vec'], g['k_vec'], g['sq_vec'], g['sk_vec'], g['v'],
                g['q_lse'], g['k_lse'], None, None, None, None, g['rtau'],
                None, None, None, None)


@torch.compiler.disable
def dism_core_hybrid(q, k, sq, sk, v, lq, lk, iq, ik, direction, hard, tau,
                     *, planner=None, layout=None, ctas=0, _lse_preabsorbed=False):
    """Complete eager training core with automatic/explicit planner ownership.

    Soft CUDA + integer hard forward; soft CUDA + event-stream hard backward.
    This explicit helper is a graph break; public default core/vocabulary APIs
    use their opaque compiled operators instead. First derivatives only.
    """
    if layout is not None and ctas:
        raise ValueError('ctas is only supported for fixed-length execution')
    if not torch.is_grad_enabled() or not any(x.requires_grad for x in (q,k,sq,sk,v,lq,lk,tau)):
        raw = (q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau)
        out, _, _ = cu.core_forward(raw, None if layout is None else layout.table,
                                    ctas, False, False, _lse_preabsorbed)
        if sq.shape[1] == 0:
            return out
        return out + forward_hard(sq.contiguous(),sk.contiguous(),v.contiguous(),
                                  iq.to(torch.int32).contiguous(),ik.to(torch.int32).contiguous(),
                                  hard.contiguous(),tau.contiguous(),layout=layout,ctas=ctas)[0]
    if sq.shape[1] == 0:
        from .varlen import _VarlenCore
        return _VarlenCore.apply(q,k,sq,sk,v,lq,lk,iq,ik,direction,hard,tau,layout,_lse_preabsorbed)
    planner = default_hard_planner(q.device) if planner is None else planner
    return _MixedCore.apply(q, k, sq, sk, v, lq, lk, iq, ik, direction, hard, tau,
                            planner, layout, ctas, _lse_preabsorbed)
