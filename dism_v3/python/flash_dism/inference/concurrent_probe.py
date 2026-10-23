"""Opt-in, eager-only concurrent frozen-SAM rebuild prototype.

A single owner thread/stream appends tokens. One worker compiles a fixed prefix
and builds private summaries on a low-priority stream. KV/history prefixes are
immutable; the rolling band stays exclusively on the foreground stream. Polling
publishes only GPU-ready candidates and replays queries into the new matcher.
No model dispatch, automatic policy, or cross-version CUDA graph support.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import torch
from .build import load_summary_probe
from .gpu_runtime import GpuPlannerCache


@dataclass
class _Candidate:
    cutoff: int
    snapshot: list
    device: object
    ready: torch.cuda.Event
    begin: torch.cuda.Event
    end: torch.cuda.Event
    metrics: dict


class ConcurrentGpuPlannerCache:
    def __init__(self, heads, r, dv, capacity, tau, *, rebuild_interval=1024,
                 workers=64, slots=16, block=256, tile=128,
                 device='cuda', cache_dtype=torch.bfloat16):
        if rebuild_interval <= 0 or not 1 <= workers <= 1024 or not 1 <= slots <= 8192 or block <= 0:
            raise ValueError('positive interval/block, workers in [1,1024], slots in [1,8192] required')
        tau = list(tau)
        self.module = load_summary_probe()
        self.bootstrap = GpuPlannerCache(heads, r, dv, capacity, tau,
            rebuild_interval=rebuild_interval, device=device, cache_dtype=cache_dtype)
        self.snapshot = self.bootstrap.snapshot
        self.device = self.bootstrap.device
        self.owner = torch.cuda.current_stream(self.snapshot[0].device)
        self.background = torch.cuda.Stream(device=self.owner.device, priority=0)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dism-rebuild')
        self.history = torch.empty((capacity, 3, heads), dtype=torch.int32, pin_memory=True)
        self.heads, self.r, self.dv, self.capacity = heads, r, dv, capacity
        self.tau = list(tau)
        self.interval, self.workers, self.slots = rebuild_interval, workers, slots
        self.block, self.tile = block, tile
        self.position = self.cutoff = self._copied = 0
        self.pending = None
        self._cancelled = self._closed = self._primed = False
        self._error = None
        self._retired = []
        self.completed = []
        self.deadline_waits = 0
        self.deadline_wait_ms = 0.

    def _check_owner(self):
        if self._closed:
            raise RuntimeError('cache is closed')
        if torch.cuda.current_stream(self.owner.device) != self.owner:
            raise RuntimeError('foreground stream mismatch')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('concurrent prototype is eager-only; graph switching is unsupported')
        if self._error is not None:
            raise RuntimeError('background rebuild failed; close this cache') from self._error

    def prime(self, labels, sk, v):
        self._check_owner()
        if self._primed or self.position or self.pending is not None:
            raise RuntimeError('prime is allowed once on an empty cache')
        self.bootstrap.prime(labels, sk, v)
        self.snapshot, self.device = self.bootstrap.snapshot, self.bootstrap.device
        self.position = self.cutoff = sk.shape[1]
        self._primed = True

    def start_rebuild(self):
        """Capture a cutoff; caller chooses lead time. At most one job is live."""
        self._check_owner()
        self._collect_retired()
        if self.pending is not None:
            return False
        if self.position <= self.cutoff:
            return False
        cut = torch.cuda.Event()
        cut.record(self.owner)
        snapshot = self.snapshot
        # The background reads only immutable [0, cutoff) ranges. Record lifetime
        # on both streams even though their concurrent accesses are disjoint.
        for i in (0, 1, 7):
            snapshot[i].record_stream(self.background)
        self._cancelled = False
        self.pending = self.pool.submit(self._build, self.position, snapshot, cut, time.perf_counter())
        return True

    def _build(self, cutoff, old, cut, submitted):
        with torch.cuda.device(self.owner.device), torch.cuda.stream(self.background):
            self.background.wait_event(cut)
            started = time.perf_counter()
            self.history[self._copied:cutoff].copy_(old[7][self._copied:cutoff], non_blocking=True)
            copied = torch.cuda.Event()
            copied.record(self.background)
            copied.synchronize()  # Only this worker waits for D2H completion.
            self._copied = cutoff
            d2h_done = time.perf_counter()
            keys = self.history[:cutoff, 0].T.contiguous()
            p = self.module.SummaryProgram(keys, self.tau, self.r, self.dv,
                min(self.block, 2 * cutoff), self.tile, self.slots, True)
            compiled = time.perf_counter()
            summary = p.prepare(old[0], False)  # Private output; no diagnostic scratch.
            topology, edges, state, counts, tau = p.decode_metadata(self.history[:cutoff], self.capacity)
            snapshot = [old[0], old[1], summary, topology, edges, state,
                        old[6], old[7], counts, tau]
            device = self.module.PreparedGpuSnapshot(snapshot, cutoff)
            # All allocations (including task buffers) precede the long kernel.
            # A cudaMalloc during it can otherwise serialize foreground streams.
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            ready = torch.cuda.Event()
            prepared = time.perf_counter()
            begin.record(self.background)
            p.run(old[0], old[1], self.workers)
            end.record(self.background)
            ready.record(self.background)
            metrics = dict(cutoff=cutoff, submitted=submitted,
                worker_start_ms=(started-submitted)*1000,
                d2h_wait_ms=(d2h_done-started)*1000,
                compile_ms=(compiled-d2h_done)*1000,
                prepare_metadata_ms=(prepared-compiled)*1000,
                host_submit_ms=(time.perf_counter()-submitted)*1000,
                scratch_bytes=p.stats()['allocated_scratch_bytes'],
                output_bytes=p.stats()['output_bytes'])
            p.release_compiler()
            metrics['host_finished_ms'] = (time.perf_counter()-submitted)*1000
            # p's CPU SAM and workspace are destroyed on this worker. GPU scratch
            # was allocated/used on background; its allocator lifetime is safe.
            return _Candidate(cutoff, snapshot, device, ready, begin, end, metrics)

    def _collect_retired(self):
        self._retired[:] = [entry for entry in self._retired if not entry[0].query()]

    def try_publish(self, *, wait=False):
        """Nonblocking poll by default. Future completion alone is insufficient."""
        self._check_owner()
        self._collect_retired()
        if self.pending is None or (not wait and not self.pending.done()):
            return False
        try:
            candidate = self.pending.result()
        except Exception as exc:
            self._error = exc
            raise
        if wait:
            candidate.ready.synchronize()
        elif not candidate.ready.query():
            return False
        self.pending = None
        metrics = candidate.metrics
        metrics['summary_ms'] = candidate.begin.elapsed_time(candidate.end)
        metrics['summary_events'] = (candidate.begin, candidate.end)
        metrics['ready_seen_ms'] = (time.perf_counter()-metrics['submitted'])*1000
        metrics.pop('submitted')
        metrics['cancelled'] = self._cancelled
        if self._cancelled:
            self.completed.append(metrics)
            return False
        if not self.cutoff < candidate.cutoff <= self.position or self.position-candidate.cutoff > self.interval:
            raise RuntimeError('invalid candidate generation/horizon')
        self.owner.wait_event(candidate.ready)
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record(self.owner)
        candidate.device.activate_and_catch_up(self.snapshot[5])
        end.record(self.owner)
        # Catchup reads old state. Retire only after that read completes, not
        # merely after background summary completion.
        self._retired.append((end, self.snapshot, self.device))
        self.snapshot, self.device, self.cutoff = candidate.snapshot, candidate.device, candidate.cutoff
        metrics['handoff_position'] = self.position
        metrics['lag_tokens'] = self.position-self.cutoff
        metrics['catchup_events'] = (begin, end)
        self.completed.append(metrics)
        return True

    def cancel_rebuild(self):
        """Invalidate publication; submitted CUDA work is drained, never aborted."""
        self._check_owner()
        if self.pending is None:
            return False
        self._cancelled = True
        return True

    def step(self, labels, sk, sq, v):
        self._check_owner()
        if self.position >= self.capacity:
            raise RuntimeError('cache capacity exhausted')
        self.try_publish()
        if self.position-self.cutoff >= self.interval:
            started = time.perf_counter()
            self.deadline_waits += 1
            if self.pending is not None:
                self.try_publish(wait=True)
            if self.position-self.cutoff >= self.interval:
                self.start_rebuild()
                self.try_publish(wait=True)
            self.deadline_wait_ms += (time.perf_counter()-started)*1000
        out = self.device.step(labels, sk, sq, v)
        self.position += 1
        return out

    def check_status(self):
        self._check_owner()
        state = self.snapshot[5].cpu()
        if state[:, 6].any() or not (state[:, 0] == self.position).all():
            raise RuntimeError('GPU planner error or host/device position mismatch')
        return self.position

    def stats(self):
        """Completed jobs only; query timing without forcing a foreground wait."""
        jobs = []
        for metrics in self.completed:
            row = {k: v for k, v in metrics.items() if k not in ('catchup_events', 'summary_events')}
            events = metrics.get('catchup_events')
            if events and events[1].query():
                row['catchup_ms'] = events[0].elapsed_time(events[1])
            jobs.append(row)
        return dict(position=self.position, cutoff=self.cutoff, jobs=jobs,
                    deadline_waits=self.deadline_waits, deadline_wait_ms=self.deadline_wait_ms)

    def close(self):
        if self._closed:
            return
        # Also safe after worker errors: drain streams before releasing storage.
        self.pool.shutdown(wait=True)
        self.background.synchronize()
        self.owner.synchronize()
        self.pending = None
        self._retired.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
