"""Explicit graph planner runtime (P2); not installed in the model dispatcher.

Lifecycle: capture snapshot()/finish() without CPU work; arm(layout=...), then
runtime.replay(graph) on the same stream. Drain before optimizer mutation
or slot reuse. One generation per slot; CPU and all CUDA work retire together.
"""
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import threading

import numpy as np
import torch

from .inference.training_planner import NativeTrainingPlanner, PlanBudgetExceeded
from .kernels.planner_graph import GraphCompactBackward, sample_route


class StreamWriter:
    def __init__(self):
        self.library = ctypes.CDLL('libcuda.so.1')
        self.write = self.library.cuStreamWriteValue32_v2
        self.write.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        self.write.restype = ctypes.c_int
        self.error_name = self.library.cuGetErrorName
        self.error_name.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.error_name.restype = ctypes.c_int

    def publish(self, stream, flag, epoch):
        if (flag.dtype != torch.int32 or flag.numel() != 1 or not flag.is_cuda
                or flag.device != stream.device or flag.data_ptr() % 4
                or not 0 < epoch < 2**32):
            raise ValueError('aligned CUDA int32 flag, matching stream and nonzero uint32 epoch required')
        status = self.write(stream.cuda_stream, flag.data_ptr(), epoch, 0)
        if status:
            name = ctypes.c_char_p()
            self.error_name(status, ctypes.byref(name))
            raise RuntimeError(f'cuStreamWriteValue32_v2: {status} {name.value!r}')


@dataclass
class GraphPlanPayload:
    host: object
    device: object
    owners: list
    descriptor: torch.Tensor
    bytes: int


def upload_plan(host, *, device, stream, budget_bytes):
    # If an allocation/copy fails midway, retire queued copies before local
    # pinned owners unwind. Normal publication remains asynchronous.
    owners = [host]
    try:
        return _upload_plan(host, device=device, stream=stream, budget_bytes=budget_bytes, owners=owners)
    except Exception:
        stream.synchronize()
        raise


def _upload_plan(host, *, device, stream, budget_bytes, owners):
    """Prepare all indirect pointers before publication; caller retains owners.

    Budget includes pinned+device metadata, auxiliary indices, descriptor and
    GPU workspace. Native planner's temporary CPU algorithm storage is separate.
    """
    stats = host.statistics
    if stats['chunk_size'] != 64 or stats['tau_rows'] != 1:
        raise ValueError('graph plan requires compact C64 with per-head tau')
    blocks = list(map(int, stats['bucket_blocks']))
    streams, chunks = int(stats['multi_streams']), int(stats['multi_chunks'])
    if max([b*w for b,w in zip(blocks,(16,32,64))] +
           [streams, chunks*64, stats['batch']*stats['heads']*stats['n']*64]) > 2**31-1:
        raise ValueError('graph plan exceeds int32 task/index range')
    # IDs/owners (host+device), h/g/alpha, tau slots, descriptor (host+device).
    size = 2*host.storage.numel() + 16*chunks + 4*chunks*(2*32*64+1) + 4*(sum(blocks)+streams) + 1024
    if size > budget_bytes:
        raise PlanBudgetExceeded(f'plan/workspace requires {size} bytes, budget {budget_bytes}')
    words = []
    with torch.cuda.stream(stream):
        gpu = host.upload(device=device, stream=stream)
        owners.append(gpu)
        def upload(array):
            cpu = torch.from_numpy(np.ascontiguousarray(array)).pin_memory()
            dst = cpu.to(device, non_blocking=True)
            owners.extend((cpu, dst))
            return dst
        def empty(shape):
            dst = torch.empty(shape, dtype=torch.float32, device=device)
            owners.append(dst)
            return dst
        for i, count in enumerate(blocks + [chunks]):
            data = gpu.buckets[i] if i < 3 else gpu.multi
            record = [count, count if i < 3 else streams]
            record += [data[k].data_ptr() for k in ('rows', 'prefixes', 'weights', 'dlog', 'lengths', 'stream_heads')]
            if i < 3:
                record += [0]*7
            else:
                offsets = host.multi['offsets'].numpy()
                ids = upload(np.arange(chunks, dtype=np.int32))
                which = upload(np.repeat(np.arange(streams, dtype=np.int32), np.diff(offsets)))
                record += [data['offsets'].data_ptr(), data['resets'].data_ptr(), ids.data_ptr(), which.data_ptr(),
                           empty((chunks, 32, 64)).data_ptr(), empty((chunks, 32, 64)).data_ptr(), empty(chunks).data_ptr()]
            record.append(empty(record[1]).data_ptr())
            words.extend(record)
        desc = torch.tensor(words, dtype=torch.int64, pin_memory=True)
    return GraphPlanPayload(host, gpu, owners, desc, size)


class GraphPlannerRuntime:
    """Bounded slots and native workers. Explicit close/drain; never checkpointed.

    Notifications run independently per slot, so waiting for one layer's CPU
    plan cannot delay dispatch for another. Not a reentrant replay runner.
    """
    def __init__(self, *, device='cuda', workers=16, max_slots=8, max_bytes=2**30):
        if max_slots < 1 or max_bytes < 1024*max_slots:
            raise ValueError('positive slots and sufficient byte budget required')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA runtime required')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.native = NativeTrainingPlanner(workers)
        self.writer = StreamWriter()
        self.pool = ThreadPoolExecutor(max_workers=max_slots, thread_name_prefix='dism-graph-plan')
        self.max_slots, self.slot_budget = max_slots, max_bytes//max_slots
        self.slots, self.closed = [], False

    def slot(self, *, rows, heads, row_tokens):
        if self.closed or len(self.slots) >= self.max_slots:
            raise RuntimeError('graph planner closed or slot limit reached')
        result = GraphPlannerSlot(self, rows, heads, row_tokens)
        self.slots.append(result)
        return result

    def drain(self):
        errors = []
        for slot in self.slots:
            try:
                slot.drain()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def replay(self, graph, *, slots=None):
        """Submit a complete graph and dispatch only after launch returns.

        Each participating slot must already be armed and its finish() must be
        in this graph. This is the preferred API: even dispatch failure retains
        every slot until GPU completion. Manual launched() is for split probes.
        """
        slots = self.slots if slots is None else list(slots)
        if (self.closed or not slots or len(set(slots)) != len(slots)
                or any(s.runtime is not self or s.state != 'armed' for s in slots)):
            raise RuntimeError('replay requires distinct armed slots of this runtime')
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('cannot replay during capture')
            try:
                graph.replay()
            except Exception:
                # No coordinator may wait on a record node from a failed launch.
                torch.cuda.current_stream().synchronize()
                raise
            for slot in slots:
                slot.state = 'inflight'
            failed = False
            for slot in slots:
                try:
                    slot.future = self.pool.submit(slot._produce)
                except Exception as exc:
                    slot.error, failed = exc, True
            if failed:
                self.drain()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.drain()
        finally:
            self.pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class GraphPlannerSlot:
    def __init__(self, runtime, rows, heads, row_tokens):
        if min(rows, heads, row_tokens) < 1 or row_tokens % 256 or rows*heads*row_tokens*64 > 2**31-1:
            raise ValueError('positive int32-addressable dimensions and 256-aligned rows required')
        self.runtime, self.device = runtime, runtime.device
        self.rows, self.heads, self.row_tokens = rows, heads, row_tokens
        snapshot_bytes = 2*rows*heads*row_tokens*4 + rows*heads + heads*4
        self.budget = runtime.slot_budget - 2*snapshot_bytes - 64*8 - 12
        if self.budget < 1024:
            raise ValueError('snapshot exceeds slot byte budget')
        with torch.cuda.device(self.device):
            self.hosts = (torch.empty((2, rows, heads, row_tokens), dtype=torch.int32, pin_memory=True),
                          torch.empty((rows, heads), dtype=torch.bool, pin_memory=True),
                          torch.empty(heads, dtype=torch.float32, pin_memory=True))
            self.snapshots = tuple(torch.empty_like(x, device=self.device) for x in self.hosts)
            self.expected = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.ready, self.choice = torch.zeros_like(self.expected), torch.zeros_like(self.expected)
            self.descriptor = torch.zeros(64, dtype=torch.int64, device=self.device)
            self.input_ready, self.finished = (torch.cuda.Event(external=True) for _ in range(2))
            self.upload = torch.cuda.Stream(device=self.device)
        self.consumer = GraphCompactBackward(self.descriptor, self.choice)
        self.epoch, self.state, self.future = 0, 'idle', None
        self.payload, self.reason, self.error = None, None, None
        self.cancelled = threading.Event()

    def _check_open(self):
        if self.runtime.closed:
            raise RuntimeError('graph planner is closed')
        if self.error is not None:
            raise RuntimeError('graph planner slot failed; rebuild runtime') from self.error

    def arm(self, *, layout=None):
        """Before replay: retire old writers; privately copy CPU document table."""
        self._check_open()
        if self.state == 'armed':
            raise RuntimeError('slot is already armed; drain an unsubmitted replay first')
        self.drain()
        documents = []
        if layout is None:
            documents = [(r, 0, self.row_tokens) for r in range(self.rows)]
        else:
            if layout.tokens != self.rows*self.row_tokens:
                raise ValueError('layout token count mismatch')
            cursor = 0
            for start, length, *_ in layout.table.tolist():
                if start != cursor or length < 0 or start % 256 or length % 256:
                    raise ValueError('invalid aligned document layout')
                cursor += length
                if not length:
                    continue
                row = start//self.row_tokens
                if row >= self.rows or (start+length-1)//self.row_tokens != row:
                    raise ValueError('document crosses original training row')
                documents.append((row, start % self.row_tokens, start % self.row_tokens+length))
            if cursor != self.rows*self.row_tokens:
                raise ValueError('documents must cover packed rows')
        self.documents = np.array(documents, dtype=np.int64, copy=True).reshape(-1, 3)
        with torch.cuda.device(self.device):
            if self.epoch == 2**32-1:
                # drain above includes every old writer and reader, before ABA reset.
                self.ready.zero_()
                self.epoch = 0
            self.epoch += 1
            signed = self.epoch if self.epoch < 2**31 else self.epoch-2**32
            self.expected.fill_(signed)
        self.state, self.reason, self.payload = 'armed', None, None
        self.cancelled.clear()

    def snapshot(self, iq, ik, hard, tau):
        """Capture-only/eager GPU nodes. Never dispatch CPU work during capture."""
        r, h, n = self.rows, self.heads, self.row_tokens
        packed = iq.shape == (1, h, r*n)
        if (iq.shape != ik.shape or (not packed and iq.shape != (r, h, n))
                or hard.shape != (r, h) or tau.shape != (h,)
                or [x.dtype for x in (iq, ik, hard, tau)] != [torch.int32, torch.int32, torch.bool, torch.float32]
                or any(x.device != self.device or not x.is_contiguous() for x in (iq, ik, hard, tau))):
            raise ValueError('invalid graph planner snapshot inputs')
        def unpack(x):
            return x[0].reshape(h, r, n).permute(1, 0, 2) if packed else x
        self.snapshots[0][0].copy_(unpack(iq))
        self.snapshots[0][1].copy_(unpack(ik))
        self.snapshots[1].copy_(hard)
        self.snapshots[2].copy_(tau)
        for host, gpu in zip(self.hosts, self.snapshots):
            host.copy_(gpu, non_blocking=True)
        self.input_ready.record()

    def sample(self):
        return sample_route[(1,)](self.ready, self.expected, self.choice, num_warps=1)

    def finish(self):
        """Capture after ALL consumers; external completion bounds payload life."""
        self.finished.record()

    def backward(self, sq, sk, value, dout, iq, ik, hard, tau, *, layout=None, ctas=0):
        """P4 standalone hard backward; P5 will bind this to core/vocabulary.

        Private branch buffers are initialized each replay; gated kernels never
        clear the other branch. No host route read, wait, or planner inspection.
        """
        from .hard_backward import backward_hard_recompute
        if sq.ndim != 4 or sq.shape[2] != self.heads or sq.numel() != self.rows*self.row_tokens*self.heads*32:
            raise ValueError('graph backward vectors must match slot geometry')
        self.sample()
        event = self.consumer.backward(*[x.reshape(-1, x.shape[-1]) for x in (sq, sk, value, dout)], heads=self.heads)
        quadratic = backward_hard_recompute(sq, sk, value, dout, iq, ik, hard, tau,
            layout=layout, ctas=ctas, _graph_choice=self.choice)
        return {key: event[key].view_as(quadratic[key]) + quadratic[key].float()
                for key in ('sq_vec', 'sk_vec', 'v', 'rtau')}

    def launched(self):
        """Call only AFTER replay submission returns, never while capturing."""
        self._check_open()
        if self.state != 'armed' or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('arm then submit replay before launched()')
        self.state = 'inflight'
        try:
            self.future = self.runtime.pool.submit(self._produce)
        except Exception as exc:
            self.error = exc
            self.drain()

    def cancel(self):
        """Suppress unsubmitted upload; already enqueued work must still drain."""
        self.cancelled.set()

    def _produce(self):
        with torch.cuda.device(self.device), torch.no_grad():
            try:
                self.input_ready.synchronize()
                if self.cancelled.is_set():
                    self.reason = 'cancelled'
                    return
                labels, hard, tau = self.hosts
                host = self.runtime.native.prepare(labels, tau, hard=hard, documents=self.documents,
                    chunk_size=64, compact=True, max_metadata_bytes=self.budget//2)
                if self.cancelled.is_set():
                    self.reason = 'cancelled'
                    return
                # Keep allocations and pinned sources alive even on upload errors.
                with torch.cuda.stream(self.upload):
                    self.payload = upload_plan(host, device=self.device, stream=self.upload, budget_bytes=self.budget)
                    self.descriptor.copy_(self.payload.descriptor, non_blocking=True)
                    self.runtime.writer.publish(self.upload, self.ready, self.epoch)
            except PlanBudgetExceeded as exc:
                self.reason = str(exc)
            finally:
                self.upload.synchronize()

    def drain(self):
        """Required before optimizer/checkpoint. Errors are sticky, never fallback-only."""
        if self.state == 'inflight':
            try:
                self.finished.synchronize()
                if self.future is not None:
                    self.future.result()
            except Exception as exc:
                self.error = exc
                raise
            finally:
                # Future must finish even if GPU completion failed.
                try:
                    if self.future is not None:
                        self.future.result()
                finally:
                    self.upload.synchronize()
                    self.state, self.future, self.payload = 'idle', None, None
        elif self.state == 'armed':
            # No replay was submitted: never wait on an unrecorded external event.
            self.state = 'idle'
        if self.error is not None:
            raise RuntimeError('graph planner slot failed; rebuild runtime') from self.error
