"""Explicit CPU hard-backward planning, shared by diagnostics and eager training.

Input labels must already be CPU-readable (wait for D2H completion first).
Native workers plan independent documents/heads, compute tau derivatives and
pack directly into pinned storage. Upload is one nonblocking H2D operation.
"""
from dataclasses import dataclass
import time

import numpy as np
import torch

from .build import load_prefill


def _specification(stats):
    streams, chunks, c = stats['streams'], stats['chunks'], stats['chunk_size']
    fields = [('offsets', torch.int64, streams+1), ('rows', torch.int32, chunks*c),
              ('prefixes', torch.float32, chunks*c), ('weights', torch.float32, chunks*c),
              ('dlog', torch.float32, chunks*c), ('lengths', torch.int32, chunks),
              ('resets', torch.int32, chunks), ('stream_heads', torch.int32, streams)]
    result, offset = {}, 0
    for name, dtype, count in fields:
        size = torch.empty((), dtype=dtype).element_size()*count
        result[name] = (dtype, offset, size)
        offset += size
    return result, offset


def _views(storage, spec):
    return {name: storage[offset:offset+size].view(dtype)
            for name, (dtype, offset, size) in spec.items()}


@dataclass
class DeviceTrainingPlan:
    storage: torch.Tensor
    data: dict
    statistics: dict
    ready: torch.cuda.Event

    def wait(self):
        """Insert a device-side dependency on the consuming stream, no host wait."""
        torch.cuda.current_stream(self.storage.device).wait_event(self.ready)
        # Protect the allocation if upload and consumer streams differ.
        self.storage.record_stream(torch.cuda.current_stream(self.storage.device))


@dataclass
class HostTrainingPlan:
    storage: torch.Tensor
    data: dict
    specification: dict
    statistics: dict
    timings: dict

    def upload(self, *, device='cuda', stream=None):
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('CUDA destination required')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        stream = torch.cuda.current_stream(device) if stream is None else stream
        if stream.device != device:
            raise ValueError('upload stream device mismatch')
        with torch.cuda.stream(stream):
            gpu = self.storage.to(device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
        return DeviceTrainingPlan(gpu, _views(gpu, self.specification), self.statistics, ready)


class NativeTrainingPlanner:
    """Reusable bounded C++ worker pool. Per-row/head hard flags, not per-doc.

    prepare() releases the GIL for native planning and packing. Concurrent calls
    share the bounded pool; input arrays must not be modified until it returns.
    An optional caller-owned pinned uint8 storage buffer can be reused, but the
    caller MUST wait for prior consumers/H2D copies before overwriting it.
    Default fresh allocations avoid that reuse hazard.
    """
    def __init__(self, workers=8):
        self.native = load_prefill().TrainingPlanner(workers)
        self.workers = workers

    def prepare(self, labels, tau, *, documents=None, hard=None, chunk_size=16, storage=None):
        start = time.perf_counter()
        if isinstance(labels, torch.Tensor):
            if labels.device.type != 'cpu' or labels.dtype != torch.int32 or not labels.is_contiguous():
                raise ValueError('CPU contiguous int32 labels required; explicitly complete D2H first')
            labels = labels.numpy()
        labels = np.asarray(labels)
        if labels.dtype != np.int32 or not labels.flags.c_contiguous or labels.ndim != 4 or labels.shape[0] != 2:
            raise ValueError('labels must be contiguous int32 [2,B,H,N]')
        _, b, h, n = labels.shape
        if min(b, h, n) <= 0:
            raise ValueError('nonempty labels required')
        for x in (tau, documents, hard):
            if isinstance(x, torch.Tensor) and x.device.type != 'cpu':
                raise ValueError('all planner inputs must already be on CPU')
        tau = np.ascontiguousarray(tau, dtype=np.float64)
        if documents is None:
            documents = np.array([(row, 0, n) for row in range(b)], dtype=np.int64)
        else:
            documents = np.asarray(documents)
            if documents.dtype != np.int64 or not documents.flags.c_contiguous:
                raise ValueError('documents must be contiguous int64 [D,3]')
        if hard is None:
            hard = np.ones((b, h), dtype=np.uint8)
        else:
            hard = np.asarray(hard)
            if hard.dtype not in (np.bool_, np.uint8):
                raise ValueError('hard must be bool/uint8 [B,H]')
            hard = np.ascontiguousarray(hard, dtype=np.uint8)
        t = time.perf_counter()
        native = self.native.plan(labels, tau, documents, hard, chunk_size)
        planned = time.perf_counter()
        stats = dict(native.statistics())
        spec, size = _specification(stats)
        if storage is None:
            storage = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        elif (storage.device.type != 'cpu' or storage.dtype != torch.uint8 or storage.ndim != 1
              or not storage.is_contiguous() or not storage.is_pinned() or storage.numel() < size):
            raise ValueError('storage must be a sufficiently large contiguous pinned CPU uint8 buffer')
        storage = storage[:size]
        data = _views(storage, spec)
        buffers = {name: tensor.numpy() for name, tensor in data.items()}
        allocated = time.perf_counter()
        native.pack_into(buffers)
        packed = time.perf_counter()
        del native
        finished = time.perf_counter()
        stats.update(metadata_bytes=size, workers=self.workers)
        timings = dict(input_ms=(t-start)*1000, plan_ms=(planned-t)*1000,
                       allocate_views_ms=(allocated-planned)*1000, pack_ms=(packed-allocated)*1000,
                       release_ms=(finished-packed)*1000, total_ms=(finished-start)*1000)
        return HostTrainingPlan(storage, data, spec, stats, timings)
