"""Native batch planning and one pinned metadata upload for hard prefill."""
from dataclasses import dataclass
import time

import numpy as np
import torch

from .build import load_prefill


def _specification(stats):
    s, c, width = stats['streams'], stats['chunks'], stats['chunk_size']
    fields = [('offsets', torch.int64, s+1), ('rows', torch.int32, c*width),
              ('prefixes', torch.float32, c*width), ('weights', torch.float32, c*width),
              ('lengths', torch.int32, c), ('resets', torch.int32, c),
              ('stream_ids', torch.int64, s), ('chunk_ids', torch.int64, c),
              ('local_offsets', torch.int64, s),
              ('short_stream_ids', torch.int64, stats['short_streams']),
              ('long_stream_ids', torch.int64, stats['long_streams']),
              ('long_chunk_ids', torch.int64, stats['long_chunks']),
              ('long_local_offsets', torch.int64, stats['long_streams'])]
    spec, offset = {}, 0
    for name, dtype, count in fields:
        offset = (offset+7)//8*8
        size = torch.empty((), dtype=dtype).element_size()*count
        spec[name] = dtype, offset, size
        offset += size
    return spec, offset


def _views(storage, specification):
    return {name: storage[start:start+size].view(dtype)
            for name, (dtype, start, size) in specification.items()}


@dataclass
class HostPrefillPlan:
    storage: torch.Tensor
    data: dict
    specification: dict
    statistics: dict
    timings: dict

    def upload(self, *, device, mma_precision):
        from .prefill_triton import TritonPrefillPlan
        # Fresh pinned storage belongs to this plan, never to mutable engine
        # scratch. Tensor.to records the async copy with PyTorch's pinned
        # allocator, preventing reuse until completion without retaining a CPU
        # metadata duplicate for the entire lifetime of a prepared GPU plan.
        gpu = self.storage.to(device, non_blocking=True)
        return TritonPrefillPlan.from_device_packed(
            _views(gpu,self.specification), self.statistics,
            mma_precision=mma_precision)


class NativePrefillPlanner:
    """Reuse the training pool and Builder scratch without tau derivatives.

    Inputs are CPU-readable before plan() starts. Jobs borrow arrays only until
    it returns; output owns all storage. The low-level Python parallel planner
    remains available as a reference, but production uses no per-head Python
    tasks, lists, exported event arrays or merge pass.
    """
    def __init__(self, workers=16):
        self.native = load_prefill().TrainingPlanner(workers)

    def plan(self, labels, tau, *, chunk_size=16, dispatch_threshold=64):
        if self.native is None:
            raise RuntimeError('prefill planner is closed')
        start = time.perf_counter()
        if isinstance(labels, torch.Tensor):
            if labels.device.type != 'cpu':
                raise ValueError('complete label D2H before CPU planning')
            labels = labels.numpy()
        labels = np.asarray(labels)
        if labels.dtype != np.int32 or not labels.flags.c_contiguous:
            raise ValueError('contiguous CPU int32 labels required')
        tau = np.ascontiguousarray(tau,dtype=np.float64)
        native = self.native.plan_prefill(labels,tau,chunk_size,int(dispatch_threshold))
        planned = time.perf_counter()
        stats = dict(native.statistics())
        spec, size = _specification(stats)
        # Alignment gaps are not interpreted by kernels, but zero-initialize
        # them so the uploaded blob never includes uninitialized host bytes.
        storage = torch.empty(size,dtype=torch.uint8,pin_memory=True)
        previous = 0
        for _, offset, length in spec.values():
            if offset > previous:
                storage[previous:offset].zero_()
            previous = offset+length
        data = _views(storage,spec)
        allocated = time.perf_counter()
        native.pack_into({name: value.numpy() for name,value in data.items()})
        del native
        finished = time.perf_counter()
        timings = dict(plan_ms=(planned-start)*1000,allocate_ms=(allocated-planned)*1000,
                       pack_release_ms=(finished-allocated)*1000)
        return HostPrefillPlan(storage,data,spec,stats,timings)

    def close(self):
        self.native = None
