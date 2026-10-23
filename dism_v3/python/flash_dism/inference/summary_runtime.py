"""Opt-in synchronous persistent summary rebuild, with stream-owned scratch.

Native prime seeds complete history/band. Subsequent rebuilds compile a frozen
prefix and run the compact-slot summary interpreter, then replace the snapshot.
The native CPU SAM/pool and metadata workspaces survive across rebuilds.
"""
import time
import torch
from .gpu_runtime import GpuPlannerCache


class SummaryWorkspace:
    """Reusable scratch only; active snapshot outputs never alias this pool.

    One owner stream serializes every user (including different model layers).
    Not suitable for concurrent rebuilds on separate streams.
    """
    def __init__(self, device):
        self.stream = torch.cuda.current_stream(device)
        self.buffers = {}

    def check_owner(self):
        if torch.cuda.current_stream(self.stream.device) != self.stream:
            raise RuntimeError('summary workspace stream mismatch')

    @property
    def reserved_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.buffers.values())


class PersistentGpuPlannerCache(GpuPlannerCache):
    def __init__(self, heads, r, dv, capacity, tau, *, workspace=None,
                 summary_workers=512, summary_slots=16, summary_block=256,
                 summary_tile=128, **kwargs):
        if not 1 <= summary_workers <= 1024 or not 1 <= summary_slots <= 8192:
            raise ValueError('invalid summary workers/slots')
        if summary_block <= 0 or summary_tile not in (32,64,128,256):
            raise ValueError('invalid summary block/tile')
        if kwargs.get('sample_interval') not in (None,4*r) or kwargs.get('materialize_threshold') not in (None,5*r):
            raise ValueError('persistent summary requires default sample spacing/materialization threshold')
        super().__init__(heads,r,dv,capacity,tau,**kwargs)
        self.workspace = workspace if workspace is not None else SummaryWorkspace(self.snapshot[0].device)
        self.workspace.check_owner()
        self.heads,self.r,self.dv,self.capacity = heads,r,dv,capacity
        self.tau = list(tau)
        self.workers,self.slots,self.block,self.tile = summary_workers,summary_slots,summary_block,summary_tile
        self.rebuilds = 0
        self.last_rebuild = {}
        self.failed = False

    def prime(self, labels, sk, v):
        if self.failed:
            raise RuntimeError('persistent cache cannot be re-primed')
        return super().prime(labels,sk,v)

    def step(self, labels, sk, sq, v):
        if self.failed:
            raise RuntimeError('persistent rebuild failed; discard cache')
        return super().step(labels,sk,sq,v)

    def rebuild(self):
        self.workspace.check_owner()
        if self.failed or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('persistent rebuild requires a healthy eager cache')
        started = time.perf_counter()
        try:
            position = self.check_status()
            if position <= 0:
                raise ValueError('persistent rebuild requires nonempty history')
            old = self.snapshot
            self.cpu.native.refresh_gpu(old[7],position,old[6],False)
            refreshed = time.perf_counter()
            native_stats = dict(self.cpu.memory_stats())
            program = self.cpu.native.summary_program(self.block,self.tile,self.slots)
            compiled = time.perf_counter()
            output = program.prepare(old[0],False,self.workspace.buffers)
            self.workspace.buffers = program.workspace()
            self.cpu.native.install_summary(output)
            snapshot = self.cpu.native.gpu_snapshot()
            snapshot[6] = old[6]  # Keep the live GPU rolling-band allocation.
            device = self.extension.GpuPlanner(snapshot,position)
            prepared = time.perf_counter()
            program.run(old[0],old[1],self.workers)
            stats = dict(program.stats())
            enqueued = time.perf_counter()
            program.release_compiler()
            released = time.perf_counter()
            self.snapshot,self.device = snapshot,device
            self.rebuilds += 1
            self.last_rebuild = dict(position=position,delta_labels=native_stats['last_refresh_tokens'],
                sam_extended_keys=native_stats['last_sam_extended_keys'],
                incremental_refresh_ms=(refreshed-started)*1000,compile_ms=(compiled-refreshed)*1000,
                prepare_ms=(prepared-compiled)*1000,
                compiler_release_ms=(released-enqueued)*1000,host_ms=(time.perf_counter()-started)*1000,
                **stats)

        except Exception:
            self.failed = True
            raise

    def memory_stats(self):
        return dict(planner_backend='gpu',summary_backend='persistent',rebuilds=self.rebuilds,
            position=self.check_status(),scratch_reserved_bytes=self.workspace.reserved_bytes,
            last_rebuild=self.last_rebuild)
