"""Experimental GPU ordinary steps; CPU snapshot rebuilding remains explicit.

step() launches exactly one kernel, does not synchronize/copy to host, and returns
a REUSED FP32 output buffer. State/position live on device, so graph replay advances
correctly. A snapshot supports at most rebuild_interval new tokens. Call rebuild()
outside a graph before that boundary; check_status() synchronizes and checks errors.
Neither method is automatically run by step(). Existing NativeDecodeCache unchanged.
Rebuild changes captured pointers: discard and recapture graphs after rebuilding.
"""
import torch
from .runtime import NativeDecodeCache


class GpuPlannerCache:
    def __init__(self, *args, **kwargs):
        self.cpu = NativeDecodeCache(*args, **kwargs)
        self.extension = self.cpu.extension
        self._export(0)

    def _export(self, position):
        self.snapshot = self.cpu.native.gpu_snapshot()
        self.device = self.extension.GpuPlanner(self.snapshot, position)

    def prime(self, labels, sk, v):
        self.cpu.prime(labels, sk, v)
        self._export(sk.shape[1])

    def step(self, labels, sk, sq, v):
        return self.device.step(labels, sk, sq, v)

    def check_status(self):
        state = self.snapshot[5].cpu()
        if state[:, 6].any():
            raise RuntimeError('GPU planner capacity/rebuild horizon/reset error; create a new cache after error')
        if not torch.equal(state[:, 0], state[:1, 0].expand_as(state[:, 0])):
            raise RuntimeError('head positions disagree')
        return int(state[0, 0])

    def rebuild(self):
        position = self.check_status()
        self.cpu.native.refresh_gpu(self.snapshot[7], position)
        self._export(position)
