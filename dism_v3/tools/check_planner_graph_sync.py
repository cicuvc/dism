"""P1 only: graph D2H notification, stream publication and immutable routing.

No DISM kernels, model, optimizer or training process are involved. The driver
call uses ctypes; the small GPU consumers use Triton, so no native build or
production extension changes are needed. Generated artifacts belong in build/.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
from pathlib import Path
import subprocess

import torch
import triton
import triton.language as tl


@triton.jit
def _acquire(ptr):
    return tl.inline_asm_elementwise(
        'ld.acquire.sys.global.u32 $0, [$1];',
        constraints='=r,l', args=[ptr], dtype=tl.uint32, is_pure=False, pack=1)


@triton.jit
def _release_one(ptr, value):
    # Exactly one publisher in the sampling CTA, even if a live flag changes
    # while other lanes execute their loads. Consumers never reload that flag.
    return tl.inline_asm_elementwise(
        '{ .reg .pred p; .reg .u32 t; mov.u32 t, %tid.x; '
        'setp.eq.u32 p, t, 0; @p st.release.sys.global.u32 [$1], $2; '
        'mov.u32 $0, 0; }', constraints='=r,l,r', args=[ptr, value],
        dtype=tl.uint32, is_pure=False, pack=1)


@triton.jit
def _make_snapshot(Control, Snapshot):
    i = tl.arange(0, 256)
    epoch, count, salt = tl.load(Control), tl.load(Control+1), tl.load(Control+2)
    value = epoch*1009 + i*17 + salt
    value = tl.where(i == 0, epoch, tl.where(i == 1, count, tl.where(i == 2, salt, value)))
    tl.store(Snapshot+i, value)


@triton.jit
def _sample_route(Ready, Control, Choice, Observed):
    seen = _acquire(Ready)
    epoch = tl.load(Control).to(tl.uint32)
    tl.store(Observed, seen)
    _release_one(Choice, (seen == epoch).to(tl.uint32))


@triton.jit
def _consume(Descriptor, Control, Choice, Output, Hits,
             EVENT: tl.constexpr, BLOCK: tl.constexpr):
    choice = _acquire(Choice)
    if choice == EVENT:
        pid = tl.program_id(0)
        # Deliberately poisoned/null descriptors are safe on the fallback path.
        if EVENT:
            payload = tl.load(Descriptor).to(tl.pointer_type(tl.int32))
            count = tl.load(Descriptor+1).to(tl.int32)
            stamp = tl.load(Descriptor+2).to(tl.int32)
        else:
            count = tl.load(Control+1)
            stamp = tl.load(Control)
        for tile in range(pid, tl.cdiv(count, BLOCK), tl.num_programs(0)):
            index = tile*BLOCK + tl.arange(0, BLOCK)
            if EVENT:
                value = tl.load(payload+index, index < count, 0)
                # Both payload and descriptor must come from this publication.
                value += stamp
            else:
                value = -stamp*1009-index*17-7
            tl.store(Output+index, value, index < count)
        tl.store(Hits+EVENT*tl.num_programs(0)+pid, 1)


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
        if flag.dtype != torch.int32 or flag.numel() != 1 or not flag.is_cuda:
            raise ValueError('one CUDA int32 publication flag required')
        if flag.data_ptr() % 4 or not 0 < epoch < 2**32:
            raise ValueError('aligned flag and nonzero uint32 epoch required')
        status = self.write(stream.cuda_stream, flag.data_ptr(), epoch, 0)
        if status:
            name = ctypes.c_char_p()
            self.error_name(status, ctypes.byref(name))
            raise RuntimeError(f'cuStreamWriteValue32_v2: {status} {name.value!r}')


class SyncProbe:
    """One fixed slot, retired before reuse; full and controlled partial graphs."""
    def __init__(self, capacity=262144, ctas=32, window_cycles=20_000_000):
        if capacity < 257 or ctas < 1 or window_cycles < 0:
            raise ValueError('capacity >= 257, ctas > 0, window_cycles >= 0 required')
        self.device = torch.cuda.current_device()
        self.capacity, self.ctas = capacity, ctas
        self.window_cycles = window_cycles
        self.stream, self.upload = torch.cuda.Stream(), torch.cuda.Stream()
        self.writer = StreamWriter()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sync-probe')
        self.control = torch.zeros(3, device='cuda', dtype=torch.int32)
        self.snapshot = torch.empty(256, device='cuda', dtype=torch.int32)
        self.host_snapshot = torch.empty_like(self.snapshot, device='cpu', pin_memory=True)
        self.ready = torch.zeros(1, device='cuda', dtype=torch.int32)
        self.choice = torch.zeros_like(self.ready)
        self.observed = torch.zeros_like(self.ready)
        self.descriptor = torch.zeros(3, device='cuda', dtype=torch.int64)
        self.payloads = [torch.empty(capacity, device='cuda', dtype=torch.int32) for _ in range(2)]
        self.output = torch.empty(capacity, device='cuda', dtype=torch.int32)
        self.hits = torch.empty((2, ctas), device='cuda', dtype=torch.int32)
        self.input_ready = torch.cuda.Event(external=True)
        self.sampled = torch.cuda.Event(external=True)
        self.finished = torch.cuda.Event(external=True)
        self.upload_done = torch.cuda.Event()
        self.kernels = {}
        self.stream.wait_stream(torch.cuda.current_stream())
        self.upload.wait_stream(torch.cuda.current_stream())
        # Warm all JIT entries before capture. No coordinator runs in capture.
        with torch.cuda.stream(self.stream):
            self.control.copy_(torch.tensor([1, capacity, 19], device='cuda', dtype=torch.int32))
            self._input()
            self._sample()
            self._consumers()
        self.stream.synchronize()
        self.full = self._capture(self._full)
        self.input_graph = self._capture(self._input)
        self.sample_graph = self._capture(self._sample)
        self.consumer_graph = self._capture(self._consumers)

    def _capture(self, function):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            function()
        return graph

    def _input(self):
        self.kernels['snapshot'] = _make_snapshot[(1,)](self.control, self.snapshot)
        self.host_snapshot.copy_(self.snapshot, non_blocking=True)
        self.input_ready.record()

    def _sample(self):
        self.kernels['sample'] = _sample_route[(1,)](
            self.ready, self.control, self.choice, self.observed)
        self.sampled.record()

    def _consumers(self):
        self.output.fill_(-1)
        self.hits.zero_()
        for event in (True, False):
            self.kernels['event' if event else 'fallback'] = _consume[(self.ctas,)](
                self.descriptor, self.control, self.choice, self.output, self.hits,
                event, 256, num_warps=4)
        self.finished.record()

    def _full(self):
        self._input()
        # Finite, test-only scheduling window. It NEVER reads a CPU flag or
        # waits for publication; a missing producer still reaches fallback.
        if self.window_cycles:
            torch.cuda._sleep(self.window_cycles)
        self._sample()
        self._consumers()

    def stage(self, epoch, count, salt=19):
        if not 0 < epoch < 2**31 or not 0 <= count <= self.capacity:
            raise ValueError('probe epoch/count out of range')
        self.stream.synchronize()
        self.upload.synchronize()
        self.expected = (epoch, count, salt)
        # Deliberately keep the OLD ready flag. Generation equality, not a
        # reset-to-zero coincidence, must reject the preceding publication.
        with torch.cuda.stream(self.stream):
            self.control.copy_(torch.tensor(self.expected, dtype=torch.int32, pin_memory=True),
                               non_blocking=True)
            self.descriptor.zero_()

    def replay(self, graph):
        with torch.cuda.stream(self.stream):
            graph.replay()

    def publish_from_snapshot(self, expected, *, split_upload=False):
        with torch.cuda.device(self.device):
            self.input_ready.synchronize()
            epoch, count, salt = expected
            snapshot = self.host_snapshot.clone()
            wanted = torch.arange(256, dtype=torch.int32)*17+epoch*1009+salt
            wanted[:3] = torch.tensor(expected, dtype=torch.int32)
            torch.testing.assert_close(snapshot, wanted, atol=0, rtol=0)
            before_sample = not self.sampled.query()
            graph_incomplete = not self.finished.query()
            host = torch.arange(count, dtype=torch.int32).pin_memory()
            host.mul_(17).add_(epoch*1009+salt)
            payload = self.payloads[epoch % 2]
            descriptor = torch.tensor([payload.data_ptr(), count, epoch],
                                      dtype=torch.int64, pin_memory=True)
            with torch.cuda.stream(self.upload):
                if split_upload:
                    middle = count//2
                    payload[:middle].copy_(host[:middle], non_blocking=True)
                    torch.cuda._sleep(1_000_000)
                    payload[middle:count].copy_(host[middle:], non_blocking=True)
                else:
                    payload[:count].copy_(host, non_blocking=True)
                self.descriptor.copy_(descriptor, non_blocking=True)
                self.writer.publish(self.upload, self.ready, epoch)
                self.upload_done.record()
            # Keep all host/device owners live through upload and publication.
            self.upload_done.synchronize()
            return dict(input_matches=True, input_before_sample=before_sample,
                        input_before_graph_end=graph_incomplete)

    def check(self, expected_route=None):
        self.stream.synchronize()
        epoch, count, salt = self.expected
        route = int(self.choice.item())
        assert route in (0, 1), route
        if expected_route is not None:
            assert route == expected_route, (epoch, route, expected_route)
        seen = int(self.observed.item())
        assert route == int(seen == epoch), (epoch, seen, route)
        wanted = torch.full((self.capacity,), -1, dtype=torch.int32)
        index = torch.arange(count, dtype=torch.int32)
        wanted[:count] = index*17+epoch*1010+salt if route else -epoch*1009-index*17-7
        torch.testing.assert_close(self.output.cpu(), wanted, atol=0, rtol=0)
        hits = self.hits.cpu()
        assert torch.all(hits[route] == 1) and torch.count_nonzero(hits[1-route]) == 0
        return dict(epoch=epoch, count=count, route=route, observed=seen)

    def run_control(self, mode, epoch, count):
        self.stage(epoch, count)
        self.replay(self.input_graph)
        self.input_ready.synchronize()
        if mode == 'ready':
            self.publish_from_snapshot(self.expected, split_upload=True)
            self.replay(self.sample_graph)
            expected_route = 1
        elif mode in ('late_after_sample', 'partial_upload'):
            if mode == 'partial_upload':
                # Valid pointer/count but incomplete data, WITHOUT publication.
                # This is a deterministic partial-upload control, not timing.
                payload = self.payloads[epoch % 2]
                with torch.cuda.stream(self.upload):
                    payload.fill_(-123456)
                    payload[:count//2].fill_(epoch)
                    self.descriptor.copy_(torch.tensor([payload.data_ptr(), count, epoch],
                                                       dtype=torch.int64, pin_memory=True),
                                          non_blocking=True)
                self.upload.synchronize()
            self.replay(self.sample_graph)
            self.sampled.synchronize()
            # The live flag becomes ready BEFORE either branch starts. Both
            # must still obey the earlier fallback snapshot.
            self.publish_from_snapshot(self.expected, split_upload=True)
            expected_route = 0
        elif mode == 'stale':
            self.replay(self.sample_graph)
            expected_route = 0
        else:
            raise ValueError(mode)
        self.replay(self.consumer_graph)
        return dict(mode=mode, **self.check(expected_route))

    def run_async(self, epoch, count, *, after_graph=False, split_upload=False):
        self.stage(epoch, count)
        self.replay(self.full)
        if after_graph:
            self.finished.synchronize()
        # Submit ONLY after graph launch: event must refer to this invocation.
        future = self.pool.submit(self.publish_from_snapshot, self.expected,
                                  split_upload=split_upload)
        try:
            publication = future.result(timeout=30)
        finally:
            self.stream.synchronize()
        result = self.check(0 if after_graph else None)
        return dict(mode='after_graph' if after_graph else 'async', **result, **publication)

    def close(self):
        self.pool.shutdown(wait=True)
        self.stream.synchronize()
        self.upload.synchronize()

    def audit(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        result = {}
        for name, kernel in self.kernels.items():
            kernel._init_handles()
            path = directory/f'{name}.cubin'
            path.write_bytes(kernel.asm['cubin'])
            path.with_suffix('.ptx').write_text(kernel.asm['ptx'])
            sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '-sass', str(path)], text=True)
            path.with_suffix('.sass').write_text(sass)
            result[name] = dict(registers=kernel.n_regs, spills=kernel.n_spills,
                                calls=sum(' CALL' in line for line in sass.splitlines()),
                                acquire_sys='ld.acquire.sys.global.u32' in kernel.asm['ptx'],
                                release_sys='st.release.sys.global.u32' in kernel.asm['ptx'])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=60)
    parser.add_argument('--capacity', type=int, default=262144)
    parser.add_argument('--window-cycles', type=int, default=20_000_000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]/'build'
    if args.rounds < 1 or args.output.exists() or not args.output.resolve().is_relative_to(root):
        parser.error('positive rounds and fresh output path under dism_v3/build required')
    torch.set_num_threads(1)
    probe = SyncProbe(args.capacity, window_cycles=args.window_cycles)
    results = []
    try:
        epoch = 1
        sizes = [0, 1, 255, 256, 257, args.capacity-17, args.capacity]
        for mode in ('ready', 'late_after_sample', 'partial_upload', 'stale'):
            for size in sizes:
                results.append(probe.run_control(mode, epoch, size))
                epoch += 1
        for index in range(args.rounds):
            results.append(probe.run_async(epoch, sizes[index % len(sizes)],
                                          after_graph=index % 5 == 0, split_upload=index % 3 == 0))
            epoch += 1
        resources = probe.audit(args.output.with_suffix(''))
    finally:
        probe.close()
    report = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                  triton=triton.__version__, capacity=args.capacity,
                  window_cycles=args.window_cycles, checks=len(results),
                  event=sum(r['route'] == 1 for r in results),
                  fallback=sum(r['route'] == 0 for r in results),
                  async_event=sum(r['mode'] == 'async' and r['route'] == 1 for r in results),
                  input_before_graph_end=sum(r.get('input_before_graph_end', False) for r in results),
                  resources=resources, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
