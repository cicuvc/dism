"""Read-only latency/overlap diagnostic; production decoder is unchanged."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dism_v4.decoding import NativeDecodeCache
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

OUT = ROOT / 'dism_v4/decoding/results/synchronization'
OUT.mkdir(parents=True, exist_ok=True)


def stats(values):
    return dict(median_us=float(np.median(values)), p95_us=float(np.percentile(values, 95)))


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(432)
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  protocol='Synthetic Zipf(1.2), B1 H12 R32 DV64 N2048; ordinary steps only, no rebuild; idle stream start; wall includes final stream sync',
                  step={}, gdn={}, copies={})
    n, h, count = 2048, 12, 128
    probs = torch.arange(1, 513, device='cuda').float().pow(-1.2)
    labels = torch.zeros(n+count, 3, h, device='cuda', dtype=torch.int32)
    labels[:, :2] = torch.multinomial(probs, (n+count)*2*h, True).reshape(n+count, 2, h).int()
    cpu_labels = labels.cpu()
    sk = torch.randn(n+count, h, 32, device='cuda', dtype=torch.bfloat16)
    sq = torch.randn_like(sk)
    v = torch.randn(n+count, h, 64, device='cuda', dtype=torch.bfloat16)

    def cache():
        c = NativeDecodeCache(h, 32, 64, n+count, [.7]*h, rebuild_interval=512,
                             sample_interval=32, materialize_threshold=64, rebuild_chunk=128)
        c.prime(labels[:n], sk[:n].transpose(0, 1).contiguous(), v[:n].transpose(0, 1).contiguous())
        torch.cuda.synchronize()
        return c

    for mode in ('gpu_labels', 'cpu_labels'):
        c = cache()
        elapsed = []
        stages = []
        for i in range(count):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            c.step((labels if mode == 'gpu_labels' else cpu_labels)[n+i], sk[n+i], sq[n+i], v[n+i])
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter_ns()-start)/1000)
            if hasattr(c.native, 'timings'):
                stages.append(c.native.timings())
        report['step'][mode] = stats(elapsed[16:])
        report['step'][mode]['cache'] = c.memory_stats()
        if stages:
            names = ['validation', 'labels_cpu', 'rebuild_check', 'planner', 'packing', 'upload_enqueue', 'query_enqueue']
            report['step'][mode]['cpu_stages'] = {name: stats(values) for name, values in
                zip(names, np.asarray(stages[16:]).T)}

    if os.environ.get('DISM_DECODE_TIMING') == '1':
        (OUT / 'cpu_stages.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return

    c = cache()
    for i in range(8):
        c.step(labels[n+i], sk[n+i], sq[n+i], v[n+i])
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        for i in range(8, 24):
            with torch.profiler.record_function('native_step'):
                c.step(labels[n+i], sk[n+i], sq[n+i], v[n+i])
                torch.cuda.synchronize()
    prof.export_chrome_trace(str(OUT / 'native_step_trace.json'))

    # Measure actual copy engines and the CPU-visible completion separately.
    host = torch.empty((3, h), dtype=torch.int32, pin_memory=True)
    dev = labels[0]
    for name, dst, src in [('pinned_d2h_144B', host, dev),
                           ('pinned_h2d_144B', dev, host)]:
        elapsed, device = [], []
        start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(120):
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            start_event.record()
            dst.copy_(src, non_blocking=True)
            stop_event.record()
            stop_event.synchronize()
            elapsed.append((time.perf_counter_ns()-started)/1000)
            device.append(start_event.elapsed_time(stop_event)*1000)
        report['copies'][name] = dict(wall=stats(elapsed[20:]), event_interval=stats(device[20:]))

    graphs = {}
    for batch in (1, 8, 32):
        q = torch.randn(batch, 1, h, 64, device='cuda', dtype=torch.bfloat16)
        k, value = torch.randn_like(q), torch.randn_like(q)
        g = -torch.rand(batch, 1, h, device='cuda')
        beta = torch.rand(batch, 1, h, device='cuda')
        state = torch.randn(batch, h, 64, 64, device='cuda')
        def gdn():
            return fused_recurrent_gated_delta_rule(q, k, value, g=g, beta=beta,
                initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(10):
                gdn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = gdn()
        times = []
        for _ in range(10):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(200):
                graph.replay()
            b.record(); b.synchronize()
            times.append(a.elapsed_time(b)*1000/200)
        report['gdn'][str(batch)] = dict(graph_replay_device_interval=stats(times))
        # Profiler supplies kernel-only durations, avoiding host enqueue gaps.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as p:
            for _ in range(32):
                graph.replay()
            torch.cuda.synchronize()
        p.export_chrome_trace(str(OUT / f'gdn_b{batch}.json'))
        # Keep capture inputs and outputs alive throughout overlap measurements.
        graphs[batch] = (graph, (q, k, value, g, beta, state, result))

    # Independent same-layer GDN branch: enqueue before the blocking CPU decoder.
    # Separate stream is essential: .cpu() otherwise waits for GDN as well.
    for mode in ('serial_gdn', 'overlap_gdn'):
        c = cache()
        other = torch.cuda.Stream()
        elapsed = []
        graph = graphs[1][0]
        for i in range(count):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            if mode == 'overlap_gdn':
                with torch.cuda.stream(other):
                    graph.replay()
            c.step(labels[n+i], sk[n+i], sq[n+i], v[n+i])
            if mode == 'serial_gdn':
                graph.replay()
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter_ns()-start)/1000)
        report['step'][mode] = stats(elapsed[16:])

    (OUT / 'timings.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
