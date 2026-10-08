"""Extract device-only durations from the profiler traces, not event intervals."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1] / 'decoding/results/synchronization'
result = {}
for name in ['native_step_trace', 'gdn_b1', 'gdn_b8', 'gdn_b32']:
    events = json.loads((root / f'{name}.json').read_text())['traceEvents']
    grouped = {}
    for event in events:
        if event.get('cat') not in ('kernel', 'gpu_memcpy'):
            continue
        grouped.setdefault(event['name'], []).append(event['dur'])
    result[name] = {key: dict(count=len(values), median_us=float(np.median(values)),
                            p95_us=float(np.percentile(values, 95)))
                    for key, values in grouped.items()}
result['host_node_probe'] = json.loads((root / 'host_node_probe.json').read_text())
result['caveats'] = [
    'NativeCache benchmark: synthetic Zipf labels, B1 H12 R32 DV64 N2048, ordinary steps only.',
    'CPU stage instrumentation is a separate build; do not sum its measurements with profiler GPU durations.',
    'GDN is only the recurrent state kernel with FP32 state, not projections/convolution/gating of a complete layer.',
    'Host-node graph is a toy correctness/latency probe, not the full decoder or a hardware lower bound.',
    'CUDA event intervals for tiny copies include host submission gaps; memcpy trace durations are the actual device activity.',
]
(root / 'device_summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
