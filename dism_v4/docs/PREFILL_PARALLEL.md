# CPU batch/head parallel planning

Update: the pool now defaults to eight workers and is connected to the GPU
through [HardDismPrefill](PREFILL_BATCH.md). The measurements below isolate
the earlier CPU-only stage; the integrated API consumes its packed chunks
directly without a second packing pass.

Opt-in `decoding/prefill_parallel.py::ParallelPrefillPlanner`: reusable Python
ThreadPoolExecutor dispatches independent `(batch,head)` tasks into the existing
GIL-released C++ planner and native chunk packer. Returns ordered
`[(HardPrefillPlan, packed_chunks), ...]` in B-major/H-minor order. No new
C++ thread pool, GPU transfer, multi-head kernel or production dispatch change.
This experiment isolates CPU parallel scaling first.

```python
from dism_v4.decoding.prefill_parallel import ParallelPrefillPlanner

with ParallelPrefillPlanner(workers=8) as planner:
    results = planner.plan(q_labels, k_labels, tau, reset=reset, chunk_size=16)
```

CPU labels are `[B,H,N]`, tau `[H]` or `[B,H]`, reset bool `[B,H,N]`.
Caller must not mutate inputs during planning. Pool compilation/import happens
once before workers start. Reuse the pool across layers/calls. Plans own their
data; no native mutable state is shared between tasks. Exceptions propagate
to the caller, and close/context exit waits for outstanding tasks.

## Results

Local CPU reports Intel Core Ultra 9 285K, 14 online logical CPUs. No affinity
pinning, reusable pool, three timed repetitions after one full warmup per
configuration. B2/H8 (16 tasks), C16, tau varies across heads from .7 to ln64.
Batch 0 has no resets; batch 1 resets every 997 tokens. Zipf512 exponent1.2
and repeat labels are synthetic. This differs from earlier no-reset single-head
benchmarks. Timings include Python validation/list conversion, native planning,
native chunk packing and NumPy export; exclude GPU work and checksum tests.

| Labels | N | 1 worker ms | 2 workers ms | 4 workers ms | 8 workers ms | 8-worker speedup |
| --- | --- | --- | --- | --- | --- | --- |
| Zipf512 | 4096 | 148.2 | 86.9 | 46.5 | 26.3 | 5.65x |
| Zipf512 | 16384 | 696.2 | 357.8 | 215.2 | 124.8 | 5.58x |
| Zipf512 | 65536 | 3239.8 | 1672.3 | 1040.6 | 629.0 | 5.15x |
| repeat | 4096 | 155.3 | 80.0 | 48.5 | 28.0 | 5.54x |
| repeat | 16384 | 725.3 | 350.2 | 198.7 | 116.6 | 6.22x |
| repeat | 65536 | 2849.5 | 1459.3 | 838.6 | 526.9 | 5.41x |

Every worker configuration produced byte-identical native program and packed
chunk arrays to the corresponding single-worker run. Separate small tests
compare each head to direct serial planning, FP64 vector outputs, [H]/[B,H]
tau, reset ordering and recovery after worker exceptions.

At 64K/repeat, retained program+packed arrays total 1.15 GB. Highest sampled
process RSS was 3.24 GB with eight workers. RSS is sampled every 10 ms, includes
Torch/imported libraries and allocator-retained pages, and configurations run
in the same process: it is not an isolated per-config peak-memory measurement.
Memory pressure did not prevent this experiment.

Eight workers is the best tested setting, not a claim of the optimum across
other CPUs/head counts. Python argument conversion/export still holds the GIL;
remaining scaling limits have not been profiled. Next integration step would
consume the returned packed arrays directly in batched GPU upload/execution,
avoiding repacking them in the existing single-head Triton constructor.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 /home/cicuvc/miniconda3/envs/blkw/bin/python dism_v4/decoding/benchmark_prefill_parallel.py --output dism_v4/decoding/results/prefill_parallel.json
/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v4/decoding/test_prefill_parallel.py
```

JSON stores raw timings, metadata checksums, retained bytes, sampled RSS, actual
sequence-token throughput (`B*N/time`) and head-token throughput (`B*H*N/time`)
separately. Neither is whole-model prefill throughput.
