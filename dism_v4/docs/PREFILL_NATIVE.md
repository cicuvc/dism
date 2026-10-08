# Native offline prefill: C++ planner + CPU vector mock

For integrated batch/head CPU+GPU execution, use
[HardDismPrefill](PREFILL_BATCH.md). The lower-level CPU facilities described
here remain available for diagnosis.

The first implementation of [the prefill plan](PREFILL_PLAN.md) is available
as `dism_v4.decoding.HardPrefillPlan`. This is an opt-in, single-sequence,
single-head inference facility. It does not change model dispatch, initialize
decoding caches. A separate opt-in [Triton executor](PREFILL_TRITON.md) now
consumes its chunk event tables. No GPU is used by the CPU facility's
build or tests. Batch/head orchestration and a final `prime()` handoff remain
integration work; do not append into a nonempty decoding cache implicitly.

## API and build

```python
import numpy as np
from dism_v4.decoding import HardPrefillPlan

# int32-compatible integer labels [N]; bool reset [N].
# Reset applies before consuming this query and does not delete keys.
plan = HardPrefillPlan(q_labels, k_labels, tau, reset=reset)
# CPU numpy vectors: sq/sk [N,R], value [N,DV], arbitrary positive R/DV.
output = plan.execute(sq, sk, value)                 # FP32
output64 = plan.execute(sq, sk, value, dtype=np.float64)
events = plan.arrays()                             # independent owned copies
print(plan.statistics())
```

Only full-hard label matching with fixed per-head tau and binary reset is
supported. Finite soft delta is not supported. SQ/SK are already parameterized;
no extra SiLU, normalization or readout scaling is applied. Signed numerator
and denominator fallback 1 match the v4 reference. Negative finite tau is also
tested by this prefill path; existing decoding cache restrictions are unchanged.

`decoding/prefill.py` builds the host-only pybind extension on demand, with
`CXX` (default c++), a build lock, and atomic publication of the shared library.
It uses pybind headers bundled with Torch but does not link libtorch or compile
CUDA. Existing decoding build entrypoints are unchanged. Use conda `blkw`.

## Planner and device-facing contract

`prefill.hpp` reuses the existing SAM extension primitive and builds a full
suffix-link tree. LCA deliberately retains binary lifting. Weighted group
merging preserves chronological order; centroid traversal is iterative inside
each component, and recursive decomposition has logarithmic depth.

CPU FP64 scalar planning produces two scalar passes: log-denominator followed
by pre-normalized event coefficients. Zero-score and causally empty work is
pruned. The resulting `Program` owns only:

| Array | Type | Meaning |
| --- | --- | --- |
| offsets | int64 [streams+1] | Half-open event ranges |
| rows | int32 [events] | Nonnegative key row, or query encoded as -row-1 |
| decay | FP64 [events] | Key matrix rescaling; ignored for query events |
| weight | FP64 [events] | Key update or normalized query readout coefficient |
| logden | FP64 [N] | Diagnostic denominator including fallback |

For each stream, initialize M=0, then execute in order:

```
key j:    M = decay*M + weight*outer(sk[j], value[j])
query i:  output[i] += weight * sq[i]^T M
```

Keys precede queries at the same time. No SAM traversal, exp, log, division,
causal test or denominator calculation remains in the vector executor.
`execute<float>` casts both coefficient arrays to FP32 at use and accumulates
in FP32. `execute<double>` preserves FP64 throughout vector work. This is not
a fully FP32 planner: topology scalar summaries and coefficient construction
still use FP64. CPU mock loops are serial and deterministic.

## Memory and future GPU execution

The retained program is O(N log N) scalar/index storage; binary-lifting LCA
can add a log factor to scalar planning time. `program_bytes` reports logical
retained array bytes, not allocator capacity, Python export copies, transient
planner memory, or process peak RSS. SAM/LCA and raw stream lists are destroyed
after program construction. Peak construction includes raw and compiled events.

Vector scratch is one R*DV matrix in the mock, plus N*DV output. There is no
N*R*DV cache or per-event DV partial array. A first GPU backend can assign
streams to CTAs, keeping live matrices bounded by execution residency, and
use FP32 atomic output reduction. No shared-memory FP32 atomics are needed.
Wave scheduling/channel tiling and register pressure must be measured before
choosing that kernel. Long streams may require chunking for GPU parallelism;
this version makes no GPU throughput claim.

## Tests and initial measurements

```bash
OPENBLAS_NUM_THREADS=1 /home/cicuvc/miniconda3/envs/blkw/bin/python dism_v4/decoding/test_prefill.py --output /tmp/dism_cpp_prefill_results.json
c++ -O1 -g -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer dism_v4/decoding/test_prefill_native.cpp -o /tmp/dism_test_prefill_native
/tmp/dism_test_prefill_native
```

100 deterministic cases passed against the Python planner, exact dense
recurrence and existing Torch reference. Coverage includes all-reset/no-reset,
clone-producing random strings, repeat/periodic/distinct labels, mismatches,
tau=0/tiny/ln64/negative, signed vectors and unequal R/DV. Short-sequence basis
values expose individual normalized pair weights. Prefix invariance and exported
array ownership are checked. Long repeat numerical checks reach N=4096.
The standalone host smoke passed AddressSanitizer and UndefinedBehaviorSanitizer.
The existing online planner regression also passed (36 streams, 9252 steps).
Machine-readable rerun results live in `decoding/results/prefill_cpu_mock.json`;
timings naturally vary from the initial observations below.

Maximum absolute output errors against exact recurrence: native FP64 5.44e-13,
native FP32 6.39e-6. Existing Torch thresholded-softplus differs from exact
recurrence by up to 9.68e-10 in these cases. Tests retain distinct tolerances;
the oracle and existing CUDA acceptance tolerances were not changed.

Initial single-run, single-head planning timings (includes Python label packing,
excludes compilation), on the local CPU:

| N | Pattern | Plan ms | Retained program MB (decimal) |
| --- | --- | --- | --- |
| 4096 | random, alphabet 16 | 9.09 | 1.20 |
| 4096 | repeat | 12.57 | 1.96 |
| 16384 | random, alphabet 16 | 45.82 | 6.06 |
| 16384 | repeat | 55.79 | 9.14 |
| 16384 | distinct | 17.23 | 0.92 |

These are bring-up observations, not a warmed statistical benchmark. FP64 CPU
mock at N4096/R16/DV32 takes about 7.3 ms random / 13.1 ms repeat in the same
run. Multi-head latency, transfer cost, peak memory and GPU speed remain unmeasured.
