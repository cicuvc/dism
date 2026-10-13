# Native parallel hard-backward planning

Implemented on 2026-10-14 after prototype commit `38946ce`. This is an explicit
planning API and a consumer in the diagnostic Triton backward, not a production
model/autograd dispatch change. The original head-shared tau is unchanged.

## Implementation

- `inference/csrc/prefill.hpp`: optional analytic derivatives in the existing
  SAM/centroid builder. Inference calls keep derivatives disabled and retain
  their existing coefficients and array interface. Training precomputes log
  geometric weights/derivatives, differentiates the scalar normalizer and
  emits event-local dlog weights with frozen numerical pivots.
- `inference/csrc/prefill_training.hpp`: persistent bounded C++ worker pool,
  independent document/head jobs, deterministic concatenation layout and
  parallel direct packing into caller-owned storage. Input is read directly
  from contiguous NumPy/pinned arrays: there are no per-head Python lists,
  futures, exported-array copies or Python merge/relabel loops.
- `inference/training_planner.py`: validates host inputs, allocates or accepts
  a reusable pinned buffer, creates typed views and performs one bulk async
  H2D upload. Native planning and packing release the GIL. Tau derivatives
  are real, not the synthetic transport field used in the earlier study.
- `tools/hard_prefill_backward_triton.py`: `from_native()` consumes the
  uploaded batch plan, waits via a device event and runs the unchanged Triton
  scans on flattened BNHD vectors. A stream-to-tau-group map reduces partials
  to the original shared head gradients. Soft row/head gradients stay zero
  in this hard-only branch; integration must merge them with soft results.

Each job independently constructs its SAM, coefficients and C16/32/64 chunks.
After prefix-summing job sizes, packing workers relabel indices into BNHD and
write disjoint output ranges. Only the terminal stream offset is written by
the caller after workers finish. No GPU matrix-state snapshots are introduced.
Intermediate per-job native chunks still exist; packing copies them once into
the final pinned buffer. Eliminating those chunks is a future optimization.

The pool accepts 1–64 workers; default 8, no automatic affinity changes.
Invocations share the bounded worker pool. Callers must separately bound
outstanding layer plans and retained metadata memory. Worker failures drain
all submitted jobs before propagating; the pool remains reusable.

## API and lifetime contract

```python
from flash_dism.inference.training_planner import NativeTrainingPlanner

planner = NativeTrainingPlanner(workers=16)
host = planner.prepare(
    labels_cpu,       # int32 contiguous [2,B,H,N], D2H already completed
    tau_cpu,          # float64-compatible [H], or explicitly [B,H]
    documents=docs,   # int64 [documents,3]: (batch_row,start,end)
    hard=hard_rows,   # bool/uint8 [B,H], sampled per packed training row
)
device = host.upload(stream=copy_stream)
# Diagnostic consumer: HardPrefillBackward.from_native(device).
```

Documents must cover each row completely, in per-row order, without overlap
or gaps. Their lengths can vary; all documents in a row/head inherit the same
hard flag. The default document layout is one document per row. An all-soft
batch produces an empty hard program with offsets `[0]`. CPU planning accepts
arbitrary positive lengths like the explicit inference API; production
training operators retain their separate 256-alignment requirement.

The flat tau result has H entries for `[H]` input, or B*H for the explicit
`[B,H]` variant. This is not a per-query tau architecture change. Tests cover
both layouts to prevent accidental merging of unrelated scalar gradients.

The planner never silently transfers GPU inputs to CPU. A coordinator worker
must wait for label/tau/hard/metadata D2H completion before calling `prepare`.
Do not mutate inputs while native work is running. `upload` records a CUDA
event; `DeviceTrainingPlan.wait()` inserts a consumer-stream dependency and
records the allocation's consumer stream, without synchronizing the host.

Fresh pinned allocations are safe by default. If passing `storage=...` for
reuse, the caller must ensure all previous host consumers and H2D transfers
are complete before overwriting it. Capacity is checked; insufficient storage
raises instead of resizing an in-flight buffer. GPU/host plan ownership must
remain tied to its forward invocation, including checkpoint replay.

## Validation

- 38 new tests: single-document native/Python event and derivative equality;
  negative/zero/near-zero/positive tau; four label patterns; worker counts;
  all chunk sizes; ragged documents; row/head hard masks; empty hard plans;
  direct native-plan GPU backward; per-head and per-row/head tau grouping;
  cross-stream upload/consumption; buffer reuse; malformed/overlapping buffers;
  concurrent calls and pool recovery after a worker exception.
- An additional 12 N2048 cases (random/repeated/periodic labels,
  tau=0/.5/2/16) matched symbolic event topology and coefficients/derivatives
  under the original strict comparison. Largest dlog coefficient difference
  was about 4.08e-10.
- 213 combined native-planner, backward-prototype, CPU-reference, SAM and
  README tests passed. Existing inference bit-identical program tests passed.
- GPU memcheck on all 38 new tests reported zero errors. The Triton kernel
  bodies/launch configuration were not changed; previous CALL/spill findings
  and kernel sanitizer limitations remain in `HARD_PREFILL_TRAINING.md`.
- The standalone C++ pool/packing/error-recovery harness passed GCC
  AddressSanitizer + UndefinedBehaviorSanitizer (including leak detection)
  and separately ThreadSanitizer. The installed Clang build lacked sanitizer
  runtime libraries, so GCC was used for these harnesses. This is not a claim
  that the Python extension or every possible workload was host-sanitized.

Native compilation uses the worktree inference `build.py` JIT path with
`MAX_JOBS=2`; no training CUDA rebuild or external job was needed.

## CPU planning and full-ready latency

RTX 5090, Intel Core Ultra 9 285K (24 physical cores, hybrid P/E), default OS
scheduling without affinity pinning. B16/H6/N2048, all-hard, tau=.5, C16;
synthetic iid vocabulary512 labels or all-identical labels. Seven warm samples,
no result/topology cache, reusable pinned capacity. The native pool is reused;
import/build/thread startup is excluded. Raw first-call measurements are also
retained. No GPU training workload or other benchmark ran during this sweep.

CPU totals include input handling, SAM/event construction, analytic tau and
normalizer derivatives, chunk packing, parallel final pinned writes, and
temporary native-result destruction. Full-ready times additionally include
pinned D2H, its host completion fence, actual metadata H2D and the final
completion fence. They are measured end to end, not sums of independent medians.

| Workers | Uniform random CPU | Uniform repeated CPU | Random full ready | Repeated full ready |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 223.98 ms | 450.09 ms | 224.41 ms | 456.26 ms |
| 4 | 59.20 ms | 113.21 ms | 61.04 ms | 116.40 ms |
| 8 | 34.19 ms | 64.89 ms | 36.15 ms | 69.38 ms |
| 16 | 23.66 ms | 49.55 ms | 24.24 ms | 53.03 ms |

At 16 workers, uniform planning itself took about 22.09/42.76 ms and final
pinned packing 1.08/6.37 ms. Metadata was 26.65/161.99 MiB, including real
dlog weights and the stream-to-tau-group map. H2D upload API completion was
0.618/3.644 ms. Ragged documents 256/512/256/1024 per row took 20.21/42.59 ms
CPU and 20.56/46.59 ms full ready.

An independent B16 repeat gave CPU times 33.62/65.40 ms with eight workers,
and 23.69/49.08 ms with sixteen workers. Sixteen-worker full-ready times were
24.47/53.05 ms. Ragged CPU was 20.70/42.82 ms and full ready 20.70/46.49 ms.
Small differences between CPU-only and end-to-end medians are sampling noise,
not negative transfer cost.

The independent native async handoff run (16 workers, concurrent unchanged
dense forward) observed upload readiness at 24.03/52.67 ms for random/repeated
labels. With one dense forward queued before the label-ready event, readiness
was 25.27/54.62 ms; the worker's D2H dependency wait was 0.77/0.83 ms. The main
thread continued GPU enqueueing rather than waiting for the CPU plan. These
times include worker wakeup, pinned packing, actual derivatives, upload and
host completion observation. Raw repeat:
`build/native_plan_handoff_w16_repeat_20261014.json`.

For B1/H6 there are only six uniform jobs: eight workers gave 3.65/6.32 ms
CPU, and sixteen did not help (3.75/6.81 ms). More workers are therefore not
an unconditional default improvement. At B16, sixteen workers improved
planning but did not improve memory-bandwidth-bound final packing.

Compared with the earlier eight-worker C++-core/Python-merge preparation,
the large Python merge/staging overhead is removed, despite adding genuine
tau derivatives. However, the remaining 20–50 ms preparation is not hidden
by a single roughly 1 ms dense forward. The next integration decision needs
actual per-layer forward-to-backward slack and aggregate CPU throughput across
all DISM layers; lower-layer slack alone cannot prevent sustained worker backlog.
No claim of full-model speedup or universally hidden planning is made.

## Allocation and layout optimization (2026-10-14)

The CPU builder now skips the SAM root's zero-weight cross-component merge
tree, prunes empty/noncausal cross calls, and avoids partitioning when no
parent-side component exists. Four partition scratch vectors retain capacity
within each Builder. Event streams use ranges into six flat arrays instead
of six separately allocated vectors per stream. SAM/endpoints/query arrays
and compiled event arrays reserve capacity. Chunk boundaries are discovered
before allocating the padded slot arrays, eliminating their repeated growth.
There is no cross-job/result cache, new worker affinity, or worker-count change.
Final pinned packing and GPU kernels are unchanged.

Event ordering, stream/chunk boundaries, padding, coefficients and analytic
derivatives are preserved. An ignored before/after native comparison covered
420 cases: N=1/2/17/65/256/2048, five label patterns, seven tau values,
query resets, inference/training, and all three chunk sizes. All exported
program/chunk arrays were bit-identical. The maintained derivative test now
covers N=1/65/513; all 269 combined regression tests passed without tolerance
changes. The 420-case comparison passed ASan/UBSan with leak detection;
the existing parallel pool/pack/recovery harness passed ASan/UBSan and TSan.
GPU memcheck on the two native-plan backward/upload cases and the all-soft
empty-plan case also reported zero errors (three tests, not the entire suite).
No GPU kernel body or launch changed, so no new CALL/spill claim is made.
The initial diagnostic comparator itself called memcmp on null empty-vector
data; that harness bug was fixed and the sanitizer comparison rerun cleanly.

Single-core diagnostic allocation counting at N2048, tau=.5, C16 reduced
decomposition allocations from 86,935 to 23,790 for random labels and from
147,043 to 54,338 for repeated labels. A paired old/new standalone benchmark
gave approximately 2.00 -> 0.91 ms and 4.49 -> 2.70 ms per head respectively.
These are single-job figures, not per-stage attribution of a parallel batch.
Hardware cache/branch counters remain unavailable: perf_event_open returned
Permission denied. Allocation reduction and timing are measured; a cache-miss
or branch-misprediction bottleneck is not claimed.

The final end-to-end sweep used the same B16/H6/N2048, tau=.5, C16, all-hard
synthetic cases, reusable pinned storage and default OS scheduling as above,
with nine warm samples and no concurrent benchmark/compiler/test job.

| Workers | Random CPU | Repeated CPU | Random full ready | Repeated full ready |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 15.76 ms | 40.67 ms | 15.60 ms | 43.38 ms |
| 16 | 11.64 ms | 30.48 ms | 11.97 ms | 34.98 ms |

Against the earlier independent 16-worker baseline of 23.69/49.08 ms,
CPU preparation improved about 2.04x/1.61x. Native planning itself was
10.33/23.62 ms; pinned packing remained 1.03/6.50 ms. Metadata layout and size
are unchanged, so this does not remove H2D bandwidth cost. Sixteen-worker
ragged cases took 9.65/27.43 ms CPU and 10.45/31.85 ms full ready.
B1/H6 with eight workers took 1.94/3.73 ms CPU and 2.01/4.25 ms full ready.
Separate CPU-only/full-ready medians can differ in either direction due to
scheduling noise. Raw report:
`build/native_training_planner_optimized_20261014.json`.
An independent 11-sample, 16-worker uniform rerun gave 10.79/30.75 ms CPU
and 11.68/34.59 ms full ready; planning was 9.32/23.88 ms and packing
1.12/6.38 ms. Raw repeat:
`build/native_training_planner_optimized_repeat_20261014.json`.

This remains an explicit planner/prototype optimization, not model integration.
Preparation still exceeds a single roughly 1 ms dense forward. Full-model
overlap and sustained multi-layer CPU throughput remain unmeasured.

## Reproduction

From the workspace root, choose fresh ignored output paths:

```bash
OPENBLAS_NUM_THREADS=1 MAX_JOBS=2 PYTHONPATH=$PWD/dism_v3/python:$PWD/nanochat \
  /home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/bench_native_training_planner.py \
  --output dism_v3/build/native_training_planner_20261014.json

OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD/dism_v3/python:$PWD/nanochat \
  /home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/bench_hard_plan_handoff.py --native-planner --workers 16 \
  --batches 16 --output dism_v3/build/native_plan_handoff_w16_20261014.json

g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -pthread -Idism_v3/python dism_v3/tools/check_native_training_planner.cpp \
  -o dism_v3/build/check_native_training_planner_asan
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  dism_v3/build/check_native_training_planner_asan
```

Use `-fsanitize=thread` separately for the ThreadSanitizer harness. All result
JSON, sanitizer binaries and logs remain under ignored `build/`.
