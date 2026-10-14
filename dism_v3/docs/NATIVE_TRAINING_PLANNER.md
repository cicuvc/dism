# Native parallel hard-backward planning

## Production default promotion (2026-10-15)

Following the merge below, compact metadata and the `ready` policy are now
enabled by default in `AsyncHardPlanner` and the lazy production runtime.
The coordinator passes `compact=True` and builds
`CompactFusedHardPrefillBackward` on its H2D stream. Its final readiness event
covers metadata copy, bucket auxiliaries and tau-head construction. The backward
selector uses event kernels only after that event completes; otherwise it uses
the existing quadratic fallback. CUDA forward and sequence hard sampling do not
change. Explicit `compact=False` / `DISM_PLANNER_COMPACT=0` and
`backward_policy='event'` / `DISM_HARD_BACKWARD=event` retain comparison routes.
The standalone `NativeTrainingPlanner.prepare` API still defaults to classic
layout for callers that explicitly consume `HostTrainingPlan.data`.

Nanochat also now uses chunked document-balanced CE, with whole-step/global-rank
normalization; see [the training contract](../../nanochat/README.md#current-training-defaults-2026-10-15).
The opt-in wording and timings below describe the earlier merge stage.

## Merged planner optimization (2026-10-15)

The repaired optimization from `/tmp/dism_planner_opt/planner-opt` is now part
of this worktree. Default classic-layout preparation uses flat centroid scratch,
direct event-to-pinned-buffer packing (no intermediate padded ChunkProgram),
one completion record per worker invocation instead of per-job futures, and
reuse of identical exponential coefficients between the two scalar passes.
The classic metadata format, coefficients, launch defaults and gradient math
are preserved. No worker-count/affinity, hard sampling, checkpoint, RNG,
optimizer, dataloader or ready/event policy changes accompany this merge.

Compact layout is **opt-in**, not connected to `AsyncHardPlanner` by default:

```python
from flash_dism.kernels.hard_backward_fused import CompactFusedHardPrefillBackward

host = planner.prepare(labels_cpu, tau_cpu, documents=docs, hard=hard_rows,
                       chunk_size=64, compact=True)
uploaded = host.upload(stream=copy_stream)
consumer = CompactFusedHardPrefillBackward.from_compact(uploaded)
grads = consumer.backward(sq, sk, value, dout)  # flattened contiguous [B*N*H,D]
```

Single-chunk streams occupy fixed-width buckets16/32/64; multi-chunk streams
retain the classic layout and three-stage GPU algorithm. Widths below16 are
not used because the local Tensor Core contraction requires at least16.
This reduces metadata padding, not the actual event count. The compact consumer
currently requires `long_mode='three_stage'` for multi-chunk streams.

Two audit fixes are included, not just the original isolated optimization:

- The last worker decrements the completion count and notifies **under the
  waiter's mutex**. Previously the waiter could destroy the stack-owned record
  before the last worker touched its mutex. Queue-allocation failures also
  drain submitted work before propagating the exception.
- Compact construction waits for metadata upload before reading tau head IDs.
  It always records a construction-ready event, including blocking host upload
  mode; backward waits on both events and records allocation lifetimes on the
  consuming stream. Delayed three-stream tests cover both `non_blocking` modes
  and dropping the owner before GPU work completes.

Maintained coverage: `tests/test_compact_training_planner.py`, standalone
`tests/native/test_planning_workers.cpp`, and
`tools/check_compact_training_planner.py --cpu|--gpu|--smoke`.
The checker reconstructs every compact array from the classic plan with exact
equality, uses unequal ragged document lengths, and asserts GPU gradients
rather than merely printing differences. Original planner/kernel numerical
tolerances are unchanged.

The repaired isolated-tree measurements (before merge, W16/B16/H6/N2048/C64,
three warmups and seven samples) were:

| Labels | Classic / compact CPU total | Classic / compact GPU backward | Metadata MB |
| --- | ---: | ---: | ---: |
| iid512 | 11.76 / 9.28 ms | 0.388 / 0.356 ms | 105.69 / 26.73 |
| iid8 | 31.81 / 25.80 ms | 2.466 / 2.387 ms | 290.63 / 98.49 |

CPU totals exclude D2H/H2D and are not full-ready times. These figures compare
layouts **after** CPU optimization, not original versus optimized planner.
No new end-to-end 72M training speedup or compact default integration is claimed.
Prior tau cancellation and long-C64 spill limitations below remain applicable.

### Post-merge validation

Built the worktree's native prefill extension through its inference `build.py`
with `MAX_JOBS=2`; only worktree Python/CUDA packages were imported.

- 367 combined native-planner, compact, fused/scan backward, async training and
  ready-dispatch tests passed; 86 shared-inference prefill tests passed.
- The final compact-only rerun passed132 tests, including four added R16/DV32
  and R32/DV64, BF16/TF32x3, per-row/head-tau grouping cases (457 distinct
  regression cases across these runs). Log: `build/merged_compact_final_tests.log`.
- 1152 standalone CPU compact cases and 48 GPU cases passed. Four classic
  storage hashes, including B16/H6/N2048 iid512/iid8, match the pre-merge bytes.
- Worker stress (100,000 tiny submissions, error draining, concurrent callers)
  and the full classic plan/pack/recovery C++ harness passed TSan and
  ASan/UBSan. The latter also used leak detection.
- Compact bucket plus multi-chunk smoke passed memcheck/synccheck with zero
  errors and racecheck with zero errors/warnings. This does not supersede
  historical soft CUDA pipeline sanitizer limitations.
- Inspected BF16 R32/DV64 compact buckets16/32/64 use56/138/249 registers and
  zero spills. Summaries/propagation also have zero spills. The C64 long-chunk
  readout still has254 registers and **12 spills**, not newly spill-free.
  All inspected variants have zero CALLs and shared atomics.

Raw ignored artifacts: `build/merged_planner_tests.log`,
`build/merged_planner_inference.log`, `build/merged_compact_{cpu,gpu}.log`,
`build/merged_compact_{memcheck,racecheck,synccheck}.log`,
`build/merged_planning_workers_{tsan,asan}.log`,
`build/merged_native_planner_{tsan,asan}.log`,
`build/merged_planner_classic_hashes.json`, and
`build/merged_planner_resources.json`.

## Historical implementation and measurements

Implemented on 2026-10-14 after prototype commit `38946ce`. The original
planning API and diagnostics are now also consumed by an explicit eager
autograd adapter; see [async integration](#eager-asynchronous-training-integration).
The default model dispatcher and original head-shared tau are unchanged.

## Selected training integration direction (2026-10-14)

For sequence-level hard rows/heads, the agreed near-term route is the current
CUDA O(N^2) forward plus an asynchronous CPU planner and event-stream backward.
Short training sequences are the immediate target; optimizing long-sequence
training forward is out of near-term scope. Soft rows/heads retain their
existing path, and document boundaries do not resample the sequence hard flag.

The [hard forward optimization plan](HARD_FORWARD_PLAN.md) retains this O(N^2)
architecture with thread-owned diagonal segments for summary and an integer
scan in the existing scan-buffer layout for readout. The initial implementation
uses the existing direct-rolled scan layout for both summary and readout and
introduces no dependency on CPU planner completion.

Once labels and the invocation's tau/hard metadata are available, enqueue
D2H and CPU planning alongside the CUDA forward and subsequent GPU work.
Upload the completed matching structure asynchronously, retaining its buffers
and readiness event until the corresponding backward consumes it. This uses
the full interval from metadata availability to that layer's backward, rather
than requiring the planner to finish within that layer's forward alone.

This decision selects the integration architecture, not a measured guarantee
that all planning/communication is hidden. Remaining integration validation
must measure per-layer slack, bounded multi-layer worker throughput, pinned
and device memory lifetimes, checkpoint replay ownership, and backward waits.
The initial planner/kernel commits did not install a coordinator. The explicit
eager adapter below now validates invocation ownership and backward waits;
model dispatch, RNG replay, checkpoints and live training remain unchanged.

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
the caller after workers finish. The original register-scan consumer needs no
GPU matrix-state snapshots; the chunk-parallel consumer described below does.
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

## GPU-only backward comparison (2026-10-14)

Planner/layout optimization committed as `2f1b79c`. The separate diagnostic
`tools/bench_hard_backward_kernels.py` compares the native-plan Triton consumer
with the current worktree CUDA packed-varlen core. Both receive identical
synthetic labels and BF16 vectors, H6/D64/R32/DV64, tau=.5, all-hard. Uniform
documents are N2048; ragged documents are 256/512/256/1024 within each row.
Plans/upload and CUDA forward saves are prepared before timing. CUDA-graph
windows include output clearing and the GPU tau reduction, but not CPU
planning, H2D, forward, or host allocation. Scan stage windows include their
gradient zeroing; profiler records separately expose device kernel durations.
The already-completed native upload event wait is omitted only in this
benchmark: attempting to capture the original external-event wait fails with
capture isolation, so this is not evidence of capture-safe async handoff.

| B16 case | CUDA full core backward | Triton BF16 full backward |
| --- | ---: | ---: |
| Uniform random | 4.75 ms | 3.87 ms |
| Uniform repeated | 5.68 ms | 20.65 ms |
| Ragged random | 2.02 ms | 2.55 ms |
| Ragged repeated | 2.39 ms | 19.00 ms |

Uniform random CUDA B1 (dSQ/dSK/dV plus recurrence summary) was 2.36 ms,
B2 0.104 ms and B3 (dQ/dK/dLSE/dtau) 2.24 ms. Triton query scan
(dSQ plus query dtau) was 1.83 ms, reverse key scan (dSK/dV plus key dtau)
1.84 ms and final tau reduction 0.128 ms. Uniform repeated scans were
10.03/8.97 ms, with a 0.606 ms tau reduction. Stages were independently
timed and need not sum to the directly measured full call.
An earlier independent uniform run gave CUDA 4.73/5.75 ms versus BF16
3.94/19.73 ms; TF32x3 was slower at 7.18/40.63 ms. B1/H6 uniform BF16
was 0.172/1.141 ms versus CUDA 0.783/0.942 ms, so small-batch gains do not
extrapolate to B16.

Each event stream launches one eight-warp CTA and sequentially traverses its
chunks, carrying an R*DV FP32 matrix in registers. Within a chunk, 16 events
use masked Tensor Core products rather than a serial token loop. Query scan
runs forward; key scan runs backward. No chunk matrix snapshots are stored.
At B16 uniform random there are 101,257 streams and 101,259 chunks: almost no
cross-chunk recurrence. Uniform repeated has 490,464 streams and 621,120
chunks; each identical head has 95.1% single-chunk streams, a 99th percentile
of eight chunks, and a maximum of 128. Padded event-slot utilization is about
27%/41%. Long streams do have a serial matrix dependency, but massive numbers
of short, sparsely filled CTAs also perform the full matrix machinery.
BF16 query/key kernels used 173/175 registers per thread, no spills, CALLs,
local-memory operations or shared atomics in inspected SASS. TF32x3 query
used 255 registers with two spills; key used 253 with no spills.

Vector agreement with CUDA remained close (BF16 relative L2 below 0.18%
for uniform random/repeated), but CUDA is not a numerical oracle. An independent
FP64 dense identical-label derivative was added, and checked against the
existing dense reference for tau=0/.5/2, two heads, two rows and ragged 7/10
documents. At B16 uniform repeated, BF16 tau had 27.8% relative L2 error and
one of six head signs wrong; CUDA tau had 210.7% relative L2 error and three
signs wrong. Ragged repeated BF16 had 7.55% relative L2 and no sign flips.
These are reported limitations, not relaxed tolerances or production approval.

The random-direction 72M training log confirms RTX 5090, B16*N2048 per
microbatch, four accumulation steps and six rear-half DISM layers; late-step
times were about 1150-1190 ms. Even extrapolating the uniform-random GPU-only
saving optimistically to all 24 layer/microbatch calls saves only about 21 ms
per optimizer step before planner/transfer costs. This is not a full-model
speedup measurement; real document lengths, hard fractions, labels, soft work
and CPU scheduling matter. Current events are not a generally faster drop-in
replacement, especially for repeated labels or ragged batches.

Raw reports (ignored):
`build/hard_backward_kernel_compare_20261014.json`,
`build/hard_backward_kernel_compare_repeat_20261014.json`.
The benchmark and resource artifacts do not modify checkpoints or training.

## Chunk-parallel backward prototype

`tools/hard_prefill_backward_fused.py` is an opt-in consumer, not production
dispatch. The local FLA `simple_gla/chunk.py`, `common/chunk_h.py`,
`common/chunk_o.py` and `retention/chunk.py` informed the decomposition;
no FLA implementation is vendored. In particular, local `dS = dO @ V.T`
is shared by dSQ and dSK, and `SQ @ SK.T` also provides dV and analytic tau.
Single-chunk streams use five MMAs instead of the original two scans' eleven.

The execution has three stages for multi-chunk streams:

1. Parallel per-chunk forward key summary and reverse query summary.
2. Exclusive forward/reverse summary propagation, with only elementwise
   affine updates in the serial loop, no Tensor Core work. Each CTA owns an
   entire R*DV matrix and overwrites its summaries with the incoming forward
   and outgoing reverse boundary states. Reset chunks zero both the affine
   multiplier and reverse summary, but retain the forward key summary.
3. Parallel fused dSQ/dSK/dV/dtau readout using these boundaries and the local
   causal pair matrix. Local tau combines query/key log derivatives before
   reduction and before the BF16 cast of dS used for vector MMAs.

Single-chunk streams execute only stage 3, without summary allocation or
state propagation. CPU offsets create compact single/multi stream buckets;
optional CPU chunk lengths further select 16/32/64-wide single-chunk tiles.
Thus a C64 plan does not force every tiny stream to compute a 64x64 matrix.
No device metadata readback is needed. `long_mode='scan'` retains the old
register scan for explicit comparison. Cross-stream output atomics are global
FP32; there are no shared FP32 atomics.
Streams overlap in original token indices, unlike disjoint temporal chunks
in ordinary GLA, so vector contributions need a cross-stream reduction.
Atomics are an implementation choice for that reduction, not a requirement
of summary propagation. The prototype separately adds local and boundary
vector terms to shorten live ranges; it also atomically combines long-chunk
tau partials per stream and uses a head scatter reduction. A temporary-buffer
segmented reduction could replace these, but is not implemented here.

The prototype's automatic launch policy uses two warps for 16-wide local
tiles and four for 32/64-wide tiles and summaries; state propagation uses
four. Explicit 2/4/8-warp overrides remain diagnostic options. Neither the
native planner's C16 default nor production dispatch is changed.

Temporary matrix storage is `2 * multi_chunks * R * DV * 4` bytes, with the
summary-to-boundary replacement in place. B16/H6/N2048 repeated labels need
2,532,311,040 bytes at C16, 1,071,120,384 at C32 and 440,401,920 at C64.
These are matrix buffers only, excluding metadata, gradients and allocator
reservation. The initial C16 implementation with eight small scan CTAs per
stream was bandwidth/scheduling limited and slower than register scanning;
parallelizing alone is not sufficient.

### GPU results and remaining limitations (2026-10-14)

Final same-source BF16 measurements, RTX5090, B16/H6/N2048/R32/DV64,
tau=.5, five graph samples with five replays of four calls each. The timing
scope remains GPU-only, including clearing and tau reduction, excluding CPU
planning/bucketing, uploads and forward saves. All numbers are milliseconds:

| Pattern/layout | C16 three-stage | C32 three-stage | C64 three-stage | C64 register fallback |
| --- | ---: | ---: | ---: | ---: |
| Uniform random | 0.372 | 0.372 | 0.373 | 0.373 |
| Uniform repeated | 9.275 | 5.195 | 3.614 | 3.604 |
| Ragged random | 0.273 | 0.273 | 0.297 | 0.272 |
| Ragged repeated | 7.005 | 4.044 | 2.906 | 2.833 |

The fallback column still fuses/buckets single-chunk streams; only long
streams use the original two register scans. CUDA full-core measurements
across these runs were 4.72-5.33 ms uniform random, 5.64-6.27 ms uniform
repeated, about 2.02 ms ragged random and 2.39 ms ragged repeated. Therefore
C64 fixes much of the matrix-traffic problem, but at N2048 the three-stage
route is only comparable to the bucketed register route, and ragged repeated
remains slower than CUDA. For B1/H6/N8192 repeated labels, C64 three-stage
took 1.558 ms versus register fallback 1.875 ms and CUDA 7.329 ms: deeper
streams start to benefit from parallelizing the chunk matrix work.

Forcing every short stream to compute a full C64 tile was slow (about
2.05 ms even on random labels); CPU length bucketing restores small tiles.
A 32-query subtile within C64 was also tested, but increased output atomics
and did not eliminate spills; it is not retained. The retained BF16 C64
single-chunk 16/32/64-width variants use 56/138/249 registers with no spills;
the summary and propagation kernels use 99/88, also no spills. The C64
long-chunk fused output uses 254 registers and **12 spills**, with local
memory operations visible in SASS. All inspected variants have no CALLs or
shared atomics. Spill freedom is explicitly not claimed for the complete
pipeline. TF32x3 is a numerical diagnostic, not a recommended C64 fast path:
large tiles spill heavily (522/528 for the C64 single/long output variants),
and take 13.34 ms uniform repeated / 10.29 ms ragged repeated.

Against the independent dense FP64 tau reference, BF16 C64 relative L2 is
8.57% for B16 uniform repeated and 2.64% ragged repeated, with zero sign
flips among six heads in each case. B1/N8192 repeated has 34.10% relative L2,
also zero of six signs flipped. These synthetic samples do not guarantee
sign preservation on trained labels. The C16 uniform repeated comparison
still has one sign flip and 27.70% relative L2; it is not hidden by relaxed
acceptance gates. C64 TF32x3 uniform repeated has 1.14e-5 relative L2 and
zero sign flips.

Larger chunks also increase the current padded metadata: B16 uniform
random grows from 27.95 MB (C16) to 105.71 MB (C64), and repeated from
169.86 MB to 533.94 MB. Bucket index uploads are additional. GPU-only
results cannot establish planner/H2D overlap or end-to-end training gains;
compact event storage is a separate remaining layout opportunity. No
checkpoints, model dispatch or live training state were changed.

Final reports are ignored under `build/hard_backward_c16_final_20261014.json`,
`build/hard_backward_c32_final_20261014.json`,
`build/hard_backward_c64_final_20261014.json`, and
`build/hard_backward_c64_n8192_20261014.json`. Earlier sweep reports record
the unsuccessful full-width-short-stream and query-subtile experiments.

Validation: 345 combined hard-gradient, native-planner, SAM-inference and
native-frontend tests passed, including 64 new fused-kernel cases. Coverage
includes the independent FP64 dense vector oracle, BF16/TF32x3 arithmetic,
BF16/FP32 inputs, R16/32, DV32/64, C16/32/64, positive/negative/zero/high
tau, resets inside streams, nonidentity stream IDs, all-soft empty plans,
mixed hard/soft row-heads, asynchronous native upload and graph replay with
changed input values. N4096 long streams also agree with the prior scan.
Vector acceptance thresholds were not changed; tau diagnostics retain the
original-gate failure reporting described above. At B16/N2048, vector
relative L2 differences from CUDA are below 0.18%; CUDA is not the oracle.
All 64 new cases separately passed GPU memcheck, synccheck and racecheck:
zero errors, and zero racecheck warnings/hazards, without suppressions.
Sanitizer logs are ignored `build/hard_fused_{memcheck,synccheck,racecheck}_20261014.log`.

### When streams become long

A stream is a centroid matching group of key/query events, not a contiguous
token run. Concentrated labels, periodicity and repetition can create long
streams; even uniform random labels develop longer streams as N grows.
The centroid tree's O(log N) depth does not bound a stream to O(log N) events.
CPU observations at tau=.5 (one document/head; C16):

| Labels | N2048 max chunks | N8192 max chunks | N32768 max chunks |
| --- | ---: | ---: | ---: |
| Independent uniform 512 | 1 | 3 | 8 |
| Independent uniform 64 | 4 | 11 | 37 |
| Independent uniform 8 | 20 | 67 | 265 |
| Identical labels | 128 | 512 | 2048 |
| Repeated period of 16 | 8 | 32 | 128 |

These are synthetic observations, not trained-label distributions or bounds.
At N2048, repeated labels still have 95.1% single-chunk streams; the longest
stream contains 2047 events. With constant keys but independent 512-way query
labels, the longest observed stream has 964 events/61 chunks: both sides
need not be constant. Physical chunk counts also include reset boundaries,
which break the actual recurrence dependency. B/H increase the number of
streams, while document boundaries limit individual stream lengths.

## Default training operator and module integration

Following the opt-in adapter commit `3b26dbf`, hard/soft dispatch is automatic
in public `dism_core`, `voc_dism`, `DismAttention` and inherited hybrid modules.
Soft sequences retain CUDA O(N²) forward/backward. Hard sequences use integer
CUDA O(N²) forward while CPU planning and metadata upload run asynchronously,
then chunk-parallel Triton dSQ/dSK/dV/dtau. Sequence-level masks and packed-row
mapping are unchanged; direction is irrelevant only for hard sequences.
Native `cu.core_forward`/`core_backward` and low-level `forward_core`/varlen
helpers remain soft-only components. They are not the public full operator.

The default runtime is created lazily once per process/device, shared across
layers, and drained at interpreter shutdown. It is not stored on modules and
adds no parameters, persistent buffers, RNG counters, optimizer entries or
checkpoint keys. Workers default to16, physical event chunk size64, and live
plan capacity32. `DISM_PLANNER_WORKERS` and `DISM_PLANNER_MAX_INFLIGHT` may be set
before the first invocation. Capacity is a ticket count, not a byte budget;
the C64 padded-metadata cost below still applies. Capacity exhaustion raises
instead of deadlocking forward. `shutdown_hard_planners()` in `hard_training`
drains/retires default runtimes; existing graphs retain their own valid plans.
Normal training requires no manual setup/cleanup or extra forward arguments.
An explicit `hard_planner=` remains an eager override for controlled experiments.

Eager and AOT paths save a CPU int64[1] opaque handle whose **native tensor
storage owns the ticket**. Views/detach share ownership; it does not depend on
Python tensor-wrapper lifetime or a global ID registry. Backward releases it
with saved tensors, even when callers retain the output/loss; retain_graph
keeps it intentionally. Dropped graphs, partial failures and checkpoint
recomputations follow storage ownership. The native accessor verifies the
DataPtr deleter before accessing its context; cloned/serialized/offloaded
handles fail explicitly. Hooks which copy all saved tensors are unsupported.
No raw pointer is encoded in tensor values. During interpreter finalization,
remaining native handles deliberately avoid calling back into a finalized
Python runtime; ordinary graph cleanup releases tickets normally.

Core and vocabulary forward/backward custom ops have versioned hybrid names,
fake tensor contracts and explicit saved-state autograd. They preserve a
separate soft-only output/normalizer for CUDA backward and merge hard gradients
before vocabulary gradient assembly/casts. Fullgraph AOT/Inductor works for
default fixed and packed APIs/modules. No-grad evaluation computes hard/soft
forward without creating a planner. Packed shape handling, checkpoint RNG
replay, nanochat's true-direction/sequence-row sampling and GDN graph breaks
are preserved. CUDA Graph capture of asynchronous training remains unsupported.
Retained compiled backward uses Torch's `donated_buffer=False` configuration;
the library does not change this compiler setting globally.

The previous one-layer handoff timings below are not a performance claim for
the integrated model. In particular, runtime integration does not remove C64
metadata padding, bound it in bytes, guarantee enough per-layer slack, or make
CPU throughput sufficient for an arbitrary number of layers/microbatches.
No live training/checkpoint or dataset state is modified by this integration.

Subsequent checkpoint-based full-model replay is recorded in
[TRAINING_BENCHMARK.md](TRAINING_BENCHMARK.md#72m-full-model-training-replay).
At72M/B16/N2048 with four accumulation steps, the default16-worker path took
about1106–1108ms per step versus1114ms for the old quadratic core with the same
current interpolation: no substantial end-to-end improvement was established.
Planner tail waits, padded metadata traffic and the larger convolution/CE
costs remain visible despite the reduced DISM GPU work. These results supersede
the earlier statement that full-model throughput had not yet been measured;
they are bounded resident-batch replay, not live dataloader training throughput.
Those initial numbers use the fused convolution frontend. The user's usual
linear/FLA frontend was subsequently compared with explicit in-memory weight
and Adam-moment layout conversion: default hybrid938.2ms versus old-core932.8ms,
with current interpolation on both sides. No stable core-path gain is established
there either. Shortconv is no longer the dominant GPU category; CE, metadata
traffic and frontend scalar-read synchronization deserve separate attribution.
See the linear/FLA comparison in the same report for numerical checks and limits.

Validation on RTX5090 (sm120a), conda `blkw`, 2026-10-14:

- 342 operator/model training tests and 109 module/nanochat/resume tests passed.
  Coverage includes fixed/packed soft/hard/mixed masks, eager and fullgraph
  AOT/Inductor, checkpoint replay, retained/discarded graphs, handle lifetime,
  noncontiguous vectors, int64 labels and unchanged module state keys.
- A seven-case ownership/checkpoint/long-hard-stream subset passed memcheck,
  synccheck and racecheck separately: zero errors; racecheck also zero warnings.
  This subset does not reclassify the historical soft-pipeline ARRIVES reports.
- An expanded CPU/reference decode regression exposed the pre-existing
  `forward_torch()` unexpected `vocab_temperature` keyword mismatch, also present
  at `3b26dbf`. It remains unfixed and is outside this CUDA training integration.
- No device kernel math or launch defaults changed. Previous C64 long-readout
  spill counts and tau-cancellation limitations still apply; these tests do not
  establish full-model training throughput or universal dtau sign accuracy.

## Eager asynchronous training integration

Historical opt-in stage (`3b26dbf`); superseded by the default integration above.

`flash_dism.hard_training.AsyncHardPlanner` and the public
`dism_core(..., hard_planner=planner)` / `voc_dism(..., hard_planner=planner)`
provide an explicit complete eager training route. Fixed and packed paths use
the same coordinator and chunk-parallel kernels. The default model/nanochat and
`compiler.py` custom-op dispatcher remain unchanged and soft-only. This is not
a claim of end-to-end compiled training integration or speedup.

One planner per device owns a single coordinator thread, the existing bounded
native worker pool (default 16), and separate D2H/H2D streams. Forward snapshots
labels, head tau and row/head flags on the producer stream, queues pinned D2H,
and proceeds with soft CUDA plus integer hard forward. Only the background
coordinator host-waits the D2H event, prepares the native plan, buckets streams
and uploads metadata. Backward first queues soft CUDA work, then waits for the
CPU future and inserts GPU dependencies for both bulk metadata and bucket
uploads. All vector contributions accumulate in FP32 before the autograd BF16
boundary. Hard Q/K/LSE/vocabulary derivatives remain actual zero tensors.
The soft backward retains its original soft-only output/normalizer, not the
sum with hard output. Tau derivatives come from the event-local analytic terms.

Packed labels are rearranged into original equal-length row/head/token order
before D2H; planner documents use row-local offsets, while event rows still
address the original flattened BNHD vectors. Documents never cross training
rows; empty documents are omitted. Each invocation retains its own pinned
buffers, GPU plan and upload events, including checkpoint recomputation and
retained-graph backward. No mutable-label cache or global plan-ID registry is
used. Source in-place mutation after forward is caught by saved-tensor version
checks. Cross-stream consumers record metadata allocation lifetimes.

`max_inflight` (default 32) bounds live tickets, including already-prepared
plans held by graphs. Saturation raises immediately: blocking in forward could
deadlock before backward releases a slot. This is a count bound, **not a byte
budget**: C64 repeated-label metadata is large. Size it for live layers and
microbatches/checkpoint replays, and retain no unnecessary output graphs.
Dropping a graph before planning completes does not cancel the running job;
the job retains its slot/buffers until completion. `close()` drains CPU jobs;
existing tickets remain valid. Exceptions are propagated at backward and the
coordinator remains usable after a failed invocation. No RNG, checkpoint,
annealing, optimizer or dataloader state is touched.

Native frontend core, hard-forward, operand/embedding preparation and layout
construction now release the GIL around ATen/CUDA work, reacquiring it only
to construct Python return containers. The first integrated memcheck run
exposed a lock inversion: main thread held the GIL while waiting for the CUDA
allocator, and the upload thread held the allocator lock while a sanitizer
callback requested the GIL. GDB stacks are retained in
`build/hard_training_allocator_deadlock_20261014.log`; that stalled run was
terminated, not counted as a sanitizer pass. The GIL fix changes no kernel
math, launch parameters or precision. An eight-live-plan allocation-overlap
regression covers the concurrent native/frontend path.

The Triton implementations now live in `python/flash_dism/kernels/`; tools
retain compatibility imports and the independent symbolic fixture. Their math
and launch settings are unchanged. Nonblocking bucket uploads retain pinned
sources and add their own event/lifetime dependency. Existing synchronous
preplanned graph tests remain supported; the **async training adapter rejects
CUDA Graph capture**, and introduces an explicit graph break under Dynamo.
Fullgraph/AOT state transport, module/nanochat opt-in, sustained multilayer CPU
throughput and a byte-bounded memory strategy remain follow-up work.

RTX 5090, B16/H6/N2048/D32/R32/DV64, all hard, tau=.5, C64, 16 workers.
Q/K width D32 is only used by the skipped soft branch in this handoff fixture;
these are not full 72M-model step measurements.
Two measured repetitions after one warmup per condition; synthetic uniform
512-way random or identical labels. Packed documents are256/512/256/1024 per
row. Single outstanding invocation; no model/vocabulary/optimizer timing.

| Layout / labels | CPU prepare (ms) | D2H (ms) | H2D window (ms) | Metadata (MB) | Exposed host wait, 0 / 50 ms slack |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed / random | 16.99 | .080 | 2.51 | 105.81 | 18.21 / .004 |
| Fixed / identical | 65.20 | .077 | 11.98 | 533.94 | 73.54 / 24.83 |
| Packed / random | 15.21 | .077 | 1.86 | 78.55 | 16.19 / .005 |
| Packed / identical | 55.04 | .079 | 11.69 | 522.52 | 63.06 / 13.28 |

CPU/copy columns use the zero-slack condition. H2D is an event window covering
bulk plus bucket uploads and intervening host bucketing/launch gaps, **not pure
PCIe bandwidth**. Exposed host wait measures `future.result()` only; any final
GPU event dependency is additional. Slack is an artificial host sleep, not
measured overlap with other model layers. With 50 ms slack random-label full
backward wall time is1.31/1.27 ms fixed/packed; identical-label wall time remains
33.08/20.83 ms. Forward host return is approximately .95–1.37 ms. These numbers
do not establish sustained throughput or hide the padded C64 metadata cost.

Resource inspection of the packaged BF16 kernels reproduces the prior results:
short widths16/32/64 use56/138/249 registers, summary99, propagation88, all
without spills. Long C64 readout still uses254 registers and **12 spills**.
All inspected kernels have zero CALLs and zero shared atomics. No tolerance or
oracle changes accompany integration. Reports are ignored under
`build/hard_training_async_final_20261014.{json,log}` and `build/hard_async_cubins/`.

Final selected regression: **317 passed**, including26 integration tests.
Coverage includes fixed/packed all-soft/all-hard/mixed reference gradients,
unchanged default compiler custom ops, zero hard vocabulary gradients,
reentrant/non-reentrant checkpoint replay, retained graphs, source mutation,
document-row rejection, bounded tickets, early graph discard, injected planner
failure/recovery, cross-stream backward and multiple live plans. Existing
gradient/forward acceptance thresholds remain unchanged; this is not a
universal guarantee of shared-tau sign preservation on trained labels.
Memcheck:15 integration cases, zero errors. Unfiltered synccheck and racecheck:
three focused fixed/packed long-stream and cross-stream cases each, zero
errors/hazards/warnings, no suppressions or ARRIVES exceptions. The latter
cases select all-hard rows; they do not supersede the known soft-forward
racecheck limitations documented in SOFT_ONLY_CUDA.md. Logs are
`build/hard_training_all_final_20261014.log` and
`build/hard_training_{memcheck,synccheck,racecheck}_final_20261014.log`.

Reproduce the handoff experiment (artificial slack explicitly labelled):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2 \
PYTHONPATH=$PWD/dism_v3/python:$PWD/nanochat \
  /home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/bench_hard_training_async.py --repeats 2 \
  --output dism_v3/build/hard_training_async_new.json
```

## Reproduction

From the workspace root, choose fresh ignored output paths:

```bash
OPENBLAS_NUM_THREADS=1 MAX_JOBS=2 PYTHONPATH=$PWD/dism_v3/python:$PWD/nanochat \
  /home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/bench_hard_backward_kernels.py --batches 16 \
  --patterns random repeat --layouts uniform ragged --precisions bf16 tf32x3 \
  --chunk-size 64 --fused-warps auto --fused-long-modes scan three_stage \
  --skip-register-baseline \
  --output dism_v3/build/hard_backward_c64.json

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
