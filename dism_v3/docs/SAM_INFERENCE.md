# SAM full-hard inference

Production code: `python/flash_dism/inference/`; native sources: `csrc/` below
that package. Public examples and complexity bounds are in the main README.

## Provenance and scope

Restored from `main:7687cf5`, formerly `dism_v4/decoding`. The hard matching
algorithm is shared with v3 and does not require v4 training operators. The
migration retains CPU parallel prefill planning, chunked Triton execution,
CPU-planned CUDA decoding, GPU ordinary-step planning and periodic rebuilds.
It changes packaging/imports, removes the Python diagnostic DecodeCache that
depended on a test module, and replaces source-directory compilation with
installed native extensions plus user-cache JIT fallback. Kernel and planner
mathematics are unchanged.

The optional binary query reset stops the incoming match before the current
query; it does not delete historical keys and is not a document-isolation
mechanism. For ordinary v3, omit it. Start a new engine plan/cache per document.
No finite-delta or soft-row inference is included.

The diagnostic [hard-prefill training study](HARD_PREFILL_TRAINING.md) checks
analytic vector/tau gradients and describes a possible training backend.
It does not change these inference APIs or production training dispatch.

## Integration boundary

`HardDismPrefill` consumes int32 BHN labels and contiguous BNHD SQ/SK/V, and
returns FP32 BNHD. Its persistent native worker pool defaults to 16 threads
(1–64 supported); planning and direct packing release the GIL. It shares the
training pool/Builder implementation, disables derivative computation, and
uploads one pinned metadata blob including dispatch indices. The GPU grid
combines all heads. `ParallelPrefillPlanner` retains the older Python-threaded
interface for low-level callers and reference comparisons.
`HardDismDecoder` consumes the same layouts (it packs per-token slices), with
BF16 payloads by default. `prime` requires complete prompt history and is
separate from prefill output computation. Tau is fixed over cache lifetime.

This is an explicit operator backend, not an implicit replacement for
`DismAttention.forward(use_cache=True)` or nanochat generation. Model adapters
must supply the original projections, labels, readout transforms, convolution
states, GDN state and output gate. No model defaults are changed here.

Low-level `GpuPlannerCache.step` has a reused output allocation and supports
graph replay between rebuilds. Call `check_status` to check device errors;
rebuild/recapture at the horizon. Do not replay graphs across a rebuild, use a
cache across streams, or reuse it after a capacity/reset error.

## Long-stream sequence parallelism

Each prefill `Program` stream is a first-order linear recurrence on an `R×DV`
state with a scalar per-chunk decay:

```
M_{c+1} = a_c * M_c + S_c,   a_c = exp(last_prefix),  S_c = sk_c^T (v_c*scale_c)
history  = (sq_c @ M_c) * scale_c
```

That is exactly the chunkwise form of SSD / Simple GLA (`S_t = a_t S_{t-1} +
k_t v_t^T`, `o_t = q_t^T S_t`). The associative operator is
`combine((a1,S1),(a2,S2)) = (a2*a1, a2*S1 + S2)` with identity `(1,0)`; the
carried element is one scalar plus an `R×DV` matrix, generally full rank.

Measured stream shape (`C=16`): ~89% (random) / 95% (all-same) of streams are
one chunk, the mean grows like `O(log N)`, and the single longest stream is
`Theta(N)` chunks (all-same ~`N/16`, random ~`N/64`). Only that tail needs
sequence parallelism. Its state passing is the FLA `chunk_h_parallel` pattern:
(1) per-chunk summaries in parallel over chunks, (2) a serial reduction over
`(K,V)` tiles that applies the inter-chunk scalar decay and stores the prefix
state, (3) `chunk_fwd_o` per chunk in parallel. Passing state stays `O(R*DV)`
live (or `O(sqrt(L))` with checkpointing) instead of materializing all `L`
chunk states. Reference: `fla/ops/common/chunk_h.py` and `chunk_h_parallel.py`.
The DISM-specific parts a reused kernel would not cover are non-uniform chunks
(per-key gates, `k_c <= C`), event-stream indexing instead of token order, and
the FP32 atomic reduction across overlapping streams.

Prototype: `TritonPrefillPlan.execute_parallel`
(`python/flash_dism/inference/prefill_triton_parallel.py`) implements the three
phases forward-only, sharing the exact chunk packing and coefficients with the
serial `_execute`, and switches BF16 / TF32x3 like the serial kernel. Every
kernel takes an explicit id list, so `execute_dispatch` can route streams by
chunk count (`dispatch_threshold`, default 64): streams at or below the
threshold keep the register-carried serial kernel, longer streams go through
the chunk-parallel phases. `HardDismPrefill.__call__` uses this routing, so a
pathological single long stream no longer carries the full per-chunk work on
one CTA; workloads without streams above the threshold run the serial kernel
exactly as before. Both paths are checked against the serial reference and the
FP64 oracle (`test_prefill_parallel_state`, `test_prefill_dispatch`).

Performance is deliberately not tuned: the serial kernel already launches O(N)
stream CTAs, so the GPU is throughput-saturated before any single stream
dominates, and at N=16k..64k on one head dispatch is 1.0-1.7x the serial time.
The threshold defaults high to keep the common case on the cheaper serial
kernel; the parallel phases only bound the worst case. They still materialize
`[long_chunks, BR, BD]` FP32 sums/states. Checkpointing or fusing the passing
kernel, and a calibrated threshold, are not implemented yet.

## Validation

```bash
CUDA_HOME=/usr/local/cuda-13.4 PYTHONPATH=dism_v3/python python -m pytest \
  dism_v3/tests/test_sam_inference.py dism_v3/tests/test_readme.py
```

The regression oracle independently computes the dense FP64 causal recurrence,
including zero-value fallback and signed SQ/SK readout. Tests cover repeated,
random and mismatched labels, zero/positive tau, nonaligned sequence lengths,
BF16 and TF32x3 prefill, serial/parallel planners, prime/append across multiple
rebuilds, and CPU/GPU decode planners. CPU FP64 event execution is also checked.
Separate tests cover graph replay/recapture and device horizon/capacity guards.
No training gradient tolerances or kernel defaults are changed by this migration.

Migration validation on RTX 5090 / conda blkw (2026-10-08): 23 SAM/README
checks passed from source. A separately installed wheel passed these plus
the 92 multi-configuration operator checks (115 total). That run disabled the
JIT builder and asserted both SAM extensions loaded from the installation
directory, not an old checkout. The source archive includes all 12 native SAM
source/header files and the inference regression tests. This migration does
not claim a new checkpoint-level quality or throughput measurement.

## Prefill performance baseline (2026-10-23)

Baseline implementation: `825b5f2`, including the shared SAM/CSR improvements
from `8e19eb2`, but before porting the training planner's native batch scheduler,
worker-local scratch reuse and direct pinned packing to inference. No planner
or GPU kernel behavior was changed for this measurement.

Tool: `tools/bench_inference_prefill.py`. RTX 5090, Core Ultra 9 285K (24 cores),
conda `blkw`, one Torch CPU thread, no CPU affinity or locked GPU clocks.
H=12, R=32, DV=64 match the current model; input vectors are BF16 and output
is FP32. Production defaults are retained: eight planner workers, C=16,
BF16 MMA, dispatch threshold 64. Every batch row is one independent document,
full-hard, tau=.5. Random query/key labels are independently uniform over 512
symbols (seed 120); repeat means all query/key labels are zero. Neither is
a checkpoint-derived label distribution.

Labels, vectors and tau start on GPU. End-to-end is the unchanged public
`HardDismPrefill.__call__`, followed by completion synchronization, including
D2H, CPU preparation, metadata upload, output allocation/zeroing and dispatched
vector execution. Native loading/JIT and the cold first call are excluded;
the report retains the cold-call time separately. There are two warmups and
seven measured calls. GPU-only medians use seven sets of ten prepared-plan
executions with CUDA events, including output zeroing and long-stream scratch;
these are eager launches, not CUDA Graph replay. Kernel launch gaps can be
included in event intervals.

Stage measurements separately mirror `prepare()`, with explicit fences:

- D2H includes reset creation, label stacking, tau conversion and transfer.
- Plan/pack includes Python list conversion, native planning, per-head chunk
  compilation and NumPy export; it is not pure native planner time.
- Merge includes row relabeling, NumPy concatenation and releasing per-head
  programs/packed arrays.
- Upload/setup includes the existing separate field uploads, allocations and
  CPU dispatch-index preparation; it is not pure PCIe wire time.

Stage medians and end-to-end medians are independent measurements and need
not sum. All times below are milliseconds for one DISM operator; projections,
vocabulary interpolation, GDN/FFN, decode-cache prime and model logits are
excluded. These are not model tokens/s or time-to-first-token measurements.

| B | N | Labels | D2H | Plan/pack | Merge | Upload/setup | GPU dispatch | End-to-end |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 2048 | random | .11 | 3.41 | 1.16 | .40 | .060 | 6.14 |
| 1 | 8192 | random | .25 | 14.25 | 4.81 | 1.07 | .174 | 20.05 |
| 1 | 16384 | random | .33 | 29.34 | 11.62 | 2.12 | .597 | 44.95 |
| 1 | 2048 | repeat | .16 | 8.08 | 5.52 | 1.75 | .375 | 16.71 |
| 1 | 8192 | repeat | .25 | 34.24 | 40.70 | 5.99 | 2.606 | 84.63 |
| 1 | 16384 | repeat | .36 | 77.14 | 92.33 | 12.40 | 5.730 | 186.53 |
| 8 | 2048 | random | .34 | 25.68 | 10.25 | 2.18 | .710 | 43.01 |
| 8 | 8192 | random | 1.81 | 95.65 | 49.02 | 6.66 | 2.339 | 158.87 |
| 8 | 16384 | random | 1.58 | 193.60 | 104.99 | 13.52 | 4.890 | 319.88 |

CPU plan/pack and merge dominate. At B=1/N=8192, random labels generate
389,004 events and 9.47 MiB of device metadata including routing indices;
repeat generates 2,408,940 events and 74.17 MiB. At N=16384 the corresponding
metadata sizes are 22.53/153.36 MiB. Metadata expansion and host copying are
therefore important even though the input label D2H is small.

All random cases above have no long streams. For the repeated-label cases,
prepared serial/dispatch GPU times are .368/.375 ms (2K), 2.164/2.606 ms (8K),
and 4.374/5.730 ms (16K). Dispatch is slower here; no threshold or kernel was
tuned. Its CPU overhead is still much larger than this execution difference.

The 8K random-label worker sweep gives:

| Workers | B=1 plan/pack (ms) | B=1 end-to-end (ms) | B=8 plan/pack (ms) | B=8 end-to-end (ms) |
|---:|---:|---:|---:|---:|
| 1 | 48.31 | 54.51 | 385.45 | 441.51 |
| 4 | 18.20 | 24.56 | 116.80 | 174.13 |
| 8 | 14.25 | 20.05 | 95.65 | 158.87 |
| 16 | 15.09 | 22.51 | 104.75 | 162.92 |
| 24 | 15.75 | 21.95 | 104.39 | 161.66 |

Eight workers uses the default/batch runs above; the remaining configurations
use the separate sweep process. More workers do not help in this measurement,
but this is not evidence that all 24 CPU cores are compute-saturated. List
conversion/export retains the GIL and merge is serial; CPU affinity and
P/E-core scheduling were not controlled. Native scheduler/workspace reuse and
eliminating intermediate exports/merge copies are the next candidates to
measure, rather than increasing the default worker count.

The existing `test_sam_inference.py -k 'prefill or binary_query_reset'` suite
passes 87 cases (15 decode cases deselected), including the FP64 oracle,
BF16/TF32x3, reset, parallel-state and dispatch checks. Existing tolerances are
unchanged. Every measured shape's initial output was also checked finite.
No new kernel or sanitizer claim is made by this baseline-only change.

Reproduction (set PYTHONPATH to this worktree only):

```bash
conda activate blkw
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
export TORCH_EXTENSIONS_DIR="$PWD/dism_v3/build/torch_extensions"
export OMP_NUM_THREADS=1 MAX_JOBS=2
python dism_v3/tools/bench_inference_prefill.py --backend legacy \
  --output dism_v3/build/prefill_baseline/default.json
python dism_v3/tools/bench_inference_prefill.py --backend legacy --batches 8 --patterns random \
  --output dism_v3/build/prefill_baseline/batch.json
python dism_v3/tools/bench_inference_prefill.py --backend legacy --batches 1 8 --tokens 8192 \
  --patterns random --workers 1 4 16 24 \
  --output dism_v3/build/prefill_baseline/workers.json
```

Use fresh output paths when rerunning. JSON retains raw samples, program sizes,
source hashes, commit, command, CPU information and the loaded native module
path. All results/logs are ignored under `dism_v3/build/prefill_baseline/`.

## Native prefill preparation (2026-10-23)

`HardDismPrefill.prepare` now calls `NativePrefillPlanner` rather than creating
Python tasks per head. The native batch interface borrows contiguous CPU label
and tau arrays directly. It shares `PlanningWorkers`, worker-local `Builder`
scratch, boundary discovery and direct event packing with training, but passes
`gradients=false` and preserves inference's optional query-reset plane. No
derivative buffers or tau derivative work are introduced. Training constructor
defaults and its packed output are unchanged.

All row relabeling, padding and dispatch routing are performed natively. The
event arrays and routing indices are packed directly into disjoint ranges of
one pinned allocation. This removes per-head list conversion, exported NumPy
arrays, serial concatenation and repeated field uploads. Padding and event
order are byte-identical to the old layout. The ordinary no-reset call also
omits allocating/transferring the reset plane. The async upload is a single
`Tensor.to(non_blocking=True)`; the pinned allocator tracks copy completion,
so a prepared GPU plan need not retain a CPU metadata duplicate. Each plan has
independent storage, with no engine-owned mutable buffer that could overwrite
an earlier prepare. GPU kernels, math, launch configurations and dispatch
threshold remain unchanged, as does the explicit model-integration boundary.

The benchmark now defaults to `--backend native`; `--backend legacy` retains
the old preparation pipeline only in the benchmark for contemporary A/B.
The following comparison uses the same settings as the baseline above and
fresh processes for each batch/backend, seven measured calls after warmup.
Both paths share the same unchanged core Builder math and GPU kernels. The
native timings include final pinned-storage lifetime handling. End-to-end
times are medians, in milliseconds; default worker count remains eight.

| B | N | Labels | Legacy | Native | Speedup |
|---:|---:|---|---:|---:|---:|
| 1 | 2048 | random | 5.140 | 2.581 | 1.99× |
| 1 | 8192 | random | 20.549 | 10.275 | 2.00× |
| 1 | 16384 | random | 44.484 | 23.053 | 1.93× |
| 1 | 2048 | repeat | 15.981 | 7.547 | 2.12× |
| 1 | 8192 | repeat | 82.004 | 31.618 | 2.59× |
| 1 | 16384 | repeat | 182.722 | 61.181 | 2.99× |
| 8 | 2048 | random | 42.822 | 12.356 | 3.47× |
| 8 | 8192 | random | 156.814 | 50.669 | 3.09× |
| 8 | 16384 | random | 315.352 | 114.415 | 2.76× |

At B=1/N=8192/random, the native prepare stages are .168 ms label D2H,
9.362 ms planning/allocation/direct packing and .323 ms upload/setup. There
is no merge pass. The contemporary legacy stages are .241/13.855/4.812/1.055 ms
(D2H, per-head plan/pack, merge, upload/setup). As before, separate stage medians
do not sum exactly to an independently timed public call. Remaining host time
is predominantly native planning, rather than Python task submission or export.

The same 8K random-label worker sweep on the new path gives:

| Workers | B=1 plan/pack (ms) | B=1 end-to-end (ms) | B=8 plan/pack (ms) | B=8 end-to-end (ms) |
|---:|---:|---:|---:|---:|
| 1 | 39.768 | 40.474 | 328.253 | 334.934 |
| 4 | 13.548 | 14.314 | 80.917 | 86.645 |
| 8 | 9.362 | 10.275 | 44.402 | 50.669 |
| 16 | 5.611 | 6.297 | 28.671 | 33.372 |
| 24 | 5.177 | 6.174 | 22.308 | 27.578 |

Eight workers uses the default runs; the other rows use a separate sweep.
More workers now help, especially for batches, unlike the previous Python
pipeline. These measurements do not prescribe a global default: B=1 has only
12 independent head jobs and CPU affinity/P/E-core placement was not fixed.

Validation:

- 343 tests passed across native inference packing, SAM prefill, native
  training planning and compact training planning (15 decode cases excluded).
  After finalizing pinned storage lifetime, all 117 inference cases passed
  again. One additional delayed-upload/reallocation test also passed, for 344
  distinct passing cases in total. Existing numerical tolerances were not widened.
- Thirty-one new inference cases include exact packed-byte and routing comparisons
  across changing lengths, resets, negative/zero/positive tau, per-row tau,
  widths 16/32/64, thresholds 0/1/64, mismatch/streamless heads, concurrent
  submissions, error recovery, multiple prepared plans, stream guards and
  prepared-plan graph capture/replay. Old FP64 GPU-oracle tests still pass.
- The standalone `tests/native/test_inference_packing.cpp` verifies 100
  concurrent plans against the original per-head packing. ASan/UBSan and TSan
  both finish with exit code zero, including routing and empty-stream cases.
- Targeted GPU memcheck and synccheck of upload ownership, two prepared plans,
  stream ordering, graph replay and delayed H2D followed by immediate pinned
  allocation/fill each report zero errors. No GPU kernels
  changed; this does not claim whole-model sanitizer or inference validation.

Build used the version's `python -m flash_dism.inference.build --component
prefill` entry point, with `MAX_JOBS=2` and a worktree-local extension cache.
Native code remains Torch/CUDA-free. Logs, source hashes, exact commands and
all raw samples are under ignored `dism_v3/build/prefill_native/`:
`b{1,8}_legacy.json`, `b{1,8}_native_final.json`, `workers_native.json`,
`tests.log`, `lifetime_tests.log`, `native_sanitizers.log`, `memcheck.log` and
`synccheck.log`. The stronger delayed-upload tests are recorded in
`async_{pytest,memcheck,synccheck}.log`. Earlier `b{1,8}_native.json` are
preliminary measurements.
No checkpoint, training job, default model dispatch or decode implementation
was changed.

## Decode analysis: ordinary steps versus rebuild (2026-10-23)

Prefill optimization is committed as `51c7e45`. This section records the decode
baseline before the CPU rebuild optimization described below. `HardDismDecoder` still defaults to
`planner_backend='cpu'`; the measurements explicitly select `'gpu'`.
The low-level GPU ordinary step launches one fused kernel with no host copy
or synchronization. The high-level adapter additionally creates/packs labels
and stacks output, and manages the Python position/rebuild schedule. Only the
low-level device-state path is suitable for direct graph replay; capture does
not repeatedly execute the adapter's Python position updates.

`tools/bench_inference_decode.py` measures BF16 R=32/DV=64, H=12, tau=.5,
independent uniform-512 labels or all-zero repeated labels. This is a resident
operator probe, not a model-generation benchmark. At N=8192/random/I=128,
ordinary-step wall times are approximately 46.01 us through `append`, 16.06 us
through raw `GpuPlannerCache.step`, and 17.15 us in a captured whole-window
probe (CUDA-event time 17.00 us). The extra ~30 us in the adapter includes both
CPU dispatch and additional GPU packing/copy kernels, not just CPU work.

The window graph contains precomputed inputs for I distinct tokens; its device
timing removes host submission gaps but cannot represent autoregressive model
throughput. Graph capture time is excluded and recorded separately, and the
graph is discarded/recreated after every rebuild. Full-model graph recapture,
projections, GDN/FFN, vocabulary selection and output logits are not measured.

### Rebuild interval sweep

The main comparison uses `--fixed-context`: each of three samples starts from
the same 8192-token prompt in a newly primed cache (prime is outside timing),
executes I new tokens and rebuilds at N+I. Capacity is the same N+4096 across
I=128..4096. The table reports medians of paired ordinary-window plus rebuild
wall time divided by I, in **us per batched decode step** (B tokens), not us
per token aggregated across the batch. Ordinary windows use the graph probe;
capture costs are excluded. CPU/GPU clocks and affinity were not fixed.

| Interval | B=1 random | B=1 repeat | B=8 random |
|---:|---:|---:|---:|
| 128 | 371.25 | 443.47 | 3001.15 |
| 512 | 116.84 | 171.99 | 808.63 |
| 1024 | 83.43 | 141.68 | 447.59 |
| 2048 | 78.47 | 152.16 | 282.81 |
| 4096 | 111.91 | 212.73 | 236.28 |

For B=1/random, moving from 128 to 1024 changes ordinary device time from
16.68 to 34.21 us and rebuild time from 45.36 to 50.36 ms. At 2048/4096,
ordinary device time rises to 51.10/95.53 us; the amortization benefit eventually
loses to the growing frontier work. For B=1/repeat, rebuild itself rises from
51.96 ms at 128 to 86.95/131.45/235.42 ms at 1024/2048/4096, in addition to
the ordinary-step increase. A larger interval is clearly useful, but it is not
monotonically better. At B=8, CPU rebuild scales poorly across 96 heads, so a
larger interval remains beneficial through 4096 in this probe. The best tested
interval is workload-specific, not an established global optimum.

### Where rebuild time goes

Before the CPU rebuild optimization, `GpuPlannerCache.rebuild` performed:

1. Device-state D2H/status checks.
2. `refresh_gpu`: copy **all** label history back, clear/refill CPU histories,
   reconstruct the recent band through `Planner.initialize`, rebuild SAM and
   scalar topology sequentially for every head, then rebuild GPU summaries.
3. `gpu_snapshot`: construct fresh padded CPU topology/transition arrays, sort
   each node's transitions, zero/fill a capacity-sized history tensor, and copy
   multiple fields to the GPU.
4. Allocate new GPU planner workspaces and reset task marks.

One same-position breakdown at ~8.5K/random/B=1/I=128 was .10 ms status,
30.04 ms refresh, 13.33 ms snapshot export/upload and .05 ms workspace setup.
These are separate samples, not a decomposition of the table's median. The
random case has **zero summary matrices**. A separate profiler run observes
only ~.53 ms of total device copy/fill activity during rebuild (two D2H and
30 H2D operations). CPU construction/export dominates; PCIe payload time does
not explain the ~45 ms unprofiled rebuild. Profiling changes wall time, so its
wall duration is not substituted into the timing table.

For 8K/repeat/B=1/I=128, the profiler counts **13,824 summary kernels**:
768 each of Euler scan, scan totals, subtree init and sample save, plus 10,752
ancestor steps. R*DV=2048 with `rebuild_chunk=32` means 64 channel tiles per
head; each tile invokes scan/init plus 14 ancestor levels and sample save.
Thus summary reconstruction has substantial launch and GPU work even though
ordinary decode is already fused. The profile measures ~26.8 ms summed
summary-kernel time; this is diagnostic under profiler overhead, not an exact
unprofiled time attribution.

The baseline CPU rebuild uses `dism_decode::Planner` with one `unordered_map` per
SAM node and nested child vectors. It does **not** use the optimized flat
`PrefillMatcher`/CSR workspace used by prefill/training. `initialize` also walks
back matching diagonals for every entry in the recent band, costing O(N*I)
on all-repeat histories. Increasing I exposes this CPU cost. Ordinary GPU
steps update the I-entry band and iterate the new-token frontier; random
inputs still pay the loop cost to skip zero-weight candidates.

### Optimization priorities

1. Use 1024 as a candidate for further correctness/model testing; test adaptive
   or batch-dependent choices rather than changing the default solely from
   synthetic measurements. Re-run the sweep after rebuild optimization because
   cheaper rebuild shifts the optimal interval downward.
2. Parallelize CPU rebuild per head with a bounded pool, reuse flat SAM/CSR
   storage and pack snapshot fields directly into pinned storage. Separate CPU
   jobs from owner-stream CUDA launches. Skip topology/coefficient upload when
   a head has no summary matrices.
3. Keep GPU history across snapshots and transfer only new labels to retained
   CPU histories; recover the already-computed band from GPU state rather than
   re-running long diagonal searches. Preserve exact reset and complete-history
   semantics when implementing this.
4. Batch/fuse summary reconstruction across heads/channel tiles to reduce the
   13K-launch repeated-label case. Increasing channel tile width alone trades
   launches for scratch/occupancy and needs measurement and numerical checks.
5. Only then tune the ordinary-step frontier traversal/compaction and adapter
   packing. Stable graph storage would also require stable pitches or dynamic
   device parameters, not just reusing pointers: snapshot dimensions currently
   enter captured kernel arguments by value.

The 15 existing decode/graph/horizon regression cases pass. Large-interval
performance cases additionally check finite output, exact device position and
error flags after each window, but **do not** establish long-context numerical
equivalence; an interval default change still needs numerical regression.
No kernel math, launch default, cache API or planner default was changed in this
baseline analysis.

Artifacts under ignored `dism_v3/build/decode_analysis/`: `ordinary.json` for
adapter/raw/graph comparisons; `fixed.json`, `fixed_batch.json`,
`large_fixed_capacity.json`, `large_batch_fixed_capacity.json` for the table;
`random_rebuild_profile.json` and `repeat_rebuild_profile.json` plus Chrome
traces for kernel/copy counts; `tests.log` for regressions. `intervals.json`,
`batch.json` use growing histories and `large*.json` without `fixed_capacity`
use different reserved capacities, so they are exploratory, not table inputs.
Every benchmark JSON records command, commit and source hashes.

```bash
python dism_v3/tools/bench_inference_decode.py --fixed-context \
  --prompts 8192 --modes graph --intervals 128 512 1024 2048 4096 \
  --output dism_v3/build/decode_analysis/fresh_interval_sweep.json
```

## Decode CPU rebuild parallelism and reuse (2026-10-23)

The native cache now uses a persistent per-cache worker pool for per-head SAM
rebuild, scalar metadata preparation and snapshot packing. `rebuild_workers=8`
is the default for this new option; valid values are 1..64, capped at BH. The
pool implementation is shared with prefill/training, but pool instances are
owned by individual decode caches. All Torch operations and CUDA launches stay
on the calling thread and owner stream; callers must serialize operations on a
cache. Existing planner selection, rebuild interval (128), summary chunk size,
kernel math and launch configurations are unchanged.

The implementation removes repeated work at the explicit rebuild boundary:

- Retain SAM nodes/transitions and extend only new keys. Reuse child vectors,
  traversal scratch and per-head export workspaces. Scalar topology, query state
  and summaries are still recomputed over the whole snapshot, so rebuild is not
  O(new tokens) overall. The SAM still uses per-node `unordered_map`; this is not
  a conversion to the flat prefill matcher.
- Keep the device history allocation across snapshots. `refresh_gpu(history,
  position, band)` imports only new labels and the live parity of the GPU band,
  avoiding full-history D2H and the O(N*I) repeated-label diagonal reconstruction.
  The existing two-argument call still replaces complete history and reconstructs
  the band. Prime still requires complete history.
- Pack summary metadata and snapshot fields directly into reusable pinned
  arenas, with one H2D upload for each nonempty packed blob. Skip summary metadata
  entirely for heads without summary matrices. A CUDA event protects each arena
  from CPU overwrite until its previous upload completes.
- Keep fresh GPU topology/state allocations for each exported snapshot. Retained
  history does **not** make old graphs reusable: topology, summary pointers and
  dimensions can change, so graphs must still be discarded and recaptured.

Pinned arenas retain their high-water allocation until cache destruction, grow
by at least 1.5x, and round to 16 KiB. At N=8192/I=1024, retained arena capacity
was 10.58 MiB for B=1/random, 29.64 MiB for B=1/repeat and 84.61 MiB for B=8/random.
These figures exclude CPU SAM/workspace memory and GPU storage. Multiple layer
caches each retain their own pool and arenas; select worker counts with the
whole process in mind.

### Paired performance results

RTX 5090, BF16 R=32/DV=64, H=12, tau=.5, fixed 8192-token prompt, capacity 12288,
three independent freshly primed samples per case, GPU planner explicitly
selected. The old native binary was saved from this worktree at `51c7e45` before
rebuilding, and loaded in a separate process through `--legacy-extension`.
Baseline and optimized jobs ran serially. CPU/GPU clocks and affinity were not
fixed. Numbers are medians; ordinary-window timing uses the synthetic graph
probe described above. Prime and capture are outside timing.

| Case | Interval | Old rebuild ms | New rebuild ms (8 workers) | Old amortized us/step | New amortized us/step |
|---|---:|---:|---:|---:|---:|
| B=1 random | 128 | 45.197 | 9.623 | 370.16 | 91.67 |
| B=1 random | 512 | 48.228 | 10.188 | 116.58 | 43.10 |
| B=1 random | 1024 | 50.863 | 10.773 | 80.53 | 41.84 |
| B=1 repeat | 128 | 51.591 | 37.348 | 437.16 | 325.90 |
| B=1 repeat | 512 | 66.464 | 40.227 | 171.81 | 120.74 |
| B=1 repeat | 1024 | 87.279 | 40.291 | 141.82 | 95.86 |
| B=8 random | 128 | 380.324 | 64.126 | 2994.30 | 522.13 |
| B=8 random | 512 | 398.539 | 67.204 | 805.48 | 158.40 |
| B=8 random | 1024 | 418.990 | 73.128 | 447.87 | 108.17 |

A step means one **batched decode step**, producing B tokens, not one token
aggregated across the batch. These are operator measurements, not model token/s.
At I=1024, random rebuild improves 4.72x for B=1 and 5.73x for B=8; the paired
amortized step improves 1.92x and 4.14x. Ordinary device time remains comparable
(30.84→31.29 us for B=1/random, 36.57→36.66 us for B=8/random), as expected with
unchanged ordinary kernels.

The worker sweep separates reuse benefits from parallel speedup at I=1024:

| Case | Old serial rebuild ms | New 1 worker | New 8 workers | New 16 requested | New 24 requested |
|---|---:|---:|---:|---:|---:|
| B=1 random | 50.863 | 39.013 | 10.773 | 10.153 | 10.784 |
| B=8 random | 418.990 | 278.684 | 73.128 | 57.745 | 56.488 |

For B=1, requests for 16/24 workers both cap at 12 heads. B=8 still gains from
8→16 workers, while 16→24 gives little improvement in this run. The default
remains 8 rather than using all cores in every layer cache. The recorded stats
confirm actual incremental work: at B=1/I=1024, `last_refresh_tokens=1024`,
`last_sam_extended_keys=12288` and `last_refresh_band_reused=1`; B=8 extends 98304
keys across 96 heads. GPU history pointers remain stable across rebuilds.

For repeated labels, CPU work is much cheaper but unchanged GPU summary
reconstruction remains substantial (the prior profile counted 13,824 kernels
at 8K/H12/I128). Its remaining cost calls for a separate kernel optimization;
this change does not fuse those launches. With cheaper CPU rebuild, B=1/random
I512 and I1024 are now close (43.10 vs 41.84 us/step), whereas B=8 still benefits
strongly through 1024. Re-evaluate intervals with real model workloads before
changing the default; this sweep does not establish a global optimum.

`rebuild_breakdown_ms` in the JSON is an additional **same-position** refresh
and export after the timed window. In the optimized version it extends zero
SAM keys and can reuse already-sized buffers, so it must not be treated as a
decomposition of the real-window rebuild table. `window_cache_stats` is saved
before that diagnostic refresh; `cache_stats` is saved after it.

### Validation and reproduction

- 359 existing SAM inference, native prefill and native/compact training planner
  regression cases passed. The 10 new `tests/test_decode_rebuild.py` cases passed,
  covering incremental band/history import, intervals 11/128/512/1024, resets,
  repeated/periodic labels, workers 1/4, queued asynchronous upload reuse, old
  snapshot ownership, two-argument refresh and argument validation.
- `tools/check_decode_rebuild.py` compared 9 saved old-binary cases with 1/4/8
  workers: all 27 combinations were bitwise equal for outputs, summaries,
  topology, transitions, state, band, counts, tau and live history. Cases include
  intervals up to 1024, prompt lengths up to 2049, two full rebuild windows,
  tau 0/.35/1.5 and single-step graph replay with changing inputs. This checks
  equivalence to the previous implementation, not a long-context FP64 oracle.
  The small CPU-label case also retains the independent FP64 oracle check at
  existing tolerances.
- The standalone native test (`tests/native/test_decode_rebuild.cpp`) passed
  ASan/UBSan with leak detection and TSan: 32 concurrent planners, 4 rebuilds each,
  replacement histories and metadata packing, compared to the saved old header.
- Compute Sanitizer memcheck and synccheck each passed 5 selected GPU tests with
  zero errors, covering asynchronous CPU-label uploads, legacy refresh,
  graph/rebuild boundaries and incremental I512 cases. This is selected-case
  sanitizer coverage, not exhaustive coverage of every shape. CUDA device
  translation units are unchanged; no new device CALL/spill behavior is introduced.

All generated binaries, fixtures, logs and result JSON remain ignored under
`dism_v3/build/decode_rebuild/`. Paired tables use `b1_before/after.json` and
`b8_before/after.json`; worker sweeps use `workers_1/16/24.json`. The baseline
binary and headers must be saved before rebuilding to reproduce an old/new
comparison. Commands below assume this worktree's Python paths and conda blkw:

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
export TORCH_EXTENSIONS_DIR="$PWD/dism_v3/build/torch_extensions"
export OMP_NUM_THREADS=1 MAX_JOBS=2
python -m flash_dism.inference.build --component decode \
  --output dism_v3/build/decode_rebuild/dism_decode_cuda.so
python dism_v3/tools/bench_inference_decode.py --fixed-context --prompts 8192 \
  --modes graph --batches 1 --intervals 128 512 1024 --rebuild-workers 8 \
  --output dism_v3/build/decode_rebuild/fresh_b1.json
python dism_v3/tools/bench_inference_decode.py --fixed-context --prompts 8192 \
  --modes graph --batches 8 --patterns random --intervals 128 512 1024 \
  --rebuild-workers 8 --output dism_v3/build/decode_rebuild/fresh_b8.json
python dism_v3/tools/check_decode_rebuild.py \
  --reference dism_v3/build/decode_rebuild/baseline.pt --workers 1 4 8
python -m pytest -q dism_v3/tests/test_decode_rebuild.py
```

## Opt-in summary program prototype (2026-10-23)

`csrc/probe/summary_program_bind.cpp` compiles an explicit host program;
`csrc/probe/summary_program.cu` executes it with persistent CTAs. This is an
isolated feasibility probe, enabled only with `DISM_BUILD_PROBES=1` and the
`summary-probe` build component. Neither `NativeDecodeCache` nor model dispatch
selects it. The existing production summary kernels are unchanged.

### Program and dependencies

For a suffix-link-tree node u, the computation is a subtree sum T[u] of raw
SK outer V matrices, followed by the affine recurrence
`P[u] = weight[u] * T[u] + rho[u] * P[parent[u]]`. Only selected subtree and
sampled-prefix matrices are outputs. The scalar coefficients come from the
existing CPU planner, including tau=0 handling.

The host builds four kinds of coarse instructions:

1. **Local Euler prefix:** each task processes at most S Euler positions for
   one matrix tile. Raw KV stays on the device.
2. **Block carries:** one task per head/tile scans the ceil(nodes/S) block totals.
3. **Local chain:** heavy paths are divided into segments of at most S nodes.
   Each segment reads subtree sums from Euler intervals and computes its local
   affine recurrence. Materialized subtree outputs are stored here.
4. **Boundary completion:** each segment reads a host-compiled list of ancestor
   segment boundary values and completes its sampled-prefix outputs.

The default S is ceil(sqrt(2*token_count)), using the SAM node-count bound;
S can be overridden in the probe. This is a two-level decomposition, not a
recursive log-depth parallel scan. A long chain has O(S + nodes/S) sequential
work per segment/stage. Light edges from heavy-path decomposition add branch
boundaries. An attachment can be **inside** a parent segment, so the compiler
saves the local affine value at the exact parent node, not necessarily at the
segment tail. This distinction is numerically essential even when positive tau
makes an incorrect carry appear small.

When a head has no sampled prefixes, the compiler emits only batched interval
queries for its materialized subtree outputs; it does not run an unused path
recurrence. Heads with no summaries emit no GPU work. Without this pruning,
8K/vocabulary-4 generated about 1.59 million tiny tasks and was much slower than
the baseline. The final S=256 program has 10,384 tasks for the same 12-head case.

All CTAs consume ready tasks from a bounded global queue. Completion counters
are per head/tile/stage; the last producer publishes the next stage. Vector
writes are fenced before release/acquire integer atomics publish completion.
CTAs never reserve an unready dependency and wait for a producer that has not
been scheduled. Shared storage carries the CTA command only; there are no FP32
atomics. The interpreter uses compensated FP32 sums for prefixes and affine
accumulation. It changes the reduction order and is not bitwise equivalent to
the production kernels.

An execution consists of **two kernels plus one device-to-device copy**: reset
queue state, copy the immutable initial task list, then run the interpreter.
Output and scratch are reused. Calls are stream-bound and serialized; retained
outputs must be cloned. This prototype has not been tested concurrently with
model decode.

### Available parallelism and measurements

RTX 5090 (170 SMs), BF16 R32/DV64, tau=.5, synthetic resident inputs. A tile is
128 matrix elements, giving 16 tiles per head. At 8K/12 heads/repeated labels:

| S | Chain segments | Independent local-chain tasks | Maximum ancestor segments | Total tasks | Estimated structural span |
|---:|---:|---:|---:|---:|---:|
| 64 | 1,548 | 24,768 | 128 | 74,496 | 449 |
| 128 | 780 | 12,480 | 64 | 37,632 | 513 |
| 256 | 396 | 6,336 | 32 | 19,200 | 833 |

The span column counts node-loop iterations along the staged critical path;
it is not GPU cycles and assigns equal cost to different instructions. Even
S=256 exposes thousands of independent tasks. Smaller S reduces the structural
span but increases queue traffic and work duplication; the measured optimum
is not the minimum-span program.

With 512 persistent CTAs, S=256 and three repetitions, median timings in ms:

| 8K / 12 heads | Production eager | Program eager | Production graph | Program graph |
|---|---:|---:|---:|---:|
| Repeated labels | 29.012 | 5.838 | 23.330 | 5.795 |
| Period-7 labels | 27.978 | 7.178 | 28.432 | 6.680 |
| Uniform vocabulary-4 | 14.432 | 3.205 | 14.066 | 3.190 |

Uniform vocabulary-512 generated no summary matrices; both paths skip GPU
summary work. Repeated/periodic production cases use 13,824 kernel launches,
while vocabulary-4 uses 2,304. The baseline calls the unchanged production
`launch_parallel_rebuild` with identical topology/coefficients, but reuses its
output and scratch allocations. It excludes production cache export and final
concatenation. Graph capture is outside timing. Eager timing is synchronized
host wall time; graph timing uses CUDA events. Clocks/affinity were not fixed.

For repeated labels, the final S=256 program takes 16.454 / 9.439 / 5.838 /
6.833 ms at 128 / 256 / 512 / 1024 CTAs. More workers are not monotonically
better. At a single head it takes 1.051 / .595 / .643 / .871 ms; the unchanged
baseline takes 2.099 ms. The default sqrt-sized S=128 program at 12 heads and
512 CTAs takes 8.299 ms, whereas S=64 takes 21.169 ms. These sweeps indicate
sufficient parallel work, with task granularity and scheduling cost now material.
They do not separately attribute queue contention versus cache/memory effects.

At 16K/12 heads, default S=182 and 512 CTAs, eager times are 12.648 / 13.880 /
6.284 ms for repeat / periodic / vocabulary-4, versus production 37.748 /
44.413 / 23.585 ms. This supports the feasibility of sqrt blocking beyond 8K;
it is not a universal block-size or worker-count recommendation.

### Costs and validation limits

The current prototype deliberately expands scratch over all active heads and
matrix tiles. At 8K/12 heads/S=256, scratch is about **775 MiB** for repeat and
periodic labels, and **1,249 MiB** for vocabulary-4. Most is the full Euler prefix
array. This is much larger than production's reused 32-coordinate slab. Program
metadata is about 4.3 MiB / 4.2 MiB / 6.2 MiB respectively. Scratch budgeting,
reuse of intermediate storage and bounded batches are needed before integration.

The probe host compiler currently builds full SAMs serially and also exports
production baseline metadata. `compile_ms` includes those diagnostics, and
`prepare` includes baseline buffer preparation/uploads. They are not measurements
of an incremental production compiler. The timing tables isolate prepared GPU
execution; they do **not** establish total rebuild or model-throughput speedup.

Validation for the final boundary-correct, compensated implementation:

- 24 BF16-input probe tests pass: production-baseline bitwise comparison, independent CPU
  FP64 tree evaluation, output comparisons after replacing a production decode
  snapshot's summaries, queue reuse, workers 1/8/32/64/256 and graph replay.
  Decode checks retain atol=2e-5/rtol=2e-3; small summary checks use
  atol=2e-5/rtol=2e-4. No acceptance threshold was loosened after failures.
- 8K FP64 checks include repeat, period-7 and vocabulary-4 at tau=0 and .5.
  Program summary relative L2 errors range from 6.8e-8 to 2.11e-7. At tau=0,
  large unnormalized prefix matrices have max absolute errors .00966 (repeat)
  and .00706 (periodic), so these are not claims of uniform tiny absolute error.
  The long-summary regression uses relative L2 <2e-6; it complements rather
  than replaces the elementwise decode checks. Full-model generation and
  arbitrary long-context numerical equivalence remain unverified.
- CUDA resource inspection reports 40 registers/thread, zero stack/local memory,
  and no CALL/LDL/STL instructions in the probe binary. Device sources are
  Torch-free.
- Compute Sanitizer memcheck, synccheck and racecheck each pass 21 selected tests
  with zero reported errors/hazards (racecheck also reports zero warnings).
  The three 8K aggregate-reference cases are excluded from sanitizer runs;
  the selected cases include 1,025-token branch attachments, tau=0, repeated
  queue use and graph replay. Host ASan/UBSan and FP32-input numerical coverage
  were not run for this isolated prototype.

Artifacts live under ignored `dism_v3/build/summary_program/`. Final performance
and FP64 results are `verified_main.json`, `verified_sqrt.json`,
`verified_smallblock.json` and `verified_reference.json`; earlier `initial`,
`coarse`, `block_*`, `long` and `reference` files are developmental measurements,
including an invalid pre-fix branch-boundary result, and are not final evidence.

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
export TORCH_EXTENSIONS_DIR="$PWD/dism_v3/build/torch_extensions"
export DISM_BUILD_PROBES=1 MAX_JOBS=2 OMP_NUM_THREADS=1
python -m flash_dism.inference.build --component summary-probe \
  --output dism_v3/build/summary_program/probe.so
python -m pytest -q dism_v3/tests/test_summary_program_probe.py
python dism_v3/tools/bench_summary_program.py --heads 1 12 --lengths 8192 \
  --block 256 --tile 128 --workers 128 256 512 1024 \
  --output dism_v3/build/summary_program/fresh.json
```

### Scratch lifetime / in-place reduction analysis

The prototype above is committed as `8a0a473`. CPU read-set analysis of the same
8K/H12/R32/DV64/S256 synthetic labels shows that the dense Euler prefix accounts
for more than 99% of scratch. The current interpreter already updates block
totals into exclusive carries in place and updates sampled outputs from local
to complete affine values in place. Subtree values and local affine accumulators
live in registers; there is no separate dense subtree or path matrix array to
remove by aliasing.

| Pattern | Dense prefix MiB | Prefix rows actually read | Prefix MiB with read-set compaction |
|---|---:|---:|---:|
| Repeat | 768.094 | 98,316 / 98,316 | 768.094 |
| Period-7 | 768.094 | 98,316 / 98,316 | 768.094 |
| Vocabulary-4 | 1,243.531 | 972 / 159,172 | 7.594 |

Only interval endpoints are read by subtree queries. Heads without sampled
prefixes need endpoints for materialized outputs only, hence the large savings
in vocabulary-4. A further backward liveness pass through sampled-prefix
ancestors reduces the repeat/periodic prefix estimates to 756.188/684.750 MiB;
that alone does not solve long-chain memory use. Endpoint compaction can preserve
all current summation order by retaining the same scan and writing only selected
prefix positions. These are static read-set counts, not implemented GPU results.

The most direct general solution is a bounded pool of `head x matrix-tile`
workspace slots. There are currently 192 such jobs at H12/tile128. A dense slot
holds at most 8,193 x128 FP32 values (4.0005 MiB) for repeat/periodic, or
13,333 x128 (6.5103 MiB) for the vocabulary-4 case. Eight / sixteen / thirty-two
slots therefore require about 32 /64 /128 MiB of prefix for long chains.
Sixteen live jobs expose 528 initial Euler tasks at S256 for the long-chain
case, compared with the measured 512-worker configuration. This suggests a
reasonable first budget, but is not a prediction of unchanged throughput;
the carry phase and the admission schedule can still limit utilization.

A slot's prefix and block carries can be overwritten by the next job after **all
local-chain tasks of its current job finish**. Final boundary-completion tasks
read only saved boundary values, provisional output and immutable metadata, so
they need not hold the prefix slot. Keep those small boundary arrays separately
at first. The last local-chain completion can release the slot and enqueue the
next job entirely on device, preserving one persistent interpreter launch.
Commands must retain their original job identity; slot reuse must not change
metadata observed by an older boundary-completion task. Completion publication
must cover every consumer's final read before a new producer overwrites storage.

Estimated total scratch for sixteen slots, keeping all boundary values and the
existing full task queue, is approximately 68 MiB for long-chain cases. For
vocabulary-4 it is about 105 MiB with dense slots, or 1.3 MiB when combined with
endpoint compaction. These estimates exclude output matrices, program metadata,
allocator rounding and the diagnostic baseline's allocations. No throughput or
sanitizer claim is made for this unimplemented scheduling change.

Blindly overwriting dense prefixes with subtree values is unsafe: one prefix
position can still be read by several other subtree tasks. A pure chain can use
saved block boundaries and an ordered in-place traversal, but the current
interpreter does not allocate a dense subtree array anyway; that transformation
alone would retain the dominant prefix allocation. Reusing old snapshot output
buffers also cannot remove this prefix scratch. When decode runs concurrently,
its active summaries must remain immutable until retirement; incremental SAM
updates can additionally change suffix links, cloned nodes and materialization
IDs, so direct additive updates require separate correctness work.

Read-set counts and the standalone CPU analysis are saved under ignored
`dism_v3/build/summary_scratch/`. No kernel or production default was changed by
this analysis. Recommended implementation order: bounded workspace slots, then
endpoint compaction, followed by actual latency/memory and concurrent-decode
measurements.

### Bounded workspace slots and prefix compaction implemented

The opt-in `SummaryProgram` now accepts trailing `slots=0, compact=False`
constructor arguments. Zero slots retains the fully expanded prototype as a
comparison path. Positive slots bound the number of admitted head/tile jobs;
`compact=True` additionally stores only prefix positions read by local-chain
instructions. Compaction requires positive slots. Requested slots are capped at
the number of active jobs, including zero when there are no summaries. Production
decode dispatch and all existing defaults remain unchanged.

The device scheduler assigns an immutable job ID to each head/tile, records its
workspace slot, and admits another job when the last local-chain task releases
that slot. Pending final-completion tasks keep their original job ID and use only
global boundary/output storage. They never read the reassigned prefix/carry slot.
Every consumer completes its reads and writes before its completion counter is
published; the final acquire/release completion publishes the next slot owner
before its tasks become visible. No extra host synchronization or kernel launch
is added. Execution remains two kernels plus the initial task-list D2D copy.

Each slot has a prefix pitch equal to the largest active head's required prefix
row count and a carry pitch equal to its largest Euler-block count. Thus mixed
heads share a conservative fixed-size pool; the very small vocabulary-4 figures
below apply to that workload, not to a mixture containing a dense long-chain
head. Boundary matrices remain globally allocated, as proposed above. Immutable
program metadata and final output matrices are not counted as scratch.

`stats()` now reports `prefix_bytes`, `carry_bytes`, `boundary_bytes`,
`workspace_slots`, `compact_prefix`, `prefix_rows_per_slot` and `initial_tasks`.
After `prepare()`, `allocated_scratch_bytes` sums the actual scratch tensors and
is checked against the estimate. `output_bytes` and
`diagnostic_baseline_scratch_bytes` are reported separately. These are live tensor
sizes, not the CUDA allocator's reserved high-water mark or total process VRAM.

Same-device comparison: RTX 5090, 8K, 12 heads, BF16 R32/DV64, tau=.5, S256,
tile128, 512 persistent CTAs, median of three CUDA-event graph replays. Capture
and host compilation are excluded; jobs ran serially, clocks/affinity were not
fixed. All times below measure prepared summary execution, not model throughput.

| Pattern | Expanded scratch MiB | 16 compact slots MiB | Expanded ms | 16 compact slots ms |
|---|---:|---:|---:|---:|
| Repeat | 774.557 | 67.635 | 5.906 | 5.902 |
| Period-7 | 774.290 | 67.368 | 6.645 | 6.098 |
| Vocabulary-4 | 1,248.615 | 1.256 | 3.190 | 2.895 |

Long-chain scratch falls about 11.5x without a material loss in this measurement;
vocabulary-4 falls about 994x. The layout, active working set and scheduling all
change, so modest timing improvements cannot be attributed solely to one cause.
Uniform vocabulary-512 still produces no summary work.

The slot budget matters. With compact storage and 512 CTAs:

| Slots | Repeat MiB / ms | Period-7 MiB / ms | Vocabulary-4 MiB / ms |
|---:|---:|---:|---:|
| 8 | 35.502 / 12.906 | 35.235 / 13.247 | .728 / 6.018 |
| 16 | 67.635 / 5.902 | 67.368 / 6.098 | 1.256 / 2.895 |
| 32 | 131.900 / 5.823 | 131.634 / 5.936 | 2.310 / 2.553 |
| 192 | 774.557 / 6.121 | 774.290 / 7.071 | 12.857 / 2.262 |

Eight slots limit ready work too much for this worker count. With 256 CTAs,
its repeat/periodic/vocabulary-4 times improve to 9.437/10.291/4.987 ms, but
remain slower than sixteen slots at 512 CTAs. Sixteen slots are a useful first
tradeoff; more slots can be useful when the compact rows are inexpensive.
Noncompact sixteen-slot measurements allocate 67.635/67.368/104.779 MiB,
confirming that prefix compaction supplies most of the vocabulary-4 savings.

At 16K with default S182 and 512 CTAs:

| Pattern | Expanded MiB / ms | 16 compact slots MiB / ms |
|---|---:|---:|
| Repeat | 1,554.069 / 13.544 | 138.163 / 12.972 |
| Period-7 | 1,553.600 / 14.149 | 137.694 / 12.460 |
| Vocabulary-4 | 2,505.148 / 6.292 | 2.328 / 5.799 |

Validation preserves the arithmetic sequence and existing tolerances:

- All 37 probe tests pass. New tests compare bounded/compact output **bitwise**
  against expanded output with BF16 and FP32 inputs, 1/3/16 slots, mixed active
  and empty heads, 1/64 workers, repeated execution, and graph replay with changing
  inputs. Empty programs, invalid options and allocation accounting are checked.
- The three 8K tau=0 regressions also compare a three-slot compact run bitwise
  with the expanded run before the independent FP64 summary check. The separate
  8K repeat/periodic/vocabulary-4 sweep at tau=0/.5 has the same FP64 errors as
  the prior prototype (relative L2 6.8e-8..2.11e-7). No tolerance was loosened.
- BF16 and FP32, expanded and bounded kernels all compile to 40 registers/thread
  with zero stack/local memory. SASS inspection finds no CALL/LDL/STL instructions.
  CUDA device files remain Torch-free; no FP32 atomic operation was added.
- Compute Sanitizer memcheck, synccheck and racecheck each pass all 13 new
  workspace/compaction tests with zero reported errors or hazards; racecheck
  also reports zero warnings. These selected tests exercise both input dtypes,
  a single reused slot, mixed heads, input-changing graph replay and empty jobs.
  The 8K/16K performance sweeps are not sanitizer runs.

These results do not establish concurrent model-decode latency, end-to-end rebuild
speedup or host compiler efficiency. Host ASan/UBSan coverage remains outstanding.
The scheduler is still a prototype, explicitly selected through its constructor
or benchmark flags, rather than a new production default.

Generated artifacts are ignored under `dism_v3/build/summary_slots/`:
`slots_*_compact_*.json` for the 8K sweep, `long_0/16/32.json` for 16K,
`reference.json`, `tests_final.log`, sanitizer logs and CUDA resource/SASS dumps.
The saved old extension and source snapshot are under its `baseline/` directory.

```bash
# Same environment/build instructions as the original probe above.
python dism_v3/tools/bench_summary_program.py --heads 12 --lengths 8192 \
  --block 256 --tile 128 --slots 16 --compact --workers 128 256 512 \
  --output dism_v3/build/summary_slots/fresh.json
python -m pytest -q dism_v3/tests/test_summary_program_probe.py
```

### Concurrent frozen-snapshot rebuild prototype

`flash_dism.inference.concurrent_probe.ConcurrentGpuPlannerCache` is an explicit,
opt-in eager wrapper around the existing GPU planner and bounded summary program.
It requires `DISM_BUILD_PROBES=1`; no model/cache dispatch or production defaults
change. The production vector/planner kernel is linked into the probe unchanged.
The only new device kernel replays query/reset labels into a new frozen matcher.

The owner thread/stream continues decoding against snapshot S. `start_rebuild()`
records a cutoff C on that stream and submits one background job. That job:

1. Waits for the cutoff event on a separate, priority-0 stream; copies only the
   newly appended label-history range to a persistent pinned host buffer. Only
   the worker waits for D2H completion. KV/history below C are immutable.
2. Compiles the full prefix into a private `SummaryProgram`, replays queries to C
   on its CPU SAM, packs a private frozen topology/state/counts/tau snapshot,
   and uploads metadata. This first prototype still builds heads serially and
   does not reuse the production incremental CPU SAM or its worker pool.
3. Allocates all private summary/scheduler/task buffers **before** launching the
   persistent interpreter. `prepare(prototype, diagnostic=False)` returns its
   private output tensor and skips diagnostic baseline allocations. Sixteen
   compact scratch slots are used by default; slot reuse is internal to this job.
4. Launches summary execution, records a CUDA-ready event, and releases the large
   CPU compiler with the GIL released. Constructor, metadata packing and prepare
   also release the GIL. A completed Python Future alone is not GPU readiness.

`step()` polls readiness without waiting. Once both the Future and CUDA event
are ready, foreground waits on that event, runs query-only catchup for `[C,P)`,
then switches to the new snapshot. Catchup reads the old device position and
sticky error flag, validates bounds/resets, advances the new frozen SAM, and
invalidates its task cache. The live rolling band is **shared and never rewritten
by the worker or catchup**. It continues to carry exact recent-history matches.
The new horizon is C+interval, not P+interval.

Old summary/topology/state/output/task storage stays alive through a foreground
retirement event recorded after catchup. Background allocations consumed by
foreground are registered with the CUDA caching allocator, including private C++
task buffers. Raw KV/history also have their background use recorded. No active
snapshot summary is overwritten; bounded prefix/carry scratch remains private to
background summary execution. This is the intended compatibility with the
previous in-place workspace-slot reuse scheme.

Only one rebuild can be pending. `start_rebuild()` returns false for a pending
job or an unchanged cutoff. The caller chooses the proactive trigger; `step()`
forces completion or starts a synchronous fallback before it would cross the
old horizon. `cancel_rebuild()` invalidates publication but drains submitted
work rather than freeing its buffers early. Worker exceptions propagate and
poison further operations; `close()` drains worker/streams even after failure.
Use a context manager or explicitly close the cache. Prime is allowed once and
requires complete history. Inputs, tau, sample spacing and materialization
threshold must remain fixed; this wrapper uses the production default spacing
4R and threshold 5R only. Output is a reused FP32 buffer: clone to retain tokens.

The wrapper rejects capture and is bound to one foreground stream. Cross-version
CUDA graph replay, prefill reentry, reset/reprime of an existing cache, automatic
lead-time selection, a fixed double-buffer pool, and model integration remain
outside this prototype. The bootstrap `GpuPlannerCache` is retained for the
wrapper's lifetime, including its initial metadata and diagnostic/native scratch;
total VRAM is therefore larger than just two snapshots plus the bounded scratch.

#### Measurements and remaining latency

RTX 5090, BF16 R32/DV64, H12, tau=.5, repeat labels, 8192-token prime followed by
1536 eager tokens. Interval=1024; both modes request rebuilds at the same cutoffs
8704 and 9216. S256, tile128, 16 compact slots, 256 persistent CTAs. Blocking uses
**the same prototype compiler and summary kernel**, waiting immediately at each
cutoff; it is not a comparison with the optimized production CPU rebuild path.
Times exclude prime/JIT/warmup/final draining. Each token waits only for its
foreground completion event, never for a device-wide synchronize. Three serial
runs per mode; table entries are medians of run-level metrics, without locked
clocks/CPU affinity.

| Foreground workload | Rebuild mode | 1536 tokens ms | Token p50 ms | Token p99 ms | Maximum token ms |
|---|---|---:|---:|---:|---:|
| Planner/operator only | Blocking | 166.85 | .060 | .074 | 38.37 |
| Planner/operator only | Concurrent | 121.63 | .075 | .118 | 1.03 |
| Operator + 4 BF16 2048-square GEMMs/token | Blocking | 756.73 | .443 | .457 | 39.24 |
| Operator + 4 BF16 2048-square GEMMs/token | Concurrent | 708.88 | .445 | .592 | 9.48 |

Concurrent wall time falls about 27% / 6.3% respectively. Both jobs publish and
there are zero deadline waits in all six concurrent runs above. GEMMs are a
synthetic compute proxy, **not measured model throughput**. Rebuild events are
rarer than 1% of tokens here, so p99 alone hides the blocking spikes. Concurrent
work raises some ordinary-token latencies while reducing the larger pauses.

At 256 CTAs, CPU compile is approximately 19–23 ms, preparation/metadata/task
allocation 5–7 ms, summary execution 8–10 ms, and compiler release another 5–6 ms
(overlapped with GPU summary). Delta D2H plus its worker-side wait is about
.07–.31 ms in these warmed runs; the first full-history copy is excluded by the
benchmark warmup. Cutoff-to-ready observation is roughly 34–41 ms. Catchup for
about 60–420 tokens costs .06–.22 ms in the unprofiled runs. These ranges describe
this synthetic pattern; compiler work and catchup depend on SAM/query structure.

A separate two-run async worker sweep with the GEMM proxy gives:

| Background CTAs | Summary ms | Tokens completed across each summary interval | Ordinary token during overlap ms |
|---:|---:|---:|---:|
| 64 | 32.8–35.5 | 51–55 | about .65 |
| 128 | 16.4–18.0 | 26–28 | about .67 |
| 256 | 9.0–9.6 | 14–16 | about .68 |

All six worker-sweep runs avoid deadline waits and have maximum token latency
1.28–1.68 ms. However, the independent three-run comparison above still exhibits
approximately 9 ms spikes with the GEMM proxy; that result is retained rather
than replaced by the smoother sweep. Moving allocations before the persistent
kernel removes an avoidable synchronization opportunity but does not establish
a hard tail-latency bound. Stream priority is not a reservation of GPU resources.
The earlier 64-CTA operator-only run misses a deadline and rejects the second
scheduled request while the first is pending; its total time is not comparable
to runs completing both jobs. The benchmark reports rejected requests, published
jobs and pending work separately.

Nsight Systems 2025.5.2 confirms actual kernel overlap: the two measured summary
kernels last 8.79/9.64 ms, with other-stream kernels executing during 8.31/9.18 ms
of those intervals. There are 13/14 foreground GPU-planner calls and 52/57 GEMM
calls overlapping them. The traced run is evidence of concurrency, not a source
of the unprofiled throughput table. The older `nsys` on PATH (2023.4) did not
capture CUDA activities for this environment; use the CUDA 13.1 bundled version.

At cutoffs 8704/9216, actual bounded interpreter scratch is 71.86/76.09 MiB and
private output summaries 12.66/13.41 MiB. Those numbers exclude topology, task
buffers, raw KV/history, the retained bootstrap, pinned host buffers and CUDA
allocator reservation. Output/topology must coexist across versions; scratch
slot reuse does not remove that requirement.

#### Validation and reproduction

The concurrent tests compare BF16/FP32 decode against the independent FP64 dense
oracle at the existing `atol=2e-5, rtol=2e-3`. Coverage includes repeat, periodic,
branching and empty-summary patterns; mixed tau=0/.35/1.5; query resets; multiple
versions; a forced 240-token lag; unchanged old snapshot/live band during
handoff; automatic deadline fallback from an unprimed cache; cancellation;
closing pending work; injected worker failure; stream/capacity guards.

Catchup compiles to 30 registers/thread, zero shared/stack/local memory; SASS has
no CALL/LDL/STL instructions. The unchanged interpreter remains at 40 registers
and zero stack/local memory, with no CALL/LDL/STL. Device sources remain
Torch-free and add no FP32 atomics. All **51 tests** pass (14 concurrent plus
37 existing summary regressions). Compute Sanitizer memcheck and synccheck each
pass all 14 concurrent tests with zero errors; racecheck passes the same 14 with
zero errors, warnings or hazards. The ordinary pytest dependency/deprecation
warnings remain visible in the logs. Host ASan/UBSan and a full-model concurrent
run are not covered; racecheck alone does not prove cross-stream global-memory
correctness, which also depends on the immutable-prefix and event-lifetime rules.

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
export TORCH_EXTENSIONS_DIR="$PWD/dism_v3/build/torch_extensions"
export DISM_BUILD_PROBES=1 MAX_JOBS=2 OMP_NUM_THREADS=1
python -m flash_dism.inference.build --component summary-probe \
  --output dism_v3/build/concurrent_rebuild/summary_probe.so
python -m pytest -q dism_v3/tests/test_concurrent_rebuild_probe.py \
  dism_v3/tests/test_summary_program_probe.py
python dism_v3/tools/bench_concurrent_rebuild.py --workers 256 \
  --proxy-gemms 0 4 --repeats 3 \
  --output dism_v3/build/concurrent_rebuild/reproduce.json
```

Use the `blkw` conda environment. Generated results stay under ignored
`dism_v3/build/concurrent_rebuild/`: `final_latency.json`, `worker_sweep.json`,
`overlap_final.nsys-rep` and its SQLite export/analysis, resource/SASS dumps,
and test/sanitizer logs. The older `latency.json` used different requested
rebuild cadences and is exploratory only; `fair_latency.json` precedes the final
allocation-order change. Neither supplies the comparison table above.

### Optional SAM prefill in downstream evaluation

`flash_dism.inference.model_prefill.use_sam_prefill(model, engine)` explicitly
selects SAM for the DISM core of an existing model forward. The scope restores
previous dispatch even when a final-norm hook exits by exception. It requires
`eval()`, disabled gradients, eager aligned CUDA inputs, fixed true direction,
all-hard flags and equal query/key head counts. Cached execution and capture are
rejected. The caller owns the reusable `HardDismPrefill` engine; do not use one
model concurrently while changing its backend context.

This integration retains the existing model's convolution/projections,
`prepare_embedding` + `emb_fwd_true` vocabulary routing, readout activation,
value residual, tau scaling, GDN branch, gating, postnorm and FFN. Only the
quadratic hard core is replaced; SAM FP32 results are cast back to the existing
BF16 core-output contract. It does not use the separate experimental
`HardInferenceEngine`, whose complete-model implementation has a different scope.
Equal-length packed rows become independent batched SAM streams. Ragged packed
documents are grouped by exact length, skipping empty documents, so neither
cross-document matches nor synthetic padding enter the SAM state.

```python
from flash_dism.inference import HardDismPrefill
from flash_dism.inference.model_prefill import use_sam_prefill

with HardDismPrefill(workers=8, mma_precision="tf32x3") as prefill:
    with torch.inference_mode(), use_sam_prefill(model.eval(), prefill):
        result = model(tokens, cu_seqlens=boundaries)
```

The local `/home/cicuvc/downstream-eval/adapter_layout.py` and
`adapter_layout_bounded.py` support `--hard-backend quadratic|sam`,
`--prefill-precision tf32x3|bf16`, and `--prefill-workers`. **Quadratic remains the
default, as requested after the paired measurements below.** SAM defaults to
TF32x3 and 16 CPU workers when explicitly selected. The legacy `adapter.py` is unchanged.
`run_layout_comparison.py` forwards these arguments and records backend,
precision, worker count and inference source hashes in provenance. New results
default to `results/layout_20261023_quadratic` or
`results/layout_20261023_sam_<precision>`; the original `layout_20261023` result
root is protected. Model checkpoints, task definitions and saved results are
unchanged. These suite scripts are local files outside this Git repository.

PG19/GovReport still use independent rolling likelihood windows, and generation
still recomputes each bounded prefix. Neither path initializes a SAM decode
cache or uses concurrent rebuild. CE, truncation, fixed physical batch size,
256-token padding, dummy rows and greedy output selection are unchanged. The
backend and precision must be held fixed when comparing model layouts.

#### Paired checkpoint checks

RTX 5090, physical B4, six DISM layers, local PG19 text from four books, full
model forward through final norm (CE excluded from timing). Three warmed timed
calls per backend, median shown; the retained JSON includes first-use/outlier
samples. These are diagnostic samples, not full downstream task scores.

| Layout | Context | Quadratic ms | SAM TF32x3 ms | SAM BF16 ms |
|---|---:|---:|---:|---:|
| gggddddddggg | 256 | 23.03 | 31.39 | — |
| gggddddddggg | 2048 | 22.97 | 65.40 | 65.92 |
| gggddddddggg | 8192 | 80.98 | 220.21 | 216.68 |
| ggggggdddddd | 2048 | 22.70 | 66.74 | — |
| ggggggdddddd | 8192 | 81.46 | 239.38 | — |

The BF16 SAM columns come from a separate paired run, whose quadratic medians
are 23.86/80.31 ms. BF16 gives little end-to-end improvement here; this experiment
does not isolate planner, transfer and GPU execution costs. The current SAM
integration is slower on these B4 workloads, despite its different asymptotic
algorithm. Do not infer a model speedup from standalone prepared-core timings.

Mathematical hard semantics are shared, but floating-point evaluation order and
rounding differ. The first DISM layer's vocabulary IDs are identical between
backends on every paired case. Small core-output differences then affect later
vocabulary argmax choices. At 8K with TF32x3, final DISM-layer key/query mismatch
rates are 1.83%/1.75% for gggddddddggg and 2.05%/2.64% for ggggggdddddd.
Final-norm hidden relative L2 differences are 3.25%/3.50%; sampled greedy-logit
agreement is 95.31%/93.75% (128 sampled positions per case).

Mean token NLL on those 8K diagnostic inputs changes from 3.844632 to 3.844846
for gggddddddggg, and 3.807029 to 3.807461 for ggggggdddddd. Mean absolute
per-token NLL changes are about .050/.048, so cancellation in mean NLL must not
be mistaken for identical predictions. Four short eight-token greedy continuations
match between quadratic and TF32x3 SAM for each layout; with BF16 SAM, two of
four change on gggddddddggg. Short-request summed likelihood also changes. Thus
SAM is an explicit alternative evaluation backend, **not a bitwise replacement**
for historical quadratic results. No tolerance was relaxed to hide these changes.

All 151 selected regressions pass (new integration tests, existing module tests
and SAM inference tests). Eight new integration tests cover direct batches, equal/ragged/empty packed
documents, bitwise document isolation within SAM, existing quadratic hard-output
criteria, dispatch restoration on exception, unchanged module state and
training/gradient guards. Real checkpoint adapter smoke checks pass for both
layouts with TF32x3 and for gggddddddggg with BF16. Core arithmetic kernels are
unchanged; this change adds Python dispatch rather than new CUDA kernels. Full
task reruns and new sanitizer runs are not part of this integration validation.

Reproduce the paired checks with `tools/validate_eval_prefill.py` (requires the
local suite/harness and existing PG19 parquet, never downloads):

```bash
PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat:/home/cicuvc/lm-evaluation-harness" \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 \
python dism_v3/tools/validate_eval_prefill.py --checkpoint /path/to/model_022888.pt \
  --precision tf32x3 --lengths 256 2048 8192 \
  --output dism_v3/build/eval_sam/paired.json
# Explicitly select SAM for a future layout suite run:
python /home/cicuvc/downstream-eval/run_layout_comparison.py \
  --hard-backend sam --prefill-precision tf32x3 --prefill-workers 8
```

Ignored artifacts are under `dism_v3/build/eval_sam/`: paired JSON/logs named
by layout and precision, regression logs and backups of the edited local suite
scripts. The validator reports numerical drift explicitly; it does not label
model-level agreement as bitwise or infer task accuracy from four books/prompts.

#### Real-model SAM cost attribution

`tools/profile_eval_prefill.py` profiles the same local PG19 inputs and
`gggddddddggg` checkpoint, B4/H12, six DISM layers, TF32x3, eight CPU workers.
Three repetitions, median of the six-layer sum. It first fences pending model
work before copying labels, then separately fences upload and vector execution.
These extra fences alter overlap; ordinary unfenced timings are reported in the
same JSON. They are close here: at 8K ordinary SAM is 222.71 ms, profiled SAM
221.75 ms, and ordinary quadratic 80.11 ms.

| Six-layer sum | 2K ms | 8K ms |
|---|---:|---:|
| CPU native planning, including event/chunk construction | 31.76 | 138.44 |
| Pinned metadata allocation / views | .66 | 1.01 |
| CPU metadata packing and native-program release | 4.67 | 12.15 |
| Label/tau D2H path, including stack/cast/host wrappers | .85 | 1.95 |
| Metadata H2D and upload setup, wall time | 1.41 | 4.16 |
| GPU vector execution, CUDA events | 3.36 | 12.32 |
| Wait for preceding model GPU work at prefill entry | .37 | 2.30 |
| Complete profiled model forward | 67.89 | 221.75 |

The CPU planner wrapper totals 37.12/151.65 ms; the three CPU detail rows are its
components, not additional time. At 8K it consumes about 68% of the full forward.
Measured label/upload paths total only 6.11 ms (about 2.8%); adding the explicit
preceding-GPU wait gives 8.41 ms. GPU execution wall time is 12.52 ms versus the
12.32 ms CUDA-event measurement. Stage medians need not add exactly; all rows
exclude CE and preserve the rest of the model pipeline.

D2H moves approximately 18 MiB of labels/tau across the six layers; H2D uploads
about 128.5 MiB of metadata. In a representative 8K repetition, the six native
plan times are 10.8, 12.8, 17.0, 28.5, 35.3 and 34.1 ms. The corresponding event
counts are approximately .062, .416, .729, 1.530, 1.943 and 2.095 million.
The last three DISM layers account for most planning work. This timer includes
SAM/query matching, event generation and chunk-boundary discovery in the CPU
pool; it does not isolate those native subphases. Thus the dominant current
bottleneck is CPU planning rather than PCIe transfer. The synchronous dependency
chain places that planning on the model's critical path: a later layer's labels
are unavailable before the current layer's output. It cannot be hidden in the
same way as an independent decode snapshot rebuild without further design work.

Numerical limitation exposed by this diagnostic: the initial assumption that
adding fences would preserve final hidden tensors bitwise was **false**. At 2K,
the first check differed in 381,752/6,291,456 elements, maximum absolute 1.890625.
The diagnostic now records drift explicitly instead of treating fences as a
numerical equivalence test. Normal **uninstrumented SAM repeats also differ**:
relative hidden L2 .0044–.0061 at 2K and .0148–.0149 at 8K in this run; quadratic
repeats are bitwise identical. Fenced versus unfenced SAM hidden L2 differences
are .0064–.0090 / .0139–.0170. SAM GPU output uses FP32 atomic accumulation
(`prefill_triton.py` and the parallel path), whose order is a plausible source;
this attribution has not yet been isolated experimentally. Small core variation
can additionally change later hard vocabulary routes. Earlier cross-backend
model differences therefore also include SAM repeat variation, and must not be
interpreted as a pure deterministic backend bias. Existing regression tolerances
were not changed, and this profile does not establish deterministic SAM eval.

The measured stage data and numerical drift are in ignored
`build/eval_sam/profile_gggddddddggg.json` and its log. Reproduce with the same
local environment as `validate_eval_prefill.py`, substituting
`tools/profile_eval_prefill.py --checkpoint ... --lengths 2048 8192 --workers 8
--output dism_v3/build/eval_sam/profile.json`. No model defaults, production
kernels, evaluation results or checkpoint state were changed for profiling.

#### Prefill versus training planner and CPU scaling (2026-10-23)

To separate workload differences from implementation differences, capture the
six layers' actual labels and tau once, then reuse those exact inputs in every
CPU control. Inputs are from the preceding B4/H12 `gggddddddggg` checkpoint,
first four local PG19 books, at 2K and 8K. Each row is one document and all heads
are hard: 48 `(row, head)` jobs per layer. The 8K inputs produce 6,775,044 events
across six layers. This is a full-hard comparison, not a replay of training's
sampled hard mask or packed-document distribution.

The isolated C++ executable uses copies of the current production headers,
the same persistent worker pool and thread-local builder, with two clock reads
per planning job. It submits the six layers serially, matching prefill's layer
dependency. Outputs use ordinary `new[]`, not pinned memory; there is no Python,
CUDA, transfer or GPU execution in these measurements. Two warmups and five
measured repetitions per configuration; configuration order is shuffled. Tables
report medians of six-layer sums. The host has 8 P cores and 16 E cores, no SMT.

| Planner control, 8K | Plan, 8 workers ms | Plan, 16 workers ms | Plan, 24 workers ms |
|---|---:|---:|---:|
| Current prefill, C16 classic, no gradients | 121.97 | 83.95 | 70.48 |
| Same builder, C16 classic, no inference routing | 122.11 | 85.47 | 69.87 |
| Training builder, C16 classic, gradients enabled | 123.94 | 81.95 | 71.45 |
| Training configuration, C64 compact, gradients enabled | 125.71 | 85.95 | 72.19 |
| Prefill, C16, diagnostic logweight table | 116.66 | 73.45 | 63.22 |

Inference does not perform substantially more planning work than training on
the same inputs and worker count. Inference-only short/long stream routing adds
about 2 ms of **packing**, not tens of milliseconds of planning: at eight workers,
packing is 15.36 ms versus 13.20 ms without routing. Their ordinary allocations
are 134,788,208 versus 123,916,376 bytes. Training C16 additionally materializes
derivatives and allocates 162,553,944 bytes. Stage differences across independent
runs include noise; they are not exact instruction-level attribution.

There is one missed optimization: training precomputes logweights, while
`gradients=false` repeatedly calls `logweight(length)` during event emission.
An isolated ablation precomputes the original scalar function for lengths 1..N
on each job, without derivatives, cross-call caching or approximate math. It
saves 5.31/10.50/7.26 ms of planning at 8/16/24 workers here. At one worker,
planning falls from 731.65 to 682.41 ms, a 6.7% reduction. This is useful but not
the majority of the CPU cost. Neither ablation nor worker defaults were applied
to production. All 38 sweep controls have identical canonical forward-program
hashes for each sequence length. An additional literal `cmp` verifies the entire
156,505,024-byte 8K canonical dump (offsets, rows, decay, weights, log denominators)
between baseline prefill, table prefill and training C16. It excludes training
derivatives and layout-specific packed arrays, and does not establish GPU
determinism or end-to-end numerical equivalence.

| Current prefill workers, 8K | Plan ms | Pack ms | Total CPU wall ms | Mean busy cores during plan |
|---|---:|---:|---:|---:|
| 1 | 731.65 | 60.36 | 795.32 | 1.00 |
| 4 | 195.97 | 20.67 | 219.40 | 3.76 |
| 8 | 121.97 | 15.36 | 139.78 | 6.49 |
| 12 | 104.68 | 15.60 | 122.98 | 8.71 |
| 16 | 83.95 | 15.20 | 101.91 | 10.30 |
| 24 | 70.48 | 16.07 | 88.93 | 12.55 |

Mean busy cores is process CPU time divided by planning wall time, including
the caller. Total includes ordinary allocation, packing and release; it is not
model forward time. At 2K, current prefill totals are 37.35/25.59/22.83 ms for
8/16/24 workers. Binding eight workers to P cores gives 120.91 ms planning at 8K,
close to the unbound 121.97 ms; placement alone does not explain the bottleneck.

Eight workers are a meaningful limitation: moving to 16/24 reduces 8K planning
by 31%/42%. However, 24 cores are not uniformly saturated. Each layer has only
48 indivisible builder jobs, and expensive heads create a long tail. Even with
one worker (removing inter-worker contention), the fifth DISM layer's job
medians range from .83 to 12.36 ms. With 24 workers that layer takes 19.34 ms,
its longest job is 15.57 ms, and summed worker elapsed time divided by 24 is
only 9.30 ms. The longest job thus consumes about 81% of the layer deadline.
Other late layers show the same effect. Per-job elapsed time includes OS
descheduling, but the single-worker control confirms substantial intrinsic
workload skew. The existing pool already dynamically claims jobs; it is not a
static equal-count assignment. Queue-lock time was not independently measured,
so these data do not claim zero scheduler overhead or identify cache/frontend
stalls. Packing also stops scaling above eight workers; this measurement alone
does not distinguish bandwidth, allocation/page faults and packing imbalance.

The training runtime defaults to 16 workers, C64 compact, and overlaps backward
planning with subsequent GPU forward work. Prefill used eight workers for this comparison,
C16 classic, and needs the current plan to produce the current layer output
before the next layer's labels exist. It therefore exposes planner time directly
on the forward critical path. Actual training may also plan fewer hard heads.
These differences are more significant than extra inference-only operations.
Changing prefill to C64 alone gives almost unchanged planning and worse packing
(20.88 ms at eight workers); the CPU-only compact control likewise does not
remove the planning bottleneck. GPU consumers were not changed or benchmarked
for these layout controls.

Practical next steps are to validate 16/24 workers end to end, integrate the
small exact logweight table optimization, then examine cost-aware job ordering
or parallel work within expensive heads. Ordering can reduce when a heavy job
starts but cannot shorten its own serial critical path. More workers alone
cannot eliminate that bound. No end-to-end speedup is asserted for these CPU
controls.

Reproduction tools are `tools/capture_prefill_planner_inputs.py`,
`tools/build_prefill_planner_compare.py`, `tools/compare_prefill_planner.cpp` and
`tools/run_prefill_planner_compare.py`. With the same local environment as the
model profile:

```bash
python dism_v3/tools/capture_prefill_planner_inputs.py \
  --checkpoint /path/to/model_022888.pt \
  --output dism_v3/build/prefill_planner_compare/inputs
python dism_v3/tools/build_prefill_planner_compare.py \
  --output dism_v3/build/prefill_planner_compare/bin
python dism_v3/tools/run_prefill_planner_compare.py \
  --root dism_v3/build/prefill_planner_compare \
  --output dism_v3/build/prefill_planner_compare/sweep.json
```

Ignored artifacts include input SHA256 manifests, copied headers, build source
hashes, all raw samples in `build/prefill_planner_compare/sweep.json`, and
`byte_comparison.json`. Setting `DISM_PLANNER_PROGRAM_DUMP` on the C++ executable
emits the first warmup's canonical bytes for direct comparison. Existing model
SAM nondeterminism described above remains unresolved; freezing inputs avoids
letting that variation contaminate these CPU comparisons.

#### Prefill worker default update (2026-10-24)

`HardDismPrefill` and `NativePrefillPlanner` now default to 16 workers, based on
the frozen-input CPU comparison above. The local downstream evaluation adapters
and runner also default `--prefill-workers` to 16. Explicit worker counts remain
supported; quadratic remains the evaluation default. Historical measurements
and their eight-worker reproduction commands are unchanged. This is a worker
count change only: no work stealing, P/E affinity, job splitting or production
logweight-table optimization is included. The legacy Python reference planner
and training/decode defaults are unchanged. CPU planning improved in the
comparison; full-model speedup at the new default has not yet been measured.
