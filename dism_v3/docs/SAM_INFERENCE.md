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
returns FP32 BNHD. Its persistent native worker pool defaults to eight threads
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
