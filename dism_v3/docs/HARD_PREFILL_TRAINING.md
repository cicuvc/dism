# Hard-prefill training: numerical study and implementation plan

Status (2026-10-14): CPU reference and opt-in Triton backward prototype. No
production dispatch/kernels, training configuration, checkpoints, or existing
acceptance tolerances were changed. The existing full-hard inference planner
provides the event topology; this study differentiates its operator, including
the original head-shared tau. Per-query tau changes are out of scope.

The subsequent [native parallel CPU planner](NATIVE_TRAINING_PLANNER.md) now
implements full derivative planning, batch/head parallelism and pinned packing;
its timings supersede the Python-based preparation estimates below.

## What has been validated

`tools/hard_prefill_grad_reference.py` mirrors the native SAM/centroid planner
in Python while retaining symbolic match lengths. Every tested plan has the
same event offsets/rows as the native planner; its coefficients and log
normalizers are checked numerically against native arrays. This duplicated
planner is a test helper, not a proposed production implementation.

For one independent document, each stream carries an R-by-DV matrix M:

```
key j:    M = a * M + b * outer(SK[j], V[j])
query i:  O[i] += c * SQ[i] @ M
```

The vector backward consists of a forward replay and a reverse scan:

```
forward query: dSQ[i] += c * M @ dO[i]
reverse query: A += c * outer(SQ[i], dO[i])
reverse key:   dSK[j] += b * A @ V[j]
               dV[j]  += b * A.T @ SK[j]
               A *= a
```

Consequently dSQ is exactly the existing prefill applied to `(dO, V, SK)`.
Neither scan needs saved per-event matrix states in its serial formulation.
Chunk-parallel execution has a separate checkpoint/state-memory tradeoff.

Tau is not detached. For match length l, the physical scalar weight is
`W_l = sum(exp(r*tau), r=1..l)`, including the denominator's zero-value
fallback of weight one. The helper computes log W and its derivative with a
stable recurrence, including negative tau, zero and values near zero.

Numerical rescaling pivots can be held constant during differentiation:
the represented operator is invariant to their choice. With those pivots
frozen, `da/dtau=0`, key `dlog(b)/dtau` is its physical log-weight derivative,
and query `dlog(c)/dtau` includes **minus dlog(normalizer)/dtau**. Then:

```
dTau = sum_query dlog(c)/dtau * dot(SQ[i], local_dSQ[i])
     + sum_key   dlog(b)/dtau * dot(SK[j], local_dSK[j])
```

These are event-local vector contributions, before reduction across streams;
one cannot recover this expression by scaling the final summed gradients.
It needs no additional matrix state in the proposed backward. The helper also
propagates a tangent matrix as an independent numerical check, not as a
requirement for production execution.

With fixed R/DV, the event scans preserve the planner's near-linear event
complexity instead of visiting all token pairs. Their constant factors,
irregular access and reductions still require measurement on sm120.

## Results and limitations

Environment: local RTX 5090, conda `blkw`, worktree packages. Baseline HEAD:
`3fb6269`. The JSON reports include source hashes and library versions.

The main scan covered 256 cases: N=1/17/65/256, R16/DV32, two seeds, random,
constant, mismatched and periodic labels, and tau in
`[-2, -1e-8, 0, 1e-8, .5, 2, log(64), 16]`. Another 16 cases covered
N=512/2048, R32/DV64, tau=0/.5 and the same four label patterns.

All 272 cases passed the additional exact logaddexp FP64 oracle checks and
the **unchanged production gradient acceptance policy** against the maintained
v3 oracle. Pure FP32 event gradients also passed that production policy in
all 272 cases. These are CPU arithmetic checks, not a new GPU backward.

| Check | Main 256 cases | Long 16 cases |
| --- | ---: | ---: |
| FP64 vector gradient max absolute error, exact oracle | 7.94e-14 | 1.83e-14 |
| FP64 tau max absolute error, exact oracle | 4.45e-12 | 1.53e-10 |
| FP32 vector gradient max absolute error, maintained oracle | 5.24e-8 | 6.73e-8 |
| FP32 tau max absolute error, maintained oracle | 7.96e-6 | 1.37e-4 |

The following discrepancies remain visible in the reports:

- One main-scan case failed the strict FP64 micro-check against the maintained
  oracle: periodic labels, N256, tau=.5, seed318, tau error 1.42e-9.
  The maintained oracle uses thresholded softplus (linear above 20), whereas
  native hard prefill uses geometric sums. The separate logaddexp oracle
  diagnoses this difference; it does not replace the maintained comparison.
  No reference code or production tolerance was modified.
- Five main-scan cases failed the strict FP32 micro-check, all on tau.
  Positive/negative event contributions can nearly cancel. For example,
  periodic N256, seed319, tau=log(64) has about 4.95% relative tau error;
  at tau=16 the true derivative can be close to zero. Passing the existing
  relaxed tau policy is not sufficient evidence of high-accuracy training.
- Two small-step tau finite differences failed their strict micro-check due
  to cancellation. The independent larger step agrees closely with the
  analytic result; both measurements and the original failure are retained.
  Analytic event/tangent and exact-oracle checks passed.

Existing GPU prefill was reused for dSQ in eight N256/R32/DV64 cases (four
patterns, tau=.5/2). All passed the unchanged vector-gradient policy. BF16
relative L2 error was at most 0.178%; TF32x3 at most 1.71e-7. Mismatched
labels produced exact zeros. This verifies GPU dSQ reuse only, not a complete
GPU dSK/dV/tau implementation.

The unchanged production forward/backward was also checked on those inputs.
Both direction values produced bit-identical hard outputs; the core Q/K/LSE
gradients were zero. However, periodic labels at tau=2 failed the **existing
production tau acceptance policy** for both directions: absolute error
0.241824 versus allowed bound 0.0296903. This is an existing-core observation,
not a failure of a newly installed backward. It must remain a regression
fixture when comparing replacement kernels.

Document isolation was tested by independently planning two documents from
one hard training row, accumulating their shared tau gradient, and comparing
both directions. A joined plan intentionally gives a different result.
The inference query-reset flag is not a document-isolation substitute.

The CPU-reference phase added no device kernel and claimed no new CALL/spill
inspection or sanitizer run. Device validation for the subsequent prototype
is documented below. Neither phase establishes training convergence,
checkpoint equivalence, or end-to-end throughput.

The new reference tests plus existing SAM/README regressions passed: 134 tests
on the local environment. The tau discrepancies above are reported by the
diagnostic sweep and are not reclassified as strict numerical passes.

## Triton backward prototype

`tools/hard_prefill_backward_triton.py` provides explicit single-document,
single-head `HardPrefillBackward(...).backward(SQ, SK, V, dO)` execution. It
returns FP32 dSQ/dSK/dV, scalar dtau and diagnostic tau partials. It is not
registered with autograd or any model dispatch. CPU symbolic planning/upload
must happen before execution or graph capture.

The query kernel computes dSQ and the query contribution to dtau; the reverse
key kernel computes dSK/dV and the key contribution to dtau. Both carry matrix
state in registers and use chunk-local MMA. Across streams, vector gradients
use global FP32 atomics. Tau has two scalar partial arrays and a GPU reduction;
there are no per-event/per-chunk matrix snapshots or shared FP32 atomics.
Physical normalization is differentiated; numerical pivots remain frozen.

R16/32, DV32/64, BF16/FP32 inputs and chunk sizes 16/32/64 are supported by the
prototype. C16 is the initial configuration, without autotuning. BF16 MMA is
the performance candidate; TF32x3 (the diagnostic default) is the accuracy
comparison. Stream execution is serial across chunks, with no long-stream
parallel backward yet. The final scalar reduction uses PyTorch GPU operations.

The main GPU sweep comprises 384 cases (192 per precision): N1/17/65/256,
R32/DV64, two seeds, four label patterns, tau=-2/0/.5/2/log(64)/16.
Every vector gradient and tau passed the unchanged production acceptance
policy. Maximum vector relative L2 error was 3.21e-7 for TF32x3 and 0.259%
for BF16. Maximum tau absolute error was 1.56e-5 and 0.0352 respectively.
Among reference tau gradients with magnitude above 1e-6, neither mode reversed
the sign; their maximum relative errors were 0.670% and 33.7% respectively.
There were four TF32x3 and three BF16 raw sign reversals below that floor.
The floor only labels diagnostic categories: original gate results, raw values
and all sign reversals remain in the JSON, without modifying shared tolerances.

An additional 48 GPU cases covered N512/2048, tau=0/.5/2, four patterns and
both precisions. All passed the original gradient policy, with no tau sign
reversals. Maximum tau absolute/relative error was 5.52e-4 / 1.05% for TF32x3
and 0.914 / 51.0% for BF16; the latter can pass the original policy because
it includes an absolute allowance. Maximum vector relative error stayed below
2.89e-7 / 0.218%. This is consistent with accepting approximate tau magnitude,
not with claiming a universally high-accuracy BF16 tau estimate.

On the default R32/DV64/C16 configuration, SASS inspection found no CALL or
shared atomic instructions. BF16 query/key kernels have zero spills; TF32x3
key has zero spills, while TF32x3 query has two reported spills (one local load
and store instruction). This remaining diagnostic-path spill is not hidden.
Larger chunk configurations are correctness probes, not tuned configurations.
Additional SASS inspection of the FP32-input test configurations also found
no CALL/shared atomic instructions, but substantial spills: C16/R16/DV32
query/key counts were 14/30 (TF32x3) and 0/10 (BF16); C32/R16/DV64 was
140/150 and 0/44; C64/R32/DV32 was 336/1522 and 16/180. Those variants are
not performance-ready. The zero-spill BF16 result above is specifically the
baseline BF16-input R32/DV64/C16 configuration, not every supported variant.

The 41 new tests cover masked tails, empty/mismatched plans, all four vector
dimensions, all three chunk sizes, FP32/BF16 input, an interior reset that
splits chunks, independent document plans, graph replay, and input validation.
All 41 passed separately under memcheck, synccheck and racecheck, with zero
errors and zero racecheck warnings. Logs are in ignored `build/`.
The combined new/reference/SAM/README regression run passed 175 tests.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD/dism_v3/python:$PWD/nanochat \
  /home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/check_hard_prefill_backward_triton.py \
  --output dism_v3/build/hard_backward_triton_final_20261014.json
```

## Overall implementation plan

1. **Harden the GPU backward prototype before production integration.**
   The initial two-scan implementation is available above. Keep original
   shared tau; the requested practical priority is avoiding sign reversals
   for meaningful gradients, with amplitude/cancellation errors still
   reported. Benchmark prepared-plan execution and improve batching and
   launch/resource choices. Do not immediately replicate parallel prefill's
   large per-chunk matrix allocations. Recheck bounds, synchronization,
   CALL/spills and sanitizers after kernel changes. No shared FP32 atomicAdd.

2. **Split the complete hard/soft operator, not only B1 or B3.**
   Route using the existing sequence-level flag per packed training row,
   head and layer. Each document inherits that flag but retains its own
   causal history. Preserve random direction and its RNG draws/replay;
   direction affects the soft path but need not enter hard arithmetic.
   For hard work, vocabulary interpolation only needs labels, not LSE or
   its backward; the DISM branch contributes zero score-Q/K/LSE/vocabulary
   gradients. SQ/SK/V, tau, readout transforms and surrounding model layers
   still require backward. Shared/tied parameters can receive gradients
   from other branches. Preserve zero-versus-None optimizer behavior.
   Keep the current soft kernels and atomic interpolation backward.
   Add explicit row/head work lists and gradient/output merges, then verify
   all-soft, all-hard and mixed batches plus packed-document boundaries.
   Preserve checkpoint counters, annealing, optimizer schema and data position.

3. **Overlap native CPU planning with the existing dense hard forward first.**
   The preferred next experiment keeps the current quadratic hard forward,
   starts asynchronous planning as soon as labels are available, and consumes
   the event program only in backward. This does not require the entire
   subquadratic forward to be integrated first. Move the derivative planner,
   batch/head parallelism, coefficient generation and packing into a native
   batch API; the Python symbolic planner is only a correctness scaffold.
   That native API is now implemented and measured in the linked report;
   production overlap and per-layer deadline measurement remain to be done.
   Keep pinned buffers, copy streams, completion events and plan lifetime
   explicit. A worker waits for label D2H completion; the training thread
   continues enqueueing GPU forward work. Backward waits on the upload event
   only when needed. Measure the actual forward-to-backward deadline per layer,
   worker backlog and memory pressure; upper layers have less slack. A dense
   backward fallback for late plans is an option, not implemented behavior.

4. **Measure the crossover and remove any remaining planning bottleneck.**
   Current label D2H, CPU SAM/event construction, detached tau and coefficient
   H2D are unsuitable as an assumed-free per-layer training operation.
   Separate label-dependent topology from tau-dependent coefficients and
   differentiate normalization explicitly. Reuse a plan within the same
   forward/backward or valid checkpoint replay, not across changed labels.
   The existing GPU decode planner is not an offline prefill-planner drop-in.
   Prototype device-side planning only after the backward/cost breakdown is
   sound. For long streams, compare replay/checkpointing against stored
   chunk states. Keep a dense-hard candidate for short documents.

Benchmark the **whole operator**: label extraction, transfers/planning,
coefficient generation, forward, backward, reductions/merges, and peak
memory. Sweep document lengths 256 through 32768, representative row packing,
hard fractions 0/.5/.95/1, R32/DV64 and the baseline head count, using both
learned label distributions and adversarial repeated/periodic labels.
Calibrate dense-hard versus SAM dispatch empirically; do not equate a long
packed row with a long independent document. Only then compare layer/step
throughput and decide whether remaining B1/B3 micro-optimizations matter.

## Transfer and host/device handoff measurements

`tools/bench_hard_plan_transfers.py` measures warmed pageable/pinned copies,
native parallel planning, CPU merging/staging and metadata upload independently.
`tools/bench_hard_plan_handoff.py` additionally exercises worker wakeup,
event synchronization and upload alongside the unchanged dense hard forward.
Both are diagnostics, not production scheduling changes.

The native inference planner **already is C++**. These initial tests used eight CPU
worker threads across independent document/head jobs. However, per-job Python
list conversion, exported arrays and batch metadata merging remain. The new
analytic derivative planner was still Python at that stage. Native derivative generation
is not included in the native timings: an explicitly synthetic float32 dlog
field accounts for its transfer bytes only. Thus the following is not a
complete training-plan latency benchmark or a limit on a fully native batch API.

Local RTX 5090, B16/H6/row length2048, all-hard, synthetic iid vocabulary512
labels or all-identical labels, five CPU samples and twenty transfer samples.
Values below are median completed wall milliseconds, including the final host
completion wait. Copy buffers are preallocated/pinned outside the copy timing;
their CPU fill is measured separately. CUDA-event windows include host enqueue
gaps and are not profiler-isolated DMA durations.

| Uniform documents | Random labels | All-identical labels |
| --- | ---: | ---: |
| Label payload | 1.5 MiB | 1.5 MiB |
| Metadata including synthetic dlog | 26.3 MiB | 160.1 MiB |
| Label D2H, pageable | 0.145 | 0.141 |
| Label D2H, preallocated pinned | 0.071 | 0.070 |
| Native planning/chunking, eight workers | 39.4 | 58.6 |
| Python merge/relabel | 11.7 | 51.1 |
| CPU copy into one pinned metadata buffer | 1.50 | 11.9 |
| Metadata H2D, pageable fields | 1.27 | 9.69 |
| Metadata H2D, pinned fields | 0.625 | 3.660 |
| Metadata H2D, one pinned buffer | 0.609 | 3.582 |

An independent B16 repeat measured native planning 35.0/60.3 ms, merge
12.6/51.8 ms and single-buffer H2D 0.595/3.573 ms. Ragged documents
256/512/256/1024 per row reduced metadata to 19.6/149.5 MiB and pinned H2D
to 0.455/3.351 ms in the first run. Measurements also include B1/B4.
The Python derivative scaffold alone took roughly 27.5/98.6 ms for one
2048-token random/repeated document: it is not suitable for an overlapping
training pipeline without the planned native implementation.

The isolated copy tests start on an idle device and therefore exclude work
queued before the labels become available. The handoff test adds those
dependencies explicitly: separate D2H/H2D streams, events, a coordinator
worker, native planner pool, dynamic pin staging, and a concurrent dense
forward with saved training state. It does not run a full model or backward.
For B16 uniform documents, the observed plan-upload readiness was about
54.0/132.6 ms after submission for random/repeated labels. Dense forward's
isolated completed latency was about 0.90/0.92 ms; that one operation alone
does not hide current preparation. Full-model per-layer slack remains unmeasured.

An idle `cudaDeviceSynchronize` wrapper cost approximately 0.002–0.020 ms in
these runs. Adding a preceding dense forward before the label-ready event
caused about 0.67–0.82 ms of worker D2H-completion wait at B16. That wait
includes unfinished GPU dependency work, not a fixed synchronization tax.
The main/training thread does not wait for D2H in this diagnostic; it can
still be delayed by Python scheduling/GIL contention, recorded separately.
Host-observed readiness includes worker scheduling and completion observation,
not just DMA execution. No claim of fully hidden planning is made.

Next: a native batch/head-parallel derivative planner, minimizing Python
round trips and directly filling reusable pinned packed storage. Then repeat
the handoff test with actual layer deadlines and bounded outstanding plans.
Only after that comparison should a GPU planner be prioritized over CPU/GPU
overlap. Raw reports are `build/hard_plan_transfers_20261014.json`,
`build/hard_plan_transfers_repeat_20261014.json` and
`build/hard_plan_handoff_20261014.json` (all ignored by Git).

## Reproduction

Run from the workspace root; choose fresh output paths:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD/dism_v3/python:$PWD/nanochat \
  /home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/check_hard_prefill_training.py --gpu-check \
  --output dism_v3/build/hard_prefill_training_final_20261014.json

OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD/dism_v3/python:$PWD/nanochat \
  /home/cicuvc/miniconda3/envs/blkw/bin/python \
  dism_v3/tools/check_hard_prefill_training.py \
  --tokens 512 2048 --taus 0 .5 --seeds 318 --readout-dim 32 --value-dim 64 \
  --output dism_v3/build/hard_prefill_training_long_20261014.json

OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD/dism_v3/python:$PWD/nanochat \
  /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q \
  dism_v3/tests/test_hard_prefill_training_reference.py \
  dism_v3/tests/test_sam_inference.py dism_v3/tests/test_readme.py
```

Result JSON stays under ignored `build/`; no generated artifacts are tracked.
