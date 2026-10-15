# sm120 training performance baseline

## Host-device synchronization audit (2026-10-15)

Purpose: close the previous section's "remaining per-microbatch metadata
synchronization" candidate. Result: **every host-device synchronization point on
the production training path is fully hidden under GPU execution. Removing them
is worth 1% or less, so no source change is proposed and the candidate is
retired in favour of launch count.**

Method. All measurements are read-only replays of resident checkpoint batches
(no dataloader, IO, W&B or checkpoint writes), and three harness options were
added, each off by default:

- `--sync-probe` wraps `fla.ops.gated_delta_rule.chunk.prepare_chunk_indices`
  with a `perf_counter` bracket and classifies every call as a cache miss or hit
  by return identity, so host blocking is measured without profiler overhead.
- `--profile-stack` records Python stacks in the torch profile.
  `analyze_model_profile.py` still reports kernel groups per phase; call-site
  attribution for scalars and copies needs a time-containment join from
  `cpu_op`/`cuda_runtime` events to the enclosing `python_function` frame, which
  was done ad hoc and is not part of the committed analyzer.
- `--host-delay-ms` inserts a synthetic host stall after gradient accumulation
  to calibrate how much extra host work a step absorbs.

A CUDA-only profile (no CPU activity) of one step gives the GPU activity union
without the CPU-side profiler inflation a full profile adds.

Inventory and cost, production configuration (valres 72M, checkpoint 19074,
direct CUDA shortconv, B16/N2048, four accumulation microbatches):

| synchronization point | per step | measured cost | critical path |
| --- | --- | --- | --- |
| FLA `prepare_chunk_indices` pageable D2H + H2D | 4 misses, 44 hits | 116.7 ms host blocking | no |
| `clip_grad_norm_(...).item()` | 1 | below the 6 ms window noise | no |
| training loss `.item()` | 1 | below the 6 ms window noise | no |
| Inductor in-graph scalar read for Triton arguments | 48 | 0.15 ms | no |
| `metrics.update_rms_max_per_group` pinned readback | 4 | about 0.04 ms | no |
| hidden-RMS `tolist()` of 12 floats | 1 | under 0.1 ms | no |
| hard-planner staging on background threads | 4 | 1.0-2.4 ms H2D, 0.07 ms D2H | no (host wait 0.0002 ms) |
| `document_loss._sum_gradients` `active.tolist()` | 1 if `dist` is initialized | not exercised on one rank | - |

The FLA helper is the only material one, and only because its result is cached
by identity in a **single-entry** cache. Inside one microbatch the adapter's
shared boundary snapshot makes all twelve mixers hit the same entry, so the
blocking device round trip happens exactly once per microbatch: four per step at
116.7 ms of host blocking, against 0.22 ms for the other 44 calls.

Evidence that it is hidden:

- Supplying the chunk-index tables from the host instead of letting FLA compute
  them, with everything else unchanged, gives **604.37 +- 6.14 ms to
  599.91 +- 5.96 ms** over 11 resident steps: -4.5 ms, -0.7%, inside one
  standard deviation. The substituted tables were verified element-wise against
  the FLA result for all nine resident batches (512 chunks each), and per
  microbatch loss and gradient norm reproduce the unmodified path exactly
  through five steps.
- The same helper blocked 173 ms per step in the historical fused-shortconv
  configuration (step about 810 ms); removing it there also changed nothing
  (808.6 to 814.6 ms, both inside the 10 ms window spread).
- Host-slack calibration in the production configuration (mean of five steps):
  +20 ms gives +2.3 ms, +40 ms gives +9.1 ms, +60 ms gives +35.0 ms, +80 ms
  gives +52.3 ms, +120 ms gives +94.0 ms. A step absorbs about 25-35 ms of extra
  host work before it starts costing step time one for one.

Why. A CUDA-only profile of one step reports 15 311 kernels and a GPU activity
union of 550-560 ms against a step of about 600 ms, so the GPU timeline is
occupied roughly 92% of the step. The remaining about 45 ms is not one stall but
the sum of sub-3 us gaps between launches, which host-side work cannot fill. The
phase event windows (forward 247 ms, backward 350 ms, normalization, optimizer
and probes 6.9 ms) already account for the whole step.

Consequence: remaining host-side and synchronization work on this path has a
hard ceiling of about 5%, and nothing measured comes close to it. The live lever
is launch count and per-kernel cost.

Artifacts, all under ignored `build/`: `syncprof_base_20261015.{json,trace.json}`,
`syncprof_stack_20261015.{json,gz,breakdown.json}` and
`hostsync_probe_20261015.json`. The stack trace is kept gzipped because the plain
chrome trace is 214 MB and this partition is nearly full; it can be regenerated
with `--profile --profile-stack`. The "supply tables from the host" and
host-slack experiments ran from a scratch copy of the replay harness outside the
worktree; the three harness options above are their committed form. No kernel
math, launch default, tolerance, checkpoint schema or training semantics were
changed by this audit.

## GDN metadata, prefetch and session closeout (2026-10-15)

Implementation commit: `4726418`. No kernel math, launch defaults, checkpoint
schema or normalization semantics changed.

- GDN previously received a new `prepare_pack` boundary clone in each hybrid
  layer, defeating FLA's identity cache. The adapter now owns one immutable
  boundary snapshot per forward and shares it across GDN layers. DISM still
  uses its validated outputs; subsequent forwards get new snapshots, including
  when the caller overwrites its input buffer in place.
- A 72M checkpoint-4000 replay (document-balanced CE, four resident microbatches,
  launch threads on P cores) reduced chunk-index rebuilds28 to4 and associated
  stream synchronizations56 to8 per step. Three eight-step windows averaged
  619.37 ms before versus610.75 ms after. The remaining four misses are one per
  microbatch. Synchronization duration includes queued GPU work, not just idle.
- Producer-side pinned staging reduced ready-queue host consumption from93–95 us
  to32–33 us per microbatch. This is only about0.24 ms of host work per step,
  not a measured end-to-end saving of that size.
- Affinity alone: unrestricted623.46 ms, P-core service/E-core planner623.41 ms,
  only launch threads on P cores619.37 ms. These small, single-session differences
  do not justify changing production defaults. Actual restarted training stayed
  unrestricted, with16 planner workers.

Validation:18 tests passed (compiled metadata identity/invalidation and gradient
parity, CUDA shortconv, checkpoint resume); the prefetch CPU/CUDA and resume suite
passed9 tests separately. No new native kernels; no new sanitizer campaign.
Results under ignored `build/`: `affinity_*_20261015.json`,
`prefetch_staging_20261015.json`, `gdn_boundary_reuse_tests_final.log`,
`gdn_conv_diagnostic_20261015.trace.json`, and
`gdn_boundary_after_20261015.{json,trace.json}`. These new traces are PyTorch
profiles, not Nsight captures. Replay excludes live data loading and W&B.

Fresh training smoke run `dism-fwedu-valres-72m-2500m-tokenbal-20261015-r3`
used the exact previous r2 model specification and training arguments:72,035,808
parameters, fixed true direction, packed-row sequence hard sampling,
token-balanced CE with global valid-token accumulation normalization,2048-token
context, device batch16, accumulation4, planned19074 steps/2.5B tokens.
It was explicitly stopped by the user after logged step294 (PID508050 exited).
Steps195–294 averaged619.84 ms, about211.5k packed tokens/s; this is logged step
time and excludes some logging overhead, not a controlled comparison to replay.
Final logged loss4.773583, gradient norm0.578479. No checkpoint was produced
before the first scheduled save at2000; this run cannot resume at294. The prior
r2 stopped at1129 without a checkpoint too. Older checkpoint files are untouched.

Run directory:
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m-tokenbal-20261015-r3`.
Its `README.md`, `train.sh`, `model_spec.json`, `logs/train.log` and provenance
remain available; W&B run ID`1vsezazx`. No training is left running at closeout.
Next candidates: GDN eager gate preparation. The per-microbatch metadata
synchronization candidate was measured and closed by the later host-device
synchronization audit above. Do not infer a new training restart.

## Direct CUDA shortconv and GPU sequence IDs (2026-10-15)

The user stopped the fresh 72M run after logged step484 (no checkpoint yet),
requested removal of the FLA shortconv wrapper's host metadata construction,
and then explicitly authorized a fresh restart after validation.

`linear_fla/cuda` already selected Tri Dao's CUDA kernels through FLA. The new
frontend calls those same CUDA forward/backward kernels directly through local
functional custom ops. The installed upstream autograd.Function could not be
traced fullgraph because its custom op mutates a noncontiguous `out=` tensor;
the local wrapper supplies fake/autograd registrations without modifying FLA,
causal_conv1d, CUDA kernel math or launch settings. Parameter names, shapes,
ordering, initialization RNG, tied projection aliases and optimizer schema
remain unchanged. The historical config name is retained for compatibility.

The model constructs one int32 `[1,T]` document map with GPU searchsorted per
forward and passes it through every DISM/GDN frontend, including first-layer
value residual and diagnostic traversal. Right-sided search handles empty
documents; no stale-pointer cache is used. Hard flags remain packed-row/head
flags, independent of these convolution document IDs. Standalone convolution
calls can build their own GPU map when not supplied. This removes the FLA
per-document GPU-scalar arange loop, not all metadata work: native DISM exact
layout validation and FLA GDN preparation still have host synchronization.

Validation: **139 tests passed**, covering direct CUDA versus old FLA CUDA
FP32/BF16 outputs and gradients, exact eager whole-model forward equality,
document isolation, empty documents, in-place boundary replacement, actual
fullgraph Inductor output/gradient checks, shared metadata identity across all
19 convolutions in a four-layer value-residual model, diagnostics and checkpoint
resume. Compiled shortconv profiling has no `aten::item` or
`aten::_local_scalar_dense`. Logs: `build/cuda_shortconv_extended_tests.log`.
The initial unsupported fullgraph attempt is retained in
`build/cuda_shortconv_tests.log`; it is not counted as passing.
Focused memcheck, racecheck and synccheck each pass7 tests with zero errors
and zero racecheck warnings (`build/cuda_shortconv_{memcheck,racecheck,synccheck}.log`).
These cover the direct convolution and eager metadata, not a new sanitizer
campaign for the full DISM pipeline; its documented limitations remain.
The generated arange/searchsorted/sub metadata kernels inspected in the local
Inductor cache use12 or17 registers, zero shared/local/stack storage, no CALL
and no LDL/STL instructions (`build/shortconv_sequence_ids_sass.log`).
The full-model trace contains4 such launches totaling8.77 microseconds per step.

Paired unprofiled 72M replay: checkpoint19074, same four resident saved batches,
B16/N2048, four accumulation microbatches, true direction, 16 unbound planner
workers, compact/ready, document-balanced CE, value residual and RMS diagnostics.
Four warmups, two phase samples, two eight-step throughput windows. Scratch
optimizer updates only; no checkpoint or dataloader state writes.

| Measurement | Before | Direct CUDA + shared IDs |
| --- | ---: | ---: |
| Forward event window mean | 359.31 ms | 274.00 ms |
| Backward event window mean | 371.39 ms | 376.92 ms |
| Normalization/optimizer/probes mean | 14.11 ms | 14.17 ms |
| Sustained step, window1 / window2 | 755.00 / 754.93 ms | 670.08 / 689.07 ms |
| Sustained mean | 754.97 ms | 679.58 ms |
| Throughput | 173.6k tokens/s | about192.9k tokens/s |

This is about10.0% lower step time /11.1% higher throughput. The shortened
forward leaves less planner overlap: diagnostic routes changed20 event/4
quadratic to16 event/8 quadratic. Whole compiled-model execution is not claimed
bitwise identical: first replay loss2.75736 versus2.75703, gradient norm.2663
versus.2667. Primitive convolution outputs and eager whole-model outputs are
exact; kernel/gradient tolerances have not been relaxed.

The new PyTorch trace records113 scalar reads and145 stream synchronizations,
versus1086/1118 in the earlier documented FLA-wrapper trace. The old trace is
historical, not a fresh paired profiler capture. The new instrumented step is
740.37ms, with591.04ms GPU activity union and149.33ms without recorded GPU
activity; these are profiling diagnostics, not uninstrumented idle estimates
or occupancy. Artifacts: `build/conv_metadata_{before,after}.{json,log}`,
`build/conv_metadata_after.trace.json`, and its `.breakdown.json`.
The warmed CUDA/NVTX/OSRT Nsight capture is retained as
`build/conv_metadata_after_nsys2025.nsys-rep` (Nsight2025.5.2, CPU sampling off),
with its matching JSON/log. The system-default Nsight2023 invocation did not
produce a capture and is not used as evidence.

## Live-run diagnostic repair (2026-10-15)

The first new 72M/2.5B document-balanced run completed training step0, then
failed in `probe_qk_vocab_usage`: the optional histogram traversal still passed
the removed token-level hard mask. The resident-batch benchmark does not exercise
this histogram callback. No checkpoint was published; its logs and source
snapshot remain in `dism-fwedu-valres-72m-2500m-docbal-20261015` on the run disk.

The probe now passes `[original_packed_rows,H]` hard flags and propagates the
first GDN value through subsequent value-residual layers, following model eval
semantics. It also forwards the model vocabulary temperature. This changes
diagnostic traversal only, not training kernel math or optimizer state.
Nine new GPU tests cover fused/linear_fla, value residual on/off, train/eval mode,
multiple documents per row, and exception restoration. Every layer input/output
matches real all-hard model evaluation exactly; selection counts, model state,
RNG and existing gradients are checked. Together with checkpoint resume tests:
**16 passed** (`build/metrics_probe_regression.log`). No new kernel sanitizer
campaign was needed for this Python-only diagnostic change.

The user authorized a fresh retry in the independent sibling directory
`dism-fwedu-valres-72m-2500m-docbal-20261015-r2`; no failed-run state is silently
resumed, and historical checkpoints remain untouched.

## Default compact/ready/document-balanced integration (2026-10-15)

The production nanochat DISM/SWA adapters now use chunked CE with the actual
packed document boundaries. `mean` is the local valid-document mean;
`document_sum` returns a differentiable sum of document means plus a detached
valid-document count. The base training loop and resident-batch benchmark use
the latter and normalize across the **whole optimizer step**, including all
accumulation microbatches and distributed ranks. Empty/ignored-only documents
are excluded; partially ignored documents divide by their valid-token count.
Unreduced token-NLL/BPB validation stays unchanged. Targets are not shifted again.
The separate HF model's loss API is outside this nanochat integration.

The temporary microbatch backward scale is 1/A (A = accumulation steps).
After all backwards, gradients are SUM-reduced across ranks and multiplied by
A/global_valid_documents before AMP unscaling and clipping. Each rank then
holds the same normalized global gradient. This is important because the DISM
adapter uses ordinary AdamW, which performs no collective reduction itself.
There is no additional batch prefetch or optimizer/checkpoint-schema migration.
Logged loss now uses global numerator/count instead of the final microbatch's
loss. Historical scalar loss values using that older logging rule are not a
like-for-like comparison. Checkpoints record the normalization convention;
resuming older token-mean training warns about the intentional objective change.

The hard runtime defaults to compact metadata plus ready selection, retaining
classic/event comparison switches. The benchmark's NVTX annotations and stream
counts support both layouts. No CUDA/Triton kernel math or launch configuration
changes accompany this promotion; earlier numerical/spill limitations remain.

Validation: 64 CE/normalization/async/ready tests plus 18 CE subtests, 21 actual
model/compiler/resume tests, and 176 extended operator/compact/varlen/registry
tests passed. A final 15-case CPU normalization suite includes six added explicit
dual-path cases: **267 distinct regression cases** total. Two-process Gloo
tests cover unequal rank counts, an empty-valid rank, rank-local unused parameters
and normalization before clipping. Multi-GPU NCCL performance is not measured on
the single 5090. All original numerical tolerances remain unchanged.
Compact async default plus unnormalized CE accumulation smoke passed memcheck,
synccheck and racecheck: zero errors and zero racecheck warnings. These two
focused cases do not supersede historical soft-pipeline sanitizer limitations.

Ignored logs: `build/default_training_{initial,integration,extended}_tests.log`,
`build/default_document_loss_final.log`,
`build/default_training_defaults_final_tests.log`, and
`build/default_training_{memcheck,racecheck,synccheck}.log`.

### 72M default-path replay

Checkpoint19074, 72,035,808 parameters, B16/N2048, four accumulation
microbatches (131072 packed tokens/step), linear_fla,16 unbound planner workers,
compact + ready, document-balanced chunked CE and fused tanh value residual.
Four warmup steps, two phase samples, two eight-step throughput windows;
resident saved batches only, scratch optimizer updates, no checkpoint writes,
loader advancement or live training restart. Hidden-RMS probes remain enabled.

| Measurement | Result |
| --- | ---: |
| Forward event window, mean | 359.03 ms |
| Backward event window, mean | 371.80 ms |
| Normalization/optimizer/probes window, mean | 13.87 ms |
| Phase wall, two steps | 743.37 / 747.60 ms |
| Sustained wall, two windows | 754.95 / 752.32 ms per step |
| Sustained throughput | 173616 / 174224 tokens/s |
| Peak allocated CUDA memory | 18.398 GB |

Diagnostic step:20 event /4 quadratic calls. Prepared plans visible to the
diagnostic total540.7 MB metadata (not a complete transfer-byte accounting for
abandoned/unavailable plans). First restored-step global document loss2.75736,
gradient norm about0.2663. These are finite correctness/throughput observations,
not a training-convergence result or a controlled speedup against an old loss
objective. Phase windows include host launch gaps; they are not pure GPU busy time.

The first replay completed timing windows but failed in the final diagnostics:
the compact consumer lacked classic `single_count`/`multi_count` attributes.
Those diagnostic aliases and compact-upload NVTX labeling are now supplied,
and a complete rerun exited successfully. Only the final report is considered
a completed run: `build/model72m_default_compact_ready_document_final.{json,log}`.
The existing hidden-RMS-hook Dynamo eight-specialization warning is still present;
it was not suppressed or counted as fullgraph coverage.

The sections below first preserve the historical single-operator baseline.
See [full-model training replay](#72m-full-model-training-replay) for the later
complete hard/soft operator measurement and controlled core-path comparison.

Measured locally on 2026-10-14, source HEAD `cdbd33c31f9f702d6b6244a253a2e23c10ae0773`.
This is a **single DISM training operator** benchmark, including vocabulary
interpolation and its backward. It is not full-model training throughput.
Projections, short convolution, GDN, FFN, loss, optimizer, gradient accumulation,
data loading and multi-GPU communication are excluded.

## Baseline and environment

The historical run's `train.sh` and `meta_019074.json` were inspected read-only:
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m`.
They agree with `nanochat/configs/sequence_true_72m_valres.json`: 12 layers,
width384, H6, D/DV64, R32, vocabulary512, context2048, device batch16,
fixed true direction and sequence-level hard sampling. No checkpoint was loaded
or modified and no training was restarted.

- GPU: RTX5090, compute capability12.0, 170 SMs, approximately32GiB, power limit500W.
- Driver615.71.09; Torch2.13.0+cu130; Triton3.8.0; clang23 git `0cad9b7585d9`;
  compilation toolkit13.1 (`ptxas`13.1.115).
- Rebuilt with `dism_v3/build.py`, `DISM_BUILD_JOBS=4`, all eight production
  configurations, sm120a, BF16 output, no optional probes or FP32 diagnostics.
- Extension SHA256 after rebuild:
  `e9e7bd6895c5f841f82594967f5afcc761d502acc77e65a34e838e43dd098a1e`.
  It matches the preexisting extension. No kernel/default/math changes.
- Normal desktop session, unlocked clocks, no other compute job observed before
  measurement. GPU clock/temperature/power snapshots accompany every JSON case.
  This is not a measurement on RTX PRO6000; use the same script there before
  extrapolating absolute throughput.

## Measurement contract

`tools/bench_training.py` calls the `compiler.py` vocabulary forward/backward
custom operators used inside compiled training. It does not wrap the surrounding
model in `torch.compile`. Both forward paths save training checkpoints; backward
consumes those saves and computes all eight input/codebook/tau gradients.
Backward-only timing reuses immutable saves, with fresh output allocations and
no parameter `.grad` accumulation. The pair measurement runs a fresh forward
followed by backward on that forward's saves.

Inputs are seeded synthetic SiLU(Gaussian) BF16 vectors, FP32 SiLU(Gaussian)
codebooks, tau0.5, beta1 and BF16 output gradients. They are not activations or
learned codebooks from the historical run. Numerical distributions and repeated
hard chains can affect timing; this fixture does not establish trained-model
quality or long-chain numerical accuracy. Each case checks finite output and all
returned gradients.

Packed B=1 transport represents original training batches B1/B4/B16:

- `uniform`: each original row is one2048-token document.
- `ragged`: each original row contains documents256/512/256/1024.
- Hard probability0/.5/.95/1; a flag is sampled once per original row/head and
  expanded across that row, including its document boundaries. Direction is true.
- The two layouts have equal token counts, but ragged has only0.34375 times
  uniform's sum of squared document lengths. Neither is a measured dataset
  length histogram. Layout construction is outside timing, metadata caches hot.
- Realized hard fractions are recorded: B16 at p=.95 has91/96 hard row/head
  planes (94.79%). B1 at p=.95 happened to be all-hard.

Timings use CUDA-event windows around repeated calls, with synchronized wall
times also recorded. These include allocations and possible host launch gaps,
not just summed kernel execution. No CUDA Graph replay is used. Independent
stage medians need not add to a full-call median. Separate profiler captures
provide instrumented device-duration attribution and are not substituted for
unprofiled latency. Every case has an untimed settling workload plus warmup.

## Sweep and independent repeat

Primary comparison point: B16, p=.95, seven windows of100 calls after20 warmups
and two seconds of settling. Medians below, with the range of window means in
parentheses, in milliseconds:

| Layout | forward | backward | fresh forward+backward |
|---|---:|---:|---:|
| uniform |1.251 (1.214–1.279)|8.581 (8.238–8.694)|9.902 (9.855–9.917)|
| ragged |.805 (.706–.844)|5.763 (5.528–6.009)|6.430 (6.360–6.686)|

Native core forward/backward medians from the same run: uniform .945/4.885ms,
ragged .416/2.055ms. Isolated B1/B3: uniform2.435/2.373ms,
ragged1.004/1.005ms. Host and clock variation remain visible even with longer
windows; always compare repeated measurements and retain their ranges.

Milliseconds per operator invocation; each entry is the median of five windows
of20 calls, after10 warmups and one second of settling. Repeat timings are
collected before the separate profiler capture for that case.

| B16 layout | hard probability | sweep forward | sweep backward | sweep pair | repeat forward | repeat backward | repeat pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform |0|1.235|7.908|9.055|1.231|7.764|9.061|
| uniform |.5|1.228|8.252|9.326|—|—|—|
| uniform |.95|1.334|8.366|9.950|1.220|8.321|9.561|
| uniform |1|1.221|8.351|9.581|1.220|8.397|9.566|
| ragged |0|.802|5.455|6.600|.714|5.324|6.035|
| ragged |.5|.728|5.414|6.321|—|—|—|
| ragged |.95|.711|5.493|6.237|.710|5.546|6.356|
| ragged |1|.807|5.632|6.269|.808|5.532|6.952|

The short-window sweep has visible timing variation, including approximately
11% pair variation for ragged all-hard. Do not interpret small differences as
hard/soft speedups. In particular, the current all-hard training path still
launches interpolation forward/backward; it does not become a SAM inference path.

At p=.95, sweep pair latency for B1/B4/B16 is1.336/2.749/9.950ms (uniform)
and0.764/1.923/6.237ms (ragged). These are synthetic single-layer operator rates,
not whole-model tokens/second.

## Device attribution and resource evidence

Separate three-call profiler captures, B16 p=.95, microseconds per call:

| Kernel | uniform | ragged |
|---|---:|---:|
| interpolation forward `emb_fwd` |272.5|270.3|
| forward summary |223.1|106.4|
| forward readout |670.3|237.3|
| interpolation backward `_interp_bwd` |3256.5|3087.7|
| B1 backward summary |2795.0|969.2|
| B3 backward Q/K |2428.6|943.0|
| all backward device events |9069.2|5492.8|

Profiler results have instrumentation/run variability, especially on uniform;
they establish attribution, not precise unprofiled latency. The main backward
targets are interpolation and B1/B3. For ragged, interpolation backward accounts
for56.2% of device duration, B1+B3 for34.8%. Forward interpolation is only about
0.27ms here, so removing it alone cannot yield a20x training speedup.

Current sm120a cubin resources for R32/D64/DV64 packed kernels:

| Kernel | registers reported | stack bytes |
|---|---:|---:|
| forward |168|16|
| B1 |168|304|
| B3 |168|192|

No CALL found in these three disassemblies. Stack is a static compiler resource,
not dynamic spill traffic; register count under setmaxnreg is not a sufficient
optimization metric. Dynamic shared memory is not represented by the1024-byte
static shared resource reported by cuobjdump.

## Memory and validation

B16 uniform/ragged `VarlenLayout.checkpoint_bytes(6)` budgets are286.5/97.5MiB.
Measured forward+backward peak allocation increments are607.3/511.3MiB over
the live input/saved-state fixture. This increment excludes the already-live
fixture and allocator reservations; it is neither whole-model peak memory nor
the checkpoint budget. All raw allocated baselines and retained increments are
in JSON, preventing these different quantities from being conflated.

After rebuilding:121 tests passed across `test_multi_config.py`,
`test_voc_glue.py`, and `test_native_frontend.py`, covering all configurations,
fixed/packed oracle comparisons and fused-glue rounding. All benchmark outputs
and returned gradients were finite. No sanitizer or strict-gradient campaign
was run for this benchmark-only change; existing BF16 delta/tau limitations in
BACKWARD_DIMS.md remain applicable.

## Reproduction and artifacts

From repository root, with a fresh output filename:

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v3/tools/bench_training.py \
  --interp-backward baseline --warmup 10 --repeats 20 --samples 5 \
  --output dism_v3/build/bench_training_sm120_20261014.json

/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v3/tools/bench_training.py \
  --interp-backward baseline --batches 16 --hard-probs 0 .95 1 --warmup 10 --repeats 20 --samples 5 \
  --profile --output dism_v3/build/bench_training_sm120_20261014_repeat_profile.json

/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v3/tools/bench_training.py \
  --interp-backward baseline --batches 16 --hard-probs .95 --warmup 20 --repeats 100 --samples 7 \
  --settle-seconds 2 --output dism_v3/build/bench_training_sm120_20261014_reference.json
```

JSON includes the exact configuration, source revision, extension and benchmark
hashes, environment, sampling fractions, document lengths, timing samples,
memory and optional kernel attribution. Generated results remain ignored under
`dism_v3/build/`; only the benchmark source and this maintained report belong in Git.

The production default subsequently changed to atomic split-attention backward
with autotune. The explicit `--interp-backward baseline` above preserves this
report's original implementation. See [INTERPOLATION_BACKWARD.md](INTERPOLATION_BACKWARD.md)
for the optimized implementation and current measurements.

## 72M full-model training replay

The initial measurements below use the selected checkpoint's default **fused**
convolution frontend. They do not represent runs configured with `linear_fla`,
which the user normally uses for training. See the later linear/FLA comparison
before using the original shortconv bottleneck ranking to prioritize work.

Measured 2026-10-14 on RTX5090/Ultra9 285K, Torch2.13.0+cu130, driver615.71.09.
GPU power limit was **450W** during all new comparisons (not the historical
500W operator baseline); no power/clocks/affinity system defaults were changed.
Source is `3b26dbf` plus the working-tree default hybrid integration. The
benchmark does not modify production kernels, launch defaults or tolerances.

`tools/bench_model_training.py` strictly loads read-only copies of the valres
checkpoint at19074, including its 72,035,808 parameters, optimizer topology and
AdamW state, RNG counter76296 and hard probability.95. It replays four real
checkpoint-owned prefetched batches, each B16/N2048, four microbatches per
131072-token optimizer step. Document counts are34/30/32/43; their quadratic
work ratios versus16 full2048-token documents are .76172/.71289/.52930/.40625.
Direction remains true, with first-half GDN/rear-half GDN+DISM, postnorm and
value residual unchanged. This is the selected fixed-true baseline, not a
claim about the random-direction checkpoint or other architectural variants.

Only disposable in-memory model/optimizer copies update. The original RNG,
weights and optimizer are restored before each measurement phase; counters
advance normally within a phase, while the final checkpoint annealing values
stay fixed. No checkpoint is written, dataloader resumed or consumed position
advanced. Inputs are resident on GPU: data loading, tokenization, W&B, disk IO
and validation are excluded. Repeated batches intentionally overfit during
the bounded throughput window; this is not continued training or a convergence
test. Loss and global gradient norm remain finite, and live plan handles return
to zero. Outputs, not training artifacts, live under ignored `build/`.

Compilation matches `base_train.py`: dynamic Inductor, duck shapes disabled,
dynamic output shape capture, GDN graph breaks retained. Three complete warmup
steps exclude JIT/autotuning. Phase events surround each forward/backward with
**no intervening synchronization**, preserving overlap; numbers below sum four
microbatches. Sustained measurements are separate three-window runs of ten
optimizer steps each, with gradient norm synchronization, AdamW, hidden-RMS and
update-RMS probes included. Phase events include host launch starvation/waits,
not just GPU-active kernel time.

### Controlled core-path comparison

The old extension is the already-built local artifact
`build/lib.linux-x86_64-cpython-312/cu_flash_dism.cpython-312-x86_64-linux-gnu.so`.
Its SHA256 equals the recorded baseline `e9e7bd68...98a1e` above; the comparison
helper rejects any other hash. It is explicitly loaded under a separate module
name, never substituted on PYTHONPATH or copied over the production extension.
`tools/bench_legacy_training.py` overrides only this benchmark process's
vocabulary custom-op implementations to use the old full-quadratic CUDA core,
expanding sequence masks into the legacy input format. Both sides retain
**current atomic interpolation**, the same surrounding model, temperature
handling, inputs, checkpoint and optimizer. This isolates core-path changes;
it is not a reconstruction of every historical training component.

All times milliseconds; forward/backward columns are sums over four microbatches.
Sustained ranges are the range of ten-step window means, not per-step extremes.

| Path | Forward | Backward | Optimizer/probes | Sustained step | tokens/s |
|---|---:|---:|---:|---:|---:|
| Old quadratic CUDA core, current interpolation |285.7|801.8|13.6|1113.5 (1109.0–1119.7)|117707|
| New hybrid,16 workers |307.4|795.3|13.6|1106.2 (1103.8–1108.7)|118483|
| New hybrid, independent repeat |303.4|790.2|13.1|1108.5 (1102.3–1112.0)|118247|
| New hybrid, no hidden/update-RMS probes |299.6|792.2|2.5|1092.7 (1092.2–1093.3)|119952|

**There is no demonstrated substantial end-to-end improvement from the new
core route at this scale:** the sustained difference is only0.5–0.7%, comparable
to run variation. The historical last100 training steps averaged1208.7ms
(108439 tokens/s), but comparing that log to the new1106ms does not isolate this
change: interpolation, data/probe costs, inputs and GPU power settings differ.
Peak allocated GPU memory in the phase measurement was16.42GiB hybrid versus
16.31GiB old core, excluding allocator reservations and pinned host memory.

### Bottleneck attribution

Separate one-step profiler captures provide attribution, **not additive latency
budgets**: copies and kernels can overlap, and CPU waits include queued GPU work.

- DISM native GPU stages fell from90.7ms (old core) to25.4ms, plus13.6ms for the
  hard event readout/summarization/passing kernels: about51.7ms less GPU work.
  The event readout alone is9.3ms, not the complete event backward.
- At16 workers, the24 plans consumed395–418ms summed CPU wall time per step,
  with35–47ms exposed `future.result()` wait. In the first measurement,329ms
  was native planning,79ms packing, and about10ms allocation/release/input work.
  Waits occur chiefly on the last DISM layer's backward: all six plans were
  queued FIFO, but backward starts at the last layer. Forward-to-backward slack
  does not guarantee that every layer's plan is ready.
- D2H event windows totaled1.9ms. Metadata H2D windows totaled52–60ms including
  enqueue gaps; the separately profiled H2D device copies totaled41ms. They
  must not be added to the step time as though all transfers were exposed.
- Metadata totals1.76GB/step. Only18.1% of padded C64 event slots are occupied,
  and96.9% of streams fit one chunk. Compact/bucket-sized short-stream storage
  is a concrete target for reducing packing/PCIe traffic without changing math.
- Short-convolution/projection kernels total276ms: `bwd_mma_gemm`176ms,
  `bwd_short_conv_silu`58ms, forward43ms. Code inspection shows the packed grid
  uses2048 for every document, or2.17 times actual rows on this fixture. The
  projection backward still iterates channels and issues dweight atomics for
  fully masked sequence tiles; its dweight atomic mask is only channel-based.
  The benefit of eliminating these padded tiles is **not yet measured**.
- Fused vocabulary CE kernels total244ms: forward61ms, dlogits65ms,
  dhidden79ms, dweight40ms. Their small fixed tiles and long vocabulary loops
  are a larger remaining GPU target than further event-readout optimization.
- `float(beta)` in compiled vocabulary backward causes24 scalar-read syncs.
  Their425.7ms profiler CPU time mostly waits for previously queued GPU work;
  **it is not425.7ms of recoverable overhead**. Packed-layout and FLA metadata
  reads create other sync boundaries. Removing/reusing these reads requires a
  controlled follow-up measurement, not adding their profiler durations.
- Hidden-RMS hooks hit Dynamo's default8-specialization limit at layer8, so
  some code falls back to eager. The no-probe control improves only about1.2%
  here; it is not the dominant source of the795ms backward duration.

The immediate planner priority is metadata footprint and tail scheduling;
the larger full-model GPU targets are projection/shortconv and CE. No changes
to those production paths were made in this measurement task. Existing tau
cancellation/long-stream numerical limitations still apply; no new sanitizer
or kernel-resource campaign was needed for benchmark-only code.

### CPU utilization and process-local affinity

The285K's PMU topology reports P cores0–7 (`cpu_core/cpus`) and E cores8–23
(`cpu_atom/cpus`), with no SMT. A12-second `pidstat -t` sample of the unbound
24-worker run averaged633% process CPU, or6.33 cores out of24. Each planner
worker averaged21–23%; main/autograd/coordinator averaged43/56/6%. This is
bursty planner work, not sustained all-core saturation. Native planning/packing
release the GIL; the worker queue mutex is released before task execution.
Low average CPU utilization does not exclude a late plan delaying backward.

The benchmark can pin each native worker to one CPU and isolate service threads
(main/autograd/coordinator) on CPUs0–1. Only this process's threads are affected;
production defaults and system affinity remain unchanged. Worker TIDs are
discovered after warming native imports; ambiguous thread discovery fails.

| Configuration | Sustained step, ms | Explicit planner wait, ms/diagnostic step | Worker migrations |
|---|---:|---:|---:|
|16 unbound, original/repeat|1106.2 /1108.5|47.2 /35.1|not captured|
|16 unbound, later control|1122.6|37.7|335|
|16 pinned2–17, service0–1|1107.8|11.7|0|
|22 pinned2–23, service0–1|1104.1|0.085|0|
|24 unbound|1100.8|1.31|not captured|

Affinity runs and the later control use two10-step windows. Migration counters
cover three phase steps plus20 throughput steps and the intervening restore,
not just the throughput windows. Pinning eliminates migration, but a large
stable end-to-end gain is **not established**. Changing worker count and service
isolation together does not isolate affinity's causal effect. An8-worker
unbound control was substantially worse:1325.2ms/step and269ms exposed wait.

### Detailed Nsight Systems and PyTorch stage captures

Saved warmed, complete optimizer-step captures (four forward/backward pairs,
gradient norm, optimizer and probes), using Nsight Systems2025.5.2.266 from
CUDA13.1. CUDA/NVTX/OSRT tracing works without root; CPU IP/backtrace and context
switch sampling were disabled because this environment's perf permissions do
not allow them. Thus these captures do not diagnose CPU cache or branch misses.

Primary artifacts, all retained locally under ignored `build/`:

- `model72m_hybrid16_stages_20261014.nsys-rep`: default16 unbound workers.
- `model72m_affinity22_stages_20261014.nsys-rep`:22 pinned workers, service0–1.
- `model72m_affinity22_torch_stages_20261014.trace.json`: PyTorch CPU/CUDA trace,
  with input shapes and explicit microbatch/stage ranges.
- Both Nsight reports have matching `.sqlite`, run `.json`/`.log` and exported
  CUDA kernel/memory/API, NVTX and OSRT CSV tables. The cross-thread analyses are
  `model72m_{hybrid16,affinity22}_nsys_analysis_20261014.json` and
  `model72m_affinity22_torch_analysis_20261014.json`.

**Attribution caveat:** built-in NVTX GPU projection is same-thread and misses
most backward launches from the autograd thread. `tools/analyze_model_profile.py`
instead maps CUDA correlations to CPU launch timestamps inside the outer
forward/backward ranges, across threads. It counts only actual kernels/copies/
memsets, not GPU user annotations. CPU scope duration is not GPU duration:
gradient-norm CPU scopes span36–45ms waiting for queued backward, while their
own GPU work is only0.56ms.

The PyTorch capture provides the following mean **GPU kernel duration sums per
microbatch**, B16/N2048. These are attribution, not additive critical-path time;
they exclude launch gaps and can overlap other streams. FFN GEMMs are identified
by the actual2576-wide intermediate tensor shapes, unavailable in Nsight's
kernel-only summary.

| Component | Forward, ms | Backward, ms |
|---|---:|---:|
|Short convolution + projection|10.62|57.23|
|Linear vocabulary cross entropy|15.19|44.65|
|FFN GEMMs|12.21|23.06|
|GDN recurrence/preparation|4.43|8.52|
|Vocabulary interpolation/glue|2.08|7.54|
|Soft CUDA core|0.82|3.60|
|Hard CUDA forward|1.90|—|
|Hard event backward|—|3.39|
|Other GEMMs|1.90|4.59|
|Other pointwise/norm/reductions|15.38|33.32|
|Total kernel duration|64.51|185.90|

Mean first-to-last-kernel spans are84.49ms forward and187.72ms backward;
the profiled whole optimizer step is1112.85ms. Use the unprofiled windows above
for throughput, not this one-step instrumented measurement. Shortconv/projection
and CE together account for510.8ms of kernel work per optimizer step; complete
hard event backward accounts for13.6ms. Other pointwise work is also material:
FP32 add37.7ms, BF16 cast19.7ms, direct copy19.0ms and fused SwiGLU backward17.3ms
per step. These categories include work outside the DISM core.

Nsight corroborates both the GPU ranking and exposed planner tail:

| Capture | Step scope, ms | GPU activity union, ms | No recorded GPU activity, ms | Longest gap, ms |
|---|---:|---:|---:|---:|
|16 unbound|1148.41|1010.18|138.23|17.43|
|22 pinned + service isolation|1140.02|1033.34|106.69|1.63|

GPU activity union merges kernels, copies and memsets across streams; it is
**not SM occupancy**. The default capture has gaps at step offsets138.44,
419.71 and703.43ms lasting17.00,17.43 and9.71ms. Each overlaps
`planner/wait_and_event_backward` and `planner/native_plan_pack`; GPU activity
resumes as the coordinator finishes packing and starts H2D. These are directly
observed planner-tail bubbles, totaling44.14ms. They disappear in the22-worker
capture, but many small launch/synchronization gaps remain. The two captures'
GPU active times differ too; do not interpret their step-time difference as an
isolated affinity speedup. Across these captures, actual D2H copies total1.91–
1.94ms/step and H2D copies43.63–51.83ms/step, mostly overlapping computation.
The `planner/bulk_H2D` NVTX scope measures host enqueue time, not copy completion.

No production optimization or default change was made as part of profiling.
Priority remains padded shortconv/projection work and CE, followed by compact
planner metadata/tail scheduling. Further event-kernel tuning has much less
full-model headroom at this fixture.

### Full-model reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat" \
/home/cicuvc/miniconda3/envs/blkw/bin/python -u dism_v3/tools/bench_model_training.py \
  --checkpoint-dir /media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/base_checkpoints/seqtrue72m-fwedu-valres \
  --warmup 3 --phase-steps 3 --windows 3 --window-steps 10 --profile \
  --output dism_v3/build/model72m_hybrid_new.json
```

Use `--legacy-cuda dism_v3/build/lib.linux-x86_64-cpython-312/cu_flash_dism.cpython-312-x86_64-linux-gnu.so`
for the SHA-checked old-core comparison, `--no-training-probes` for the probe
control, or `--planner-workers N` for a process-local worker sweep. Every command
requires a fresh output filename. JSON includes the full model/config, optimizer
hyperparameters, document lengths, phase samples, window throughput, planner
diagnostics and optional profiler attribution; Chrome traces are separate files.
Primary reports: `build/model72m_{hybrid,hybrid_repeat,legacy,hybrid_noprobes}_20261014.json`.

For a bounded Nsight capture, use the same environment and checkpoint arguments
with `--warmup 3 --profile-only --nsys` (omit `--profile`), prefixed by:

```bash
/usr/local/cuda-13.1/nsight-systems-2025.5.2/bin/nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=false --output=dism_v3/build/model72m_new_capture \
  /home/cicuvc/miniconda3/envs/blkw/bin/python -u dism_v3/tools/bench_model_training.py \
  --checkpoint-dir /media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/base_checkpoints/seqtrue72m-fwedu-valres \
  --warmup 3 --profile-only --nsys \
  --output dism_v3/build/model72m_new_capture.json
```

Add `--planner-workers 22 --planner-cpus 2-23 --service-cpus 0-1` for the pinned
case. For PyTorch, run Python directly with `--profile --profile-only` instead
of `--nsys`. Export Nsight SQLite with `nsys stats` or `nsys export`, then use:

```bash
/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v3/tools/analyze_model_profile.py \
  dism_v3/build/model72m_new_capture.sqlite \
  --output dism_v3/build/model72m_new_capture_analysis.json
```

The same analyzer accepts a PyTorch `.trace.json`. Choose fresh output names;
the original reports above are intentionally retained for inspection.

### Existing linear + FLA frontend comparison

The same72M checkpoint and four resident batches were replayed with the existing
`conv_impl="linear_fla", conv_backend="cuda"` implementation. This is a scratch
benchmark conversion, **not a checkpoint migration or training restart**.
`tools/bench_conv_frontend.py` maps fused `[lag, channel]` filters to FLA
`[channel, 1, tap]`, reversing the lag axis, and applies the identical permutation
to Adam first/second moments. Projection weights, tied Q/K aliases, other
parameters, group ordering/hyperparameters, counters and RNG restore semantics
are retained. Source model/optimizer fingerprints are checked before conversion;
the rebuilt optimizer must exactly match the name-mapped schema. Backend fallback
is rejected. Original checkpoints are never written.

Numerical check: a real first-layer Q convolution from the checkpoint, seeded
inputs and packed lengths256/512/256, compared both BF16 frontends to an FP32
linear/depthwise-convolution/SiLU oracle. Fused forward/dX/dW/dConv relative L2
errors were .350/.425/.423/.343%; linear/FLA errors were .351/.385/.418/.348%,
with all cosines above .99999. This is an operator diagnostic, not a full-model
gradient equivalence claim. Whole-model first replay losses were2.80590 fused
and2.80616 linear/FLA, with gradient norms .2319/.2320. Different intermediate
rounding can change hard labels; bitwise equality is not expected. All measured
steps remained finite and all plan handles were released. Three CPU conversion
tests cover filter orientation, moment/step/group preservation and rejection of
unsupported conversion inputs. No production kernel or tolerance was changed.

All times below sum four B16/N2048 microbatches; sustained windows are three
sets of ten optimizer steps, with the same probes and scratch AdamW updates.
22-worker runs pin workers2–23 and service threads0–1.450W GPU power limit.

| Frontend / planner | Forward, ms | Backward, ms | Optimizer/probes, ms | Sustained step, ms | tokens/s | Peak allocated, GiB |
|---|---:|---:|---:|---:|---:|---:|
|Fused /22 pinned, fresh control|311.7|763.2|13.4|1089.5|120303|16.42|
|Linear/FLA /22 pinned|381.1|562.3|13.3|961.2|136361|17.50|
|Linear/FLA /16 unbound default|355.5|574.6|13.2|938.2|139700|17.44|

The matched22-worker comparison reduces sustained step time by11.8%, increasing
throughput13.3%, with about1.08GiB additional peak allocation. Window means span
1085.9–1092.0ms fused and954.9–968.6ms linear/FLA. The16-worker linear/FLA run spans
932.0–941.4ms: the earlier affinity choice is not automatically best after
changing the frontend. Do not change the default worker configuration from this
limited comparison. Explicit planner waits in the separate diagnostic step were
.085ms fused22, .086ms linear/FLA22 and3.13ms linear/FLA16. The16-worker CPU planning
sum remains397.9ms, mostly hidden by forward computation.

To separate frontend gains from the hard-backward change, the same scratch
linear/FLA conversion was also measured with the SHA-verified old CUDA core and
current interpolation. With unrestricted service affinity, old-core forward/
backward/optimizer totals were313.0/607.0/13.2ms and sustained step932.8ms
(928.4–936.7),140519 tokens/s. With service CPUs0–1, the old core measured934.6ms
(932.3–936.8). It has no CPU planner, so no worker pool is created. Corresponding
hybrid values are938.2ms unbound and961.2ms pinned. Thus **linear/FLA's substantial
frontend improvement must not be attributed to the event backward**: a stable
end-to-end core-path gain is still not demonstrated, even with the frontend
normally used for training. The default unbound difference is only0.6%, within
the observed window/run variability; the pinned hybrid is2.8% slower in this
control. No production dispatch was reverted or retuned.

The existing linear/FLA path reuses the shared DISM/GDN Q/K projection in hybrid
layers; the fused path shares weights but repeats projection computation. Thus
this comparison includes that existing reuse, not just a kernel implementation
swap. The isolated frontend's memory and intermediate-rounding tradeoffs remain.

The linear/FLA PyTorch trace contains **no** `fwd_short_conv_silu`,
`bwd_short_conv_silu` or `bwd_mma_gemm` launches. FLA depthwise convolution totals
17.52ms/step (4.51 forward,13.01 backward); the linear projection GEMMs are counted
separately, not included in that17.52ms. Non-FFN GEMMs increase from25.96 to51.84ms
and include more than just these projections. CE remains246.27ms, FFN GEMMs141.02ms,
GDN recurrence/preparation51.83ms, vocabulary interpolation/glue38.18ms and hard
event backward13.54ms. The previous271ms fused-shortconv category is no longer
the dominant training path to optimize.

Forward latency increases despite lower GPU work. The linear/FLA trace has1086
`aten::_local_scalar_dense` calls and1118 stream synchronizations, versus113 and145
in the earlier fused trace. The extra973 scalar reads equal
`7 * (34+30+32+43)` documents. In the installed FLA implementation,
`prepare_sequence_ids` calls `prepare_position_ids`, whose no-CPU-metadata branch
uses one `torch.arange(n)` per GPU length scalar. Trace nesting places834 of the
extra reads inside the six hybrid-layer compiled regions and139 in the outer
forward scopes, consistent with per-boundary-object sequence-ID construction.
Reusing packed metadata is therefore a concrete follow-up target; this task does
not change FLA or production callers. CPU sync durations include queued GPU work
and are **not** directly recoverable overhead.

PyTorch instrumentation is materially intrusive on this frontend: captured step
1085.27ms versus unprofiled961.22ms. Its union of GPU kernels/copies/memsets is
824.77ms with260.50ms of no recorded activity; do not present those gaps as the
unprofiled idle budget. Use warmed window timing for throughput.

A separate default16-worker Nsight capture is retained as
`build/model72m_linfla16_stages_20261014.nsys-rep`, with matching SQLite, CSV,
run JSON/log and `model72m_linfla16_nsys_analysis_20261014.json`. Its instrumented
step is1046.35ms, GPU activity union815.78ms and no-activity time230.57ms;
the longest gap is1.20ms, with no long planner-tail bubbles in this capture.
Planner native-plan/pack wall-time scopes sum422.88ms, mostly overlapping GPU
work. CE kernels total239.30ms and depthwise convolution17.72ms. Nsight is also
intrusive here; neither its step duration nor summed idle is the unprofiled
938.24ms throughput result.

Artifacts: `build/model72m_fused22_control_20261014.json`,
`build/model72m_linfla{16,22}_20261014.json`,
`build/model72m_linfla{16,22}_legacy_20261014.json`,
`build/model72m_linfla22_20261014.trace.json` and
`build/model72m_linfla22_torch_analysis_20261014.json`, plus matching logs.
To reproduce, add `--conv-impl linear_fla --validate-conv` to the full-model
command above. Conversion and numerical diagnostics are recorded in the JSON;
neither flag changes the production model defaults.

### Standalone current linear-CE baseline

`tools/bench_cross_entropy.py` captures the LM-head input of one eager/no-grad
linear/FLA forward from the same checkpoint's first prefetched batch. It then
releases the model and measures CE alone: BF16 hidden `[1,32768,384]`, BF16 head
`[32000,384]`, softcap30, ignore_index=-1. There are3801 ignored labels and28967
valid tokens across34 documents. No optimizer is constructed, no checkpoint is
written and no dataloader is resumed. The first-batch CE loss is not the prior
full-model benchmark's reported fourth-microbatch loss.

RTX5090 at the unchanged450W limit, conda blkw,2026-10-14. Five warmups, five
samples of ten calls; table reports median CUDA-event intervals. Direct helper
forward/backward functions are compiled fullgraph to avoid eager diagnostic
label/boundary synchronizations. The public autograd column separately measures
the actual compiled loss plus `autograd.grad`, including the BF16 gradient
boundary. CUDA graphs are disabled. Inputs remain resident and repeatedly
reused: this is a warm standalone benchmark, not the full-model schedule with
planner traffic and other kernels. Do not multiply it by four and substitute
it for the observed full-model CE attribution.

| Existing entry | Forward, ms | Backward, ms | Combined helpers, ms | Public autograd, ms |
|---|---:|---:|---:|---:|
|Token mean, current nanochat training path|11.156|27.971|39.497|39.307|
|`cu_seqlens` document mean|11.540|28.209|40.025|40.995|

Public-autograd sample ranges were39.09–40.69ms and39.75–41.67ms respectively;
the small difference between paths should not be read as a robust speed ratio.
Peak additional allocated memory was134.97/135.09MiB above live inputs and saved
diagnostic tensors, excluding allocator reservations. Backward uses1048 rows
per dlogits chunk (about64MiB),32 chunks, three launches per chunk. Separate
token-mean profiling gives11.678ms linear-CE forward, .056ms mean reduction,
11.031ms dlogits,8.442ms dHidden and7.556ms dWeight per invocation. These kernel
duration sums are instrumentation-based attribution, not the unprofiled medians.

Both APIs live in `python/flash_dism/kernels/fused_cross_entropy.py`. The varlen
entry uses the **same** `_linear_cross_entropy_fwd` and three backward kernels.
It replaces the token reduction with `_document_loss_reduce` and
`_document_mean_reduce`, and supplies per-token inverse-document-valid-count
scales to dlogits. It averages over nonempty-valid documents; ignored tokens and
documents with no valid targets do not contribute. On this fixture token mean
is2.704145, whereas document mean is2.644696. Therefore passing `cu_seqlens` to
the current training call is a loss-reweighting change, not a performance-only
replacement. Per the training objective clarified after this measurement,
document-balanced is the intended weighting and the existing token-mean
training call is incorrect. This benchmark does not change that call. A future
correction must also account for unequal document counts across accumulation
microbatches/ranks; averaging their document means equally is not the global
document mean. This is separate from sequence-level hard sampling, whose unit
remains a packed training row, not a document.

A separate forward-only sensitivity probe varies **only** the benchmark launch's
BLOCK_M; production source/defaults are untouched. BLOCK_V64, BLOCK_K32,
four warps and three stages stay fixed. This directly times the row kernel,
without the final loss reduction:

| Tokens per CTA | Median, ms | Registers/thread | Compiler spills | Dynamic shared bytes |
|---|---:|---:|---:|---:|
|16, original|11.718|63|0|10240|
|32|9.104|86|0|12288|
|64|5.981|128|0|16384|
|128|5.953|252|0|24576|

BM64 is about1.96x faster than BM16 in this probe; BM128 adds little speed but
nearly doubles register use. This supports investigating larger token tiles,
not automatically promoting a new launch default. All variants retain the500
serial vocabulary iterations. LSE/NLL max absolute difference from BM16 is0
for BM32 and1.91e-6 for BM64/128. Cubin disassembly found no CALL/LDL/STL;
stack/local resources are zero. Reported dynamic shared memory comes from Triton
metadata, not cuobjdump's1024-byte static shared resource. This probe covers the
real aligned fixture, not all shapes/tails or an autotuning search. No sanitizer
campaign was run; no production kernel math, tolerance or launch changed.

The existing CE suite passed10 tests and12 subtests, including ignored labels,
softcap and packed loss/gradient checks. Both timed public autograd paths also
matched the direct helpers exactly after the existing BF16 output casts, with
finite losses and gradients. Generated JSON, cubins, PTX and SASS are retained
under ignored `build/ce72m_standalone_20261014*`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat" \
/home/cicuvc/miniconda3/envs/blkw/bin/python -u dism_v3/tools/bench_cross_entropy.py \
  --checkpoint-dir /media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/base_checkpoints/seqtrue72m-fwedu-valres \
  --warmup 5 --repeats 10 --samples 5 --profile --probe-bm 16 32 64 128 \
  --output dism_v3/build/ce72m_standalone_new.json
```

### Stock FLA fused-linear CE comparison

`tools/bench_fla_cross_entropy.py` reuses the exact read-only fixture and timing
helpers above:32768 tokens, width384, vocabulary32000, BF16, ignore_index=-1,
five warmups and five samples of ten calls, RTX5090 at450W. It measures the
installed `fla.modules.fused_linear_cross_entropy.FusedLinearCrossEntropyLoss`,
not the logits-only `FusedCrossEntropyLoss`. Installed-source paths and hashes
are recorded in the result. No production kernels, training configuration or
checkpoint state changed.

The stock FLA fused-linear interface has **no softcap and no per-token/document
weights**. Its loss reduction is mean/sum, not unreduced. Its forward computes
dx/dw immediately and saves them; backward only returns/scales them. Consequently
compare the complete public-autograd pair, not forward/backward columns in
isolation. FLA's class is compiler-disabled and runs with its natural eager
boundary, including the valid-token `.item()` and backward upstream-scalar host
synchronizations. Local public entries are fullgraph compiled as before, without
CUDA graphs. Local backward-only timing uses the prepared direct helper because
compiled autograd with donated buffers disallows repeated retained backward.
Direct local gradients match public gradients exactly after existing BF16 casts.

| Entry | Training forward, ms | Saved-state backward, ms | Complete autograd, ms | Peak extra allocated, MiB |
|---|---:|---:|---:|---:|
|Local, no softcap, token mean|10.943|27.890|39.018|134.97|
|Local, softcap30, token mean|11.206|28.209|40.147|134.97|
|Local, softcap30, document mean|11.448|28.785|40.162|135.09|
|Stock FLA,8 chunks/default, no softcap, token mean|16.795|0.359|17.053|571.02|
|Stock FLA,32 chunks, no softcap, token mean|16.506|0.289|16.843|196.00|

Complete FLA event sample ranges are16.735–17.072ms and16.814–17.238ms; host-wall
medians17.055/16.845ms. The difference between8/32 chunks is small and their
ranges overlap, while both are about2.3x faster than the matching no-softcap
local control. Upstream gradient1/4 (fresh graph each call, as FLA mutates saved
gradients during scaling) gives17.044/16.941ms. Scaled gradients exactly equal
the unit-upstream outputs divided by four. This does not test global
document-count normalization across accumulation/ranks.

FLA's actual chunks have4096/1024 rows, with250/62.5MiB BF16 logits buffers.
Measured peak allocation exceeds one buffer: eager replacement can transiently
retain both old/new chunk logits, in addition to gradients and temporaries.
Memory is peak incremental allocated above resident inputs/diagnostic state,
not allocator reservation or an end-to-end model-memory claim.

Separate profiling attributes default8-chunk FLA to12.338ms across the three
GEMM families,2.305ms CE/dlogits,1.206ms LSE, and approximately0.629ms other
device work (total16.478ms). The32-chunk run gives12.477ms GEMMs,0.790ms
CE/dlogits,0.696ms LSE and2.239ms other device work (total16.201ms); notably
FP32 dWeight addition grows from0.408 to1.508ms. These are intrusive device
duration sums, not a decomposition of the unprofiled median. The route uses
chunked library GEMMs and overwrites each chunk's logits with dlogits, avoiding
the local implementation's separate backward logits recomputation. This supports
investigating GEMM-based chunked CE with softcap/document weights, rather than
only increasing the current row-kernel BLOCK_M. It is not yet a benchmark of
that semantically complete implementation.

No-softcap losses are2.730832815 local versus2.730943203 FLA (absolute
difference1.104e-4). FLA8 dx/dw relative-L2 differences are0.742%/0.564%, with
cosines0.999973/0.999984; FLA32 gives0.711%/0.574%, cosines0.999975/0.999984.
All outputs are finite. These are comparisons to the existing local kernel,
not a new high-precision oracle or a claim of bitwise equivalence: FLA
materializes BF16 logits and BF16 chunk dWeight matmul outputs before FP32
addition. No tolerance/default was changed and no new sanitizer campaign was
run for unmodified installed kernels.

Results and per-kernel attribution are retained in ignored
`build/ce72m_fla_retry_20261014.json` and matching log. The earlier
`ce72m_fla_20261014` files record an incomplete harness attempt rejected by
PyTorch's donated-buffer/retain_graph check; they are not performance results.
No new Nsight capture was needed for this standalone comparison; earlier
full-model `.nsys-rep` files remain intact.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat" \
/home/cicuvc/miniconda3/envs/blkw/bin/python -u dism_v3/tools/bench_fla_cross_entropy.py \
  --checkpoint-dir /media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/base_checkpoints/seqtrue72m-fwedu-valres \
  --warmup 5 --repeats 10 --samples 5 --num-chunks 8 32 --profile \
  --output dism_v3/build/ce72m_fla_new.json
```

### Chunked fused linear CE with softcap and document balancing

`python/flash_dism/kernels/fused_ce_chunked.py` is the new, independent entry
required by `docs/FUSED_CE_TASK.md`; `fused_cross_entropy.py` stays in place as
the control and none of its launch defaults changed. The route is ported from
FLA's chunked fused-linear CE (`fla/modules/fused_linear_cross_entropy.py`, MIT
License, Copyright (c) 2023-2025 Songlin Yang, Yu Zhang, itself adapted from
Liger-Kernel). No FLA source is vendored and no FLA private helper is imported;
the inherited ideas are the per-chunk library-GEMM logits, the in-place
logits->dlogits reuse and the chunked FP32 dWeight accumulation. Ignored rows
are not short-circuited: their logits are already materialised, so only the loss
reduction counts them.

Per token chunk the operator (1) computes the chunk logits with one library
GEMM into one reusable `[chunk_rows, vocab]` buffer, never the full
`[tokens, vocab]` logits, (2) runs `_chunk_partial_stats` for the capped
per-(row, vocab-block) max, shifted sum-exp and target logit, (3)
`_chunk_row_stats` merges those partials into one FP32 LSE and NLL per row, (4)
`_chunk_dlogits` overwrites the chunk buffer in place with the softcap-corrected
softmax gradient times the per-token reduction weight, and (5) two library GEMMs
produce dHidden and a dWeight chunk partial that `_accumulate_fp32` adds into
one FP32 `[vocab, channels]` accumulator. Softcap uses the same capped logits in
the LSE, the target logit and the gradient, whose backward factor is
`1 - tanh(z / c) ** 2`; it is applied inside the chunk kernels, so no extra pass
and no saved tanh tensor is needed. The upstream gradient is applied to the
saved gradients at backward time, so any upstream value and repeated backward
passes stay correct - the FLA in-place rescue of saved gradients is deliberately
not reproduced.

Counting and reduction stay on the device: a per-token weight array is built
from `_valid_token_count` (valid-token mean) or from `_document_valid_counts` +
`_nonzero_count` + `_document_scale` (`1 / (D * n_d)` over documents with at
least one valid label), and `_token_nll_sum` / `_document_nll_mean` +
`_document_mean_sum` produce the numerator and the count. No per-chunk `.item()`,
D2H copy or host-device synchronisation exists in either direction of the
operator. `chunked_ce_forward` returns `(loss, mean_sum, count)` with `loss =
mean_sum / count`, i.e. the sum of document means over the number of documents
that carry at least one valid label; `normalize=False` returns the raw sum of
document means and the same count so a caller can divide by a globally gathered
document count instead. With `cu_seqlens=None` the same interface reports the
valid-token sum and count. Document boundaries are independent of hard sampling,
which remains packed-row level.

dtypes and rounding, as measured rather than assumed:

| Tensor | dtype | rounding |
|---|---|---|
| chunk logits / dlogits | `logits_dtype` (BF16 default, FP32 opt-in) | BF16 output is one rounding of an FP32 GEMM accumulation |
| dHidden | library GEMM in `logits_dtype`, saved and cast to the input dtype after the upstream multiply | one BF16 rounding of the GEMM output |
| dWeight chunk partial | library GEMM in `logits_dtype` | a BF16 chunk partial added into an FP32 buffer is *not* an FP32 GEMM output |
| dWeight accumulator / `mean_sum` / LSE / NLL / weights | FP32 | FP32 accumulation across chunks and reductions |
| returned gradients | hidden / weight input dtype | one final cast after the upstream multiply |

A BF16 chunk partial into an FP32 accumulator therefore costs one extra BF16
rounding relative to an FP32 GEMM output; the FP32 `logits_dtype` mode removes
that rounding (at roughly 3.7x the runtime) and is the way to separate the
arithmetic from the chunk-buffer rounding.

RTX5090 at the unchanged450W limit, conda blkw,2026-10-14. Same read-only
fixture and method as the sections above: 32768 tokens, width384,
vocabulary32000, BF16,34 documents,3801 ignored labels, five warmups and five
samples of ten calls, CUDA-event and host-wall timing, CUDA graphs disabled,
complete public autograd as the comparison unit (each call therefore includes
the chunked forward that prepares dHidden/dWeight). Every row below was
re-measured in one session, including the legacy control and stock FLA.
Host-wall medians agreed with the CUDA-event medians to within0.01ms in every
row (both are kept in the result JSON). The chunked and legacy rows contain no
host-device synchronisation inside the timed region; the stock FLA row keeps its
own valid-token `.item()` forward synchronisation, as described above.

| Entry | Complete autograd, ms | Upstream 1/4, ms | Eager autograd, ms | Peak extra allocated, MiB |
|---|---:|---:|---:|---:|
|Legacy control, no softcap, token mean|39.21|38.69|38.82|135.0|
|Legacy control, softcap30, document mean|39.00|39.42|39.38|135.1|
|Stock FLA,8 chunks, no softcap, token mean|17.27|17.29|n/a|571.0|
|Stock FLA,32 chunks, no softcap, token mean|16.90|16.93|n/a|196.0|
|Chunked, no softcap, token mean, chunk1024|16.67|16.42|16.78|159.2|
|Chunked, softcap30, document mean, chunk1024|16.91|16.96|17.28|159.2|
|Chunked, no softcap, token mean, chunk2048|17.98|18.10|18.30|342.1|
|Chunked, softcap30, document mean, chunk2048|18.31|18.36|18.60|342.1|
|Chunked, no softcap, token mean, chunk4096|17.43|17.56|17.67|353.3|
|Chunked, softcap30, document mean, chunk4096|17.89|17.76|18.03|353.3|

A second run of the same scan reproduced the chunk1024 rows within0.4ms
(16.41/17.03ms and 16.32/17.00ms for the two runs of the same command), so the
chunk-size ordering is a real effect: a1024-row buffer is about64MiB, which
stays in L2 across the two CE passes (0.85-0.94ms of dlogits work per run
instead of2.64ms at4096 rows), while the three GEMMs stay at12.5-13.5ms either
way. Peak extra allocation grows with the chunk and with the number of retained
chunk partials, and stays below the stock FLA 8-chunk figure. `chunk_rows=None`
now means the64MiB budget rounded down to a power of two (1024 rows here); no
production caller was switched to this entry.

Correctness against the existing local control on the same fixture. The numbers
compare the chunked operator's internal pre-upstream tensors
(`versus_local_control_internal`) with the control's public gradients, so the
final input-dtype cast of the returned gradients does not hide the arithmetic;
the public comparison is reported alongside in the result JSON:

| Case | Loss | dHidden relative L2 / cosine | dWeight relative L2 / cosine |
|---|---:|---:|---:|
|Chunked BF16, no softcap, token mean|2.730943 vs2.730833|0.694% /0.999976|0.532% /0.999984|
|Chunked BF16, softcap30, document mean|2.644690 vs2.644696|0.573% /0.999984|0.537% /0.999985|
|Chunked FP32 buffer, no softcap, token mean|2.730832 vs2.730833|0.269% /1.000000|0.221% /1.000000|
|Chunked FP32 buffer, softcap30, document mean|2.644696 vs2.644696|0.247% /1.000000|0.228% /1.000000|

Norm ratios stay within0.04% of1 and every output is finite. The BF16 numbers
carry one rounding for logits, one for dlogits and one for each GEMM output, and
land next to the stock FLA figures measured against the same control (dx0.71
to0.74%, dw0.56 to0.57%). The FP32-buffer rows isolate the remaining error: the
0.22 to0.27% left is the local control's own BF16 dlogits/dHidden rounding,
while the small-shape FP32 oracle tests put the fixture's BF16 *input*
quantization at about0.18% and attribute nothing else to the operator. The loss
agrees to1.1e-4 (no softcap) and6.4e-6 (softcap30, document mean), and the
normalizer identities are visible:28967 valid tokens,34 valid documents, sum of
document means89.919464 = 34 x2.644690. Upstream1/4 reproduces exactly0.25x the
unit-upstream gradients (max absolute0, relative L20) for every case, because
the scaling happens after the saved gradients.

Separate device-duration attribution (intrusive profiler, three iterations, not
a decomposition of the unprofiled medians):

| Chunk1024 case | GEMMs, ms | dlogits, ms | partial stats, ms | row merge, ms | FP32 dWeight add, ms | Device total, ms |
|---|---:|---:|---:|---:|---:|---:|
|No softcap, token mean|12.71|0.85|0.57|0.18|1.45|16.00|
|Softcap30, document mean|13.54|0.94|0.94|0.18|1.66|17.47|

At chunk4096 the same attribution is12.46-12.71ms of GEMMs,2.64ms of dlogits,
1.50-1.54ms of partial stats,0.14ms of row merge and0.37ms of FP32 dWeight
addition. The FP32 accumulator read-modify-write is the price of one explicit
FP32 `[vocab, channels]` accumulator per chunk (about122MiB of traffic per
chunk); letting the compiler fuse the `add_` chain measured about1ms faster at
32 chunks but retained one BF16 `[vocab, channels]` partial per chunk, i.e.
841MiB instead of159MiB of peak extra allocation, so the explicit kernel was
kept. Chunk sizes are ordinary per-call arguments; the benchmark only passes
them, never the library defaults.

Kernel hygiene for the three new Triton kernels at the default block sizes:
`_chunk_partial_stats` 79 registers, 0 spills,128B shared;
`_chunk_row_stats` 128 registers, 0 spills,128B shared; `_chunk_dlogits` 95
registers, 0 spills,0B shared. Cubin SASS contains no `CALL`, `LDL` or `STL`
instruction in any of them. `compute-sanitizer` memcheck (0 errors),
synccheck (0 errors) and racecheck (0 hazards) all ran clean over a scenario
that covers softcap, document balancing, zero-length and single-token
documents, fully ignored documents, chunk sizes3/7/23/64 and both reduction
modes. The sanitizer scenario is small: it does not cover the full-fixture
shapes, the multi-chunk tail or the compiled path, and no FP32-atomic,
shared-memory-atomic or long-running-soak check was added, because the operator
uses neither atomics nor a shared FP32 atomicAdd.

`test_fused_ce_chunked.py` passes15 tests and18 subtests: FP32 explicit
references for both reductions with chunk sizes3/7/64, empty and single-token
documents, softcap30/1.0/0.25, exactly saturated and partly saturated logits,
documents crossing chunk boundaries with a fully ignored document, the
microbatch split with unequal document counts (the naive average of microbatch
document means is shown to differ, the reported sum/count pair is shown to
match a single whole-batch call), upstream1/0.37/0.25 linearity, repeated
backward over one graph, all-ignored zero gradients with no NaN, BF16 and
FP32-buffer tolerances, `torch.compile` fullgraph equality with eager, int32
labels and the validation contract. The existing `test_fused_cross_entropy.py`
still passes its10 tests and12 subtests unchanged.

Limits and integration notes. The BF16 default loses about0.5 to0.7% of gradient
relative L2 against an FP32-internal implementation; that is inherent to a
BF16 logits/dlogits chunk buffer and is quantified above rather than hidden. The
document-balanced loss is exact for the local packed rows, but the returned
loss is normalised by the *local* valid-document count. Accumulating losses
across microbatches or ranks must not average those per-microbatch means:

```
L_step = sum over ranks and microbatches of (sum of document means)
         / (total number of documents with at least one valid label)
```

`chunked_ce_forward(..., normalize=False)` plus `mean_sum`/`count` is exactly
the pair needed for that reduction, and for a single-pass gradient the
microbatch loss can be pre-scaled by `1 / total_valid_documents` before
backward, because the per-token weight then only carries `1 / n_d`. The current
training loop was deliberately not modified: the distributed gradient average
happens in the optimizer, so any switch to document-balanced weighting has to
be checked there separately. Dataloader, RNG counters, optimizer schema and
checkpoints are untouched.

Artifacts: results, per-kernel attribution and kernel resources in ignored
`build/ce72m_chunked_final_20261014.json`, accuracy isolation in
`build/ce72m_chunked_accuracy_20261014.json`, the in-session controls in
`build/ce72m_fla_20261014_resession.json`, sanitizer logs in
`build/ce72m_chunked_{memcheck,synccheck,racecheck}_20261014.log`, kernel
cubins/PTX/SASS next to the result JSON.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat" \
/home/cicuvc/miniconda3/envs/blkw/bin/python -u dism_v3/tools/bench_ce_chunked.py \
  --checkpoint-dir /media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/base_checkpoints/seqtrue72m-fwedu-valres \
  --warmup 5 --repeats 10 --samples 5 --chunk-rows 1024 2048 4096 --kernel-probe \
  --output dism_v3/build/ce72m_chunked_final_20261014.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat" \
/home/cicuvc/miniconda3/envs/blkw/bin/python -u dism_v3/tools/bench_ce_chunked.py \
  --checkpoint-dir /media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/base_checkpoints/seqtrue72m-fwedu-valres \
  --warmup 5 --repeats 10 --samples 5 --chunk-rows 1024 --no-profile --fp32-buffer \
  --output dism_v3/build/ce72m_chunked_accuracy_20261014.json

PYTHONPATH="$PWD/dism_v3/python" /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q \
  dism_v3/python/flash_dism/kernels/test_fused_ce_chunked.py
```

## RTX PRO 6000 deployment and microbatch sizing (2026-10-16)

Purpose: port the current tree to the RTX PRO 6000 Blackwell Server Edition box
and measure real training throughput with the planner disabled
(`DISM_HARD_PLANNER=0`), i.e. the n^2 `backward_hard_recompute` path for every
hard row.

Target: `RTX PRO 6000 Blackwell Server Edition`, 188 SMs / sm_120, 97887 MiB,
driver 595.71.05, 600 W cap. Host is a 2x Xeon Platinum 8470Q (2.1 GHz base,
208 threads) with 1007 GB RAM. torch 2.12.1+cu130, CUDA 13.0, clang 23
(`~/dism125m-2500m/toolchain/bin/clang`), 92 local fwedu-100B shards. Same
`archs=['120a']` build as the 5090; `cu_flash_dism` builds unchanged.

Idle-box capability probes (so GPU limits are not confused with host limits):
bf16 8192^3 matmul 426 TFLOP/s, 1 GiB device copy 1463 GB/s. The local 5090 box
measures 419 TFLOP/s-class peaks with 1.79 TB/s HBM, so the two GPUs are close
in compute and the PRO 6000's 96 GB is the real difference.

### Microbatch sizing is the dominant throughput knob

The step is **host-launch bound**, not GPU bound. Instrumentation on a
checkpoint replay at the trained operating point (hard_prob 0.95) reports
`Self CPU 1341 ms` against `Self CUDA 754 ms` for a 796 ms step, and a live
`nvidia-smi` sample of a dbs16 run oscillates 16-100% (mean ~65%) at ~300 W of
a 600 W cap.

Launch count, not bytes, drives that: the step issues ~3485 CUDA launches and
~10-15k aten dispatches. Launch overhead is per-op, so the same work in more,
larger microbatches amortizes it away. Holding total batch at 131072 tokens and
varying only `--device-batch-size`:

| device-batch | accumulation | ms/step | tok/s | MFU | peak alloc | sampled GPU util |
|---|---|---|---|---|---|---|
| 16 | 4 | 779.5 | 168k | 38 | 16.8 GB | ~65% (16-100), ~300 W |
| 32 | 2 | 644.7 | 203k | 42 | 31.9 GB | - |
| 64 | 1 | **569.0** | **230k** | **46.9** | 61.3 GB | ~90% (74-100), ~400 W |

All three runs use the same spec, RNG seed, 92-shard dataloader and
`--total-batch-size 131072`; only the accumulation split differs. Loss at the
same step agrees within 0.03 (5.422 / 5.455 / 5.432 at step 86), which is
packing-order noise, so the objective is unchanged.

`--device-batch-size 64` is the largest legal value for a 131072-token total
batch and needs 61.3 GB, so it fits the PRO 6000 and does not fit the 32 GB
5090. For reference the local 5090 at dbs16 runs 665 ms / 197k tok/s / MFU 44;
the PRO 6000 is ~28% slower at the same dbs16 and ~17% faster than that once
microbatch sizing is corrected. The gap at equal dbs16 is host, not device:
single-thread CPU is ~1.5x slower on the Xeon (3e6-iteration float loop 0.188 s
vs 0.124 s) and the replay harness pins itself to one thread.

### Negative result: vectorizing the packed fallback gather

`backward_hard_recompute` runs **packed** on every DISM layer (2880 calls /
120 steps = 24/step = 6 layers x 4 microbatches, ~35 documents per call) and
costs ~75 ms/step of host time. Its packed branch regrouped documents by length
and then built each group with a Python loop of `x[0,s:s+length]` slices plus
`torch.stack`, i.e. ~210 tiny aten ops per call. Replacing that with one
`index_select` per group (plus a single `index_copy_` scatter) was correct -
`test_hard_ready_dispatch.py` 14/14 including the FP64 packed oracle - and
measured **3x faster in isolation** (0.236 vs 0.771 ms/call idle), but in the
trainer it was **94 ms/step slower** (873.7 vs 779.5 ms at identical settings).

The isolated benchmark only reproduces the in-situ cost once a deep queue is
simulated, where both variants converge (1.5 vs 1.3 ms/call), and the
vectorized form then adds per-call index construction, a non-contiguous label
layout that `run()` re-`contiguous()`s, and an extra dtype cast per scatter.
The change was reverted; the per-document loop stays. Conclusion: this call
site must be judged by an in-trainer A/B, not by micro-benchmarks.

### Harness notes

- Conda ships libstdc++ 6.0.29 while the JIT planner extension needs
  `GLIBCXX_3.4.30`. `LD_LIBRARY_PATH` does **not** fix it because
  `/root/miniconda3/bin/python3.12` carries `DT_RPATH=$ORIGIN/../lib`, which is
  searched before `LD_LIBRARY_PATH` for every dependency. Replacing
  `/root/miniconda3/lib/libstdc++.so.6.0.29` with the system 6.0.30 is the fix.
- `bench_model_training.py`'s diagnostic hook dereferenced `HardPlanTicket.future`,
  which is `None` for `DisabledHardPlanner` tickets; it now records those as
  `route='quadratic', plan_unavailable=False`.
- `nanochat/common.py` gained `rtx pro 6000` entries (231.7e12 bf16 peak from
  the 188/170 SM ratio, 1.6e12 HBM3e) so MFU is reported instead of 0%.

```bash
# build (remote, clang toolchain + bound parallelism)
cd /root/dism-core/dism_v3 && \
  PATH=/root/dism125m-2500m/toolchain/bin:/usr/local/cuda/bin:/root/miniconda3/bin:/usr/bin:/bin \
  DISM_BUILD_JOBS=4 python build.py

# trained-operating-point replay (planner off => n^2 fallback), single-threaded
PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat" FLA_FLASH_KDA=0 DISM_HARD_PLANNER=0 \
  python dism_v3/tools/bench_model_training.py --checkpoint-dir /root/ckpt72m \
    --step 19074 --hard-backward-policy quadratic \
    --output dism_v3/build/remote_pro6000_g1.json

# live throughput sweep: only --device-batch-size changes
--total-batch-size 131072 --device-batch-size {16,32,64}
```
