# Packed training CUDA Graphs

`scripts.base_train --cuda-graph` enables explicit microbatch forward/backward
capture for the packed DISM model with `DISM_HARD_PLANNER=0`. The selected
FFN-only MXFP8 model uses its existing fused AdamW optimizer. Without the flag,
training continues to use `torch.compile`; GDN remains outside Dynamo.

Layer order comes from the ModelSpec. The current 123M config explicitly uses
`gggddddddggg` (`g` = GDN, `d` = GDN+DISM). Earlier measurements below used the
original `ggggggdddddd`; reproduce them with a copy of the config omitting
`layer_layout`. The final section records the reordered model's measurements.

The graph returns the document-loss numerator and valid-document count and
accumulates gradients divided by the existing accumulation count. Document
normalization, clipping, learning-rate updates, optimizer/FP8 updates, evaluation,
logging and data loading remain outside the graph. No running training process
is automatically converted or restarted.

## Changing packed boundaries

`nanochat.cuda_graph.PackedTrainingGraph` owns a fixed input shape `[B,L]`.
Every replay accepts changed `cu_seqlens`, including changed document counts
and reused loader storage. Before replay it reads and validates the boundaries:
monotonic, endpoints `0` and `B*L`, 256-aligned intervals bounded by `L`, and
no document crossing an original training-row boundary.

Empty intervals are compacted and unused metadata slots contain empty intervals.
Tokens are neither padded nor moved. Empty documents contribute zero to the loss
denominator; hard sampling remains per original training row/head. Native
document/task records and FLA chunk indices are refreshed in persistent GPU
storage before replay, on the same stream.

For aligned intervals, soft/hard tile and chunk-scan task counts depend on total
tokens, not cut positions. Tau-reduction records have a fixed capacity with
unused entries skipped. Checkpoint capacities per head are
`T*((L-1)//32)`, `T*((L-1)//16)` and `T*(L//32)`. Exact document-local offsets
and kernel math remain unchanged. Memory usage can exceed ordinary exact-sized
ragged execution, especially for packs containing many short documents.

The native CPU table's Storage owns its descriptors and metadata builders;
there is no global pointer registry. `make_graph_layout`/`update_graph_layout`
are runtime APIs, not checkpoint formats. Cloning or serializing the tensor
does not preserve its graph-plan ownership.

`flash_dism.graph_gdn` calls the existing FLA forward/backward kernels with
explicit chunk indices. A fresh boundary tensor at capture prevents FLA's
identity cache from returning offsets computed in warmup; GPU offset operations
then replay using the changed contents. The installed FLA CPU index builder
mishandles internal empty documents, so empty-interval comparisons use the
semantically equivalent compact eager pack.

## State and lifecycle

Warmup restores the DISM RNG counter, CPU/CUDA RNG states and existing gradients. Actual replay
increments the counter once, reads the current GPU hard-probability buffer,
and uses current parameters/FP8 weights. Runtime MXFP8 plans are rebuilt on the
capture stream to release stale autograd nodes; referenced workspaces remain
owned across intervening eager evaluation.

Graph resources are rebuilt after checkpoint loading. Model keys, aliases,
optimizer grouping, counters and annealing buffers remain unchanged. Graph
construction never calls the dataloader; the consumed-position/next-batch
checkpoint protocol stays in the training loop. FusedAdamWMXFP8 explicitly
preserves BF16 moments on load, overriding Torch's default conversion to each
parameter's dtype without resetting any saved values or steps.

The runner is sequential, not reentrant. Construct it with no earlier backward
pending. Metadata staging, capture and replay use its owning stream; the caller
stream waits before consuming results. Returned loss/count tensors must be
consumed before the next replay. Both gradient zeroing modes are supported.

Changed batch/row shape, parameter or model-state buffer storage, FP8 weight storage, host temperature
or training mode requires a new graph. Changing host tau-scale schedules and
FP16 GradScaler are currently rejected. The selected configuration uses constant
host tau scale and BF16. Fused MXFP8 AdamW updates quantized weights in place;
optimizers replacing their storage require recapture.

## Explicit planner support

Add `--cuda-graph-planner` to `scripts.base_train --cuda-graph`, or construct
`PackedTrainingGraph(..., planner=True)`. This is an independent opt-in: the
default remains planner-disabled, and setting `DISM_HARD_PLANNER=1` alone does
not authorize graph capture. The graph planner supports one rank/GPU, BF16,
fixed true direction, original-row hard sampling, compact C64, R32/DV64 and
D32/64. It requires `DISM_HARD_BACKWARD=ready` and `DISM_PLANNER_COMPACT=1`
(both defaults); classic metadata and forced event/quadratic policy are rejected.
MXFP8 FFN and its existing eager optimizer are supported.

Each DISM layer owns a binding/slot. Captured D2H and an external event notify
its CPU worker, which prepares and uploads a fresh plan before publishing the
epoch with `cuStreamWriteValue32`. A GPU sampler makes one choice shared by the
event and quadratic backward nodes. Late publication cannot change that choice.
See [the protocol, ownership rules and validation](PLANNER_CUDA_GRAPHS.md).

`forward_backward` retires the previous generation before staging another
microbatch. It returns while CPU planning/upload may still be pending. **Call
`drain()` before document normalization, clipping, optimizer, evaluation or
checkpoint**, and `close()` on exit, including exceptions. A context manager
closes automatically. `base_train` performs these steps. Errors are sticky and
propagated after all work retires; subsequent use requires a new runner.
Returned loss/count storage is reused on the next replay, as in disabled mode.

After draining, `last_routes` reports the actual GPU choices (`e`/`q`),
`route_counts` accumulates those choices, and `last_planner_reasons` records
budget/cancellation outcomes. The existing training route logger receives the
same choices. Warmup/capture do not contribute to these statistics. An all-soft
invocation may still select `e`; this is a route count, not a hard-token count.

`--cuda-graph-planner-max-bytes` (API `planner_max_bytes`) defaults to 1 GiB
across all DISM slots. It bounds snapshots and plan payload/workspaces, not the
entire model/graph pool or native planner's temporary CPU algorithm storage.
A plan exceeding its per-slot share falls back without publication; genuine
planner/upload errors abort the step. `DISM_PLANNER_WORKERS` controls the native
pool (API `planner_workers`, default 16). Runtime threads, events, payloads,
bindings and graph handles never enter a checkpoint; reconstruct after loading
model/optimizer state and before consuming the saved next batch.

## Validation

Use conda `blkw`, `DISM_HARD_PLANNER=0` and this worktree's
`PYTHONPATH=$PWD/dism_v3/python:$PWD/nanochat`:

```bash
python -m pytest nanochat/tests/test_cuda_graph.py -q
python -m pytest nanochat/tests/test_planner_cuda_graph.py -q
python dism_v3/tools/check_training_cuda_graph.py \
  --model-spec nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32.json \
  --rows 8
# Add --planner for the explicitly enabled graph planner; keep the eager
# reference at DISM_HARD_PLANNER=0. No dataset or user checkpoint is accessed.
```

The regression suite passed 81 tests covering graph replay, checkpoint resume,
data prefetch, document loss, GDN boundary reuse, model integration, MXFP8,
disabled-planner behavior, build dependencies and Dynamo. The four graph tests
also passed Compute Sanitizer `memcheck` and `synccheck`, each with zero errors.
These checks used RTX 5090, Torch 2.13.0+cu130, Triton 3.7.1 and fla-core 0.4.1.
The GDN adapter uses FLA's internal forward/backward API; rerun these checks when
upgrading FLA. Racecheck was not part of this validation. Separate throughput
measurements are described below.

The synthetic tool creates random weights/inputs without datasets, downloads
or user checkpoints. It first stores eager references in CPU RAM and releases
the eager model before capture. Each replay receives the corresponding reference
step's master/quantized weights through in-place copies, so gradient comparisons
use identical inputs despite nondeterministic BF16 rounding between independent
optimizer trajectories. The unit tests separately cover independent optimizer
continuation, accumulation and checkpoint restoration.

The production 123,473,088-parameter configuration was checked at `[8,2048]`,
with the training backward scale `1/8`, document counts `16 -> 64 -> 8`, and
hard probabilities `0 -> ~0.475 -> 0.95`. All three replay losses matched eager;
parameter gradients passed the existing elementwise `atol=.001, rtol=.03`
comparison. Peak allocation across reference and graph phases was 12,390 MiB.
This is a correctness check, not an uncontended performance measurement.

Numerical limitation: an additional unscaled `[1,2048]` stress comparison showed
small elementwise BF16 failures, including output-bias error up to `.001953125`
with cosine about `.99997`. Repeated eager execution showed the same maximum
difference. Native cross-CTA accumulation remains nondeterministic, and subsequent
MXFP8 quantization can amplify small changes. An additional aggregate VECTOR
diagnostic applied to a full-model FFN gate gradient exceeded its L2 bound
(`.000977` versus `.000920`), even though the production-shape elementwise check
passed. These limitations remain visible; this feature changes neither kernel
math nor existing tolerances. Per-operator release gates remain separate in
`BACKWARD_DIMS.md`.

## Throughput on an idle RTX 5090

Measured on 2026-10-22 with the production 123M model above, BF16/MXFP8 FFN,
`DISM_HARD_PLANNER=0`, microbatch `[8,2048]`, eight accumulated microbatches
and 131,072 input tokens per optimizer step. Each mode runs in a separate
process with one CPU intra-op thread and the training script's TF32 setting.
The baseline uses the same `torch.compile(dynamic=True)` configuration as
`base_train`; graph mode uses `PackedTrainingGraph`.

Every step cycles document counts `8,16,32,64` twice. Per-row document lengths
are respectively `[2048]`, `[256,1792]`, four 512-token documents and eight
256-token documents. Both paths copy pinned synthetic inputs/boundaries into
reused GPU storage. Timing includes these H2D copies, metadata preparation,
forward/backward, document normalization, clipping, finite-value checks,
FusedAdamWMXFP8 updates and gradient clearing. Disk loading, evaluation,
checkpointing and optional hidden/update RMS probes are excluded. Random
weights and inputs are used; no user checkpoint is read or written.

After four warmup optimizer steps, 16 steps were timed at each fixed annealing
position. Throughput is total input tokens divided by total wall-clock time;
compilation and capture are excluded. No timing samples were discarded.

| Hard probability | Compile ms/step | Graph ms/step | Compile tokens/s | Graph tokens/s | Throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 941.1 | 876.8 | 139,280 | 149,485 | 7.3% |
| 0.475 | 939.6 | 889.9 | 139,502 | 147,282 | 5.6% |
| 0.95 | 940.6 | 865.1 | 139,345 | 151,518 | 8.7% |

The compile runs at probabilities 0 and 0.95 each included one approximately
1.16-second step. Median-step comparisons give gains of 5.6%, 5.5% and 6.9%,
respectively. PyTorch peak reserved memory during measurement was 13,138 MiB
for compile and 13,694 MiB for graph; graph replay's allocator-active statistics
do not count all memory whose addresses the captured graph retains.

A reverse-order repeat (graph first, compile second) at probability 0.475 used
eight warmup steps and 32 measured steps per mode. Compile averaged 1,008.6
ms/step (129,951 tokens/s); graph averaged 961.8 ms/step (136,272 tokens/s), a
4.9% throughput gain. Median-step gain was 4.2%. Both paths were slower than in
the first round; clocks were not locked and the GPU also serves the desktop.
The cause of this cross-round variation was not isolated. Thus the mid-anneal
result is approximately 5% faster, rather than a guarantee of one absolute
tokens/s value. Both rounds retained every sample.

Reproduce with the same environment as the correctness checks:

```bash
OMP_NUM_THREADS=1 python dism_v3/tools/bench_training_cuda_graph.py \
  --model-spec nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32.json \
  --mode compile --training-step 11444
# Repeat in a fresh process with --mode graph.
# Use --training-step 0 or 22887 for the other annealing positions.
# For the longer repeat, add --warmup 8 --steps 32 and run graph first.
```

These synthetic packs measure the training compute loop. Actual job throughput
also depends on its document distribution, loader and instrumentation.

## Component forward/backward costs during graph replay

Nsight Systems 2025.5.2 profiling on the same RTX 5090/configuration at hard
probability 0.475 used one warmup optimizer step and three measured steps.
Each measured step accumulated eight `[8,2048]` microbatches, again cycling
document counts `8,16,32,64`. The following times are **milliseconds per
optimizer step**, aggregated across all layers and all eight microbatches.
Divide by eight for the average microbatch cost for this mixture of packs.

These are GPU kernel/copy/memset duration sums, not CPU API durations. Replay
`graphNodeId` values are traced through their `originalGraphNodeId` to the
capture-time NVTX component. Autograd sequence numbers associate backward
launches, including worker-thread launches, with their forward component.
Innermost scopes own each activity exactly once; parent and child times are
not added together. All 3,169 graph kernel nodes occurred exactly 24 times.
Unmapped graph work was only 0.0213 ms/step (0.0023%), retained in Other.
The annotations introduce no extra graph timing nodes or backward hooks.

| Component | Forward ms | Backward ms | Total ms |
| --- | ---: | ---: | ---: |
| DISM core, including vocabulary interpolation | 43.186 | 220.004 | 263.190 |
| Q/K/V input projections, shared projections counted once | 36.936 | 77.540 | 114.476 |
| GDN recurrent core and its internal preparation | 37.973 | 74.721 | 112.694 |
| Attention gates, output norm and output projection | 22.524 | 67.130 | 89.654 |
| MXFP8 FFN gate/up/SwiGLU | 11.521 | 52.599 | 64.120 |
| MXFP8 FFN down projection | 9.213 | 20.144 | 29.356 |
| Output vocabulary projection and chunked CE | 124.518 | 2.419 | 126.937 |
| Depthwise causal convolution + SiLU | 10.064 | 28.418 | 38.482 |
| Block/final normalization, residuals, fused norm quantization | 15.077 | 35.269 | 50.346 |
| DISM readout SQ/SK projections | 6.110 | 12.600 | 18.710 |
| Token embedding | 0.178 | 1.370 | 1.548 |
| Other attention/model glue, parameter gradient accumulation | 6.979 | 26.495 | 33.474 |
| **Total inside graph** | **324.279** | **618.707** | **942.987** |

The CE split follows its actual execution: `ChunkedCrossEntropyFunction.forward`
precomputes local hidden/weight gradients; its backward mainly applies the
upstream scalar. Thus 124.5 ms is not a purely mathematical forward-only CE
cost, and the small backward entry does not indicate missing gradient work.

DISM accounts for 27.9% of graph GPU activity; its backward alone accounts for
23.3%. The largest kernels within that backward were hard quadratic backward
(52.53 ms/step), soft backward summary (31.66 ms), soft backward QK (30.38 ms),
vocabulary `_dkv`/`_dq` (34.58 ms combined), and hard tau reduction (13.65 ms).
The full kernel list is retained in the generated report.

The complete DISM backward breakdown from that trace is below. Percentages use
the **220.004 ms DISM backward total**, not the whole model or step. Each row is
exclusive; times sum across six DISM layers and eight microbatches. The average
is 27.500 ms per microbatch, or 4.583 ms per DISM layer per microbatch.

| DISM backward item | ms/step | Share |
| --- | ---: | ---: |
| Hard quadratic main kernel (`hard_backward_kernel`) | 52.531 | 23.88% |
| Soft backward summary | 31.656 | 14.39% |
| Soft backward QK | 30.382 | 13.81% |
| Vocabulary interpolation `_dkv` | 19.099 | 8.68% |
| Vocabulary interpolation `_dq` | 15.483 | 7.04% |
| Tensor initialization/fill kernels, all dtypes | 13.850 | 6.30% |
| Hard tau gradient reduction + dlog-weight transform | 13.740 | 6.25% |
| Hard forward-state recomputation during backward | 11.121 | 5.06% |
| Generic copies and dtype conversions | 7.968 | 3.62% |
| Token-gradient merge (`_merge`) | 7.415 | 3.37% |
| Soft delta (`varlen_delta_kernel`) | 6.342 | 2.88% |
| Generic gradient additions | 5.725 | 2.60% |
| Soft backward chunk scan | 2.231 | 1.01% |
| Interpolation delta (`delta_true`) | 2.021 | 0.92% |
| Vocabulary preparation and final cast | 0.437 | 0.20% |
| **Total** | **220.004** | **100%** |

The 11.121 ms recomputation is real backward work: with the planner disabled,
`backward_hard_recompute` calls `forward_hard(..., save_state=True)` to rebuild
the hard state. Its hard forward kernel costs 8.801 ms, hard summary 1.504 ms,
hard chunk 0.746 ms and log weights 0.071 ms. Those are separate launches from
the original model forward and are not double-counted. Hard tau costs 13.647 ms
for reduction and 0.093 ms for the dlog-weight transform.

The `_dq`/`_dkv` entries are interpolation gradients for token inputs and
codebook key/value tensors; they are separate from native soft backward QK.
The initialization/copy/addition rows classify their actual GPU kernels rather
than attributing the same shared kernel name to an unverified logical buffer.
Fill, generic copy/conversion, addition and `_merge` together cost 34.959 ms
(15.89%). The three largest main kernels together cost 114.570 ms (52.08%).

Outside-graph GPU activities totaled 3.988 ms/step: optimizer 1.454 ms,
gradient clipping 1.168 ms, document normalization 0.597 ms, graph input/gradient
preparation 0.350 ms, metadata upload 0.294 ms, and the remaining copies/loss
accumulation 0.124 ms. The profiled wall time was 972.334 ms/step; union of all
observed GPU activity was 945.641 ms, leaving 26.693 ms with no observed GPU
activity. Duration sums may overlap and should not be added to CPU scope time.

In particular, boundary GPU-to-CPU reads occupied 825.05 ms of CPU scope time,
mostly waiting for previously submitted GPU work; that is not an additional
825 ms of metadata computation. Metadata construction/upload submission itself
occupied 4.32 ms of CPU scope time per step. CPU and GPU observations overlap.

A separate unprofiled control with the same warmup/count averaged 955.005
ms/step versus 972.340 ms with profiling (1.8% difference, including cross-run
variation). These component figures are profiler measurements, not replacements
for the unprofiled throughput measurements above.

Use the CUDA 13.1 bundled Nsight binary; `/usr/local/bin/nsys` on this machine
points to an older 2023 version. From the worktree root, with the same conda and
PYTHONPATH environment used above:

```bash
OMP_NUM_THREADS=1 /usr/local/cuda-13.1/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=dism_v3/build/graph_profile/components_mid \
  python dism_v3/tools/bench_training_cuda_graph.py \
  --model-spec nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32.json \
  --mode graph --nsys --warmup 1 --steps 3
/usr/local/cuda-13.1/bin/nsys export --type sqlite \
  --output dism_v3/build/graph_profile/components_mid.sqlite \
  dism_v3/build/graph_profile/components_mid.nsys-rep
python dism_v3/tools/profile_training_cuda_graph.py \
  dism_v3/build/graph_profile/components_mid.sqlite \
  --output dism_v3/build/graph_profile/components_mid_checked.json
```

Choose fresh output names for another run. Trace, SQLite and JSON artifacts stay
under ignored `dism_v3/build/graph_profile/`; no checkpoint is accessed.

The subsequent [interpolation TMA experiment](INTERPOLATION_TMA.md) fuses
interpolation dQ into dKV. It shows no measured throughput gain and fails a
strict full-model gradient diagnostic, so the default remains separate dQ/dKV.

## Planner-enabled validation (2026-10-23)

The explicit planner runner passed the same three production-shape comparisons
above with the original tolerances. Actual six-layer routes were `eeeeee`,
`eeeeee`, `eeqqqq`; these are correctness observations, not a performance or
hit-rate estimate. The combined regression passed 171 tests (five strict-gradient
diagnostics deselected); the updated lifecycle/resume file passed 16 tests.
The two model graph test files passed 23 tests under both memcheck and synccheck,
each with zero errors. No CUDA/Triton kernel or tolerance changed.

Representative full-model racecheck passed both numerical tests but reported
69 warnings, all in PyTorch's native RMSNorm backward. A standalone eager
`torch.nn.RMSNorm(128)` reproduces the same read/write PCs without DISM, planner
or graph. This remains an unresolved warning, **not a clean racecheck result or
a classified false positive**. Earlier fixed-soft racecheck and strict numerical
limitations also remain. See [P6/P7 evidence and reproduction](PLANNER_CUDA_GRAPHS.md)
for scope, exact results and commands. No user checkpoint or training task was
accessed or restarted.

## Middle DISM layout: planner throughput (2026-10-23)

After the P6/P7 commit `4b2d32d`, the current 123M config was changed to
`layer_layout="gggddddddggg"`: layers 1–3 and 10–12 are pure GDN; layers 4–9
are GDN+DISM. The parameter count remains 123,473,088. The optional layout
field validates length, equal g/d counts and a leading g for value residual.
Omitting it preserves the old `ggggggdddddd` model; saved ModelSpecs and
fingerprints are not rewritten or migrated.

The new arrangement passed 23 targeted layout/registry/resume tests, including
12-layer MXFP8 graph versus eager loss and gradients. The full `[8,2048]` model
also passed all three synthetic correctness steps (document counts 16/64/8,
hard probabilities 0/~.475/.95) at the existing loss `.005/.02` and gradient
`.001/.03` tolerances. Kernel math, launch defaults, interpolation backend and
default planner enablement are unchanged.

Throughput used an otherwise idle RTX 5090, BF16/MXFP8 FFN and optimizer,
CUDA Graph, eight `[8,2048]` microbatches per optimizer step (131,072 input
tokens). Packing cycles 8/16/32/64 documents twice. Each of the nine conditions
ran in a separate process with eight warmup steps and 32 measured steps; all
samples were retained. Timing includes H2D, metadata staging, planner work and
drain, accumulated forward/backward, document normalization, clipping, optimizer
and gradient clearing. It excludes setup/capture, data loading, evaluation,
checkpointing and instrumentation. Inputs are repeated synthetic batches with
random initial weights and real optimizer updates, not a user training run.
CPU intra-op threads = 1, native planner workers = 16, compact C64/ready, budget
= 1 GiB, and interpolation remains split dQ/dKV.

| Hard probability | Old layout, planner on tokens/s | New layout, planner off tokens/s | New layout, planner on tokens/s | New planner-on ms/step |
|---|---:|---:|---:|---:|
| 0 | 136,724 | 137,854 | 137,192 | 955.39 |
| 0.475 | 135,741 | 135,181 | 135,494 | 967.37 |
| 0.95 | 135,414 | 139,101 | 135,883 | 964.59 |

There are 1,536 sampled layer routes per condition (32 steps × 8 microbatches ×
6 DISM layers). At probability 0 and .475 both planner-enabled layouts selected
event 1,536/1,536 times. At .95 the old layout selected event 1,085/1,536 times
(70.64%), versus 1,532/1,536 (99.74%) for the new layout. All-soft event choices
are empty hard plans, not useful-hard-work counts.

Moving DISM to the middle substantially improved readiness in the .95 test,
but measured throughput changed only +0.35% versus the old planner-enabled
layout. Against the same new layout with planner disabled, planner enabled
changed throughput by −0.48%, +0.23%, −2.31% at probabilities 0/.475/.95.
Thus this run shows no clear throughput benefit from reordering; improved
event selection alone did not produce a comparable speedup. It does not isolate
the cost responsible for the remaining overhead.

Clocks were not locked and the GPU also serves the desktop. Sub-percent changes
should not be treated as a stable gain. To reduce one-way order bias, execution
order was new-on/old-on/new-off at 0, old-on/new-off/new-on at .475, and
new-off/new-on/old-on at .95. New planner-on median step times were
955.15/967.39/963.86 ms. Peak reserved memory across the nine runs was
13,414–13,720 MiB; graph-owned storage is not fully represented by allocator-active
figures after resetting peak statistics.

Reproduce from the repository root using conda `blkw` and the local imports:

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
export DISM_HARD_PLANNER=0
export DISM_HARD_BACKWARD=ready
export DISM_PLANNER_COMPACT=1
export DISM_PLANNER_WORKERS=16
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
python dism_v3/tools/bench_training_cuda_graph.py \
  --model-spec nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32.json \
  --mode graph --planner --training-step 11444 --warmup 8 --steps 32
```

Use steps 0/22887 for the other probabilities. Omit `--planner` for the new
disabled control. For the old layout, copy this JSON into ignored scratch
storage, remove `layer_layout`, and run with that copy and `--planner`.
The tool reports actual layout, parameter count and measured-only route counts.
Generated logs/results remain under ignored `build/middle_dism_*` paths.
The prior strict numerical and sanitizer limitations remain as documented above;
this layout change did not modify any kernel or attempt to suppress those reports.

## Why planner readiness did not improve throughput (2026-10-23)

The layout/throughput work above was committed as `947efb4`. Follow-up diagnosis
used the same middle layout, RTX 5090 and synthetic workload at hard probability
.95. No production kernel, launch default or planner enablement changed.

Two fresh Nsight runs used eight warmup and three measured optimizer steps.
`profile_training_cuda_graph.py` now annotates the explicit graph binding and
the nested hard-backward paths; innermost backward scopes receive exclusive
GPU attribution. There were 24 graph replays and 144 sampled layer routes per
run. Planner-on chose event 143 times and quadratic once. These are GPU activity
duration sums per optimizer step, not independent wall-clock contributions:

| Hard backward work | Planner off (ms) | Planner on (ms) |
|---|---:|---:|
| Quadratic recomputation/backward, including buffer initialization | 100.598 | 4.535 |
| Event consumer, including buffer initialization and tau reduction | — | 97.658 |
| Route sampling, casts and branch-gradient addition | — | 11.494 |
| Total | 100.598 | 113.686 |

The event consumer spends 93.056 ms in `compact_stage`, 2.592 ms clearing FP32
outputs and 2.009 ms in tau stages. The 11.494 ms dispatch/merge cost contains
7.245 ms of addition, 4.177 ms of casts and only .071 ms of flag sampling.
The quadratic branch still allocates/clears buffers and launches gated kernels
when event wins; its 4.535 ms also includes the one real fallback. Hence flag
sampling itself is not the dominant overhead.

Measured wall time was 951.641 ms off versus 971.366 ms on, consistent in direction
with the longer uninstrumented runs above. Forward DISM activity also increased
from 40.260 to 44.060 ms with snapshot staging. Planner-on had 14.450 ms of
outside-graph GPU activity, mostly uploads, which overlaps other work and must
not be added directly to wall time. CPU drain/wait spans include GPU waiting;
these profiles do not establish a pure CPU-planning cost.

An independent same-input comparison uses `tools/bench_graph_hard_routes.py`.
After eight synthetic optimizer steps it copies SQ/SK/V/dO, labels, hard masks
and tau for six DISM layers and four packing patterns. All model replays finish
before standalone captures begin. Each case owns immutable input copies and
its own graph layout. Planning and completed uploads are outside timing.
Each route is captured separately, warmed up, and measured as the median of
five groups of 32 replays. The following are mean case medians across six
layers, in ms per layer/microbatch at eight training rows:

| Document lengths in each 2048-token row | Quadratic | Current graph event | Current dual dispatch, event chosen | Original static event |
|---|---:|---:|---:|---:|
| 2048 | 2.978 | 2.441 | 2.754 | .201 |
| 256 + 1792 | 2.598 | 2.298 | 2.612 | .183 |
| 4 × 512 | 1.240 | 1.774 | 2.089 | .129 |
| 8 × 256 | .980 | 1.249 | 1.582 | .107 |

Quadratic includes forward-state recomputation, clearing and tau reduction.
Event includes clearing and tau reduction. Dual dispatch includes both gated
branches and the same four gradient additions/casts as production. Original
static event uses `CompactFusedHardPrefillBackward` on the identical plan and
inputs with its original task-count launch grid; its already-completed upload
event is excluded from capture. It cannot replace dynamic planner dispatch
directly because its launch geometry depends on the prepared plan.

With each packing repeated twice, sums of the isolated measurements are
93.557 ms quadratic, 93.147 ms current graph event, 108.434 ms event-selected
dual dispatch, 107.173 ms quadratic-selected dual dispatch, and 7.443 ms original
static event. These are GPU-only estimates, not model throughput. All 24 cases
passed the existing relaxed `.001/.03` gradient check for all four alternatives
against quadratic. Across comparisons, maximum absolute error was 9.39e-7,
maximum relative L2 was .001685, and minimum cosine was .9999985; this does not
replace the separate strict-gradient diagnostics.

Thus current graph event is faster for long documents but slower for short
ones. Its aggregate advantage is negligible, and dual-branch overhead removes
it. The original static consumer being about 12.5 times faster than current
graph event points to the graph port's execution efficiency, not an inherent
advantage of quadratic math. Current graph consumers use a fixed 32-CTA grid
with an indirect descriptor and a task loop; static consumers use a task-count
grid. The previously recorded graph-kernel spills remain relevant. This probe
does not isolate the individual contribution of grid size, indirection or
spills, so no single cause is established and no launch default was changed.

An earlier diagnostic interleaved standalone captures with later model replays
while sharing the live model layout; its next model replay failed with CUDA
719. That diagnostic result was discarded. The final isolated snapshot version
completed all four model packs and all 24 route comparisons. The failure's
precise cause was not established; it is not classified as a tool false positive
or evidence that the production runner is sanitizer-clean.

A supplementary memcheck run of the final script (`--rows 1 --warmup-steps 1
--repeats 1 --samples 1`, compute-sanitizer 2025.4.1, `--error-exitcode 1`)
stalled before the first snapshot. After several minutes without progress,
GPU utilization was zero and native thread stacks showed `cuMemHostAlloc`
waiting on driver locks, with other workers waiting on the Python GIL.
The diagnostic process was terminated; this is an **incomplete memcheck**, not
a passing or classified false-positive result. The log and native stack are
retained under ignored `build/hard_routes_matched_memcheck*` paths. This does
not change the earlier production regression results or resolve the stall.

Reproduce the matched-input comparison using the local environment above:

```bash
python dism_v3/tools/bench_graph_hard_routes.py \
  --model-spec nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32.json \
  --training-step 22887 --warmup-steps 8 --repeats 32 --samples 5 \
  --output dism_v3/build/hard_routes_matched.json
```

For the full-model profiles, use `bench_training_cuda_graph.py` with `--mode
graph --training-step 22887 --warmup 8 --steps 3 --nsys`, once with `--planner`
and once without, under the Nsight capture/export procedure documented above.
The generated `build/hard_routes_{off,on}.*` and `build/hard_routes_matched_v4.*`
artifacts remain ignored. No user checkpoint, dataset or real training job was
accessed. Prior strict numerical and racecheck limitations remain in force.

## Graph event scheduling diagnosis (2026-10-23)

The dominant problem was the fixed **32-CTA grid**, followed by the cost of the
grid-stride task loop. At this diagnostic stage only opt-in tools and tests
changed; production still defaulted to 32 CTAs. The optimization below supersedes
that scheduling default; planner enablement remains opt-in.

The matched-input tool now accepts `--event-ctas` to sweep the existing consumer
constructor argument. All 24 cases (six layers × four packs), all gradient
comparisons and all timing samples completed. Each timing is the median of
three groups of 16 replays after warmup. For the same eight-microbatch packing
mix as above, sums of the isolated GPU times were:

| Grid | Event GPU ms per optimizer-step equivalent |
|---|---:|
| 32 (production default) | 93.959 |
| 64 | 49.749 |
| 128 | 27.704 |
| 256 | 16.903 |
| 512 | 12.064 |
| 1024 | 12.103 |
| 2048 | 11.830 |
| Original static event | 7.475 |

Increasing only the grid to 512 removes about **94.7% of the measured gap**
between current graph event and static event. This uses the same compiled
kernel specializations, including their existing spills. Larger grids plateau
and do not improve every packing: at 256-token documents the means were .179,
.180 and .190 ms for 512/1024/2048 CTAs, versus .310/.311/.293 ms for 2048-token
documents. These are isolated measurements, not an autotuned production choice.

The planner produces tens of thousands of independent C16 tasks per layer:
the first layer's four cases had 38,122 / 38,574 / 36,012 / 27,095 C16 blocks.
At 32 CTAs each CTA serially traverses roughly 850–1,200 such tasks, and only
32 CTAs are available to occupy the GPU's 170 SMs at a time. The fixed grid also applies
to other stages despite very different task counts. Thus the issue is not
primarily failure to build a plan or a missing fast single-pair arithmetic path.

`tools/graph_event_schedule_probe.py` provides opt-in ablations which reuse the
original arithmetic helpers, acquire gate and tau stages. For one DISM layer
and all four packs, it compares direct versus descriptor-loaded pointers,
512 versus exact task-count grids, and a grid-stride loop versus one task per
CTA. Task-count/direct-pointer variants are frozen-plan controls; they are not
valid replacements for changing plans in production. A covering grid is
required for the one-task variant, and zero-count records still execute no task.

For the 2048-token document case, graph event at 512 CTAs took .310 ms, the
indirect task-count grid retaining the loop took .266 ms, and removing the loop
on that same task-count grid took .214 ms. Original static event took .202 ms.
Keeping the one-task grid and replacing descriptor pointers with direct
arguments took .216 ms. Direct pointers alone do not explain the large gap.

A second four-pack ablation also skips entire empty records using the frozen
CPU plan. At 512-token documents, the indirect one-task path fell from .150 to
.140 ms; at 256-token documents it fell from .144 to .134 ms. Long-document
cases had no empty records and no consistent change. Thus empty launches add
about 10 microseconds per layer/microbatch in these short-document controls,
but do not explain the original millisecond-scale deficit. Direct one-task
timing at 256 tokens was .132 ms, indicating a smaller remaining contribution
from pointer/compiler differences. Empty-record skipping is a diagnostic
control, not a host decision added to dynamic graph replay. Both ablation runs
passed all four gradient comparisons against quadratic for every variant.

The compiled resource difference supports the loop result:

| Stage | Current loop: registers / spills | Indirect one-task: registers / spills | Original static: registers / spills |
|---|---:|---:|---:|
| C16 backward | 157 / 0 | 64 / 0 | 56 / 0 |
| C32 backward | 240 / 0 | 127 / 0 | 136 / 0 |
| C64 backward | 255 / 22 | 250 / 0 | 248 / 0 |
| Multi summaries | 255 / 10 | 185 / 0 | 181 / 0 |
| Multi scan | 96 / 0 | 96 / 0 | 78 / 0 |
| Multi backward | 255 / 50 | 255 / 0 | 255 / 10 |

`spills` is Triton's `n_spills` field, not ptxas spill bytes. All 44 distinct
compiled kernels dumped across both ablation runs were disassembled; none
contained a SASS CALL instruction. No shared FP32 atomic operation was added.
These controls establish that the loop increases resource pressure; they do
not separately quantify time spent spilling versus occupancy or scheduling.

Four new scheduling-probe tests passed normally and independently under
compute-sanitizer memcheck and synccheck (four passes, zero errors for each
tool). They cover vocabulary sizes 8/64/256/512 at `[2,2048,2]`, mixed hard
flags, bucket/multi tasks, direct/indirect and loop/one-task variants, skipping
empty records, repeated replay and the frozen inactive choice. They compare
against the unchanged consumer using its existing `1e-4/1e-4` cross-consumer
contract. These tests exercise isolated frozen plans, not the asynchronous
full-model lifecycle; they do not resolve the prior full-model memcheck stall
or RMSNorm racecheck warning. Reproduce with:

```bash
python -m pytest -q dism_v3/tests/test_graph_event_schedule_probe.py
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -m pytest -q dism_v3/tests/test_graph_event_schedule_probe.py
```

Replace `memcheck` with `synccheck` for the synchronization check. Logs are
ignored under `build/graph_event_schedule_{tests,memcheck,synccheck}.log`.

An end-to-end confirmation used three fresh processes, in the order 512 CTA,
32 CTA, planner off. Each ran eight warmup and 32 measured optimizer steps at
hard probability .95 with the same model/config/packing as above, and includes
planning, transfers, drain, normalization, clipping and optimizer work:

| Mode | Mean ms/step | Input tokens/s | Event / fallback choices |
|---|---:|---:|---:|
| Planner off | 893.735 | 146,657 | 0 / 1536 |
| Planner on, production 32 CTA | 910.247 | 143,996 | 1419 / 117 |
| Planner on, diagnostic 512 CTA | 842.224 | 155,626 | 1437 / 99 |

The 512-CTA run improved throughput by 8.08% against 32 CTAs and 6.12% against
planner off. These are single-run measurements with unlocked clocks and a
desktop GPU, not a guarantee across shapes, hard probabilities or real training
data. Absolute times and route rates differ from the earlier run; comparisons
above use this new set of fresh processes. Faster isolated event backward does
not translate proportionally to whole-model speed because other components,
both gated branches and planner synchronization remain.

For this end-to-end experiment only, a scratch driver changed
`GraphCompactBackward.__init__.__kwdefaults__['ctas']` to 512 and invoked
`bench_training_cuda_graph.py` with `runpy.run_path(..., run_name='__main__')`.
The override existed only in that process. Reproduce with the usual benchmark
arguments `--mode graph --planner --training-step 22887 --warmup 8 --steps 32`;
at that revision, running the script normally used 32 CTAs. Omit `--planner` for the disabled
control. Logs remain under ignored `build/graph_event_{512,32,off}_training.log`.

The following grid-sweep command records the pre-optimization experiment;
reproducing that table requires the consumer source from `947efb4` with the
diagnostic tool additions. On the optimized consumer, `--event-ctas` controls
the overflow/other-stage grid while the C16 prefix remains enabled.

```bash
python dism_v3/tools/bench_graph_hard_routes.py \
  --model-spec nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32.json \
  --event-ctas 64 128 256 512 1024 2048 --samples 3 --repeats 16 \
  --dump-kernels --output dism_v3/build/graph_event_cta_sweep.json
```

For the ablations, use `--event-ctas 512 --schedule-ablations --case-layer 3
--samples 5 --repeats 32` and a fresh output path. Resource reports are embedded
in result JSON; optional PTX/cubin dumps sit beside it in an ignored directory.
A layer-filtered result sums only the selected layer, not all six DISM layers.
All measurements use synthetic scratch weights/inputs and exclude planning
and upload from isolated GPU timing. Clocks are not locked. The earlier strict
numerical and sanitizer limitations remain distinct from these scheduling tests.

## Graph event scheduling optimization (2026-10-23)

The production event consumer now uses 512 CTAs for grid-stride stages and a
fixed C16 prefix with one task per CTA. The prefix grid is
`min(8192, ceil(flat_vector_rows / 4))`, where flat rows include training rows, tokens and
heads. It is computed from tensor shape at capture, never from a live CPU plan.
The prefix processes `[0, min(count, prefix_capacity))`; a separate 512-CTA
grid-stride launch processes `[prefix_capacity, count)`. Thus a plan exceeding
the prefix remains complete, and changing counts/pointers requires no recapture.
Other bucket/multi stages retain their existing arithmetic and traversal.
An uncapped prefix launched about 49K empty CTAs per layer at the current full
shape and regressed the all-soft end-to-end check by 1.4%. The 8192-task cap
bounds this overhead; the overflow path still handles every remaining task.

Four private gradient clears and the multi tau clear are fused into one kernel.
The clear runs even when quadratic is selected, but only a successful acquire
of the frozen event choice permits descriptor/multi-tau access. Single-bucket
tau clearing is removed: each task overwrites its slot, including the
single-pair path; the disjoint prefix/overflow partitions cover every task.
Multi tau still needs initialization because its chunks accumulate atomically.
The four tau reductions are one launch with a second grid dimension selecting
the descriptor record. Private gradient addition to quadratic is unchanged.

This reduces event GPU launches from 18 to 9: one fused clear, seven compute
launches (C16 prefix/overflow, C32, C64, three multi stages), and one tau reduce.
The C16 prefix has 64 registers and no spills, versus 157 registers for the old
persistent C16 kernel. Overflow retains 157 registers. Clear/reduce use 30/34
registers and zero spills. C32/C64 and multi resources remain as previously
reported, including their existing spills; those are not claimed eliminated.
All nine production specializations dumped at full shape were disassembled
with zero SASS CALL instructions. No shared FP32 atomics were introduced.

Attempts using disabled loop-invariant motion, out-of-line task calls, and
separate single-pair/matrix loops did not improve measured performance and were
discarded. Only the measured fixed-prefix/overflow strategy is retained.
`prefix_tasks=False` remains an explicit diagnostic constructor control;
planner enablement, model configuration, checkpoints and optimizer defaults
are unchanged. The default planner is still off.

An intermediate uncapped-prefix 24-case comparison (six DISM layers, four packing patterns, eight
synthetic optimizer warmups, three samples × 16 replays) passed the existing
`.001/.03` gradient checks. GPU time summed for eight microbatches was 9.533 ms
event, 11.673 ms with its prefix disabled, 7.503 ms original static event,
24.382 ms event-selected dual dispatch and 94.804 ms quadratic recomputation.
These are GPU-only measurements excluding plan construction/upload and are not
wall-clock step times. Generated timing, numerical and resource evidence is
under ignored `build/graph_event_optimized_matched*` paths. Final capped-prefix
measurements are reported separately below.

The final capped-prefix operator/binding/probe/model graph regression passed 96 tests,
with five existing strict-gradient diagnostics deselected. New checks cover
odd output tails, rejected invalid descriptors, changing plans, empty plans,
more C16 tasks than prefix capacity plus overflow grid, and NaN-poisoned tau
slots before repeated event/quadratic/event choices. Tolerances were not changed.
The known strict-gradient and full-model RMSNorm racecheck limitations remain.

The production `[8,2048]` model also passed all three eager/graph comparisons
at hard probabilities 0/~.475/.95 and document counts 16/64/8, with the existing
loss `.005/.02` and per-parameter gradient `.001/.03` tolerances and exact
RNG/annealing buffer checks. Routes were `eeeeee`, `eeeeee`, `eeeqqq`; this is
correctness coverage, not a throughput estimate. Operator/probe memcheck and
synccheck each passed 27 tests with zero errors (five strict diagnostics
deselected). These checks do not resolve the earlier intrusive full-model
memcheck stall or classify the existing RMSNorm racecheck warnings.
Final logs are ignored under `build/graph_event_final_{tests,memcheck,synccheck,full_model}.log`.

The final capped-prefix 24-case comparison also passed all gradient checks:
maximum absolute difference 7.42e-7, maximum relative L2 .001688, and minimum
cosine .9999985 across alternatives versus quadratic. Summed GPU times were
11.226 ms event, 11.845 ms without its prefix, 7.481 ms static event, 26.777 ms
event-selected dual dispatch and 95.614 ms quadratic. Compared with the earlier
93.959 ms original event measurement, the final event path is about 8.4× faster.
The cap trades some isolated long-document speed for bounded empty-grid cost.
Final timings/resources and the nine-kernel SASS inspection are under ignored
`build/graph_event_final_matched*` paths; all nine have zero CALL instructions.

End-to-end results on the same synthetic changing-pack workload:

| Hard probability | Original planner-on ms/step | Optimized planner-on ms/step | Original tokens/s | Optimized tokens/s | Throughput change |
|---|---:|---:|---:|---:|---:|
| 0 | 904.149 | 899.898 | 144,967 | 145,652 | +0.47% |
| ~.475 | 911.073 | 862.046 | 143,866 | 152,048 | +5.69% |
| .95 | 906.522 | 839.400 | 144,588 | 156,150 | +8.00% |

Each condition runs in a fresh process with 131,072 input tokens/optimizer step.
At .95 there are eight warmup and 32 measured steps; at 0/.475 there are four
warmup and 16 measured steps. All samples are retained. Timing includes H2D,
planner work/drain, forward/backward, normalization, clipping and optimizer.
The original control loads only `planner_graph.py` from commit `947efb4` in an
ignored scratch module and substitutes its `GraphCompactBackward` in both the
kernel module and runtime before invoking the benchmark; other source and
configuration are identical. This restores the original 32-CTA/18-launch
consumer rather than merely changing the optimized consumer's grid parameter.
Final capped tests ran in order 0/.95/.475, after the controls and the uncapped
screen. Clocks were not locked, and the GPU serves the desktop; the all-soft
sub-percent change is consistent with noise, not a claimed stable speedup.

At 0/.475 both variants selected event 768/768 times. At .95 the original
selected event/fallback 1429/107 times versus 1443/93 for the optimized version.
The measurements cover synthetic scratch training, not a resumed checkpoint or
real dataset. Logs and aggregate JSON remain ignored under
`build/graph_event_opt_before_*`, `build/graph_event_opt_capped_*` and
`build/graph_event_opt_capped_throughput.json`.

Reproduce final correctness and component timing with the local environment
above:

```bash
python -m pytest -q dism_v3/tests/test_planner_graph.py \
  dism_v3/tests/test_planner_graph_binding.py dism_v3/tests/test_graph_event_schedule_probe.py \
  nanochat/tests/test_cuda_graph.py nanochat/tests/test_planner_cuda_graph.py
python dism_v3/tools/check_training_cuda_graph.py --planner \
  --model-spec nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32.json \
  --rows 8 --iterations 3 --microbatches 8
python dism_v3/tools/bench_graph_hard_routes.py \
  --model-spec nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32.json \
  --samples 3 --repeats 16 --dump-kernels --output dism_v3/build/graph_event_recheck.json
```

For sanitizer reproduction, prefix the two operator/probe test files with
`compute-sanitizer --tool memcheck --error-exitcode 1` or `--tool synccheck`.
For optimized throughput, invoke `bench_training_cuda_graph.py --mode graph
--planner --training-step 22887 --warmup 8 --steps 32` with the same model spec;
use steps 0/11444 and `--warmup 4 --steps 16` for the other rows.
