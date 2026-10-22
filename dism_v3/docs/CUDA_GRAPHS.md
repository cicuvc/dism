# Packed training CUDA Graphs

`scripts.base_train --cuda-graph` enables explicit microbatch forward/backward
capture for the packed DISM model with `DISM_HARD_PLANNER=0`. The selected
FFN-only MXFP8 model uses its existing fused AdamW optimizer. Without the flag,
training continues to use `torch.compile`; GDN remains outside Dynamo.

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

Warmup restores the DISM RNG counter and existing gradients. Actual replay
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

Changed batch/row shape, parameter storage, FP8 weight storage, host temperature
or training mode requires a new graph. Changing host tau-scale schedules and
FP16 GradScaler are currently rejected. The selected configuration uses constant
host tau scale and BF16. Fused MXFP8 AdamW updates quantized weights in place;
optimizers replacing their storage require recapture.

## Future planner support

Planner-enabled capture remains explicitly unsupported. Async submission,
completion, per-invocation handle ownership and event/quadratic routing cannot
be frozen during capture. Future support must stage each invocation's plan
outside replay and define backward dispatch/lifetimes. The packed metadata
interface is independent of that planner and can be reused.

## Validation

Use conda `blkw`, `DISM_HARD_PLANNER=0` and this worktree's
`PYTHONPATH=$PWD/dism_v3/python:$PWD/nanochat`:

```bash
python -m pytest nanochat/tests/test_cuda_graph.py -q
python dism_v3/tools/check_training_cuda_graph.py \
  --model-spec nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32.json \
  --rows 8
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
