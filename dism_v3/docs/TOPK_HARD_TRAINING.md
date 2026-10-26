# Opt-in top-k hard training

2026-10-25. Implementation follow-up to [the checkpoint readout screen](TOPK_HARD_READOUT.md).
This is a training baseline, not evidence of trained quality. No long training
or checkpoint modification was performed during implementation.

## Current support boundary (2026-10-27)

Model configuration accepts `hard_query_topk=1`, `2` or `4`. K1 retains the
native path. K2/4 use the same committed Q/K argmax history and mix the selected
local Q reads. The existing K4 selector arithmetic and tie order are preserved.
K2 now uses two-slot kernels; `tools/eval_top2_from_top4.py` remains a separate
inference-only ablation of four-slot checkpoints.

| Capability | Native top-one | Model top-two / top-four |
| --- | --- | --- |
| CUDA BF16 quadratic forward/backward, D32/64, R16/32, DV32/64 | Supported | Supported |
| Native soft path and original-row hard/soft scheduling | Supported | Preserved |
| Packed 256-aligned documents within equal training rows | Supported | Supported, including mutable graph metadata |
| Eager autograd and planner-off CUDA Graph training | Supported | Supported |
| CPU planner + event backward in training | Supported | Connected to the direct decoupled planner; R32/DV64 |
| Live planner CUDA Graph, ready/fallback arbitration | Supported | Supported; R32/DV64, D32/64, one rank |
| Hard-path Q selector / Q-codebook gradient | No argmax derivative | Preserved through candidate output weights on both routes |
| Hard K selector gradient | No argmax derivative | Unchanged, no estimator added |
| Checkpoint schema, RNG counters, annealing, consumed data position | Preserved | Preserved |
| Explicit SAM prefill/incremental decode sessions | Available with their documented restrictions | Not connected; entry points reject top-k |
| DISM Dynamo compilation / standalone Torch fallback | Available on their native supported paths | Explicitly unsupported |
| Direction false / mixed directions | Available in native operators | Fixed true only |

These changes connect **training backward**. Model hard forward still uses the
quadratic implementation and its N²/32 history storage. They do not connect
subquadratic model prefill or incremental top-k decode. Generic hybrid
`use_cache` is not implemented by the explicit native SAM APIs either.

Eager event planning uses `DISM_HARD_PLANNER=1` and the existing
`DISM_HARD_BACKWARD=ready|event|quadratic` policies. The model graph entrypoint is
`PackedTrainingGraph(..., planner=True)`; the training CLI is
`--cuda-graph --cuda-graph-planner`. Graph policy remains compact/ready and
single-rank, with the same explicit drain before optimizer/checkpoint boundaries.
There is no top-k DDP validation or automatic planner enablement.

With `hard_query_topk=4, hard_query_sampled=true`, the
[sampled-CV backend](SAMPLED_HARD_TRAINING.md) retains its existing semantics and
counter/seed domain. It still requires planner off. This integration does not
change its estimator, sampling distribution or evaluation behavior.

## Training event integration

The production CPU extension is built by
`python/flash_dism/inference/build.py --component topk-training`; wheel builds
include `_dism_topk_training`. Checkout builds use the worktree sources and a
user-cache extension. The port removes runtime source patching: its direct
count/pack pass builds only hard row/head/document jobs, maps Q reads to virtual
candidate rows, and keeps K/V rows physical. The threshold and 64×64 decoupled
block sizes match the validated backward prototype. Short streams reuse compact
16/32/64 event kernels. Planner metadata is packed directly into one pinned
arena and uploaded once.

Graph snapshots include argmax Q/K, candidate IDs, original-row hard flags and
head tau. A replay coordinator waits for that slot's external D2H event, then
builds and uploads its plan and publishes the epoch with `cuStreamWriteValue32`.
One GPU sample freezes the route for the entire hard backward. Event kernels
acquire that choice before reading any descriptor; quadratic kernels exit when
it selects event. Both branches own zeroed outputs. Late plans, cancellation and
budget rejection take quadratic; unexpected planner/upload errors surface at
drain. Payloads, upload work and graph work retire before slot reuse. Mutable
packed document tables retain the binding's existing capture/replay checks.

Router probabilities stay on device. After readiness is acquired, a small
kernel scales only Q-event coefficients by `pi`; immutable source coefficients
allow repeated execution without accumulating the scale. K summaries remain
shared. SQ and dO are replicated exactly along the candidate axis, avoiding a
BF16 pre-multiplication of dO by pi. After event backward, candidate dSQ values
sum to the physical SQ gradient. Since each candidate output is linear in SQ:

```
u_c = dot(SQ, dSQ_c) = pi_c * dot(dO, O_c)
dz_c = u_c - pi_c * sum_j u_j
```

The existing router backward maps `dz` to Q and its codebook. This avoids a
second candidate forward solely for router gradients. BF16 dot-product rounding
means this recovered moment need not be bitwise identical to the quadratic
pairwise moment; validation compares direction, norm and bias. Global FP32
atomics and the existing long-chain tau limitations remain visible. This is not
a batch-invariance or deterministic-gradient change.

Validation (2026-10-27):

- 164 regression tests passed under `dism_v3/pytest.ini`, including K2/4 eager
  event autograd, planner graphs with MXFP8, forced ready/late routes, mutable
  document boundaries and hard masks, empty ready plans, budget/cancel/error
  handling, bounded replay, capture failure, no-grad evaluation, checkpoint RNG,
  optimizer schema and prefetched/consumed dataloader state.
- K1/2/4 event SQ/SK/V/tau/router gradients passed the independent FP64 recurrence
  at the existing top-k thresholds (relative L2 < .025, cosine > .999, with exact
  structural-zero checks). Random and repeated-chain cases also compare against
  the quadratic kernels. No acceptance thresholds were changed.
- CPU direct packing: 972 coefficient/derivative cases, including nonzero
  batch/head/document offsets and all tile sizes, passed ASAN/UBSAN. Maximum
  coefficient/derivative errors were 2.75e-7 / 1.30e-6. The direct count/pack
  subcases also compare byte-for-byte with the earlier converter.
- GPU memcheck, synccheck and racecheck passed for the K1/2/4 R32/DV64 consumer,
  route gates, quadratic kernels and selector/router kernels in the bounded
  diagnostic. A full two-DISM-layer K4 model graph also passed memcheck with
  both forced event and late/fallback routes (zero errors). This is not
  exhaustive sanitizer coverage of every shape.
- That full-model check exposed a host-side GIL/driver-lock deadlock under
  Compute Sanitizer 13.1: `torch.tensor(words, pin_memory=True)` held the GIL in
  `cudaHostAlloc`, while a concurrent H2D sanitizer callback held the driver lock
  and tried `PyGILState_Ensure`. Both legacy and top-k descriptor builders now
  allocate with `torch.empty(..., pin_memory=True)` before copying CPU words.
  The factory releases the GIL during allocation. The same full-model memcheck
  then completed in 24 seconds with zero errors; descriptor contents, ownership
  and publication order are unchanged.
- The first racecheck run found a K4 normalizer WAW at shared `STS.128`, not an
  ARRIVES exception. Its separate 2D DLOG load was being broadcast through a
  redundant shared-layout conversion. Candidate-masked LOG/DLOG loads now use
  the candidate layout directly, preserving the formula and launch geometry;
  racecheck then reported zero hazards. This fix is retained as a synchronization
  correction, not dismissed as a tool warning.
- SASS audit found zero CALLs in the audited kernels. Spills remain: the K4
  normalizer reports 48 slots after the layout correction; persistent split
  stages report 32/10, and reused compact variants up to 50. These are visible
  optimization targets, not a zero-spill claim.
- Five pre-existing `strict_gradient` diagnostics remain separately selectable.
  An all-tests diagnostic reproduced the native event `boundary` pointwise BF16
  mismatch (1/16384 elements; absolute error .01526 against atol .008), while the
  old/new event consumers agreed at their existing 1e-4 comparison. The standard
  suite uses the repository's existing default exclusion; this change neither
  suppresses that diagnostic nor relaxes its thresholds.

Reproduce the numerical/resource microbenchmark with
`tools/check_topk_event_training.py`; optional `--input` accepts the saved
`top4_event_probe` binary symbols. Its result JSON and cubins belong under
`build/topk_event_training`, never Git. `--check-only` is the sanitizer entrypoint.
CPU sanitizer compilation of `tools/check_decoupled_readout.cpp` uses
`-DDISM_PRODUCTION_TOPK` and the production `inference/csrc` include directory.
GPU kernel timings exclude CPU overlap and are not LM throughput guarantees.

### Bounded microbenchmark

On RTX 5090, saved `n2048_layer0.bin` Q/K symbols (B4/H12/N2048/R32/DV64),
synthetic BF16 vectors and router weights, captured GPU-only backward medians:

| K | Plain quadratic | Ready event + gated fallback + merge | Late event + quadratic + merge |
| --- | ---: | ---: | ---: |
| 2 | 1.097 ms | 0.387 ms | 1.273 ms |
| 4 | 1.224 ms | 0.489 ms | 1.402 ms |

This includes candidate expansion, probability-weighted metadata, SQ/selector
moment reduction and route merging; it excludes forward, Q/codebook router
matrix gradients common to both paths, D2H/H2D and CPU overlap. Four-worker CPU
plan measurements for this one layer were approximately 10.5/13.0 ms for K2/4;
these are not six-layer scheduling or end-to-end training throughput numbers.
The late route still costs about 0.18 ms over plain quadratic. More work remains
on fallback overhead and persistent register pressure. Candidate expansion and
collapse already use the frozen route to skip unnecessary copying/reduction
when the plan is late. Raw records: `build/topk_event_training/final_real2k`.

### Full optimizer-step throughput (2026-10-27)

Implementation commit `b1059e9`, RTX 5090, 123,473,088 parameters. Use the
recent top-four training spec
`nanochat/configs/sequence_true_123m_layernorm_sigmoid_top4_slimpajama_3b.json`:
`ggggggdddddd`, LayerNorm + calibrated sigmoid gate, postnorm, 2K context,
8 rows × 8 accumulation microbatches (131,072 tokens/optimizer step), MXFP8
FFN/optimizer, BF16 CE/projections. K and fixed hard probability are explicit
benchmark overrides; graph is enabled for both planner-off and planner-on.
CPU planning uses 16 workers and Torch uses one CPU thread.

`tools/bench_topk_model.py --compare-planner --topk {1,2,4}` initializes two
scratch models identically and alternates measurement order. Each receives
four warmup and ten measured optimizer steps. Timing includes input/metadata
H2D, forward/backward, document-loss normalization, gradient clipping,
optimizer, scalar checks and planner drain. Route-log collection is outside
the timer. No checkpoint or dataset is accessed. Synthetic batches repeat;
data loading, evaluation, checkpoint I/O and training instrumentation are
excluded, so this is a full compute-step benchmark, not measured sustained
corpus-training throughput or a trained-symbol-distribution benchmark.

Changing packing cycles through 2K, 256+1792, 4×512 and 8×256 documents per
original training row. Hard sampling still uses original rows, not documents.

| Hard probability | K | Off step (ms) | On step (ms) | Off token/s | On token/s | Planner throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .5 | 1 | 1020.2 | 978.1 | 128,473 | 134,003 | +4.30% |
| .5 | 2 | 1039.9 | 1057.8 | 126,044 | 123,909 | -1.69% |
| .5 | 4 | 1062.0 | 1093.5 | 123,418 | 119,869 | -2.88% |
| .95 | 1 | 1000.6 | 944.6 | 130,990 | 138,766 | +5.94% |
| .95 | 2 | 1060.0 | 1061.4 | 123,654 | 123,491 | -0.13% |
| .95 | 4 | 1094.4 | 1109.8 | 119,769 | 118,099 | -1.39% |

All six planner-on cases selected event for 480/480 measured layer/microbatch
decisions; no quadratic fallback was selected. K2 at .95 is effectively tied
(1.4 ms mean difference versus roughly 3.6–3.8 ms within-run step standard
deviation). K2/4 therefore have no demonstrated whole-step speedup on this
packing distribution despite the isolated long-sequence backward speedup.
The losses and gradient norms stayed finite. This does not isolate CPU planning,
copy, drain, event-kernel or short-document overhead; attributing the regression
to a particular component requires profiling. Readiness misses do not explain
these measurements.

For K4 with full 2K documents (`--packing full`), the additional paired runs
were also 480/480 event decisions:

| Hard probability | Off step (ms) | On step (ms) | Off token/s | On token/s | Planner throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| .5 | 1182.6 | 1191.0 | 110,836 | 110,053 | -0.71% |
| .95 | 1229.6 | 1204.5 | 106,594 | 108,818 | +2.09% |

Full documents improve the paired K4 planner gain relative to changing packing,
but even the .95 case gives only a modest 2.1% whole-step benefit. Document
distribution matters; the isolated backward speedup must not be extrapolated
to the whole training loop. These paired on/off comparisons are stronger than
cross-process comparisons between different K values. No layout changes,
kernel tuning or default planner enablement were made for this measurement.

Reproduce with the spec above, `--compare-planner --topk K --probability P
--rounds 10 --packing changing` and a fresh `--output` under `dism_v3/build`.
Use only this worktree on `PYTHONPATH` and `DISM_HARD_PLANNER=0` (the graph
constructor explicitly enables planning for the paired on model). Raw paired
records are ignored `build/topk_event_training/step_old_k{1,2,4}_p{50,95}.json`;
the two full-document records have the suffix `_full.json`.

## Definition and shared work

Keep the native v3 soft path, K argmax and one committed Q argmax history. At
query i, select the native Q argmax plus the three highest other Q logits. Only
that current read branches; the other candidates never enter future history.
The four logits are renormalized over the selected set, not over the full
vocabulary with missing mass discarded. SQ/SK/V/tau are shared.

For candidate c, with document/causal masking implicit:

```
A_ij = exp(tau) * (1 + H_hard[i-1,j-1])
D_ic = 1 + sum_{j: k_symbol[j]=c} A_ij
alpha_ij = pi_ic * A_ij / D_ic,  c=k_symbol[j] in selected[i]
alpha_ij = 0,                  otherwise
O_i = sum_j alpha_ij * dot(sq_i,sk_j) * v_j
```

The partition by the single K symbol lets all four candidates share one
SQ@SK.T and one weighted V read per pair tile. There are no four copies of the
history recurrence, vector forward, or vector backward. A scalar first pass
computes the four denominators before the vector readout pass.

With upstream g_i and `f_ij=alpha_ij*dot(sq_i,sk_j)*dot(g_i,v_j)`, the same
backward traversal also computes:

```
U_ic = sum_{j: k_symbol[j]=c} f_ij
(dL/d selected_logit)_ic = U_ic - pi_ic * sum_c U_ic
(dL/d tau) = sum_ij f_ij * (dlog(A_ij)/dtau - dlog(D_ic)/dtau)
```

Thus router backward needs four scalars per query, not saved four-candidate
output vectors. Selected logits feed Q and Q-codebook gradients, using BF16
operands like the native vocabulary projection. Discrete support changes,
committed Q history and K labels remain nondifferentiable. Native soft gradients
are added normally; codebook and input-projection sharing are unchanged.

## Implementation

- [topk_hard.py](../python/flash_dism/kernels/topk_hard.py): GPU selection,
  hard-history checkpoints, stable per-candidate normalizers, shared vector
  readout/backward, and router Q/codebook backward.
- [topk_training.py](../python/flash_dism/topk_training.py): autograd integration
  reusing native soft/interpolation forward and backward. The default native
  dispatch is unchanged.
- `hard_query_topk=2|4` in the nanochat model spec opts in. Omitted/1 preserves
  existing behavior and legacy spec identity. Only 1, 2 and 4 are supported.
  No parameters, state_dict keys, RNG counters, schedules or optimizer groups
  are added. `hard_prob_fixed=0.5` is a separate, explicit experiment choice.
- Supported: aligned CUDA BF16 training/eval, fixed true direction, D32/64,
  R16/32, DV32/64, validated 256-aligned documents inside equal training rows,
  eager autograd and `PackedTrainingGraph`. Planner/event support uses R32/DV64
  as described above; planner-off quadratic paths retain the wider dimensions.
- [topk_planner.py](../python/flash_dism/topk_planner.py) and
  [topk_event.py](../python/flash_dism/kernels/topk_event.py) implement the new
  invocation-owned CPU plans and fixed-grid event consumers.
- Unsupported paths reject explicitly: Dynamo, Torch/cache fallback and SAM
  prefill/decode. Sampled-CV also rejects planner bindings.

History checkpoints occupy `B*H*floor((N-1)/32)*N` int32 elements, where N is
training-row length. No dense N-by-N weight matrix or N-by-N-by-top-k tensor is
materialized. This still has quadratic/32 saved history storage, in addition
to the native soft state. Candidate IDs/probabilities/log-denominators/derivative
moments occupy O(B*H*N*4). Router vocabulary backward uses a tiled reduction
without contended per-token codebook atomics.

The hard vector pass uses query tiles of 32 and key tiles of 64; normalization
and backward use 32-by-32 tiles. The initial quadratic audit had zero spills
for these passes; the current K4 normalizer's race correction and event consumer
spill counts are reported above. Segment handling relies on the validated 256-alignment:
32/64-wide tiles contain no interior document boundary. No shared FP32 atomicAdd
is introduced. Global FP32 gradient atomics remain, so this is not a new claim
of bitwise/batch-invariant gradients.

## Initial quadratic validation (2026-10-25)

`check_topk_training.py` compares forward, dSQ/dSK/dV/dtau and selected-logit
gradients against an independent FP64 sequential oracle, at N256/512/1024,
random and all-equal symbols, tau=0/.1, and document boundaries. Output relative
L2 is about .00235; vector gradients about .00165; tau/logit errors below 4e-6
in the checked N512/1024 cases. Vector cosine exceeds .999997.

`check_topk_router.py` independently checks Q and Q-codebook gradients against
FP64 differentiation with BF16 operands, including mixed hard/soft rows, D32/64
and C4/8/512. Relative errors are below 1e-5. A packed all-hard vocabulary
integration check also compares the complete autograd result against FP64:
output/vector/Q relative L2 is below .0024, Q-codebook error about 2.4e-7,
and tau error about 1.4e-6; K and K-codebook hard gradients are exactly zero.

The nanochat regression tests cover unchanged all-soft outputs/gradients,
configuration identity/parameter alias/no-decay schema, eager versus graph
replay with changing documents and hard probabilities, optimizer-state restore,
RNG/annealing buffers, pre-existing gradients and capture-state preservation.
They use scratch models only. The existing native compiled checkpoint-resume
regression also passes, checking that the default compilation path remains intact.

`audit_topk_training.py` exercises the new kernels under graph replay while
changing segments/masks and then clearing all hard flags. At B2,N512,H2,D64,
R32,DV64,C512, Compute Sanitizer memcheck and synccheck report zero errors;
racecheck reports zero hazards/errors/warnings. No tool exception is applied.
SASS audit finds no CALL instructions in all eight new kernels. Normalization,
vector forward/backward and router gradients have zero spills; selection has
8 spill slots and the history-boundary kernel has 2 for this audited shape.
This is a bounded-shape audit, not exhaustive coverage of all launch shapes.

These checks do not establish accuracy for arbitrarily long identical-symbol
chains at large tau. Log denominators prevent direct exponential overflow, but
subtracting nearby FP32 tau moments can lose relative accuracy when the true
gradient is tiny. The native BF16/relaxed-gradient limitations are unchanged.
No convergence/quality claim follows from the finite synthetic training steps.

## Initial quadratic throughput (2026-10-25)

RTX 5090, 123,473,088 parameters, `gggddddddggg`, 2K context, 8 rows x 8
microbatches, MXFP8 FFN/optimizer, BF16 CE, graph on. Final measurements use
`bench_topk_model.py`: same initialization and synthetic inputs, four warmup
and ten measured optimizer steps for each of two resident models. Measurement
order alternates each round. Timings include metadata/H2D, forward/backward,
document normalization, clipping, planner draining where enabled, and optimizer;
exclude data loading, training instrumentation, evaluation and checkpoint I/O.
In these earlier measurements, top-four used its GPU quadratic path with planner disabled.

| Hard probability | Packing | Top-1 planner | Top-1 step | Top-4 step | Top-1 token/s | Top-4 token/s | Throughput drop |
|---|---|---|---:|---:|---:|---:|---:|
| .5 | Full 2K documents | Off | 1000.8 ms | 1058.7 ms | 130,971 | 123,805 | 5.47% |
| .5 | Changing document lengths | On | 854.1 ms | 938.6 ms | 153,464 | 139,647 | 9.00% |
| .95 | Changing document lengths | Off | 875.3 ms | 968.1 ms | 149,745 | 135,392 | 9.58% |

Changing lengths cycles through 2K, 256+1792, 4x512 and 8x256 documents within
each training row. The final kernels find each tile's containing document on
device and traverse only that interval. This removes the earlier full-row
masked work on short documents; mutable graph metadata remains supported.

The fixed-.5 baseline meets the under-10% target in the measured comparisons,
including the stronger planner-on top-one baseline for changing documents.
The margins near 9–10% are small: these are measured synthetic configurations,
not a guarantee across layouts, document distributions or hard probabilities.
In particular the .95 comparison uses planner-off top-one; no .95 planner-on
penalty is established. Mean step overhead and throughput loss are different:
9% throughput loss corresponds to about 9.9% longer steps.

Earlier separate-process measurements, before document-range skipping, gave
2.44%/7.28% loss against planner-off top-one on full/changing documents, but
15.0% loss against planner-on top-one. Separate-run clock variation made the
full-document figure optimistic; use the interleaved final results above.
The same graph geometry measured about 5.4 GiB peak allocated per scratch model;
no dense pair-weight storage was added by the interval optimization.

Ignored manifests/configs/logs live in `build/topk_training/`. Final paired
reports are `paired_p50_full_docskip.json`, `paired_p50_planner_docskip.json`, and
`paired_p95_changing_docskip.json`. Earlier `final_top*` and `limit_top*` logs
are development measurements, not final performance claims. Reproduce with
`tools/bench_topk_model.py`, the 123M model spec, `--probability .5`, and explicit
`--packing full|changing` / `--top1-planner` as applicable. Always benchmark with
no other GPU workload. The script creates scratch models and never loads a
checkpoint or dataset.

## Further work

Remaining opportunities include fusing candidate selection into the existing
Q vocabulary projection, reducing late-route overhead/register pressure, and
connecting subquadratic hard forward/prefill. Backward planner integration is
now implemented above. Incremental top-k decode and sampled-CV event planning
remain separate work; all alternatives must preserve the single committed
history rather than creating a growing beam.

## Fresh LayerNorm/sigmoid run (2026-10-25)

The user selected a fresh run after the requested early checkpoint was not
available, then explicitly chose the original linear hard-probability schedule.
The maintained spec is
`nanochat/configs/sequence_true_123m_layernorm_sigmoid_top4_slimpajama_3b.json`:
its only model-config addition over the original scheduled LayerNorm/sigmoid
spec is `hard_query_topk: 4`. Hard probability rises from 0 to .95 across the
original 22,888-step horizon; it is not fixed at .5.

Layout `ggggggdddddd`, all-mixer LayerNorm/calibrated sigmoid gate, postnorm,
MXFP8 FFN/optimizer, BF16 projections/CE, data/shuffle seeds, LR8e-4, warmup500,
warmdown.25, weight decay.01, clip1, 8x2048 with 8 accumulation microbatches,
and 2,999,975,936-token budget retain the original recipe. CUDA Graph is on;
planner was explicitly off because top-four had no planner support at launch. Validation and dual hard/soft loss, 8K NIAH and Q/K histograms remain every
500 steps; saves remain every 2000. No source checkpoint is loaded or modified.

The exact old-layout LayerNorm/top-four configuration passed a full-size graph
preflight at step11444 (hard probability about .475), with finite losses and
gradients. Ignored log: `build/top4_ln_launch_preflight.log`.

Run directory basename:
`dism-slimpajama123m-h768-h12-i1152-r32-mxfp8-ffnonly-3b-docce-niah-clip1-lr8e4-ggggggdddddd-ln-sigmoid-top4-linear-20261025-234828`.
Pointer: `build/top4_layernorm_sigmoid_training_run.txt`. The separate run holds
its exact spec, train/supervisor scripts, source snapshot with SHA256 manifest,
PID/status files and logs. Implementation commit: `a7ee16f`; the newly selected
run config is additionally captured by the source manifest.
