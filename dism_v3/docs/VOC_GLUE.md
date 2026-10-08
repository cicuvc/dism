# Fused compiled vocabulary/core glue (2026-09-30)

Scope: opaque compiler.py voc_forward/voc_backward internals. Default fused
path; no user-facing performance toggle. Eager voc_dism remains the numerical
comparison path. Scan, interpolation MMA, core GEMM, RNG, checkpoint format,
parameter gradient accumulation and tolerances are unchanged.

## Five Triton primitives

`python/flash_dism/kernels/voc_glue.py`:

- prepare_embedding: aliases BNHD Q/K without copying; one launch casts both
  codebooks to BF16. Handles shared2D and independent3D strided codebooks.
  Native prepare_embedding(materialize=False) performs the original validation
  without first allocating/materializing the outputs. Default materialize=True
  preserves the eager ATen/autograd behavior.
- select_operands: one launch chooses direct/interpolated Q/K by direction and
  writes both core operands in contiguous BNHD.
- split_interpolation_gradients: converts core dA/dB to BF16, applies direction,
  writes interpolation dOq/dOk directly in contiguous BNHD.
- merge_token_gradients: rounds interpolation/direct gradients to BF16
  separately, adds in FP32 and rounds the result to BF16, writing contiguous
  BNHD. Also casts dsq in the same launch. It MUST NOT add unrounded FP32
  gradients and round just once. dsk/dv already have the required BF16 layout.
- cast_vocabulary_gradients: BF16 round then convert to original parameter
  dtype for both tables in one launch. Shared-codebook head reduction retains
  Torch's original order and post-cast precision; it is not moved before rounding.

Backward native prepare_operands now accepts absorbed=True (defaultFalse),
eliminating the previous zero-tau allocation and two LSE-minus-zero operations.
No further changes to the preabsorbed LSE-tau interface.

Helpers are internal no-grad operations inside registered autograd custom ops.
They do not replace PyTorch parameter `.grad` accumulation. Zero-token transport
and vocabulary tails are handled; the module still requires positive aligned N.

## Verification

-16 helper cases pass exact (atol=rtol=0) comparison to the prior Torch
  expressions: R16/32,D32/64, batch2 with mixed per-head directions, shared and
  independent strided vocabularies, N0/256 and vocabulary size37.
-36 full Dynamo/module tests passed after integration, including fixed/packed,
  changing document count without recompilation, saved-layout ownership and
  pure/hybrid empty-tail gradient equivalence.
-Expanded32 opaque-op gradient comparisons passed: all8 R/D/DV shapes x
  fixed/packed x shared/independent vocabulary (overlaps16 of the36 above).
-2 nanochat integration tests passed, including actual compiled two-layer
  optimizer/save/restore and next-step gradient/parameter replay.
-17 selected memcheck tests passed with0 errors, including all16 exact helper
  cases and prepared-layout mutation/retained-backward test.
-No tolerance changes. Logs: build/glue_{unit,dynamo_final,all_vocab,memcheck_final}.log;
  nanochat runs/dism72m_3k/glue_resume.log. Build uses build.py; device CUDA
  kernels did not change, only the frontend object and Triton glue.

## Same nanochat configuration benchmark

71.96M, width384,12 layers,H6,D/DV64,R32,N1024,micro4,compile+varlen,
4-thread tokenizer. Same initialized-model probe with hard_prob at step1500;
32 warm forward/backward microbatches. No optimizer/eval/checkpoint timing.

| Unprofiled ms/microbatch | Before glue fusion | Fused run1 | Fused run2 |
| --- | ---: | ---: | ---: |
| Cached GPU batch |44.95|43.29|42.78|
| Real loader |52.51|48.52|48.44|

Cached improvement ~4–5%; live-loader result also includes loader timing
variation, so do not attribute the entire ~8% gain solely to fusion.

Separate four-microbatch profiler captures:
- Kernel launches per microbatch2709->2325 (-384,14.2%).
- Generic ATen elementwise1134->666 (-468); fused helpers add84 launches
  (7 per layer: two prepares, two selects, split, merge, vocabulary cast).
- Everything outside these groups stays1575 launches/microbatch.
- Remaining ATen operations include327 FP32 parameter-gradient accumulations
  per noninitial accumulation microbatch; these were intentionally left alone.

Artifacts in nanochat runs/dism72m_3k:
utilization_glue_fused{,_repeat}.{log,json}; comparison baseline
utilization_optimized_isolated.{log,json}. Old nsys_optimized.nsys-rep predates
this fusion and must not be treated as a new capture. No training restarted.

## BNHD interpolation follow-up (2026-09-30)

The Triton interpolation now consumes and produces BNHD token vectors, including
its FP32 token gradients. Scalar LSE/top1/labels stay BHN, tables stay HVD.
The JIT's stride arguments remain in semantic batch/head/token order; wrappers
pass the corresponding BNHD strides. Token gradient stores use their own output
strides rather than reusing input base offsets, which also supports strided inputs.

Both native eager preparation/selection and compiled glue use this contract.
Q/K preparation returns input aliases and only casts the vocabularies; selection,
gradient split and merge now access vector operands in the same linear order.
Compiled operators have new internal `_bnhd` names to invalidate stale Inductor
cache contracts for the differently shaped saved tensors. Public API and model
state dictionaries are unchanged; restart existing Python processes after rebuild.

- 89 tests passed: embedding layout, native frontend, glue and full Dynamo suite.
- 8 new interpolation tests compare forward and all four gradients bit-for-bit
  against unchanged v2 BHND kernels: D32/64, vocab37/512, contiguous/strided Q/K.
- 24 embedding/glue tests passed under memcheck, zero reported errors.
- 2 nanochat integration tests passed, including compiled optimizer checkpoint
  save/restore and next-step replay (runs/dism72m_3k/embedding_bnhd_resume.log).
- No changes to scan, interpolation math, BF16 rounding or test tolerances.

Logs: build/embedding_bnhd_{build,tests,memcheck}.log.

Same nanochat probe (32 warm forward/backward microbatches, no optimizer):

| ms/microbatch | Previous fused BHND run1 / run2 | BNHD run1 / run2 |
| --- | ---: | ---: |
| Cached GPU batch |43.29 / 42.78|43.89 / 43.06|
| Real loader |48.52 / 48.44|53.82 / 51.77|

No demonstrated end-to-end throughput improvement; cached timings are close,
while live-loader timing varied more (loader-only means5.27/11.00ms in new runs).
These are sequential historical comparisons, not an interleaved controlled A/B.
Launch count stays2325/microbatch because the removed Q/K copies were already
fused with the still-required codebook cast. Profiled preparation time falls
137.1->51.1/51.9us per microbatch, selection grows112.5->154.9/155.1us;
interpolation forward/backward kernel time remains approximately unchanged.
Profiled durations are separate from the unprofiled wall-clock table above.

Artifacts: nanochat runs/dism72m_3k/utilization_embedding_bnhd{,_repeat}.{log,json}.

### LSE gradient view follow-up

Native core_backward now exposes its BHN LSE-gradient storage as BNH views,
without materializing contiguous BNH copies. The compiled interpolation path
transposes these views back to contiguous BHN before any precision conversion.
This removes the two BHN->BNH->BHN copy round trips without changing the public
BNH shape or adding a dispatch option. dK-LSE's required BF16->FP32 conversion
remains; dQ-LSE is already FP32. Interpolation's contiguous checks are now no-ops
on this path. Kernel math and gradient rounding are unchanged.

The native frontend regression checks fixed/varlen gradient strides and aliases,
and profiles their preparation to reject clone/layout copies while allowing the
necessary dtype conversion. Logs: build/lse_gradient_views_{build,tests}.log.
Validation: all73 native frontend, BNHD interpolation and Dynamo tests passed,
including fixed/varlen compiled forward/backward and dynamic document counts.
