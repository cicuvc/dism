# Fixed-true vocabulary interpolation

The explicit `direction_true=True` CUDA training specialization computes
`beta * q @ E[p(k)](Q_vocab) - LSE(beta*q@Q_vocab.T)` without constructing the
unused `E[p(q)](K_vocab)` vector. Q and K still both need vocabulary scores;
both labels remain available to hard forward and the asynchronous CPU planner.

## Implementation

- `kernels/interp_true.py`: two independent launches. Q computes LSE and argmax
  (plus the existing top1 diagnostic); K computes the ordinary interpolated
  vector, LSE and argmax. Same64-token/64-codeword tiling and online softmax
  conventions as the dual forward. The unused interpolation output is an empty
  tensor, not an allocated/zero-filled BNHD vector.
- `kernels/interp_bwd_split.py`: Q backward specializes `LSE_ONLY=True`. It
  reconstructs probabilities and forms `dscore=p*dLSE`; no dO/delta/value load,
  dP matrix product or value-gradient product/atomic. K retains normal FA2
  backward. The two sides still accumulate into the shared global FP32 table
  gradients, preserving the existing autotune restore semantics. LSE_ONLY is
  part of the tuning key; generic launch candidates are unchanged.
- One-sided delta preprocessing reads only the retained K->Q output. Compiler
  glue bypasses direction-dependent operand selection/split when fixed true.
  Input/gradient BF16 rounding and head-shared table reductions are retained.
- Host-beta custom ops are versioned v2 for the new boolean specialization and
  empty auxiliary output. Autograd saves the selected mode with the invocation.

## Dispatch contract

`compiler.voc_forward(..., direction_true=True)` is a semantic override: it
creates all-true directions, rather than checking a GPU mask with all()/item().
It requires host scalar beta. Explicit tensor-beta callers retain the generic
compatibility entry. `DismAttention` exposes the same keyword for uncached CUDA
training/evaluation; cached/reference inference is unchanged.

Nanochat selects the fast path for its statically known true training policy
unless direction_flip is requested, and for its all-true evaluation direction.
Random, cycle/late-true and flipped training calls keep the generic path, even
if a particular sampled mask happens to be all true. Existing sampling/RNG
counter updates still execute unchanged. No new checkpoint field or model
parameter is introduced; hard masks remain packed-row/head decisions.

## Native varlen specialization

The same host `direction_true` selection now reaches the native packed core.
F1, F3, B1 and B3 each retain their runtime-direction instance and additionally
instantiate a `DirectionTrue=true` kernel. The static instance does not load
the direction flag, publish/read its shared-memory copy, load K-LSE, or execute
the K-LSE branch. B3 omits dK-LSE reduction/writeback; its already zero-initialized
output remains zero. dQ-LSE, tau accumulation, soft score math, causal scans,
TMA/cp.async slot/barrier protocols, launch geometry and hard-row skips are
unchanged. Chunk passing and dedicated hard kernels are direction-independent
and are not duplicated. Fixed-length CUDA continues to use the generic path.

`forward_varlen(..., direction_true=True)` and native `core_forward` /
`core_backward` expose an explicit host assertion, not a GPU-mask detector.
`backward_varlen` saves/reuses the forward selection; old states default to the
generic path. Native per-phase entrypoints append an optional boolean defaulting
to false, retaining old positional calls. Generic paths remain usable even with
all-true input tensors. The production vocabulary custom op propagates its
existing semantic override to both native phases; there is no new model option,
RNG counter, checkpoint buffer, host-device synchronization or task upload.

Tests: `tests/test_varlen_direction_true.py` compares all eight R/D/DV shapes,
soft/mixed/hard row masks, empty documents/packs, nonconstant LSE, saved boundaries
and all eight gradients against the retained general kernels. Poisoned unused
K-LSE and overridden native direction flags test branch independence. Existing
codegen checks now require both instances and retain the no-CALL/Torch-free
device-TU checks. `tools/bench_varlen_direction_true.py` alternates both variants
on identical operands to report F1/F3/B1/B3 CUDA-event times (including native
allocation/launch gaps, not kernel-only sums).

Validation on RTX 5090 (sm120), 2026-10-15:

- All eight configurations built with `build.py`, `DISM_BUILD_JOBS=2`.
- Initial numerical/codegen/interpolation regression: 108 passed, 18 skipped
  (optional FP32 output instances disabled). The subsequently added nondefault
  stream/packed-row test is included in each sanitizer run below.
- Memcheck, synccheck and racecheck each pass all 27 new core tests, zero errors;
  racecheck reports zero warnings/hazards. This is not a claim that historical
  FP32/debug instances or all unrelated model kernels were sanitized.
- Both variants across all eight shapes pass no-CALL and Torch-free device-TU
  checks. R32/D64/DV64 per-thread stack bytes (generic -> static true):
  F1 16 -> 8, F3 24 -> 32, B1 304 -> 296, B3 176 -> 160. Existing spills remain;
  notably F3's stack grows, so specialization is not claimed spill-free.
- At T32768/H6, R32/D64/DV64, sixteen 2048-token documents, all soft, median
  phase times over 20 alternating samples (generic -> static true, ms):
  F1 .2465 -> .2426; F3 .5523 -> .5301; B1 1.9306 -> 1.8495;
  B3 1.8873 -> 1.7600. This is an isolated-stage measurement, not whole-model MFU.

Ignored artifacts: `build/varlen_true_build.log`, `varlen_true_tests.log`,
`varlen_true_{memcheck,synccheck,racecheck}.log`, `varlen_true_resources.log`,
and `varlen_true_bench.{json,log}`. Existing gradient tolerances and BF16
delta/tau cancellation limitations are unchanged.

72M checkpoint-4000 scratch replay (r3 config, four microbatches, metrics probes
enabled, three warmups and three six-step sustained windows) measured 623.956ms
generic versus 622.776ms static true. This 0.19% difference is within short-run
variation, **not evidence of a significant end-to-end throughput improvement**.
Both runs keep the already optimized interpolation and metrics; the baseline
overrides only the final native core direction-specialization boolean. The
post-timing Torch profile confirms 24 calls each to static-true F1/F3/B1/B3 and
no generic instances of those phases. Both runs end with zero live plan handles.
Artifacts: `build/varlen_true_model_before.{json,log}` and
`build/varlen_true_model_after.{json,log,trace.json}`. No training restart or
checkpoint mutation occurred; repeated scratch batches are not a convergence
test.

Expanded regression (`build/varlen_true_regression.log`) is **197 passed,
9 failed**, not a clean full-suite pass. Eight failures are
`test_multi_config::test_vocabulary_wrapper` R32 cases: its `torch_voc` reference
explicitly zeroes compact-mask hard heads (the old soft-only contract), whereas
the public production wrapper now computes hard attention. Those tests do not
select this new specialization. The ninth is `test_core_autograd[hard]`, a
packed-versus-separate-doc tau comparison with absolute error 5.98e-5 against
the unchanged 2e-6 + 2e-5 relative acceptance. A diagnostic forcing the existing
hard `event` policy passes that case; the default ready policy may select the
quadratic versus event backward at different times. That explanation is
consistent with the diagnostic, not a claim of a new numerical fix.

The isolated diagnostic asserts every native core call has direction_true=false
(20 forwards / 12 backwards) and reproduces the eight vocabulary failures while
the pure-hard test passes: `build/varlen_true_fallback_diagnostic.log`. In that
diagnostic run no oracle, tolerance, xfail or hard-policy production default
was changed. Module Dynamo, direction-policy and checkpoint tests in the expanded
run passed. The full-suite limitations remain separate from the new-path
equivalence and sanitizer results above.
`build/varlen_true_oracle_diagnostic.log` additionally confirms all eight
fixtures' soft-head outputs meet the original tolerance; only the four R32
fixtures contain a hard head, whose reference output is identically zero while
production's maximum magnitude is .106--.161. This diagnostic does not replace
the original failing tests.

Follow-up test correction: `torch_voc` now expands compact sequence masks to
`[B,H,N]` and evaluates the full hard attention via the unchanged `dism_ref`,
instead of masking hard outputs to zero. Legacy token-mask oracle behavior is
unchanged. Complete vocabulary wrapper coverage now spans soft/mixed/hard,
all eight shapes and fixed/varlen execution (48 cases), with an explicit nonzero
hard-reference assertion on matching-label fixtures. The whole
`test_multi_config.py` passes 124 tests using the original output/gradient
tolerances (`build/hard_oracle_tests.log`); its separate low-level soft-only
core tests still require hard heads to return zero. This resolves the eight
stale-oracle failures, not the separate ready-policy hard tau comparison above.

## Numerical expectations

Splitting the forward changes compiler reduction/FMA layout. The initial exact
FP32 comparison failed at about7.2e-7 absolute LSE error; that diagnostic remains
in `build/interp_true_tests.log`. The new tests require exact retained BF16 output
and labels on their fixtures, standard Torch FP32 closeness for LSE/top1, and the
existing independent FP64 gradient acceptance policy. This is not a general
bitwise-equivalence guarantee for whole-model training. Existing production
gradient tolerances and tau/delta numerical limitations are unchanged.

No training restart or checkpoint mutation is part of this implementation.

## RTX5090 measurements

Isolated B1/T32768/H6/D64/V512, beta1, three alternating-order benchmark rounds:
forward about0.277->0.228ms (18% less), backward about0.991->0.764ms (23% less).
The unused saved BNHD interpolation is24MiB per layer/microbatch. See
`build/interp_true_bench.json` and `interp_true_bench_resources.json`;
`tools/bench_interp_true.py` reproduces the timing and selected-kernel inspection.

72M r3 checkpoint4000, hard probability0.1992, same four resident saved batches,
B16/N2048, four accumulation microbatches, document-balanced CE, direct CUDA
shortconv, RMS probes, compact ready planner with16 workers. Three warmups,
two phase samples and two six-step throughput windows; scratch optimizer updates
only, no dataloader advance or checkpoint writes.

| Measurement | Generic true-direction tensors | Fixed-true specialization |
| --- | ---: | ---: |
| Step window1 / window2, ms | 656.06 / 652.39 | 646.29 / 644.23 |
| Mean step, ms | 654.22 | 645.26 |
| Throughput, tokens/s | about200.3k | about203.1k |
| Forward event-window mean, ms | 261.35 | 260.85 |
| Backward event-window mean, ms | 374.85 | 365.25 |
| Peak allocated CUDA memory, MiB | 17205.53 | 17062.25 |

About1.4% lower step time and143.3MiB lower peak allocation in this short replay,
not a sustained pretraining convergence result. Phase event windows include
launch gaps and asynchronous dependencies, not additive kernel-only budgets.
Both diagnostic steps used24 event backwards and ended with zero live handles.
The final Torch profile still has65 scalar reads/97 stream synchronizations:
this specialization adds no direction-mask readback. Initial printed loss/norm
match; repeated scratch updates are not bitwise identical.

Raw model artifacts: `build/interp_true_model_before.json`,
`interp_true_model_before_retry.log`, `interp_true_model_after.json`, `.log`,
`.trace.json`. The before run explicitly forces the generic branch through a
process-local wrapper. The first baseline launcher failed on a runpy/Dynamo
Python-global guard, before producing measurements; its original log is retained.

## Kernel safety and code generation

Final regression:169 tests pass (`build/interp_true_final_tests.log`). Includes
fixed/tuned primitive FP64 gradients, tied-score/aliased-QK regression, soft/mixed/
hard fixed and packed operators, actual fullgraph Inductor, generic interpolation
oracle/autotune/CUDA-Graph regression, module shape tests, model direction-policy
and flip dispatch, diagnostics, RNG counters and checkpoint resume.

Memcheck passes9 primitive cases with autotune, zero errors. Racecheck and
synccheck each pass9 fixed-launch primitive cases, zero errors; racecheck has
zero warnings. These cover D32/64, vocab37/512, strided inputs and three beta
values, not a new sanitizer campaign for the unchanged native DISM pipeline.
The final fixed-launch memcheck rerun also covers tied labels and aliased Q/K:
10 cases pass with zero errors (`build/interp_true_memcheck_final.log`).
The first racecheck run with autotune was stopped due to instrumentation cost;
only `interp_true_racecheck_fixed.log` is a completed racecheck result.

Selected production-sized kernels: Q-LSE forward120 registers, K-interpolation
forward186, delta40, Q dQ/dtable136/134, K dQ/dtable190/123. All seven selected
kernels have zero register spills and no CALL instructions. The wider cache
inspection found some autotune candidates with up to32-byte stacks; this is
not a claim that every candidate is spill-free. Resource data and SASS checks
are in `interp_true_bench_resources.json` and `interp_true_sass.log`.
