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
