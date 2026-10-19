# Planner hard-backward tau precision investigation

Investigated from detached worktree `fe026fcb6dc87cdbd61afaa54be3ddaa038c9fc9`
on an RTX 5090, using conda `blkw`. The native CUDA extension was the explicitly
permitted, read-only prebuilt binary in the main tree. Python sources, native
CPU planner JIT sources, Triton kernels, caches, and logs came from this worktree.
No training, checkpoint, live-run, production configuration, RNG, or optimizer
changes were made. `DISM_HARD_PLANNER=0` still bypasses planning.

## Finding and fix

The discrepancy is cancellation amplified by two BF16 rounding sites in the
three-stage event backward, not an incorrect prefill derivative table or a
missing cross-chunk derivative. Both compact and ordinary fused plans use these
kernels:

1. `_chunk_summaries` rounded weighted `sk*ks` and `dout*qs` to BF16 before
   constructing the forward and reverse summaries H/G.
2. `_chunk_backward` rounded carried H/G to BF16 for vector contractions, then
   reused those rounded contractions to compute tau.

For a cross-chunk pair, tau adds a query term proportional to `dlogF_q-dlogZ`
and a key term proportional to `dlogF_k`. These can each grow with match length
while their sum is small. Independent rounding in the two contractions destroys
cancellation. The local-pair branch already adds the derivatives *before*
contracting tau, using FP32 dot results; it does not incur this particular
carried-state rounding defect.

The fix constructs H/G with FP32 operands and TF32x3 MMA, and separately contracts
tau against those FP32 states with TF32x3. Vector readout MMAs remain BF16 under
`mma_precision='bf16'`, and local-pair math is unchanged. Vector outputs can
change at BF16 rounding scale because their shared summaries are more accurate.
This is a deliberate precision change required by the fix. Launch shapes,
chunk sizes, stream ordering, metadata, buffers, global reduction structure,
CUDA C++ sources, and dispatch defaults are unchanged. No quadratic substitution
is used for event requests.

Ablation on one hard row, two heads, N=512, all labels equal, tau=.5,
seed=918 vectors and seed=77 dO (both scaled by .2):

| Computation | rtau relative L2 vs maintained FP64 autograd |
| --- | ---: |
| Original BF16 event, compact | 2.1812e-2 |
| Original BF16 event, ordinary | 2.1812e-2 |
| Original TF32x3 event | 1.10e-5 |
| Accurate tau readout only | 1.3768e-2 |
| Accurate summaries only | 9.8768e-3 |
| Both changes | about 1e-5 |
| Quadratic recompute | 2.0522e-6 |

FP64 rtau is `[-0.290117925696, 0.780322601649]`; the original BF16 event
returns approximately `[-0.274483, 0.771087]`. The compact plan has 2516
single-chunk blocks and 26 multi-chunk streams (84 multi chunks), with a maximum
of eight chunks per stream. The fixture therefore exercises the requested long
matching suffix with just one training row. Peak PyTorch allocation for the
complete detailed diagnostic is about 92 MiB.

## Algebra and boundary audit

Let `F_s = sum(l=1..s, exp(l*tau))`. `prefill.hpp` stores `log(F_s)` in
`logweights` and `D_s = d log(F_s)/d tau` in `derivatives`. Its recurrence is

```
D_s = exp(logF_(s-1)-logF_s)*D_(s-1) + exp(s*tau-logF_s)*s.
```

Thus the stored derivative is logarithmic, not `dF/dtau`. At tau=0 it is
`(s+1)/2`; it remains stable for positive and negative tau. The scalar prefill
pass accumulates each stream's weighted mass and derivative, then combines them
into `logden` and `dlogden` using normalized log-sum weights. The denominator
starts at log(1)=0 with derivative zero: the zero-valued fallback bucket appears
exactly once in Z and requires no numerator event. There is no extra or omitted
fallback-gradient term.

Key event `dlogweight` stores `dlogF_k`; query event `dlogweight` stores
`dlogF_q-dlogZ`. Numerical pivot rescaling is a stop-gradient choice: key,
query, and decay factors cancel pivots in the represented pair weight. Adding
pivot derivatives separately would double-count a numerical gauge. Dlog arrays
are computed in FP64 on the CPU and stored in FP32 for the GPU; finite FP32
coefficient/reduction error remains after this fix.

`chunk_program` cuts at reset events, resets each chunk's local log-prefix to
zero, and copies event derivatives without accumulating them a second time.
The three-stage scan uses exclusive forward/reverse boundaries. A reset sets
alpha and the reverse summary to zero, while retaining the current forward key
summary. Reverse state therefore cannot leak past a reset. Local pairs are
counted once in the local contraction; cross-chunk query and key derivative
terms are each counted once in their respective boundary contractions. Finally,
chunk contributions are atomically summed per stream and scattered once to the
correct shared/per-row tau group. Compact single-block and multi-stream tau
slots are disjoint. The unchanged native, compact, reset, and graph tests cover
these layouts.

The separate register-scan implementation in `hard_backward_scan.py` uses the
same derivative algebra and correct reset logic. Its optional BF16 arithmetic
also rounds states/vector contractions; it is a diagnostic alternative, not the
three-stage implementation selected by `AsyncHardPlanner`. It is not repaired
by this patch. Its TF32x3 comparison remains covered by the existing tests.

## Forward

This defect affects backward tau. Training event and quadratic policies both
call the unchanged `forward_hard`; derivative metadata never feeds forward.
The minimal fixture's BF16 forward relative L2 against FP64 is 2.35255e-3,
cosine 0.999997234, norm ratio 1.000044302, signed mean error -2.29762e-7.
Forward metrics are identical before/after. Native derivative/weight tests and
the FP64 event evaluator also validate the geometric sum and normalization.

The old sweep's `block/chunk` ratio is only a topology proxy. Identical query/key
labels still match along the complete diagonal across alternating-symbol block
boundaries, and a planner chunk counts **events**, not tokens. The tool and test
comments now make this distinction explicit.

## Commands and evidence

All commands below run from the isolated worktree root. Logs, generated resource
JSON, SASS, binaries, and JIT caches remain ignored under `build/`.

```bash
PY=/home/cicuvc/miniconda3/envs/blkw/bin/python
export PYTHONPATH="$PWD/dism_v3/python:$PWD/dism_v3/tests:$PWD/dism_v3/tools"
export TORCH_EXTENSIONS_DIR="$PWD/build/torch_extensions"
export TRITON_CACHE_DIR="$PWD/build/triton"
export MAX_JOBS=2 OMP_NUM_THREADS=1
ln -s /home/cicuvc/cs/project/dism-core/dism_v3/python/cu_flash_dism.cpython-312-x86_64-linux-gnu.so \
  dism_v3/python/cu_flash_dism.cpython-312-x86_64-linux-gnu.so

"$PY" dism_v3/tools/route_tau_sweep.py detailed
"$PY" dism_v3/tools/route_tau_sweep.py chunk
"$PY" -m pytest -q -s dism_v3/tests/test_hard_route_consistency.py
"$PY" -m pytest -q dism_v3/tests/test_hard_prefill_backward_fused.py \
  dism_v3/tests/test_hard_training.py dism_v3/tests/test_hard_ready_dispatch.py \
  dism_v3/tests/test_disabled_hard_planner.py \
  dism_v3/tests/test_native_training_planner.py \
  dism_v3/tests/test_compact_training_planner.py \
  dism_v3/tests/test_hard_prefill_training_reference.py
"$PY" -m pytest -q dism_v3/tests/test_disabled_hard_planner.py
"$PY" -m pytest -q -m strict_gradient dism_v3/tests/test_backward.py \
  dism_v3/tests/test_backward_qk.py dism_v3/tests/test_backward_summary.py \
  dism_v3/tests/test_voc_backward.py
```

Only the CPU planner needed checkout JIT compilation (`MAX_JOBS=2`); no native
CUDA source changed, so rebuilding `cu_flash_dism` with `build.py` was unnecessary.
Triton compiled the changed GPU kernels in this worktree's cache.

## Final gradient measurements

Signed bias below is `mean(actual-reference)` over every tensor element.
Cosine and ratios are undefined for structural zero q/k gradients; their
maximum absolute error is exactly zero. Values vary slightly with FP32 atomics.

| Actual vs reference | Component | Cosine | Relative L2 | Norm ratio | Signed bias |
| --- | --- | ---: | ---: | ---: | ---: |
| event / quadratic | q_vec | N/A | N/A | N/A | 0 |
| event / quadratic | k_vec | N/A | N/A | N/A | 0 |
| event / quadratic | sq_vec | 0.999999971 | 0.00024130592 | 1.00000195 | 4.75387196e-09 |
| event / quadratic | sk_vec | 0.999999973 | 0.000233470694 | 1.00000161 | -1.1155862e-08 |
| event / quadratic | v | 0.999999968 | 0.000251305607 | 1.00000114 | -6.04694606e-09 |
| event / quadratic | rtau | 1 | 9.01604329e-06 | 1.00000758 | 4.47034836e-06 |
| event / FP64 | q_vec | N/A | N/A | N/A | 0 |
| event / FP64 | k_vec | N/A | N/A | N/A | 0 |
| event / FP64 | sq_vec | 0.999997275 | 0.00233441421 | 1.00001717 | -3.66240783e-07 |
| event / FP64 | sk_vec | 0.999997244 | 0.00234822816 | 1.00003643 | -2.02143733e-07 |
| event / FP64 | v | 0.999997169 | 0.00238017204 | 0.999937896 | -7.10319933e-09 |
| event / FP64 | rtau | 1 | 8.22961231e-06 | 1.00000578 | 4.55282096e-06 |
| quadratic / FP64 | q_vec | N/A | N/A | N/A | 0 |
| quadratic / FP64 | k_vec | N/A | N/A | N/A | 0 |
| quadratic / FP64 | sq_vec | 0.999997277 | 0.00233391312 | 1.00001522 | -3.70994655e-07 |
| quadratic / FP64 | sk_vec | 0.999997251 | 0.0023451681 | 1.00003482 | -1.90987871e-07 |
| quadratic / FP64 | v | 0.999997167 | 0.00238106554 | 0.999936753 | -1.05625328e-09 |
| quadratic / FP64 | rtau | 1 | 2.05223719e-06 | 0.999998197 | 8.24726041e-08 |

The detailed command also prints ordinary-plan and TF32x3 event results.
The 18-case symbol-block/chunk-size sweep (C16/32/64; block16..512) reduced
the largest observed event/quadratic tau relative error to 2.29e-5.
The original long-block C16 case was 1.61e-1.

## Regression and diagnostic outcomes

- Original route suite: 4 passed, 4 expected failures. With the new regressions
  and original kernel restored temporarily: 11 failed, 3 passed (8 real route
  failures and 3 synthetic boundary failures). The latter bypass the native
  planner entirely, isolating the numerical fault to GPU carried-state math.
- Final route + fused suite: **88 passed** (`focused-final.log`). This includes
  random/equal/no-match patterns, single/exact/partial/multiple chunks, multiple
  streams, packed documents, compact/ordinary plans, negative/zero/positive tau,
  supported vector dimensions, graph replay, and reset boundaries. Existing
  pointwise, gradient-acceptance, and strict numerical gates were not loosened.
- Broader integration/native/compact/CPU-reference command: **365 passed,
  5 failed** (`validation.log`). Four failures are the existing GPU-prefill
  diagnostic's obsolete token-level hard mask; all four also fail on the
  original kernel (`legacy-prefill-before.log`). The fifth is a test-isolation
  issue: `test_disabled_planner_matches_the_quadratic_policy[hard]` sees old
  `'e'` entries in the process-global route census after earlier tests. Its
  gradients pass, and the disabled-planner file passes **3/3** in a fresh process
  (`disabled-isolated.log`). These unrelated tests were not edited or hidden.
- Original strict diagnostics: **6 failed, 72 skipped, 21 deselected**, both
  before and after (`strict-before.log`, `strict.log`). The six vocabulary
  fixtures use N=65 and fail the existing 256-token alignment contract before
  comparing gradients. The prebuilt extension has `fp32_enabled() == False`,
  so 72 optional FP32 tests cannot run. No FP32 CUDA rebuild was attempted.
- A multiplication-order diagnostic initially kept the scalar weight inside
  each tau inner reduction. Two new synthetic cases then failed at relative
  errors 2.39e-4 and 2.29e-4, even though sanitizers reported zero hazards.
  The final fix factors that scalar outside the inner reduction and passes
  the same 2e-4/2e-6 test without changing its data or tolerance. Intermediate
  failures remain in `*-mulorder.log`.

The final focused command was:

```bash
"$PY" -m pytest -q dism_v3/tests/test_hard_route_consistency.py \
  dism_v3/tests/test_hard_prefill_backward_fused.py
```

The failing-before command (run with only `hard_backward_fused.py` temporarily
restored from the original source, then restored in a `finally` block) was:

```bash
"$PY" -m pytest -q dism_v3/tests/test_hard_route_consistency.py \
  dism_v3/tests/test_hard_prefill_backward_fused.py \
  -k 'multi_chunk_streams_route_invariant or bf16_tau_cancellation_at_chunk_boundaries'
```

## Sanitizers, compiled resources, and limits

PyTorch 2.13.0+cu130, Triton 3.7.1, Compute Sanitizer 2025.4.1.0. Final unfiltered
memcheck, synccheck, and racecheck each ran **8 passing tests**. Memcheck and
synccheck reported **0 errors**; racecheck reported **0 hazards / 0 warnings**.
No unsupported-instruction warning or sanitizer exception was observed in these
runs. These are small smoke tests on sm120, not exhaustive synchronization
proofs or validation on other architectures.

```bash
for tool in memcheck synccheck racecheck; do
  /usr/local/cuda/bin/compute-sanitizer --tool "$tool" \
    --target-processes all --error-exitcode 86 "$PY" -m pytest -q \
    dism_v3/tests/test_hard_training.py \
    dism_v3/tests/test_hard_prefill_backward_fused.py \
    -k 'async_hard_long_stream_safety or bf16_tau_cancellation_at_chunk_boundaries'
done
```

Resource inspection covered C16/32/64, R16/32, DV32/64, and both BF16 and TF32x3.
N=65 for C16/32 and N=129 for C64 ensures multi-chunk kernels are actually
launched. `inspect_kernel` disassembles each cubin with
`/usr/local/cuda/bin/cuobjdump -sass`. Every inspected kernel had **0 CALLs and
0 shared atomics**, with at most 32768 shared bytes. The summary scan remains
an elementwise, disjoint-owner scan with no inter-CTA communication.

The BF16 multi-chunk readout's resource change is real:

| C | R/DV | Registers before → after | `n_spills` before → after |
| ---: | --- | ---: | ---: |
| 16 | 32/64 | 130 → 255 | 0 → 0 |
| 32 | 32/64 | 157 → 255 | 0 → 10 |
| 64 | 32/64 | 254 → 255 | 10 → 70 |
| 64 | 16/64 | 254 → 255 | 10 → 78 |

All summary kernels were spill-free. The optional full-TF32x3 kernels already
have much larger spill counts (up to 674 in this matrix). No launch tuning,
shared-memory spill mitigation, or relaxed precision was introduced to conceal
this cost. Resource diagnostics also checked the original gradient-acceptance
gates for all 24 cases and reported no failures. The exact inspection loop was:

```bash
"$PY" - <<'PY'
from pathlib import Path
import json
import torch
from check_hard_prefill_backward_triton import check_case

torch.set_num_threads(1)
root = Path('build/planner-investigation/resources')
rows = []
for c in (16, 32, 64):
    for r, dv in ((16, 32), (16, 64), (32, 32), (32, 64)):
        for precision in ('bf16', 'tf32x3'):
            result = check_case('repeat', 129 if c == 64 else 65, .5,
                                r=r, dv=dv, chunk_size=c, precision=precision,
                                fused=True, resource_dir=root)
            rows.append(result)
            print(c, r, dv, precision, result['kernels'])
root.with_suffix('.json').write_text(json.dumps(rows, indent=2)+'\n')
PY
```

Remaining limits: FP32 metadata, summation, and cross-CTA atomics still incur
finite precision and nondeterministic last-bit changes; this is not an exact
arbitrary-length proof. The legacy BF16 register-scan option is not fixed.
Optional FP32 native instances and other GPU architectures were not rebuilt or
tested. The new MMA/spill cost has not been benchmarked under an idle GPU; the
live training job was left running and test processes stayed below a few GiB
(observed broad-test process usage about 1.1 GiB including context). Production
planner backward remains disabled/configured exactly as before. Stale strict
fixtures and route-census test isolation remain unimplemented separate fixes.

Changed files are `python/flash_dism/kernels/hard_backward_fused.py`,
`tests/test_hard_route_consistency.py`, `tests/test_hard_prefill_backward_fused.py`,
`tools/route_tau_sweep.py`, and this report. No source outside `dism_v3` changed.
