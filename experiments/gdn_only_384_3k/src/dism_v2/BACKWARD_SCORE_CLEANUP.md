# Backward score and finite-zero cleanup experiments

Subsequent user selection: OPT13 is now the default (stages2). Earlier
default11 statements below describe the experiment before this decision.
The documented strict numerical failures remain visible and tolerances
are unchanged. OPT14's independent dA writer remains an opt-in experiment.

User-requested follow-up after the OPT11 optimization cycle. Default remains
OPT11. Downstream evaluation focuses on `DISM_TILE_LSE=tanh_finite`; full
results below belong to the already completed OPT12 comparison, not a
requirement to optimize full mode further.

## OPT12: preloaded score metadata and one FFMA

The branch-heavy `transposed_score` in the two WS source files is OPT0-only.
OPT11 already dispatches direction/mode/label width at compile time and uses
`bwd_metadata::score`. OPT12 additionally caches `scale2=scale*LOG2E` and
`bias2=(tau-lse)*LOG2E`, then uses `fmaf(dot,scale2,bias2)` per score.
Held-key metadata is loaded before B/V ready wait; distributed query
metadata before RNG and A/dO ready wait. No extra shared staging, no change
to TMA or mailbox protocols, B1 and B3 use the same recomputation formula.

Real SASS gate covers B1/B3 D64 mixed int32 policies in both directions:
32 score FFMAs per warp tile, no score-expression FADD/FMUL/LDG/BRA.
Full/finite each pass both codegen tests, including whole-extension noCALL,
180-instance native-TMA and register-role checks. Spill is recorded, not
repaired. Full D64/DV64 mixed B1 stack48/24B (q/k), B3 zero in both.

Numerical results, unchanged tolerances:

- Full B3 suite plus codegen:129 passed/22 failed, adding bounded-soft N64
  and65 strict independent-G/BF16-GEMM failures to the existing20.
- Finite B3+dV suite plus codegen:117 passed/131 failed. Frozen OPT11 on
  the same numerical suite:119 passed/128 failed (one fewer codegen test).
  Added IDs: B3 bounded-soft N2049 and dV reverse-summary bounded-soft N2049
  in both directions. No previously failing IDs disappear.
- Selected autograd wiring/training/replay:507 passed/398 deselected in
  each mode. This is not a claim that strict oracle accuracy passes.

Two alternating rounds,30 CUPTI samples per arm/direction/mode, actual
CUDA embedding and fixed forward states, B64/H4/N1024/D64/DV64/V512,
mixed.5, tau3, scale1, no concurrent GPU timing or clock locking:

| finite kernel/direction | OPT11 us | OPT12 us |
|---|---:|---:|
|B1 q_from_k|657.405|651.980|
|B1 k_from_q|650.428|636.669|
|B3 q_from_k|1477.335|1484.120|
|B3 k_from_q|1346.872|1334.729|

No convincing net B3 gain. Raw samples/resources/failure details are in
`benchmarks/backward_score_ffma_{timing,validation}_sm120a.json`.
Retain OPT12 as an explicit experiment, not a new default.

## OPT13: finite zero invariants and redundant masks

Builds on OPT12, enabled only for the finite sentinel path. No new full-mode
semantics. Input/state assumptions match the existing bounded finite
experiment: reachable W stays far above LOG_ZERO, finite operands/dP/delta,
normalizer includes the fallback, and reverse terminal G is zero.

| Location | Change and reason |
|---|---|
|score validity|`q<n && k<=q` already implies `k<n`; remove the latter.|
|padding query metadata|Set invalid query normalizer to+INF (a normalization mask, not a scan sentinel), making its probability exactly zero. Global metadata accesses remain guarded.|
|probability|Remove per-element q/k bounds: invalid q is suppressed by its normalizer; invalid k with valid q is noncausal, whose whole diagonal has unreachable W and zero EX2.|
|coefficient LOG_ZERO checks|At -1e6, TANH saturates to-1 and FTZ EX2 returns0, naturally giving alpha0/beta0 for a hard break.|
|coefficient validity|Keep q/k validity for alpha: missing coordinates must remain affine identity alpha1, not hard-break alpha0. Beta requires no extra mask under the probability invariant.|
|Gsoft/scalar reductions|Invalid/noncausal diagonals have beta0 and terminal G0, so reconstructed G is exactly0. Omit repeated value masks, retain hard/soft selection and output address bounds. Scalar helper opts in only from WS B3.|
|forward padding identities and all loads/stores|Keep their bounds and identity handling; normalization masking does not make out-of-bounds memory access legal.|

Validation compares OPT13 to OPT12, not the older score association:
`DISM_BWD_REFERENCE_OPT=12` selects the comparison in the existing
equivalence/epoch/int64 suite. No comparison tolerances are changed.
Initial sandbox execution skipped all tests because CUDA was unavailable;
that run is not validation evidence. GPU-enabled rerun:194 passed (192
equivalence/epoch/int64 cases plus2 codegen gates). dV, affine summaries and
passing boundaries agree bitwise with OPT12; operand/scalar gradients pass
the unchanged3e-5 tolerance. Selected autograd507 passed/398 deselected,
bitset62 passed. Memcheck/racecheck/synccheck each3 passed, zero errors/
hazards, covering mixed D32/DV32 N129, mixed D128/DV128 N257, and D64/DV64
N65 multi-workload epochs. No new shared storage or barrier count changes.

Both main B3 directions remain zero stack; B1 stack16B in both. All180
instance codegen gates pass; recorded resources and validation are in
`benchmarks/backward_zero_masks_validation_sm120a.json`.

Finite main-shape controlled CUPTI comparison, two alternating rounds of
40 samples per configuration (80 pooled samples), same fixture as OPT12:

| Kernel/direction | OPT11 us | OPT12 us | OPT13 us | OPT13 vs12 |
|---|---:|---:|---:|---:|
|B1 q_from_k|659.676|654.444|595.565|-9.00%|
|B1 k_from_q|652.284|638.237|591.100|-7.39%|
|B3 q_from_k|1481.944|1486.712|1444.456|-2.84%|
|B3 k_from_q|1347.017|1341.752|1287.177|-4.07%|

Raw samples: `benchmarks/backward_zero_masks_timing_sm120a.json`.
This mask cleanup preserves OPT12 results on the tested comparisons, but
OPT13 still includes OPT12's score reassociation and its known added strict
precision failures. Do not call it a fully passing strict-oracle replacement
for OPT11; leave default11 unchanged pending an explicit acceptance/selection.
Performance scope is the measured finite main shape, not all shapes or a
whole training-step throughput claim.

After adding both experiments, the unchanged default OPT11 finite path was
rebuilt and checked:508 passed/399 deselected (507 selected autograd cases
plus the whole-extension codegen gate). No default promotion or commit.

```bash
DISM_BWD_OPT=13 DISM_TILE_LSE=tanh_finite /home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.benchmark_backward_kernels --direction q_from_k --repeats 40
DISM_BWD_OPT=13 DISM_BWD_REFERENCE_OPT=12 DISM_TILE_LSE=tanh_finite /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_backward_trim.py tests/test_dism_v2_backward_optimization.py
```
