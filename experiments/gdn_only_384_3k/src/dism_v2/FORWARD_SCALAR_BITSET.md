# Forward scalar cleanup and hard_bits A/B (2026-09-09)

Default OUTPUT Q reuse is now conservative `DISM_OUTPUT_Q_ALIAS=kv`, as
requested for the mixed main workload. This only changes D64/DV64; explicit
`none` and experimental `k` remain available. `DISM_ROW_BITSET` remains **0**.

## Scalar changes and validation

Summary and OUTPUT mixed specializations call `row_hard<true>`, omitting
probability==0/1 checks. Compile-time soft/hard modes already avoid this RNG.
Unspecialized backward/embedding retain their cheap endpoint checks. RNG
identity, generator consumption and bitset encoding are unchanged.

Forward online softmax and shared log-affine use explicit
`ex2.approx.ftz.f32`; results below FP32 normal range flush to zero. This is
intentional, not an assertion of IEEE subnormal equivalence. Chunk passing
still uses max + log1p(exp2(-abs)) * LOG2E (now FTZ exp2), not tanh LSE.
Shared affine also affects backward recomputation. Embedding and other direct
backward weight exp2f sites were not changed in this scoped forward cleanup.

D64/DV64 mixed q-direction production SASS, static counts before → after:

| Mode/kernel | FMUL | FSETP | MUFU.EX2 |
|---|---:|---:|---:|
| full OUTPUT |467 → 211|315 → 185|128 → 128|
| full summary |157 → 81|155 → 115|38 → 38|
| finite OUTPUT |227 → 91|75 → 5|68 → 68|
| finite summary |5 → 5|3 → 1|0 → 0|

BRA.DIV and WARPSYNC counts unchanged; removed endpoint checks had compiled
as comparisons rather than reducing those branch counts. All summary
instances remain zero-spill; existing OUTPUT spill resource records are
identical before/after (4 full and 6 finite instances, D128/DV128).
Codegen gates pass for CALL, native TMA and register reallocation.
Raw: `benchmarks/forward_ex2_codegen_sm120a.json`.

Current default-kv validation:

- full core/replay/codegen/bitset/label suites:263 passed;
- finite replay/codegen/bitset/label suites:130 passed;
- new isolated exp2 boundary/RNG equivalence/SASS probe:3 passed;
- full backward/recompute:257 passed, existing backward zero-spill gate
  deselected because it conflicts with previously accepted spills;
- selected full/finite autograd:507 passed each,398 deselected each; these
  are wiring/training/replay checks, not a claim that known strict precision
  failures disappeared;
- default-kv replay memcheck/racecheck/synccheck:10 each, zero errors/hazards.

No tolerance changes. Existing strict-oracle approximation/quantization
limitations remain. Scalar tests live in `tests/test_dism_v2_forward_scalar.py`.

Before/after scalar cleanup: finite, bitset OFF, two rounds ×30 launches,
order reversed in round2; actual CUDA embedding inputs. Pooled launch medians:

| Direction/kernel | Before us | After us |
|---|---:|---:|
| q_from_k summary |221.967|222.575|
| q_from_k OUTPUT |416.078|405.358|
| k_from_q summary |233.439|233.470|
| k_from_q OUTPUT |448.094|446.878|

OUTPUT improves about2.6% /0.3%; summary effectively unchanged.
Raw: `benchmarks/forward_ex2_timing_sm120a.json`. Before binaries are saved
outside the repository under `/tmp/dism-before-ex2-ftz/`.

## hard_bits performance with the updated code

RTX5090, CUDA13.1/sm120a, BF16 B64/H4/N1024/D=DV64, vocabulary512,
scale1, rtau3, hard_prob0.5, tanh_finite, lineinfo enabled, default kv.
CUDA embedding forward and paired CUDA embedding backward, token32 vocabulary
backward configuration selected by current defaults. Same replay RNG state
for both modes. Bitset is32KiB, generated in the embedding epilogue with no
extra kernel launch; backward reuses it. Triton embedding does not produce
this bitset, so setting the flag alone with that backend does not enable it.

CUPTI, A/B then B/A,10 warmups per measurement,50 iterations per measurement:
100 samples per mode/direction. Values below are mean GPU kernel time per
iteration, **not CPU wall time or a complete three-layer training benchmark**.
Forward-only and forward+backward are separately measured contexts.

| Forward component | q off us | q on us | k off us | k on us |
|---|---:|---:|---:|---:|
| embedding |779.118|780.136|772.083|776.372|
| summary |231.149|232.853|249.676|246.400|
| passing |38.140|32.375|32.486|36.716|
| OUTPUT |426.089|427.101|479.155|472.868|
| total |1474.495|1472.465|1533.400|1532.356|

Full forward+backward GPU totals: q14280.122 →14200.758us, k14055.513
→14023.659us. Forward total gains are0.14%/0.07%; forward+backward gains
are0.56%/0.23%. Backward value kernel improves2458.359 →2405.743us(q)
and2401.854 →2357.022us(k); operand kernel3413.263 →3382.339us(q)
and3183.460 →3124.547us(k).

No locked clocks. Even the unchanged passing kernel shows several-us shifts,
so these small total gains are not evidence of a stable throughput advantage.
RNG is already cached outside the key loop in forward; replacing it by a
global bitset read need not improve that kernel. Do not default-enable based
on this batch. Keep the explicit experiment for future backward optimization.

Raw files: `benchmarks/row_bitset_forward_{q,k}_sm120a.json` and
`benchmarks/row_bitset_training_{q_from_k,k_from_q}_sm120a.json`.
Reproduce (omit `--forward-only` for full backward; repeat other direction):

```bash
DISM_TILE_LSE=tanh_finite DISM_LINEINFO=1 /home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.benchmark_row_bitset --forward-only --direction q_from_k --repeats 50
```
