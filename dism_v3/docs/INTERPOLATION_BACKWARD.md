# Split-attention interpolation backward on sm120

Measured on the local RTX5090 on 2026-10-14. Environment, baseline configuration,
and measurement caveats are in [TRAINING_BENCHMARK.md](TRAINING_BENCHMARK.md).
Production interpolation backward now uses the atomic split-attention backend
with autotune. Interpolation forward, core kernels, training state and gradient
acceptance tolerances are unchanged. The original three-way experiment is
preserved in commit `5ad8226`; its measurements below remain historical evidence.

## Production atomic backend

Both eager autograd and compiled training reach `emb_kernel.emb_bwd_wrapper`,
which calls `kernels/interp_bwd_split.py`. Only the global FP32 atomic reduction
remains in that implementation; partial buffers, the final reduction kernel,
and the register-only double-scan implementation have been removed. The old
combined backward remains private as `_emb_bwd_legacy`, solely for tests and
baseline measurements, and is not selected by production dispatch.

Triton autotunes five dQ schedules and fifteen dK/dV schedules. dQ uses32/64
row and vocabulary tiles, four warps and two/three stages. dK/dV uses32x64,
64x32 or64x64 tiles, four warps, three stages and1/4/8/16/32 token partitions;
partition counts larger than the number of token tiles are pruned. Cache keys
include device ordinal, B/N/H/V/D, tau presence, input strides and tensor dtypes.
Selected configurations are cached in the process, not persisted across runs.
Expect first-use compilation/tuning overhead; warm each layout before CUDA
Graph capture. Fixed launch parameters remain available with `autotune=False`
for diagnostic comparisons, not as a separate reduction implementation.

Gradient outputs are zeroed once before either attention. Every autotune trial
uses `restore_value` for BOTH accumulated codebook gradients. Using
`reset_to_zero` instead would erase the first attention's contribution if the
second attention has different strides and also needs tuning. Trial snapshots
exist only during tuning, not on cache-hit launches; there is no split-gradient
workspace. Global FP32 atomic summation remains nondeterministic.

Current paired seven-window,50-repeat production measurement:

| Tokens | Legacy ms | Atomic autotuned ms | Fixed atomic16 ms |
|---:|---:|---:|---:|
|2048|.2127|.1027|.0886|
|8192|.7814|.2464|.2472|
|32768|3.1287|1.0125|1.0057|

Autotune is not guaranteed to beat hand tuning: small-shape timing is sensitive
to host dispatch overhead and tuning noise. In this run dQ selected32x32 at2048
tokens,64x64 at8192/32768; dK/dV selected64x32 with8/8/32 splits respectively.
At32768, full-operator backward changed8.306→6.116ms uniform and5.533→3.393ms
ragged; fresh forward+backward changed9.468→7.281ms and6.533→4.011ms.
Warm isolated peak allocation remained99MiB at32768 tokens. Worst isolated
gradient relative-L2 difference from legacy was3.81e-5.

All66 candidate cubins compiled during this benchmark were inspected: no spills
or CALL. dQ registers80–190, shared26624–49152B; dK/dV registers77–246,
shared20480–58880B. These resource checks are for the measured H6/D64/V512
configurations, not every possible shape or other GPU.

Production validation includes cold/warm autotune with deliberately different
Q/K layouts and changed upstream gradients, FP64 oracle checks, legacy
regressions, compiled custom-op autograd and warmed CUDA Graph replay. Final
regression run:290 passed across `test_interp_bwd_split`, `test_embedding_bnhd`,
`test_voc_glue`, `test_multi_config`, `test_native_frontend`, `test_module`,
`test_module_dynamo`, `test_voc_training`, `test_varlen_autograd` and `test_readme`.
This includes dynamic packed layouts, shared vocabulary, all eight native
configurations and short synthetic optimizer smoke tests, not a resumed
training run or a convergence study. The historical exact BNHD/BHND
layout-migration test still exercises the legacy kernel without relaxing its
bitwise assertions; production atomic correctness is checked separately because
its FP32 reduction order differs. New memcheck (allocator disabled, padding4096)
and synccheck runs each passed37 selected oracle/cold-warm tests with zero
errors. No racecheck or new strict-gradient campaign was run; existing BF16
limitations remain applicable.

## Historical implementations and contract (`5ad8226`)

Interpolation is two attentions: `(q, E_q, E_k)` and `(k, E_k, E_q)`.
Each attention has a row-owned dQ kernel and a vocabulary-owned dK/dV kernel.
The latter can partition the token axis into16 splits. Preserve saved natural
log LSE-minus-tau, the extra dLSE term, BF16 dot inputs, FP32 accumulation and
beta placement. Tau-gradient ownership remains in the core backward.

- Partial: write four FP32 split-gradient arrays, then reduce in fixed order.
- Atomic: directly accumulate into two zeroed final arrays with global FP32
  atomics; summation order is nondeterministic. No shared FP32 atomics.
- Register: each CTA owns one codebook tile, scans all tokens for its key-role
  gradient, then rescans the opposite attention for its value-role gradient.
  Both contributions stay in one register accumulator until the final store.
  Both codebooks share a launch, but no token splitting is possible.

Default experiment tiles: dQ64x64, dK/dV64x32, four warps, three stages.
Register-only additionally tried64x64 vocabulary tiles. Strided BNHD/HVD inputs,
B>1 and masked vocabulary tails are covered. Training N remains256-aligned.

## Results

Interleaved, alternating-order CUDA-event windows including allocation/dispatch;
seven windows of50 invocations, H6/D64/V512, all interpolation gradients active:

| Tokens | Original ms | Partial16 ms | Atomic16 ms | Register64x32 ms |
|---:|---:|---:|---:|---:|
|2048|.2134|.0957|.0885|.1327|
|8192|.7866|.2672|.2485|.4477|
|32768|3.146|1.073|1.024|1.777|

A longer11x100 repeat at32768 tokens measured partial1.068ms
(1.048–1.165) and atomic1.041ms (1.029–1.159). The small2.6% median advantage
has overlapping ranges; the48MiB reduction-scratch saving at16 splits is robust.
Isolated measured peak allocation increments at32768 tokens were147MiB partial
versus99MiB atomic, including other backward outputs/scratch.

Register tile tuning improved32768-token latency to1.542ms with64x64 tiles;
the same run measured atomic.987ms and partial1.009ms. Eliminating reduction
communication did not compensate for repeated scans and lower CTA parallelism.
128x16/128x32 register candidates exceeded the101376-byte shared-memory limit
and were explicitly recorded as unsupported, not treated as numerical passes.

Full operator at B16, p=.95, original versus partial/atomic/register64x32:

| Layout | Metric | Original ms | Partial ms | Atomic ms | Register ms |
|---|---|---:|---:|---:|---:|
|uniform|backward|8.272|6.144|6.109|6.876|
|uniform|fresh forward+backward|9.560|7.425|7.315|8.103|
|ragged|backward|5.506|3.542|3.339|4.151|
|ragged|fresh forward+backward|6.239|4.219|4.112|4.897|

These are synthetic single-operator timings, not full-model or RTX PRO6000
throughput. No checkpoint was loaded, modified or used for training.

## Validation and resources

106 tests passed across `test_interp_bwd_split.py`, `test_embedding_bnhd.py`
and `test_voc_glue.py`. Tests compare both implementations against FP64 math
with existing gradient acceptance, plus a relative-L2 regression bound1e-4
against the original. Coverage includes beta/tau, strides, optional dLSE,
hard/soft paths, directions and real Inductor autograd. Core results are reused
for isolated glue checks to avoid confusing native atomic variability with an
interpolation difference; unchanged SQ/SK/V/tau gradients are checked exactly.

Worst observed isolated benchmark gradient relative-L2 difference was about
3.8e-5. This is not a trained-model accuracy study; BF16 delta/tau limitations
in [BACKWARD_DIMS.md](BACKWARD_DIMS.md) still apply.

Compute Sanitizer memcheck passed partial25, atomic24 and register12 selected
tests with zero errors (padding4096, caching allocator disabled). Partial
synccheck passed25 selected tests; atomic/register synccheck passed36 selected
tests with zero errors. Racecheck has not been run.

Inspected cubins: dQ190 registers/49152B shared; partial dK/dV125 registers/
46592B shared; atomic123 registers/46592B shared; default register112 registers/
79872B shared, best register236 registers/92160B shared. No spills or CALL were
reported for these viable configurations. Shared storage is compiler-generated
dot operand/inter-warp communication, not shared FP32 atomic reduction.

## Reproduction and artifacts

Use this worktree on PYTHONPATH and conda `blkw`:

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v3/tools/bench_interp_backward.py \
  --splits 16 --full-operator --samples 7 --repeats 50 \
  --output dism_v3/build/interp_atomic_autotune.json
```

`--tune-tiles` adds fixed atomic tile comparisons. `bench_training.py` defaults
to `--interp-backward atomic` (production autotune); `baseline` selects the old
combined reference and `atomic-fixed --interp-splits 16` disables tuning. These
benchmark overrides are process-local. To reproduce the old three-way results,
use the source in `5ad8226` with its `--compare-reduction`/`--tune-register` flags.

Raw JSON, cubins, SASS and sanitizer logs remain ignored in `dism_v3/build/`.
Evidence files include `bench_interp_three_reductions_20261014.json`,
`bench_interp_atomic_confirm_20261014.json`, and
`bench_interp_register_tiles_20261014_v2.json`. The first commit of this
experiment preserves all three implementations for historical reproduction.
The production comparison is `bench_interp_atomic_autotune_20261014.json`.
