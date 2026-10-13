# Split-attention interpolation backward on sm120

Measured on the local RTX5090 on 2026-10-14. Environment, baseline configuration,
and measurement caveats are in [TRAINING_BENCHMARK.md](TRAINING_BENCHMARK.md).
This experiment changes only interpolation backward through an opt-in Python
backend; production dispatch, interpolation forward, core kernels, training
state and acceptance tolerances are unchanged.

## Implementations and contract

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
  --splits 16 --compare-reduction --full-operator --samples 7 --repeats 50 \
  --output dism_v3/build/interp_three_reductions.json
```

`--tune-register` and `--tune-tiles` add tile sweeps. `bench_training.py` also
accepts `--interp-backward split --interp-reduction atomic --interp-splits 16`.
All overrides are process-local, not persistent training configuration changes.

Raw JSON, cubins, SASS and sanitizer logs remain ignored in `dism_v3/build/`.
Evidence files include `bench_interp_three_reductions_20261014.json`,
`bench_interp_atomic_confirm_20261014.json`, and
`bench_interp_register_tiles_20261014_v2.json`. The first commit of this
experiment preserves all three implementations for historical reproduction.
