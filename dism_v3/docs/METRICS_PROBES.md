# Training RMS probes

The optional nanochat probes keep their existing definitions and frequencies:

- Hidden RMS: RMS along the hidden dimension, then max over tokens, one value
  per layer. The final microbatch overwrites the previous microbatch's value.
- Update RMS: RMS of each parameter's actual optimizer update, then max over
  parameters within each layer/module group (not a pooled group RMS).
- Tied parameters are deduplicated; frozen parameters are not snapshotted.

Snapshots now use multi-tensor clone. FP32 CUDA updates use multi-tensor
subtraction/L2 norm/division; other dtypes retain subtraction in parameter dtype
before FP32 reduction. Group results are read back once per device instead of
once per group. Snapshots still own independent storage. No pointer metadata
cache, optimizer modification, RNG use, or checkpoint schema change is involved.

Contiguous CUDA residuals use a two-stage Triton reduction (16 token rows per
CTA, followed by max/sqrt). Both max reductions explicitly propagate NaN.
Empty/noncontiguous or unusually large shapes retain the PyTorch path. Hooks
capture scalar output views instead of Python layer indices, avoiding the
per-layer Dynamo specialization/cache-limit fallback. This changes compilation
coverage: complete training trajectories are not claimed bitwise identical;
ordinary compiler fusion/FP32 reduction ordering can change low bits. Probe
numerical comparisons use 1e-6 relative tolerance, without changing any existing
operator tolerance. QK histogram diagnostics are unchanged.

## RTX 5090 measurement, 2026-10-15

Read-only checkpoint-4000 replay of the 72M r3 run, 32768 tokens/microbatch,
four accumulation microbatches, fixed true direction, all probes enabled.
Three warmup steps, two phase samples, two six-step sustained windows; profiler
runs separately after timing. Old helpers loaded from commit `9fe61b8` for the
baseline; no checkpoint or training job was changed.

| Measurement | Original | Optimized |
|---|---:|---:|
| Mean sustained step | 646.58 ms | 623.77 ms |
| Sustained throughput | 202.7k tokens/s | 210.1k tokens/s |
| Optimizer + probes phase (CUDA event, mean) | 13.53 ms | 6.96 ms |
| Optimizer + probes CPU scope (profiled) | 24.88 ms | 8.34 ms |
| Kernels within that synchronized scope | 1459 | 39 |
| GPU kernel sum within that scope | 3.74 ms | 2.49 ms |
| Whole-step aten::item / stream sync calls | 65 / 97 | 50 / 83 |

About 3.5% less step time / 3.7% higher throughput in this short paired test.
The improvement includes avoiding the hook's compiler-cache fallback, not just
the RMS reductions. Profiled CPU times include profiler overhead; they are not
additive unprofiled step budgets. This is scratch replay, not resumed training
or a convergence experiment.

Artifacts (ignored): `build/metrics_rms_before.{json,log,trace.json}` and
`build/metrics_rms_final.{json,log,trace.json}`. The intermediate `after` run
preceded the scalar-view hook fix and is not the final measurement. The initial
`metrics_rms_tests.log` records the NaN propagation failure that was fixed.

Validation: 31 tests pass across `nanochat/tests/test_metrics_rms.py`,
`test_metrics.py`, and `test_checkpoint_resume.py` (including a 12-layer
fullgraph hook test requiring exactly one compiled graph). Final memcheck and
synccheck each pass 13 non-compile RMS cases with zero errors. Production BF16 residual shape
`[1,32768,384]` uses 38 / 23 registers for partial / final reduction, zero
spills and no SASS CALL instructions in either kernel. Logs are
`build/metrics_rms_final_tests_v3.log`, `metrics_rms_memcheck_final.log`,
`metrics_rms_synccheck_final.log`, and
`metrics_rms_resources.log`. These checks cover the new diagnostics; they do
not constitute a new sanitizer campaign for the existing model kernels.

Racecheck initially reported write/write hazards in the width-13 partial
reduction. This was **not** treated as a sanitizer false positive: widths below
32 now use one warp, avoiding that cross-warp layout. The six FP32/BF16
residual cases then pass racecheck with zero errors/warnings
(`metrics_rms_racecheck_v2.log`). Width 384's launch and the timing above are
unchanged. The original failing `metrics_rms_racecheck.log` is retained.
