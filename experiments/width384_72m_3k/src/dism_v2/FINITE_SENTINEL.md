# Bounded finite log-zero experiment (SM120a)

Enable with `DISM_TILE_LSE=tanh_finite` before importing Python modules. Default
`full` and guarded `tanh` are unchanged. Forward/backward must use the same mode;
saved scan boundaries must not cross modes. Embedding kernels are unchanged.

Core masks, unreachable boundaries and the affine identity's second component
use FP32 `-1e6`. Tile LSE uses the existing `rl/lse.cu::approx` tanh expression
without its two infinity guards. Full chunk-passing `logadd2` is unchanged.
This is an explicitly approximate representation, not an exact algebraic zero.

For a conditional bound of log2 score <=7 and N<=65536, a path containing a
masked edge has upper bound `-1e6 + N*7 + log2(N) = -541232`, far below FP32
exp2 underflow. This is not a runtime validation or an unconditional bound on
arbitrary low-level core inputs; rtau alone does not bound arbitrary dot/LSE
arguments. The intended embedding Jensen bound and rtau<=ln(D) motivate it.

## Numerical checks

- 66 fixed-input all-CUDA end-to-end cases vs guarded tanh: all nine D/DV
  combinations at N139, plus D64/DV64 N513/1024, both directions, hard_prob
  0/.37/1, rtau=ln(D). Output and all six gradients finite, no new tau sign flips.
  Worst relative L2: output 4.99e-5, dq 9.36e-4, dk 9.42e-4, dV 3.51e-5,
  dtau 7.25e-3, dq_vocab 1.223e-3, dk_vocab 1.217e-3.
  Raw per-case metrics: `benchmarks/finite_sentinel_comparison_sm120a.json`.
  Atomic accumulation introduces some run-to-run gradient differences.
- Six matching-chain cases: N1025/8193, rtau=-1/0/ln(32). Compared with guarded
  tanh, outputs are identical; maximum normalization difference 1.1921e-7 and
  reachable W2 boundary difference 4.7684e-7.
- The six new strict analytic-oracle chain tests remain ordinary failures:
  outputs pass, but normalization exceeds the full-LSE 2e-5 tolerance
  (maximum observed absolute discrepancy 0.001508). The guarded-tanh comparison
  above localizes this primarily to the existing tanh approximation, not sentinel
  drift. No thresholds were relaxed. Boundary assertions follow normalization
  and therefore were not reached in these failed tests.
- 27 existing training-smoke / replay / tail / no-embedding-fallback checks pass.
  These are not a new full copy-task convergence experiment.
- Memcheck: 19 finite tests pass, zero errors, including all-unmatched output
  exactly zero and 12 persistent cross-workload reuse cases. Six strict chain
  checks excluded from this sanitizer run. Racecheck/synccheck not yet repeated.
- Forward codegen check passes: no CALL or spill, native TMA/setmaxnreg retained.
  Backward SASS has no CALL; known backward spills are not addressed here.

Reproduce comparisons in separate processes:

```sh
DISM_TILE_LSE=tanh python -m dism_v2.compare_tile_lse --write-baseline --baseline /tmp/tanh-new.pt
DISM_TILE_LSE=tanh_finite python -m dism_v2.compare_tile_lse --baseline /tmp/tanh-new.pt
DISM_TILE_LSE=tanh python -m dism_v2.compare_finite_chain --baseline /tmp/chain-new.pt
DISM_TILE_LSE=tanh_finite python -m dism_v2.compare_finite_chain --baseline /tmp/chain-new.pt
DISM_TILE_LSE=tanh_finite python -m pytest -q tests/test_dism_v2_finite_sentinel.py
```

## Initial summary performance

RTX5090, CUDA13.1, Torch2.13+cu130, B64/H4/N1024/D=DV64/V512, actual CUDA
embedding inputs, scale1, rtau3, hard_prob=.5. CUPTI kernel durations, 20 warmups,
30 samples, medians, no per-launch synchronization; outliers retained.

| Direction | guarded tanh | finite tanh | speedup |
|---|---:|---:|---:|
| q_from_k | 328.299 us | 227.327 us | 1.44x |
| k_from_q | 337.342 us | 246.719 us | 1.37x |

These are summary-only timings, not end-to-end training speedups. Current D64
SASS BSSY count drops 67->29 (row LSE) / 68->30 (column LSE); BRA including
predicated branches drops 125->87 in both. Two TMA sites and 41 MUFU sites
remain. Counts are static instruction counts, not executed instruction totals.

## NCU repeat

Full 40-pass report: `/tmp/dism-summary-finite-sentinel-tanh-q.ncu-rep`.
Same q_from_k configuration as above; skip10/count1, cache-control none,
clock-control none. Prior report: `/tmp/dism-summary-predicated-score-tanh-q.ncu-rep`.

| Metric | guarded tanh | finite tanh |
|---|---:|---:|
| Duration | 337.22 us | 237.82 us |
| Compute SM throughput | 32.24% | 41.09% |
| Tensor pipeline utilization (elapsed cycles) | 24.4% | 37.3% |
| DRAM throughput | 22.86% | 32.81% |
| Branch efficiency | 79.90% | 99.57% |
| Average divergent branches | 4518.01 | 54.15 |
| No eligible scheduler cycles | 60.16% | 49.70% |

Finite run has zero local/shared spilling requests. Dynamic shared is47.23KB,
one CTA/SM, 384 threads; 64KiB shared configuration. Clocks were not locked:
reported SM frequency was 2.80GHz before and 2.65GHz now. Do not treat this
single replayed profile as a controlled clock-matched benchmark.

## Removing line information (2026-09-09)

The forward extension now omits `-lineinfo` by default; set `DISM_LINEINFO=1`
to restore source correlation for NCU. This switch currently affects only the
core forward extension, not embedding/backward. No `-G`/device-debug option was
present in the original build. The generated build.ninja was checked to confirm
the new compile command has neither `-lineinfo` nor `-G`.

All six summary specializations have identical disassembled instruction sequences
with/without lineinfo. Each D64 direction still has 24 WARPSYNC.COLLECTIVE and
24 ENDCOLLECTIVE instructions. The first such pair surrounds SHFL.IDX, not debug
instrumentation; removing lineinfo does not remove it under this toolchain.

Sequential q_from_k CUPTI medians (same setup above) were 227.3265us with lineinfo
and 228.351us without: no demonstrated performance gain. Codegen and 19 finite
non-chain checks pass (20 total; six known strict-chain failures excluded).
The original NCU report above still corresponds to the lineinfo-enabled build.
Saved comparison binary: `/tmp/dism-summary-finite-lineinfo.so`.
