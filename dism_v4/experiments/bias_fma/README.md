# Fused bias and log2 conversion

Based on9ede8e8, variant0/tanh. Both patches zero the MMA accumulator,
precompute FP32 bias2=-absorbed_lse*LOG2E, and use explicit fma.rn.f32
for dot*LOG2E+bias2 before the unchanged predicated hard/causal masks.
BF16 operands are not rescaled. branch.patch specializes the element loop by
direction; select.patch uses a common loop with PTX slct for bias selection.
Patches are alternatives, applied from the baseline, not cumulative.

Both pass150 tests (two EX2-only tests skip) without relaxed tolerances.
Both have168 registers, zero stack/spill and no CALL. Source-mapped codegen
checks confirm FFMA and no separate FMUL inside finish_score_pair.

RTX5090, B16/N2048/H16/D64, identical seeded inputs,100 warmup graph replays,
20 launches/replay; pooled median of18 samples from two runs, us/launch:

| Direction | hard_prob | Original pre-MMA | FMA direction branch | FMA slct |
| --- | ---: | ---: | ---: | ---: |
| query | 0 | 538.01 | 568.68 | 578.90 |
| query | .5 | 537.46 | 567.40 | 578.67 |
| query | 1 | 521.06 | 550.40 | 561.74 |
| key | 0 | 554.05 | 578.01 | 586.65 |
| key | .5 | 553.10 | 575.50 | 584.79 |
| key | 1 | 536.54 | 558.18 | 568.18 |

FMA is possible and numerically acceptable, but these implementations regress.
Original versus branch static SASS: MOV201→80, FFMA132→260, FMUL65→21;
HMMA64 and SHFL72 unchanged. Direction-specialized finalization increases code
duplication; eliminating that via slct did not improve timing. Do not infer
a unique runtime bottleneck from these static counts. Default source/extension
were restored before the separate key-metadata optimization.

Logs/samples: build/bias_fma*. Source/cubin/patch snapshots also live in
build/bias_fma_branch and build/bias_fma_select. No default dispatch is changed.
