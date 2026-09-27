# Forward-summary score: predication and metadata reuse (2026-09-08)

## Implemented scope

Only `summary_persistent` changed; output and backward recomputation remain unchanged.
There are now six summary instantiations: D32/64/128 x compile-time row/column LSE.
No pure-soft/hard endpoint specialization or full/interior tile dispatch yet.
The validated ready32/free256 synchronization and single-transfer Q/K layout are unchanged.

- Compute tau2=tau*LOG2E and scale2=scale*LOG2E outside the key loop.
- Row-LSE direction: two query-row biases per thread `(tau-lse_q)*LOG2E`, before the loop.
- Each key iteration: lane l preloads int64 labels for keys l and l+32 into registers.
  Column-LSE direction also preloads and converts the corresponding two key biases.
  Loads are placed before K-ready wait/MMA; first consumption occurs in score conversion.
  Metadata remains register-only, with lane shuffle distributing each column's value.
- Each score uses one `fmaf(dot,scale2,bias2)`; hard matching selects tau2 or negative infinity.
  Explicit PTX selp then chooses hard/soft and applies bounds/causal masking.
  No conditional global load remains inside the per-score selection.
- Hard mismatch, causal mask, and log-affine identity keep exact infinity semantics.
  No finite sentinel, multiplication by an infinity mask, or precomputed global RNG mask is added.

Both hard endpoints currently use this same predicated path, including metadata that an endpoint
specialization could avoid. Preloading both candidates is an intentional mixed-path tradeoff.

## Code generation

Full/tanh: all six summary instances zero stack/local spill, no CALL, native TMA/setmaxnreg.
Each summary still has exactly two static UTMALDG.5D sites and now four LDG.E.64 sites
(two query labels plus two key labels). These counts are guarded by the codegen test.

D64 tanh whole-kernel static counts (not dynamic instruction counts):

| Version | BSSY | BRA | LDG.E.64 |
|---|---:|---:|---:|
| Prior runtime-direction score | 129 | 235 | 66 |
| New row-LSE | 67 | 105 | 4 |
| New column-LSE | 68 | 105 | 4 |

The score source now lowers to FFMA plus predicates/FSEL. Remaining branches include scan
infinity handling, producer/tail control and pipeline waits; this is not a branch-free kernel.

## First performance comparison

RTX5090, B64/H4/N1024/D=DV64/V512, scale1, tau3, hard_prob=.5, tanh tile LSE,
actual CUDA embedding inputs. CUPTI summary-only timing,20 warmup/30 samples per run,
sequential baseline/new processes with seed777 replay and no simultaneous sanitizer.
Outliers retained; full raw samples are in `benchmarks/predicated_score_summary_sm120a.json`.

| Direction | Previous us | New us | Throughput ratio |
|---|---:|---:|---:|
| q_from_k / row-LSE | 523.261 | 327.887 | 1.60x |
| k_from_q / column-LSE | 512.237 | 337.854 | 1.52x |

This combines direction specialization, metadata reuse, reassociation and predication;
it does not separately attribute the gain to any one change. It is not full-model throughput.
Runner: `DISM_TILE_LSE=tanh python -m dism_v2.benchmark_persistent_summary`, with
`--direction k_from_q` for the second direction. Baseline binary was saved locally at
`/tmp/dism-summary-single-tma-pre-score-tanh.so`, not a repository dependency.

## Validation

Existing full-mode forward121 cases passed, including both directions, all D/DV combinations,
RNG replay and persistent reuse/tails. Added12 cases at rtau=ln(D), N513, both directions,
D32/64/128 and hard_prob=0/.37 also passed with existing tolerances.
These check summary components, resolved boundaries, saved W boundaries and output against the
same BF16 interpolation oracle. Thus the tested FP32 reassociation error remains within the
existing tolerances, despite output/backward still using the previous score evaluation order.
Full/tanh codegen checks passed, including the new four-label-load guard.
Racecheck, memcheck and synccheck each passed12 cross-workload cases with zero errors/hazards.
Full end-to-end wiring/replay/training-selected regression:513 passed.
Tanh codegen plus end-to-end replay/training-selected regression:28 passed.
Each end-to-end run retained the known PyTorch AccumulateGrad stream-mismatch warning.
These selected regressions do not replace the known failing FP32-oracle precision suite;
no tolerances were relaxed and no existing failures were marked as expected/skipped.
