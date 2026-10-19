# Planner/event hard backward performance

The follow-on [algorithm analysis](PLANNER_BACKWARD_ALGORITHM_ANALYSIS.md)
uses `8d47ebf` as its baseline and examines event-level identities, operation and
traffic counts, parallel scans, persistent recurrence, and the one-pair fast path.

Baseline: `2eb846ae92cc92fdc2572e5f1b36e098823fcdac` (the repaired tau
implementation). Hardware: RTX 5090 / sm120, conda `blkw`, PyTorch
2.13.0+cu130, Triton 3.7.1. This work optimizes the repaired implementation;
it does not compare against, or restore, the inaccurate pre-fix BF16 tau path.

## Selection and scope

Use three BF16 components for the FP32 operand of an accurate contraction when
all source vectors are BF16 and `mma_precision='bf16'`. The other operand is an
exact promoted BF16 vector. Compute the low and residual products before adding
the high product. Reuse the resulting accurate query/key contraction for vector
and tau gradients. Apply the same decomposition when constructing H/G, with
the weighted operand kept in FP32 until decomposition. Two BF16 components
are insufficient: they fail existing cancellation regressions.

FP32 or mixed input dtypes and explicit TF32x3 retain the original arithmetic.
Single-chunk math, scalar placement outside the inner tau reduction, metadata,
reset semantics, state scan, global reductions, output buffers and launch grids
are unchanged. The compiler already specializes on input dtype and the precision
flag; this adds no runtime dispatch, tuning cache, environment variable, or
separate tau launch. CUDA device translation units were not edited.

The default planner remains disabled. The explicit planner keeps C64 and its
existing warp rule (C16: two; C32/C64: four). There is no new adaptive launch
heuristic. Smaller chunks help some medium streams but increase state storage
and serial scan work on long streams; neither C8 nor C16 is a universal win.

The actual checked-in `nanochat/configs/sequence_true_72m_valres.json` has
D=64, **R=32**, DV=64, H=6 and N=2048. D is the key dimension, not the readout
rank R. Both the native CUDA core and public event consumer support R16/32.
The requested R64/DV64 and R64/DV32 measurements are diagnostic wrapper
extensions only; this patch does not advertise R64 production support.

## Isolation and method

All sources, imported `flash_dism` / `cu_flash_dism`, Triton artifacts and CPU
planner JIT files reside in this worktree. The inherited extension symlink was
replaced with an independent local copy of its bytes; the main-tree binary was
not modified. No CUDA source changed, so no native CUDA rebuild was necessary.
CPU JIT used `MAX_JOBS=2`; `DISM_BUILD_JOBS=2`, `OMP_NUM_THREADS=1`, and two native
planner workers were used. No datasets, models or checkpoints were downloaded.
The paused training process PID 2058134 and the main worktree were not modified.

`tools/bench_planner_event.py` loads the pinned baseline and diagnostic variants
from private modules in `build/planner-performance/variants/`, without rewriting
the installed implementation. Inputs use seed 318 BF16 vectors scaled by .2,
seed 120 labels, tau=.5, and complete independent packed training rows.
Repeated labels exercise long matching suffixes; random 512-symbol labels
exercise many short streams. The native planner supplies actual stream/chunk
counts and kernel grids. Planning is independent of vector rank.

Timing categories are deliberately separate:

- CPU plan: nine warmed preparations, including packing/pinned allocation,
  excluding the first native JIT call. Consumer/upload preparation is separate.
- Eager enqueue: host time for one Python `backward` call, including allocation
  and CUDA API submission. Synchronized wall time also waits for completion.
  These are not pure driver-launch latency measurements.
- GPU execute: CUDA events around warmed graph replays. Eight calls per replay
  amortize host submission; snapshots above 128 MiB use one call to bound memory.
  Full timing includes clearing, summary, scan, readouts and final tau reduction.
  Completed external upload waits are omitted only for capture; eager timing
  retains them. This is not a claim that asynchronous plan handoff is capturable.
- Individual kernels use recorded real launch arguments/grids. Scan summaries
  are rebuilt outside each measured replay; stage times are independent and
  should not be added to reconstruct full-call time. Initial screening reused
  scan buffers in place; final paired timings rebuild them.

Screening uses 15 samples, final paired runs 31, after compilation, eager warmup
and five graph warmups. Paired final runs reverse baseline/candidate order.
Median and p95 are across warmed samples, not cold-start measurements. Clock
and device information, source/binary hashes and raw result metadata accompany
the final ignored JSON artifacts. No GPU clock/power setting was changed.

C8 is not accepted by the native planner. The diagnostic splits native C16
streams into eight-event chunks, rebases prefixes, and pads each chunk to 16
for the dot K-dimension minimum. Compact and ordinary layouts are both rebuilt.
Its Python repacking cost is not a proposed production planner implementation.
The larger C8 state arena is real. An initial timing harness retained recorded
launch states concurrently with graph states and hit the 2.83-GiB PyTorch
allocation cap on B2/H6/N2048. The process reached about 3.71 GiB including CUDA
context, below the requested 4-GiB target. The harness was corrected to measure
full calls before retaining stage arguments and to use one captured call for
large snapshots. The cap was not increased. Failed initial rows are retained
in `topology.json`/`topology.log`; they are not used as speedup evidence.

## Original bottleneck and investigated alternatives

The repaired BF16 path simultaneously needed BF16 hq/hk for vector gradients
and FP32/TF32x3 tq/tk for cancellation-sensitive tau. Its carried matrix operands
and several output/layout conversions extended register live ranges. R32/DV64,
C64/W4 compiled to 255 registers and 70 spills in the multi readout. Single-chunk
kernels and summary construction were spill-free in this configuration.

The experiments preserve the outer-scalar reduction order that fixed the prior
cancellation regression:

| Variant | Change | Decision |
| --- | --- | --- |
| baseline | Repaired TF32x3 summaries/tau, BF16 vector contraction | Reference |
| reuse | One TF32x3 query/key contraction serves vector and tau | Passes 88 focused tests; modest gains |
| reorder | Consume accurate tau before computing BF16 vector contraction | Little benefit; compiler/liveness still dominate |
| split2 | Two BF16 components, shared accurate readout | Reject: 3 existing cancellation tests fail |
| split3 | Three BF16 components, shared accurate readout and summaries | Selected; validated below |
| separate | Keep vector readout; add an accurate boundary-tau kernel | Extra launch and operand reloads offset most spill savings |
| combined | Combine local/carried vectors before three global vector atomics, instead of six | Mixed results; longer live ranges and layout costs |

The separate kernel can remove readout spills without improving the whole call.
The combined variant reduces atomic instruction sites but does not uniformly
improve throughput. Neither atomics nor spill count alone predicts the winner.
No shared FP32 atomic accumulation was introduced. Reordering scalar factors
inside the cancellation-sensitive reduction was deliberately not reconsidered:
its numerical failure is already established in PLANNER_BACKWARD_PRECISION.md.

## Paired final speedup table

All values are GPU microseconds, **median/p95**; W4, BF16 input/mode, DV64.
The final source, not a source-transform candidate, is timed as `current`.
The last column gives baseline/current median speedups in forward and reversed
measurement order. R64 rows remain diagnostic. Random rows have no multi-chunk
work here and are effectively unchanged.

| B×H×N:labels | R | C | compact | baseline µs | final µs | speedup × (two orders) |
| --- | --- | --- | --- | --- | --- | --- |
| 1x1x512:repeat | 32 | 32 | 0 | 29.29/29.34 | 27.56/27.75 | 1.063 / 1.056 |
| 1x1x512:repeat | 32 | 32 | 1 | 27.22/27.33 | 25.46/25.55 | 1.069 / 1.061 |
| 1x1x512:repeat | 32 | 64 | 0 | 39.71/39.81 | 35.18/35.33 | 1.129 / 1.129 |
| 1x1x512:repeat | 32 | 64 | 1 | 37.49/37.53 | 32.88/32.98 | 1.140 / 1.139 |
| 1x1x512:repeat | 64 | 32 | 0 | 39.26/39.33 | 34.41/34.50 | 1.141 / 1.140 |
| 1x1x512:repeat | 64 | 32 | 1 | 37.20/37.28 | 32.24/32.38 | 1.154 / 1.144 |
| 1x1x512:repeat | 64 | 64 | 0 | 59.74/59.83 | 46.36/46.47 | 1.289 / 1.293 |
| 1x1x512:repeat | 64 | 64 | 1 | 57.40/57.48 | 43.90/43.98 | 1.308 / 1.304 |
| 2x6x2048:repeat | 32 | 32 | 0 | 970.29/977.16 | 911.25/914.55 | 1.065 / 1.067 |
| 2x6x2048:repeat | 32 | 32 | 1 | 927.87/932.78 | 868.62/872.64 | 1.068 / 1.070 |
| 2x6x2048:repeat | 32 | 64 | 0 | 443.22/443.69 | 387.10/387.53 | 1.145 / 1.146 |
| 2x6x2048:repeat | 32 | 64 | 1 | 423.72/424.08 | 368.63/368.92 | 1.149 / 1.151 |
| 2x6x2048:repeat | 64 | 32 | 0 | 1412.61/1432.24 | 1333.63/1357.47 | 1.059 / 1.059 |
| 2x6x2048:repeat | 64 | 32 | 1 | 1314.72/1331.41 | 1240.70/1257.47 | 1.060 / 1.062 |
| 2x6x2048:repeat | 64 | 64 | 0 | 1507.62/1511.23 | 789.89/792.86 | 1.909 / 1.911 |
| 2x6x2048:repeat | 64 | 64 | 1 | 1461.47/1466.21 | 749.24/751.32 | 1.951 / 1.950 |
| 2x6x2048:random | 32 | 32 | 0 | 46.30/46.43 | 46.26/46.46 | 1.001 / 0.998 |
| 2x6x2048:random | 32 | 32 | 1 | 44.03/44.16 | 44.10/44.21 | 0.998 / 1.001 |
| 2x6x2048:random | 32 | 64 | 0 | 46.23/46.44 | 46.40/46.52 | 0.996 / 1.001 |
| 2x6x2048:random | 32 | 64 | 1 | 43.95/44.15 | 44.03/44.20 | 0.998 / 1.001 |
| 2x6x2048:random | 64 | 32 | 0 | 51.80/51.91 | 51.82/51.93 | 1.000 / 1.000 |
| 2x6x2048:random | 64 | 32 | 1 | 49.49/49.60 | 49.48/49.53 | 1.000 / 1.001 |
| 2x6x2048:random | 64 | 64 | 0 | 51.72/51.85 | 51.82/51.94 | 0.998 / 1.000 |
| 2x6x2048:random | 64 | 64 | 1 | 49.47/49.57 | 49.44/49.55 | 1.001 / 1.000 |

For B2/H6/N2048 repeated labels, R32/C64, compact layout, the reproducible
full-device improvement is about **1.15×**. Python synchronized backward also
improves, while CPU planning is much slower than device execution in this
bounded two-worker fixture. This is not a measured model-training speedup.

## Per-kernel and host timing

The following are separate CUDA-event measurements of every kernel launch for
B2/H6/N2048 repeated labels, R32/DV64, C64/W4. `single16/32/64` is the bucket's
physical event width; `multi` is the carried-state readout. Full time includes
additional clearing/reduction kernels and cache/launch interactions. Each cell
is median/p95 µs.

| compact | kernel | actual grid | baseline µs | final µs |
| --- | --- | --- | --- | --- |
| 0 | single16 | 58320 | 95.56/95.67 | 95.34/95.63 |
| 0 | single32 | 1524 | 11.14/11.36 | 11.11/11.29 |
| 0 | single64 | 756 | 16.23/16.38 | 16.02/16.23 |
| 0 | summary | 3360 | 52.85/53.23 | 39.32/39.54 |
| 0 | scan | 708×1 | 26.62/26.66 | 26.62/26.66 |
| 0 | multi | 3360 | 140.96/141.22 | 91.53/91.76 |
| 0 | full | — | 443.22/443.69 | 387.10/387.53 |
| 1 | single16 | 58320 | 90.12/90.49 | 89.99/90.38 |
| 1 | single32 | 1524 | 10.57/10.63 | 10.57/10.64 |
| 1 | single64 | 756 | 15.47/15.56 | 15.46/15.53 |
| 1 | summary | 3360 | 52.16/52.41 | 39.37/39.64 |
| 1 | scan | 708×1 | 26.62/26.66 | 26.62/26.66 |
| 1 | multi | 3360 | 139.88/140.14 | 91.76/91.91 |
| 1 | full | — | 423.72/424.08 | 368.63/368.92 |

Host timings for the same fixture (median/p95 µs); plans are prepared before GPU timing:

| compact | version | CPU plan | Python enqueue | Python sync wall | quadratic GPU | quadratic sync wall |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | baseline | 20041.44/29192.19 | 173.25/211.62 | 500.86/516.38 | 903.22/908.99 | 923.84/929.57 |
| 0 | current | 20041.44/29192.19 | 176.54/213.48 | 444.46/460.58 | 903.22/908.99 | 923.84/929.57 |
| 1 | baseline | 18293.32/22838.63 | 165.51/201.41 | 476.47/492.92 | 903.22/908.99 | 923.84/929.57 |
| 1 | current | 18293.32/22838.63 | 170.35/213.28 | 416.51/443.24 | 903.22/908.99 | 923.84/929.57 |

The quadratic columns use **the same labels, vectors, dO and tau**, and include
its forward-state recomputation. Event timing is preplanned backward, so this
is a route-cost reference, not an equal-preparation-cost algorithm comparison.
R64 has no native quadratic configuration and is compared with FP64 instead.

## Spill/resource table

Multi-chunk readout resources, repaired baseline → compensated candidate/final.
Official R32/DV64 C32/C64 entries use final paired cubins; C16 uses the final
24-case audit. R64/DV32 and extreme W2 rows use diagnostic screening cubins.
`spills` is Triton's `n_spills`, not a measured byte count. LDL/STL are static
SASS instruction-site counts from `cuobjdump -sass`.

| R/DV | C/W | registers | spills | LDL/STL | shared bytes |
| --- | --- | --- | --- | --- | --- |
| 32/64 | 16/2 | 255 → 171 | 0 → 0 | 0/0 → 0/0 | 9216 → 11264 |
| 32/64 | 32/4 | 255 → 156 | 10 → 0 | 5/5 → 0/0 | 10240 → 14336 |
| 32/64 | 64/4 | 255 → 255 | 70 → 10 | 82/82 → 5/5 | 24576 → 32768 |
| 64/64 | 16/2 | 255 → 249 | 56 → 0 | 58/53 → 0/0 | 18432 → 20480 |
| 64/64 | 32/4 | 255 → 216 | 50 → 0 | 63/54 → 0/0 | 20480 → 24576 |
| 64/64 | 64/4 | 255 → 255 | 458 → 164 | 594/564 → 189/174 | 40960 → 49152 |
| 64/64 | 64/2 | 40 → 255 | 2062 → 320 | 8855/4949 → 423/380 | 40960 → 49152 |
| 64/32 | 32/4 | 255 → 231 | 70 → 0 | 84/76 → 0/0 | 18432 → 20480 |
| 64/32 | 64/4 | 255 → 255 | 62 → 2 | 90/81 → 1/1 | 40960 → 45056 |

All 24 final audit cases (C16/32/64 × R16/32 × DV32/64 × BF16/TF32x3)
passed the existing diagnostic gradient gates. Every inspected kernel had
**0 CALLs and 0 shared atomics**. Maximum shared storage in this supported-shape
audit is 32768 bytes. BF16 multi readout spills range from 0 to 12 (R16/DV64/C64);
R32/DV64/C64 has 10. Explicit TF32x3 remains expensive, reaching 674 spills;
that unchanged diagnostic path is not claimed to be optimized. The scan has
zero shared memory and disjoint element ownership. Single-chunk kernels and
summary kernels are spill-free in the selected BF16 supported-shape audit.

## Error table

The canonical cancellation reproducer is `route_tau_sweep.py detailed`:
B1/H2/N512, repeat labels, tau=.5, vector seed 918, dO seed 77. These are fresh
baseline/final measurements. Relative L2 below includes the autograd output
casts; zero q/k gradients remain exactly zero. Atomic last-bit differences mean
these are precision levels rather than bitwise guarantees.

| gradient | old / quadratic | final / quadratic | old / FP64 | final / FP64 |
| --- | --- | --- | --- | --- |
| sq_vec | 2.413059203e-04 | 1.991375304e-04 | 2.334414208e-03 | 2.333681825e-03 |
| sk_vec | 2.334706943e-04 | 1.794585179e-04 | 2.348228160e-03 | 2.345466141e-03 |
| v | 2.513056074e-04 | 2.513056074e-04 | 2.380172039e-03 | 2.380172039e-03 |
| rtau | 9.628331575e-06 | 6.627995307e-06 | 8.967714830e-06 | 6.418242081e-06 |

Final canonical metrics against maintained FP64 (signed bias is mean error):

| gradient | cosine | norm ratio | signed bias |
| --- | --- | --- | --- |
| sq_vec | 9.999972771e-01 | 1.000015937e+00 | -3.528191993e-07 |
| sk_vec | 9.999972501e-01 | 1.000034874e+00 | -1.841211860e-07 |
| v | 9.999971691e-01 | 9.999378962e-01 | -7.103199332e-09 |
| rtau | 1.000000000e+00 | 1.000002744e+00 | 3.777960580e-06 |

Additional shared-tau relative L2 against independent dense FP64 repeated-label derivatives:

| B×H×N | R | C | old tau rel-L2 | candidate/final tau rel-L2 |
| --- | --- | --- | --- | --- |
| 1x1x4096:repeat | 32 | 16 | 1.107e-04 | 5.386e-06 |
| 1x1x4096:repeat | 32 | 32 | 1.386e-04 | 1.392e-05 |
| 1x1x4096:repeat | 32 | 64 | 8.077e-06 | 1.984e-06 |
| 2x6x2048:repeat | 32 | 64 | 3.812e-06 | 7.196e-06 |
| 2x6x2048:repeat | 64 | 64 | 8.971e-06 | 9.975e-06 |

The canonical final tau error is still at the repaired 10⁻⁵-or-better level;
vector errors are not worsened. Two-component compensation instead fails the
existing events=32/33/64 regression at relative errors **1.48e-3, 5.60e-3,
6.42e-3**, against the unchanged `rtol=2e-4, atol=2e-6` gate. It is rejected
regardless of its speed. Neither REL_L2 nor any existing tolerance was edited.

## Chunk/topology results

W4, compact=true, DV64, median µs as baseline → split3/final. Short/long and
single/multi-head cases use native topology, not a fixed synthetic CTA grid.
Initial short/long screening uses `split3`; B2/H6 uses the final implementation.
C8 means eight logical events in a physical 16-wide tile.

| B×H×N:labels | R | C8 padded | C16 | C32 | C64 |
| --- | --- | --- | --- | --- | --- |
| 1x1x7:repeat | 32 | 6.51 → 6.51 | 6.50 → 6.52 | 6.51 → 6.49 | 6.51 → 6.50 |
| 1x1x7:repeat | 64 | 6.72 → 6.56 | 6.70 → 6.55 | 6.54 → 6.60 | 6.74 → 6.74 |
| 1x1x65:repeat | 32 | 16.04 → 15.48 | 15.22 → 14.45 | 19.31 → 17.79 | 13.68 → 13.69 |
| 1x1x65:repeat | 64 | 20.38 → 19.28 | 18.82 → 17.35 | 25.43 → 20.67 | 13.93 → 13.93 |
| 1x1x4096:repeat | 32 | 1062.83 → 1066.22 | 165.20 → 158.08 | 121.28 → 107.42 | 108.37 → 95.10 |
| 1x1x4096:repeat | 64 | 1714.59 → 1695.74 | 599.62 → 580.45 | 195.97 → 166.14 | 266.88 → 143.89 |
| 2x6x2048:repeat | 32 | 3418.27 → 3387.81 | 1721.44 → 1690.46 | 927.87 → 868.62 | 423.72 → 368.63 |
| 2x6x2048:repeat | 64 | 5056.51 → 5000.38 | 2445.06 → 2416.77 | 1314.72 → 1240.70 | 1461.47 → 749.24 |
| 2x6x2048:random | 32 | 63.68 → 61.61 | 43.91 → 43.92 | 44.03 → 44.10 | 43.95 → 44.03 |
| 2x6x2048:random | 64 | 86.11 → 80.48 | 49.26 → 49.31 | 49.49 → 49.48 | 49.47 → 49.44 |

For R32/B2/H6/N2048 repeated labels, reducing C64 to C32 more than doubles
full-device time; C16 and padded C8 are worse again. With final R32 kernels,
state snapshots grow from 52.5 MiB (C64) to 127.6875 MiB (C32), 301.875 MiB
(C16) and 698.0625 MiB (C8). R64 doubles these sizes. Smaller chunks reduce
readout register pressure but increase summary/scan traffic and serial work.
Conversely, N512 prefers C32. Random labels mostly use single16 buckets, so
changing C does little. These observations argue against changing the global
default or adding a heuristic based on one favorable microbenchmark.

Measured compact CTA topology (identical semantic stream selection in ordinary
layout). `single CTAs` lists width16/32/64 buckets; multi chunks launch the
summary and multi readout, while multi streams launch the scan. No multi stream
means those three stages are absent.

| B×H×N:labels | C | single CTAs | scan CTAs | summary/multi CTAs | max chunks/stream |
| --- | --- | --- | --- | --- | --- |
| 1x1x7:repeat | 64 | 15/0/0 | 0 | 0 | 1 |
| 1x1x65:repeat | 16 | 146/0/0 | 7 | 17 | 4 |
| 1x1x65:repeat | 64 | 146/5/2 | 0 | 0 | 1 |
| 1x1x512:repeat | 32 | 1212/31/0 | 28 | 111 | 16 |
| 1x1x512:repeat | 64 | 1212/31/15 | 13 | 42 | 8 |
| 1x1x4096:repeat | 64 | 9724/255/127 | 122 | 681 | 64 |
| 2x6x2048:repeat | 8 | 55260/0/0 | 6048 | 44676 | 256 |
| 2x6x2048:repeat | 16 | 58320/0/0 | 2988 | 19320 | 128 |
| 2x6x2048:repeat | 32 | 58320/1524/0 | 1464 | 8172 | 64 |
| 2x6x2048:repeat | 64 | 58320/1524/756 | 708 | 3360 | 32 |
| 2x6x2048:random | 64 | 12711/0/0 | 0 | 0 | 1 |

## Complete variant screening matrices

B1/H1/N512 repeated labels, compact=true, BF16; each cell is full GPU
**median/p95 µs**, from 15 warmed samples. This initial matrix covers all
C16/32/64 × W2/4/8 × R32/DV64, R64/DV64, R64/DV32 combinations. These are
screening results; only the chosen final implementation received the complete
release validation. All 162 configurations compiled and ran successfully.

| R/DV | C/W | baseline | reuse | reorder | split2 FAIL | split3 | separate tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32/64 | 16/2 | 28.85/29.03 | 28.83/28.93 | 28.85/28.93 | 26.96/27.06 | 27.56/27.61 | 30.88/31.04 |
| 32/64 | 16/4 | 27.90/28.01 | 27.85/28.09 | 28.30/28.37 | 26.75/26.83 | 27.12/27.25 | 30.10/30.22 |
| 32/64 | 16/8 | 33.14/33.22 | 33.44/33.56 | 33.44/33.75 | 28.81/29.00 | 29.54/29.62 | 33.36/33.47 |
| 32/64 | 32/2 | 33.20/33.27 | 32.66/32.77 | 33.12/33.27 | 26.94/27.00 | 27.85/28.04 | 32.34/32.44 |
| 32/64 | 32/4 | 27.08/27.31 | 27.24/27.29 | 26.99/27.12 | 25.24/25.34 | 25.57/25.74 | 29.84/29.96 |
| 32/64 | 32/8 | 31.55/31.68 | 31.35/31.43 | 31.42/31.59 | 28.29/28.42 | 28.84/28.98 | 34.91/35.03 |
| 32/64 | 64/2 | 57.21/57.33 | 54.71/54.85 | 56.19/56.34 | 43.40/43.50 | 45.92/45.98 | 50.81/50.97 |
| 32/64 | 64/4 | 37.48/37.59 | 36.21/36.31 | 37.45/37.54 | 31.44/31.59 | 33.09/33.25 | 36.46/36.66 |
| 32/64 | 64/8 | 43.42/43.50 | 41.81/41.88 | 42.90/43.16 | 37.18/37.24 | 39.03/39.11 | 43.23/43.39 |
| 64/64 | 16/2 | 44.63/44.76 | 44.18/44.29 | 44.45/44.63 | 38.91/39.09 | 40.15/40.40 | 44.99/45.11 |
| 64/64 | 16/4 | 40.66/40.86 | 40.62/40.76 | 40.75/40.91 | 38.23/38.37 | 39.10/39.28 | 43.63/43.72 |
| 64/64 | 16/8 | 47.04/47.20 | 46.71/46.77 | 46.92/47.01 | 40.09/40.31 | 41.09/41.33 | 47.25/47.47 |
| 64/64 | 32/2 | 46.24/46.37 | 44.99/45.24 | 45.99/46.10 | 34.56/34.71 | 36.48/36.61 | 43.69/43.88 |
| 64/64 | 32/4 | 37.04/37.12 | 36.72/36.89 | 37.08/37.27 | 31.31/31.44 | 32.20/32.39 | 36.97/37.07 |
| 64/64 | 32/8 | 39.51/39.64 | 39.35/39.38 | 39.50/39.61 | 33.97/34.12 | 35.21/35.31 | 41.31/41.43 |
| 64/64 | 64/2 | 248.33/248.76 | 234.35/234.75 | 252.27/252.67 | 61.37/61.50 | 72.58/72.67 | 82.00/82.08 |
| 64/64 | 64/4 | 57.50/57.71 | 51.48/51.67 | 57.17/57.31 | 40.37/40.63 | 43.94/44.04 | 48.22/48.28 |
| 64/64 | 64/8 | 75.88/75.98 | 70.02/70.24 | 74.98/75.11 | 49.34/49.53 | 56.15/56.27 | 64.67/64.85 |
| 64/32 | 16/2 | 29.30/29.38 | 29.04/29.14 | 29.33/29.55 | 27.02/27.09 | 27.84/27.99 | 31.34/31.44 |
| 64/32 | 16/4 | 28.32/28.52 | 28.26/28.42 | 28.28/28.53 | 27.29/27.50 | 27.72/27.83 | 30.81/30.94 |
| 64/32 | 16/8 | 30.83/30.95 | 30.84/30.93 | 30.88/31.07 | 29.37/29.55 | 30.06/30.30 | 33.20/33.38 |
| 64/32 | 32/2 | 32.26/32.42 | 30.37/30.58 | 31.09/31.34 | 27.04/27.26 | 27.98/28.02 | 32.15/32.24 |
| 64/32 | 32/4 | 30.89/30.98 | 29.34/29.54 | 29.53/29.60 | 26.30/26.48 | 27.26/27.46 | 29.58/29.72 |
| 64/32 | 32/8 | 36.50/36.71 | 35.46/35.57 | 35.88/36.12 | 30.65/30.77 | 32.42/32.59 | 33.97/34.05 |
| 64/32 | 64/2 | 57.82/57.97 | 55.58/55.74 | 57.43/57.59 | 47.24/47.38 | 49.28/49.38 | 52.34/52.45 |
| 64/32 | 64/4 | 37.16/37.33 | 34.87/34.97 | 35.41/35.60 | 31.35/31.42 | 32.38/32.56 | 35.95/36.06 |
| 64/32 | 64/8 | 45.65/45.74 | 43.27/43.39 | 43.89/44.00 | 38.20/38.31 | 40.56/40.65 | 45.73/45.91 |

Ordinary-layout C8 and atomic-combination matrix, same N512 repeat fixture;
all configurations passed the sampled FP64 gates. This tests the actual padded
C8 topology with every requested warp count. Each cell is median/p95 µs.

| R (DV64) | C/W | baseline | split3 | combine vector atomics |
| --- | --- | --- | --- | --- |
| 32 | 8/2 | 46.24/46.37 | 43.64/43.77 | 45.97/46.20 |
| 32 | 8/4 | 44.56/44.72 | 42.45/42.64 | 44.14/44.22 |
| 32 | 8/8 | 56.25/56.37 | 48.60/48.79 | 56.43/56.67 |
| 32 | 16/2 | 30.81/30.91 | 29.54/29.65 | 30.59/30.66 |
| 32 | 16/4 | 29.86/29.95 | 28.85/29.02 | 29.33/29.44 |
| 32 | 16/8 | 35.20/35.64 | 31.60/31.69 | 35.16/35.25 |
| 32 | 32/2 | 35.16/35.28 | 29.92/30.14 | 32.18/32.30 |
| 32 | 32/4 | 29.26/29.48 | 27.75/27.84 | 30.86/31.02 |
| 32 | 32/8 | 33.95/34.10 | 31.33/31.42 | 36.76/36.93 |
| 32 | 64/2 | 59.31/59.41 | 48.13/48.32 | 52.39/52.57 |
| 32 | 64/4 | 39.82/39.97 | 35.18/35.27 | 36.71/36.83 |
| 32 | 64/8 | 45.77/45.96 | 41.40/41.58 | 43.44/43.50 |
| 64 | 8/2 | 76.77/76.97 | 68.28/68.63 | 74.62/74.71 |
| 64 | 8/4 | 72.88/73.01 | 68.17/68.29 | 71.42/71.54 |
| 64 | 8/8 | 83.86/84.01 | 73.83/73.98 | 83.82/83.90 |
| 64 | 16/2 | 46.48/46.61 | 41.90/41.99 | 44.43/44.65 |
| 64 | 16/4 | 42.60/42.72 | 41.13/41.29 | 42.36/42.51 |
| 64 | 16/8 | 49.07/49.22 | 43.26/43.46 | 48.85/49.03 |
| 64 | 32/2 | 48.55/48.65 | 38.59/38.76 | 46.26/46.37 |
| 64 | 32/4 | 39.34/39.42 | 34.47/34.61 | 37.04/37.25 |
| 64 | 32/8 | 42.10/42.18 | 37.68/37.78 | 41.09/41.22 |
| 64 | 64/2 | 249.74/250.33 | 74.67/74.87 | 85.83/85.93 |
| 64 | 64/4 | 60.04/60.14 | 46.28/46.43 | 55.92/56.00 |
| 64 | 64/8 | 78.21/78.50 | 58.76/58.89 | 71.77/71.87 |

Combining the compensated contractions with the atomic reduction is also
measured, rather than inferred from either improvement alone. It gives only
about 2% extra on the production-rank C64 large fixture, regresses its C32 case
by about 7%, and requires a larger readout reordering. The broad compensated
change is selected; no extra shape-dependent atomic-combination path is added.
R64 C64 benefits more, but remains outside the production consumer contract.

| B×H×N | R | C | selected µs | split3 + combine µs | extra speedup × |
| --- | --- | --- | --- | --- | --- |
| 1x1x512:repeat | 32 | 32 | 25.52/25.69 | 27.00/27.21 | 0.945 |
| 1x1x512:repeat | 32 | 64 | 32.88/32.94 | 32.20/32.35 | 1.021 |
| 1x1x512:repeat | 64 | 32 | 32.20/32.42 | 32.17/32.99 | 1.001 |
| 1x1x512:repeat | 64 | 64 | 43.96/44.12 | 39.89/40.08 | 1.102 |
| 2x6x2048:repeat | 32 | 32 | 863.08/867.94 | 926.91/929.64 | 0.931 |
| 2x6x2048:repeat | 32 | 64 | 368.96/369.39 | 361.66/362.07 | 1.020 |
| 2x6x2048:repeat | 64 | 32 | 1233.18/1244.81 | 1233.44/1245.54 | 1.000 |
| 2x6x2048:repeat | 64 | 64 | 743.64/745.11 | 687.80/691.60 | 1.081 |

## Validation, failures and limits

- Final focused suite: **142 passed** (`focused-verified.log`): the original
  88 route/fused cases plus 54 explicit launch cases. The new cases cover
  C16/32/64 × W2/4/8 × compact/ordinary, BF16 inputs, general FP32 with explicit
  TF32x3, and mixed storage exercising the BF16-mode fallback. They retain the
  existing vector gate and use `rtol=2e-4, atol=2e-6` for tau against the
  maintained FP64 autograd oracle. Existing graph/reset/packed-document,
  negative/zero/positive-tau and cancellation tests remain unchanged.
- The new test's first run had an oracle-shape wiring error: native tau is a
  one-element group, while the single-head oracle returns a scalar. All 54
  failures stopped at that shape assertion. The fixture now reshapes only that
  scalar reference to its one tau group; no numerical threshold was relaxed.
  `focused-final.log` retains that unsuccessful harness run.
- Integration/native/compact/CPU-reference suite: **293 passed, 4 failed**.
  All four are `test_existing_gpu_prefill_dsq` using obsolete token-level hard
  masks. They fail with `hard must be bool [training_rows,H]; token-level masks
  are unsupported`. All four reproduce on the untouched baseline kernel
  (`prefill-baseline.log`). They were not edited or hidden.
- Disabled planner, in a fresh process: **3 passed**. No production dispatcher
  or model setting was changed.
- Strict native diagnostics: **6 failed, 72 skipped, 21 deselected**, identically
  on baseline and final. The six fixtures have N=65 and fail the established
  256-token alignment contract before gradient comparison. The 72 skips are
  the existing optional-native-FP32 gate: `cu.fp32_enabled()` is false in the
  inherited extension. No new skip/xfail was added. Triton FP32 input paths
  were tested; optional FP32 native CUDA instances were not rebuilt.
- Final unfiltered **memcheck, synccheck and racecheck each passed 8 smoke
  tests**. Memcheck/synccheck: 0 errors; racecheck: 0 hazards, 0 errors,
  0 warnings. No sanitizer exception or unsupported-instruction warning was
  used. This is a small sm120 smoke run, not an exhaustive race proof.
- Final supported-shape resource audit: **24 cases**, no diagnostic gradient
  failures, no CALLs, no shared atomics. `inspect_kernel` saved cubins and
  disassembled every inspected kernel with `/usr/local/cuda/bin/cuobjdump -sass`.
  Resource and numeric results are in `resources-final.json` and
  `final-cubins/`. Final paired benchmarks also audit their actual kernels.

The repaired cancellation is still finite-precision arithmetic. FP32 metadata,
state summation and global atomics permit nondeterministic last-bit differences;
this is not a proof for arbitrary sequence lengths or magnitudes. Other GPU
architectures were not tested. The legacy register-scan BF16 diagnostic remains
outside this change. R64 and C8 remain diagnostic-only extensions. No checkpoint,
optimizer, dataloader, RNG counter, annealing buffer, direction or hard-sampling
policy was touched. The parent training process remained stopped throughout;
after tests the GPU returned to 0% compute utilization and 15347 MiB free.

## Reproduction and artifacts

Run from this worktree root, with a **fresh output filename** for each benchmark.
Generated JSON, variants, cubins, SASS and logs are ignored under `build/`.
The benchmark validates local `flash_dism` and `cu_flash_dism` imports and limits
PyTorch GPU allocation to 9% of this 31.44-GiB card. The corrected large-C8 run
peaked at 1540.4 MiB PyTorch allocation; final paired runs peaked at 980.2 MiB.
Those figures exclude CUDA context/driver storage, unlike the earlier OOM
message's total process measurement. The original failed harness was bounded
rather than allowed to consume the paused job's remaining memory.

```bash
export PY=/home/cicuvc/miniconda3/envs/blkw/bin/python
export PYTHONPATH="$PWD/dism_v3/python:$PWD/dism_v3/tests:$PWD/dism_v3/tools"
export TORCH_EXTENSIONS_DIR="$PWD/build/torch_extensions"
export TRITON_CACHE_DIR="$PWD/build/triton"
export MAX_JOBS=2 DISM_BUILD_JOBS=2 OMP_NUM_THREADS=1

# Full variant/warp/rank screen, 162 configurations.
"$PY" dism_v3/tools/bench_planner_event.py \
  --output build/planner-performance/reproduce-sweep.json \
  --cases 1x1x512:repeat --shapes 32x64 64x64 64x32 --compact 1 \
  --variants baseline reuse reorder split2 split3 separate

# Eight-event padded chunks and combined-atomic candidate, ordinary layout.
"$PY" dism_v3/tools/bench_planner_event.py \
  --output build/planner-performance/reproduce-c8.json \
  --cases 1x1x512:repeat --shapes 32x64 64x64 --compact 0 \
  --chunks 8 16 32 64 --warps 2 4 8 \
  --variants baseline split3 combined --references

# Complete topology matrix with the corrected memory-bounded harness.
"$PY" dism_v3/tools/bench_planner_event.py \
  --output build/planner-performance/reproduce-topology.json \
  --cases 1x1x7:repeat 1x1x65:repeat 1x1x4096:repeat \
          2x6x2048:repeat 2x6x2048:random \
  --shapes 32x64 64x64 --chunks 8 16 32 64 --warps 4 \
  --variants baseline current --references

# Final paired run; repeat with --variants current baseline and a new filename.
"$PY" dism_v3/tools/bench_planner_event.py \
  --output build/planner-performance/reproduce-paired.json \
  --cases 1x1x512:repeat 2x6x2048:repeat 2x6x2048:random \
  --shapes 32x64 64x64 --chunks 32 64 --warps 4 \
  --variants baseline current --samples 31 --references

# Test the combination of both optimizations, not just each in isolation.
"$PY" dism_v3/tools/bench_planner_event.py \
  --output build/planner-performance/reproduce-combined.json \
  --cases 1x1x512:repeat 2x6x2048:repeat --compact 1 \
  --shapes 32x64 64x64 --chunks 32 64 --warps 4 \
  --variants current split3_combined --references

"$PY" -m pytest -q dism_v3/tests/test_hard_route_consistency.py \
  dism_v3/tests/test_hard_prefill_backward_fused.py \
  dism_v3/tests/test_hard_prefill_backward_launches.py
"$PY" dism_v3/tools/route_tau_sweep.py detailed
"$PY" -m pytest -q dism_v3/tests/test_hard_training.py \
  dism_v3/tests/test_hard_ready_dispatch.py \
  dism_v3/tests/test_native_training_planner.py \
  dism_v3/tests/test_compact_training_planner.py \
  dism_v3/tests/test_hard_prefill_training_reference.py
"$PY" -m pytest -q dism_v3/tests/test_disabled_hard_planner.py

for tool in memcheck synccheck racecheck; do
  /usr/local/cuda/bin/compute-sanitizer --tool "$tool" \
    --target-processes all --error-exitcode 86 "$PY" -m pytest -q \
    dism_v3/tests/test_hard_training.py \
    dism_v3/tests/test_hard_prefill_backward_fused.py \
    -k 'async_hard_long_stream_safety or bf16_tau_cancellation_at_chunk_boundaries'
done

"$PY" -m pytest -q -m strict_gradient dism_v3/tests/test_backward.py \
  dism_v3/tests/test_backward_qk.py dism_v3/tests/test_backward_summary.py \
  dism_v3/tests/test_voc_backward.py

"$PY" - <<'PY'
from pathlib import Path
import torch
from check_hard_prefill_backward_triton import check_case
torch.set_num_threads(1)
for c in (16, 32, 64):
    for r, dv in ((16, 32), (16, 64), (32, 32), (32, 64)):
        for precision in ('bf16', 'tf32x3'):
            result = check_case('repeat', 129, .5, r=r, dv=dv,
                                chunk_size=c, precision=precision, fused=True,
                                resource_dir=Path('build/planner-performance/reproduce-cubins'))
            print(c, r, dv, precision, result['kernels'],
                  result['production_acceptance_failures'])
PY
```

Primary measured artifacts: `sweep.json`, `c8-combined.json`,
`paired-final.json`, `paired-reverse.json`, `topology.json` (partial, includes
failed original harness rows), `topology-rest.json` (successful bounded retry),
`combined-split3.json`, `topology-metadata.json`, `resources-final.json`,
`detailed-baseline.log`, `detailed-final.log`, `focused-verified.log`,
`integration-final.log`, `prefill-baseline.log`, `strict-final.log`,
`strict-baseline.log`, and the three sanitizer logs. The report above preserves
measured results in Git; generated artifacts themselves are not committed.
