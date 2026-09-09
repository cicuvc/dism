# Explicit K0/K1 producer prologue (2026-09-09)

Rejected diagnostic candidate relative to commit514456b; default restored.
The patch keeps D32/64 at three K slots, D128 at one, and keeps Q-first issue,
ready32/free256, WG mailbox128, score-end release and cumulative tile phases.
It peels K0/K1 into a compile-time-unrolled prologue before the dynamic K loop.
No extra shared memory, barriers or consumer gate was added. In particular,
this does not prove K1 finishes before consumers switch workloads: the original
loop already permitted K1 prefetch, and this experiment only changes codegen
and loop control, not the logical slot-availability constraints.

## Results

- tanh_finite + lineinfo, RTX5090/CUDA13.1: label-width55 + row-bitset62 passed.
- D32/64 summary has four TMA emission sites (Q/K0/K1/steady-state), D128 two.
  The existing codegen test correctly fails its two-site assertion; it was not
  weakened. Separate SASS inspection found no CALL but did find LDL/STL.
- D64, either direction, soft and mixed(int32/int64): six instances report
  STACK8. Other instances report no stack. Spill was not optimized.
- Benchmark B64/H4/N1024/D=DV64/V512, mixed0.5/int32/q_from_k:
  original first round median220.974us (20warmups/30CUPTI samples).
  Candidate process failed to return for over70seconds with GPU utilization100%.
  The candidate and its benchmark launcher were terminated; no candidate
  timing exists, and the planned remaining rounds did not run.
- Small tests do not establish persistent-workload synchronization correctness.
  Root cause of the stall is unresolved; do not attribute it to a specific
  phase error or spill based on this evidence alone.
- No full-LSE suite or sanitizer acceptance was run for this rejected candidate.

Baseline binary: /tmp/dism-summary-before-k01.so.
Apply k01.patch with git apply to reproduce, then use
DISM_TILE_LSE=tanh_finite DISM_LINEINFO=1 with the standard summary benchmark.
Use a timeout and an otherwise idle GPU: the large candidate run may stall.
