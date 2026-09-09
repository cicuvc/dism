# GLX divergence/collective experiment (2026-09-09)

No change was applied to the external GLX checkout or the default kernel path.
Candidates were built through GLX_ROOT=/tmp/dism-glx-branchless; the original
forward cache was rebuilt with the default GLX root after the experiment.

## Candidates and results

1. `diagonal_scan.patch`: replace12 forward/reverse scan/reduce conditional
   send-value updates with unconditional construction plus GLX choose/selp.
   The selector argument order is choose(condition, false_value, true_value).
   Finite118 and full251 regression tests pass. D64 mixed int32 summary still
   has24 WARPSYNC.COLLECTIVE and24 ENDCOLLECTIVE sites; output64/64 still60 each.
2. Additionally replace FP32 shuffle intrinsics with equivalent inline PTX
   shfl.sync.idx/bfly, full mask and clamp0x1f. Finite118 tests pass; counts
   remain24/60. This does not bypass warp synchronization semantics.
3. Additionally insert __syncwarp before each BinaryElement shuffle/shuffle_xor
   (before the four FP32 component shuffles). `barrier_shuffle.patch` contains
   this complete candidate relative to the unmodified external header.
   Finite56 codegen/label-width tests pass. Counts increase to30 for summary
   and66 for output. Not selected; no sanitizer/complete backward acceptance
   was attempted for this rejected candidate.

All tested forward codegen variants remain no CALL/zero stack/local.
No approximate-math tolerances or hard/identity semantics were changed.

## Dynamic evidence

In `/tmp/dism-summary-int32-lineinfo-q.ncu-rep`, source counters show the
collective fallback sites have zero executed instructions. BRA.DIV checks
before these paths execute (147456 per displayed check). Therefore static
collective presence is not proof that the sampled workload executed divergent
shuffle synchronization. Removing the lam4 branches alone did not eliminate
the fallback or its runtime checks; the exact compiler reasoning remains open.

Inspect with:

```sh
ncu --import /tmp/dism-summary-int32-lineinfo-q.ncu-rep --page source \
  --print-source sass --metrics inst_executed,thread_inst_executed,thread_inst_executed_true
```

Same configuration as the metadata experiment, RTX5090, D64/DV64, B64/H4/N1024,
V512, hard_prob=.5, q_from_k, int32 labels, tanh_finite, lineinfo enabled,
CUPTI20warmups/30samples, sequential runs: original median222.639us,
candidate3 median221.842us. Unlocked clocks and a single round; no established
speedup, so the default path is unchanged.

To reproduce, copy the external GLX include directory into an isolated root,
apply either patch there with `patch -p1`, and set GLX_ROOT to that root.
Do not apply both patches sequentially; each is relative to the original header.
Saved original binary: `/tmp/dism-before-branchless-scan.so`.
