# GLX 16x32 compatibility and register probe

2026-09-06, RTX 5090 (sm120), CUDA 13.1.115, nvcc `-std=c++20 -O3 -arch=sm_120`.
External GLX HEAD: `beb416b11a1911b80d12a8da169126424a425d21`.
The GLX checkout/header was not modified. Its pre-existing untracked TMA probe was not changed.

## Findings

The existing GLX template supports 16x32 without a core-code change, although
upstream documentation/tests listed only 16x64, 32x32 and 16x16. This corrects
the earlier repository assessment that 16x32 was unsupported.

- `scan_compat.cu`: 12 cases, forward/reverse Add/Mul/Affine, F32x2/BF16x2,
  each on a 2x2 tile grid compared with the upstream sequential host oracle.
  All passed; largest F32 absolute error 1.91e-6. BF16 cases use deliberately
  exactly representable data, so they test layout rather than training accuracy.
- `reduce_compat.cu`: 12 cases covering the same operations/types/directions,
  scan/reduce outgoing boundary bit equivalence and unchanged reduce input,
  with nonidentity incoming boundaries. All passed.
- `log_compat.cu`: 16x32/16x64 Dism `(m,m)` with finite mixed-sign m,
  intermittent -infinity, all -infinity and causal masking, compared with an
  independent double-precision sequential recurrence across 2x2 tiles.
  Eight cases passed, maximum absolute error 9.07e-7; two additional log-affine
  scan/reduce equivalence cases passed.
- Compute Sanitizer memcheck: zero errors for all four executables.

## Register counts

The following are **scan-only one-warp probes**, not complete attention kernels.
They load runtime FP32 scores, scan/reduce them, and make both outgoing boundary
summaries observable. Forward uses log2 LSE with `log1p/exp2` and explicit
-infinity handling, not either optimized LSE approximation. Reverse uses an
ordinary FP32 affine op. No launch bounds or register cap is imposed.

| Operation | 16x32 registers/thread | 16x64 registers/thread |
|---|---:|---:|
| log-affine inclusive scan, scalar roll then duplicate | 48 | 80 |
| log-affine inclusive scan, duplicate then tuple roll | 48 | 72 |
| scalar-roll scan, also keep original score live | 64 | 112 |
| log-affine reduce, scalar roll then duplicate | 39 | 56 |
| reverse affine inclusive scan | 40 | 64 |
| reverse affine reduce | 38 | 40 |

All entries: stack=0, spill loads/stores=0, local memory=0, shared memory=0.
Counts were obtained with ptxas diagnostics, cudaFuncGetAttributes and cuobjdump.

The original-score retention variant uses opaque register values to prevent
reloading/rematerializing them and stores them after the scan. This is a
controlled liveness experiment, **not a benchmark of the old exclusive scan**.
It supports releasing the original score early, but does not predict an exact
register reduction for the old CUDA kernel.

Scalar-first roll versus tuple-first roll produces bit-identical scan output
on the probe input. Static SHFL counts for the no-retention scan are 44 versus
44 (16x32), and 66 versus 66 (16x64). These count the whole kernel, including
scan and inverse roll, not just the input roll. The compiler appears to merge
the repeated input operations already; this is an inference, not a measured
MIO throughput result. Explicit scalar-first ordering currently uses eight
more registers in the 16x64 probe, illustrating sensitivity to scheduling.

Production intent remains: FP32 scalar logM -> scalar roll -> in-register
duplicate to FP32 `(logM,logM)` -> inclusive scan -> consume second component.
No original score tile is needed to turn an exclusive prefix into W. Q/K/V
operands, PV accumulators, online softmax, RNG, TMA pipelines and multi-warp
boundaries will add pressure absent from this probe. Prefer warp_k_size=64;
retain 32 as fallback. Direct 16x128 instantiation is outside the current
GLX COL_BLOCKS constraint (2/4/8), so this experiment says nothing about
whether a redesigned or composed 128-column implementation would spill.

## Reproduction

From the repository root:

```bash
bash experiments/glx_scan/run.sh
```

`GLX_ROOT` and `CUDA_ROOT` can override dependency locations. The script builds
in a fresh /tmp directory, runs correctness/resource probes and memcheck, and
retains build logs and SASS. It reuses upstream test helpers via include paths;
it does not copy or patch the GLX library. Existing initial investigation logs
are in `/tmp/dism-glx-probe` (temporary, not a committed artifact).

Not yet verified here: actual TMA/MMA column mapping, online softmax/PV fusion,
all D/DV combinations, throughput/MIO counters, sm90 execution or 128-column
scan. Those remain integration work rather than claims of this experiment.
