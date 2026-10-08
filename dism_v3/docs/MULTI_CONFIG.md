# Multi-configuration CUDA extension

The default build contains R16/32 × D32/64 × DV32/64, each with fixed and
packed-varlen execution:16 configurations × six primary kernels. Utility
kernels (delta, and optional diagnostic probes) are additional, not part
of that96 count. Output and CTA-private gradients are BF16; cross-CTA atomic
gradients and accumulators remain FP32. FP32 validation instances are disabled
by default, not selected dynamically on device.

## Build and source organization

Use conda blkw and `python build.py`. `DISM_BUILD_JOBS=4` limits parallel compiler
processes. Each configuration has separate object/dependency directories under
`build/{object,deps}/rR_dD_vDV`. Kernels use a `KernelConfig<R,D,DV>` type and
are instantiated in independent translation units. Existing concrete aliases
are isolated in a configuration namespace to prevent cross-shape ODR violations.
The build defines a configuration selection per TU; there is no runtime shape
branch in the tile computation. This is not eight separately loaded extensions.

All kernel `.cu` files are Torch-free. Tensor validation, allocation, stream
selection and argument construction live in `src/host/*.cpp`. These compile in
CUDA host-only mode because ThunderKittens' type headers contain CUDA syntax;
Torch is never parsed in a device compilation. The CUDA TUs export opaque
kernel-address getters, and host code calls cudaLaunchKernel with the existing
by-value parameter Args. Varlen still shares one Args across all documents.
`src/frontend.cpp` is compiled once (ordinary C++) and owns native shape dispatch,
high-level validation/staging and complete three-phase forward/backward calls.

For incremental development only, `DISM_BUILD_CONFIGS=32,64,64` builds a subset;
the normal build must omit it. `DISM_ENABLE_FP32=1` can restore diagnostic
instances, but is not used for this milestone. DISM_BACKWARD_DEBUG remains an
independent compile-time diagnostic gate. Do not use the old dimension environment
variables to choose a runtime shape; the default build contains all eight shapes.

## Python API

```python
from flash_dism import dism_core, voc_dism, VarlenLayout, supported_configs

# q/k: [B,N,H,D], sq/sk: [B,N,H,R], v: [B,N,H,DV], BF16; N%256==0.
out = dism_core(q, k, sq, sk, v, q_lse, k_lse,
                idx_q, idx_k, direction, hard, rtau)

# Same argument order; packed tensors have B=1. Document boundaries are aligned.
out = dism_core(q, k, sq, sk, v, q_lse, k_lse,
                idx_q, idx_k, direction, hard, rtau, cu_seqlens=cu_seqlens)
```

Raw natural-log LSE, rtau absorption, explicit hard flags and per-head direction
are unchanged. Layout caching avoids repeated cu_seqlens synchronization.
Vocabulary interpolation is local in flash_dism/emb_kernel.py and fuses LSE-tau.
Backward infers the same shape from saved operands: mixed shapes can coexist in
one autograd graph, without changing process-global dispatch state. Unsupported
shapes fail explicitly. Diagnostic entrypoints require DISM_BUILD_PROBES=1.
Production wrappers call the native frontend, not Python get_config dispatch.
See NATIVE_FRONTEND.md for alignment, fused tau-gradient ownership and verification.

Validation: `PYTHONPATH=python:.. python -m pytest -q tests/test_multi_config.py`.
The numerical oracle is FP64 reference; gradient tests use the unchanged
existing acceptance policy. FP32 diagnostic tests are separate from production.

## Validation and resource snapshot (RTX5090 / sm120a)

- Multi-configuration correctness/codegen:100 passed. All16 execution shapes
  pass forward/backward FP64 oracle, with soft/mixed/hard cases; ragged replay,
  autograd, Triton vocabulary gradients and mixed-shape graphs also pass.
- Default suite:868 passed,278 skipped,78 deselected. Optional FP32 validation
  tests are explicitly skipped when their instances are not compiled; original
  strict-gradient tests/tolerances are unchanged. Single-build dimension rejection
  and two-precision codegen expectations were updated to the new registry.
- Exactly96 primary BF16 kernel instances were found across the eight shape
  directories. No CALL in their SASS; no Torch headers in their device dependency
  files. The extension is about50MiB including auxiliary/diagnostic kernels.
- Extension-filtered memcheck and synccheck each24 passed, zero errors: all16
  mixed fixed/varlen configurations plus eight multiple-document replay tests.
  Logs: build/multi_memcheck.log and build/multi_synccheck.log.
- Varlen-filtered racecheck: eight mixed configurations passed, zero hazards;
  build/multi_varlen_racecheck.log. Fixed racecheck's earlier ARRIVES-immediate
  warnings remain historical evidence; they are not reported as zero hazards.
- Existing spill is retained, not optimized or used to relax tolerances:

| R/D/DV | Fixed F1/F3/B1/B3 stack (B) | Varlen F1/F3/B1/B3 stack (B) |
|---|---|---|
| 16/32/32 | 0/0/128/152 | 0/24/88/64 |
| 16/32/64 | 0/0/256/192 | 0/24/264/112 |
| 16/64/32 | 0/0/160/208 | 16/32/128/144 |
| 16/64/64 | 0/0/200/208 | 16/24/240/176 |
| 32/32/32 | 0/0/152/144 | 0/24/136/104 |
| 32/32/64 | 0/0/304/200 | 0/24/288/120 |
| 32/64/32 | 0/0/160/240 | 16/24/160/176 |
| 32/64/64 | 0/8/336/344 | 16/24/304/192 |

Stack is the compiler's per-thread static frame size, not a traffic measurement.
No performance or Hopper conclusions are inferred from these counts.
Raw logs are in build/multi_all_build.log, build/multi_validation.log and
build/multi_regression_final.log (ignored build artifacts).
