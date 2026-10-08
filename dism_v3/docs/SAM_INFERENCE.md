# SAM full-hard inference

Production code: `python/flash_dism/inference/`; native sources: `csrc/` below
that package. Public examples and complexity bounds are in the main README.

## Provenance and scope

Restored from `main:7687cf5`, formerly `dism_v4/decoding`. The hard matching
algorithm is shared with v3 and does not require v4 training operators. The
migration retains CPU parallel prefill planning, chunked Triton execution,
CPU-planned CUDA decoding, GPU ordinary-step planning and periodic rebuilds.
It changes packaging/imports, removes the Python diagnostic DecodeCache that
depended on a test module, and replaces source-directory compilation with
installed native extensions plus user-cache JIT fallback. Kernel and planner
mathematics are unchanged.

The optional binary query reset stops the incoming match before the current
query; it does not delete historical keys and is not a document-isolation
mechanism. For ordinary v3, omit it. Start a new engine plan/cache per document.
No finite-delta or soft-row inference is included.

## Integration boundary

`HardDismPrefill` consumes int32 BHN labels and contiguous BNHD SQ/SK/V, and
returns FP32 BNHD. Its persistent worker pool defaults to eight threads; native
planning/chunk compilation release the GIL. The GPU grid combines all heads.
`HardDismDecoder` consumes the same layouts (it packs per-token slices), with
BF16 payloads by default. `prime` requires complete prompt history and is
separate from prefill output computation. Tau is fixed over cache lifetime.

This is an explicit operator backend, not an implicit replacement for
`DismAttention.forward(use_cache=True)` or nanochat generation. Model adapters
must supply the original projections, labels, readout transforms, convolution
states, GDN/SWA states and output gate. No model defaults are changed here.

Low-level `GpuPlannerCache.step` has a reused output allocation and supports
graph replay between rebuilds. Call `check_status` to check device errors;
rebuild/recapture at the horizon. Do not replay graphs across a rebuild, use a
cache across streams, or reuse it after a capacity/reset error.

## Validation

```bash
CUDA_HOME=/usr/local/cuda-13.4 PYTHONPATH=dism_v3/python python -m pytest \
  dism_v3/tests/test_sam_inference.py dism_v3/tests/test_readme.py
```

The regression oracle independently computes the dense FP64 causal recurrence,
including zero-value fallback and signed SQ/SK readout. Tests cover repeated,
random and mismatched labels, zero/positive tau, nonaligned sequence lengths,
BF16 and TF32x3 prefill, serial/parallel planners, prime/append across multiple
rebuilds, and CPU/GPU decode planners. CPU FP64 event execution is also checked.
Separate tests cover graph replay/recapture and device horizon/capacity guards.
No training gradient tolerances or kernel defaults are changed by this migration.

Migration validation on RTX 5090 / conda blkw (2026-10-08): 23 SAM/README
checks passed from source. A separately installed wheel passed these plus
the 92 multi-configuration operator checks (115 total). That run disabled the
JIT builder and asserted both SAM extensions loaded from the installation
directory, not an old checkout. The source archive includes all 12 native SAM
source/header files and the inference regression tests. This migration does
not claim a new checkpoint-level quality or throughput measurement.
