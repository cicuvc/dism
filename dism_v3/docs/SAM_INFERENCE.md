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

## Long-stream sequence parallelism

Each prefill `Program` stream is a first-order linear recurrence on an `R×DV`
state with a scalar per-chunk decay:

```
M_{c+1} = a_c * M_c + S_c,   a_c = exp(last_prefix),  S_c = sk_c^T (v_c*scale_c)
history  = (sq_c @ M_c) * scale_c
```

That is exactly the chunkwise form of SSD / Simple GLA (`S_t = a_t S_{t-1} +
k_t v_t^T`, `o_t = q_t^T S_t`). The associative operator is
`combine((a1,S1),(a2,S2)) = (a2*a1, a2*S1 + S2)` with identity `(1,0)`; the
carried element is one scalar plus an `R×DV` matrix, generally full rank.

Measured stream shape (`C=16`): ~89% (random) / 95% (all-same) of streams are
one chunk, the mean grows like `O(log N)`, and the single longest stream is
`Theta(N)` chunks (all-same ~`N/16`, random ~`N/64`). Only that tail needs
sequence parallelism. Its state passing is the FLA `chunk_h_parallel` pattern:
(1) per-chunk summaries in parallel over chunks, (2) a serial reduction over
`(K,V)` tiles that applies the inter-chunk scalar decay and stores the prefix
state, (3) `chunk_fwd_o` per chunk in parallel. Passing state stays `O(R*DV)`
live (or `O(sqrt(L))` with checkpointing) instead of materializing all `L`
chunk states. Reference: `fla/ops/common/chunk_h.py` and `chunk_h_parallel.py`.
The DISM-specific parts a reused kernel would not cover are non-uniform chunks
(per-key gates, `k_c <= C`), event-stream indexing instead of token order, and
the FP32 atomic reduction across overlapping streams.

Prototype: `TritonPrefillPlan.execute_parallel`
(`python/flash_dism/inference/prefill_triton_parallel.py`) implements the three
phases forward-only, sharing the exact chunk packing and coefficients with the
serial `_execute`, and switches BF16 / TF32x3 like the serial kernel. Every
kernel takes an explicit id list, so `execute_dispatch` can route streams by
chunk count (`dispatch_threshold`, default 64): streams at or below the
threshold keep the register-carried serial kernel, longer streams go through
the chunk-parallel phases. `HardDismPrefill.__call__` uses this routing, so a
pathological single long stream no longer carries the full per-chunk work on
one CTA; workloads without streams above the threshold run the serial kernel
exactly as before. Both paths are checked against the serial reference and the
FP64 oracle (`test_prefill_parallel_state`, `test_prefill_dispatch`).

Performance is deliberately not tuned: the serial kernel already launches O(N)
stream CTAs, so the GPU is throughput-saturated before any single stream
dominates, and at N=16k..64k on one head dispatch is 1.0-1.7x the serial time.
The threshold defaults high to keep the common case on the cheaper serial
kernel; the parallel phases only bound the worst case. They still materialize
`[long_chunks, BR, BD]` FP32 sums/states. Checkpointing or fusing the passing
kernel, and a calibrated threshold, are not implemented yet.

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
