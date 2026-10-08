# Native host frontend and local interpolation

`src/frontend.cpp` is compiled once, without TK/device compilation. It directly
dispatches through a C++ function-pointer registry to the eight configuration
namespaces. There is no Python per-phase shape dispatch. `core_forward` and
`core_backward` validate, prepare operands, run the three phases and format
saved states/gradients. Python retains autograd registration, state saving and
the small fixed/varlen route choice.

## Alignment contract

- Fixed inputs: positive N divisible by256. No automatic pad or trimming.
- Packed inputs: B=1; all cu_seqlens entries, including terminal T, divisible
  by256. Empty documents/packs remain supported. Layout creation and monotonicity,
  range, endpoint and alignment checks are native C++; construction from a CUDA
  cu_seqlens still synchronizes once. Reuse VarlenLayout when possible.
- Core/summary/output/backward stage host entrypoints reject nonaligned lengths.
  The Torch mathematical/decoding references remain arbitrary-length oracles.
- Contiguous vectors are aliased; noncontiguous vectors may still require a
  contiguous copy. Metadata casts/transposes remain where genuinely necessary.
  CUDA kernels still validate their low-level arguments; moving validation into
  C++ is not a claim of eliminating all host checks or allocations.

## Embedding LSE and tau gradients

The Triton implementation now lives in `python/flash_dism/emb_kernel.py`.
Production has no import dependency on dism_v2 or FlashAttention. Vocabulary
expansion/casting and direction selection use native ATen helpers with autograd
enabled. Triton kernel launch/allocation metadata remains in Python.

The voc path passes detached contiguous tau to interpolation. Both FP32 LSE
outputs directly store `lse-tau`. Core staging takes the private preabsorbed
path and does not subtract again. For probability reconstruction, interpolation
backward adds tau back at the load site within the Triton kernel. This is FP32
subtract/add, so it can introduce ordinary rounding relative to raw-LSE storage.
Neither a separate LSE subtract nor a restore tensor is materialized.

Core backward already returns the COMPLETE tau derivative, including soft and
hard branches. Interpolation therefore does not return a second tau gradient.
Its tau preparation input must be detached; C++ validation rejects a requiring-grad
tau. Do not apply an independently differentiable Python `lse-tau` before the
private preabsorbed core path, which would double-count this derivative. Public
raw-core callers continue to supply raw LSE; native staging absorbs tau under
NoGradGuard, with the same explicit core gradient ownership.

`tests/test_native_frontend.py` covers alignment rejection, zero Python pad/shape
dispatch, fused-LSE forward/backward equivalence and a profiler assertion that
voc forward has no standalone `aten::sub` or `aten::constant_pad_nd`. Existing
multi-config tests use aligned inputs and retain their numerical tolerances.

## Memory-safety verification

Previous memcheck runs used the default Torch allocator and did NOT establish
logical tensor redzones. A zero-error result alone cannot rule out writes inside
another suballocation of the same backing CUDA allocation.

Use both:

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 PYTHONPATH=python:.. \
  /usr/local/cuda-13.4/bin/compute-sanitizer --tool memcheck --padding 4096 \
  --error-exitcode 1 python -m pytest -q tests/test_multi_config.py \
  -k 'core_oracle or ragged_replay'
PYTHONPATH=python:.. python -m pytest -q tests/test_tensor_canaries.py
```

The canary tests intercept ATen factories including calls inside C++ host
wrappers, surround each allocation with4KiB prefix/suffix byte guards, and
preserve TMA alignment. They assert that actual O/normalizer allocations were
intercepted, check guards after synchronization, and compare complete outputs
and gradients. This detects writes within these guard regions, not reads;
neither finite test coverage nor guard bytes constitute a proof for all inputs.

Observed on RTX5090 with CUDA13.4 compute-sanitizer:
- `build/native_uncached_memcheck.log`:56 passed,0 errors; caching disabled and
  allocation padding4096, eight shapes x fixed/varlen x soft/mixed/hard plus ragged replay.
- `build/native_canaries.log`:18 passed; sixteen shape/layout cases plus prefix
  and suffix corruption positive controls. Baseline outputs/gradients matched.
- `build/native_embedding_memcheck.log`:fused interpolation forward/backward
  comparison passed with caching disabled, padding4096,0 errors.
- `build/native_full_tests_final.log`:513 passed,533 skipped,78 deselected.
  Skips retain optional probe/FP32 gating; strict-gradient tolerances unchanged.
- `build/native_post_move_tests.log`:161 passed after docs/tools relocation,
  covering native/frontend, all configurations, decoding and canaries.
