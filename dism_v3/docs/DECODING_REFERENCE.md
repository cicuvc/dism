# Torch decoding reference

`python/flash_dism/reference/dism_decode_ref.py` provides inference-only
`dism_decode_ref` and `dism_wrapper_decode`. They do not launch DISM CUDA kernels.
The former takes the exact argument order/layout of `dism_ref`, with an optional
keyword `cache`. Pass only new tokens, including their current K/V:

```python
from flash_dism.reference.dism_decode_ref import dism_decode_ref

# Each chunk may contain one token or multiple tokens (prefill).
output, cache = dism_decode_ref(
    q, k, sq, sk, q_lse, k_lse, idx_q, idx_k, direction, hard, v, rtau,
    cache=None,
)
next_output, next_cache = dism_decode_ref(
    next_q, next_k, next_sq, next_sk, next_q_lse, next_k_lse,
    next_idx_q, next_idx_k, direction, next_hard, next_v, rtau,
    cache=cache,
)
```

Use raw natural-log LSE, not CUDA's caller-preabsorbed `lse-rtau` operands.
Q/K here are already selected/interpolated score operands. SQ/SK already include
caller activations. The signed `SQ @ SK.T` factor affects only the numerator;
the denominator still has the unit-weight zero-value fallback.

The cache contains K, SK, V, key labels/LSE and the last causal W row, plus fixed
direction/tau. An appended step computes
`W_new = logM_new + softplus(cat([-inf], W_previous))`.
The old self-key state is included as the new self-key's predecessor. No dense
attention matrix or query history is retained. Cache storage is linear in length;
each step is linear in history, so full decoding remains quadratic time. Dynamic
`torch.cat` allocations are intentional reference code, not a production cache.

`dism_wrapper_decode` additionally computes the same shared `[V,D]` vocabulary
interpolation as `dism_wrapper`. A default direction is sampled globally on the
first call and reused thereafter. Pass explicit hard masks for reproducible
chunk partitioning; Torch RNG sampling is not CUDA Philox replay. Tables/model
parameters must remain fixed while using a cache. Direction/tau changes are
explicitly rejected. FP16/BF16 accumulates and returns FP32; FP64 is retained.

These APIs use no-grad, do not mutate input caches, and copy incoming data into
new caches. Do not edit cache tensors in place. Start each independent document
with `cache=None`; this reference has no packed-varlen or unequal batch-length
interface and does not inherit CUDA's256-token alignment requirement. It is not
the subquadratic hard-label/SAM inference algorithm.

Tests: `PYTHONPATH=python:.. python -m pytest -q tests/test_decode_reference.py`.
CPU tests cover dense-oracle equivalence, token/chunk decoding, FP64/FP32/BF16,
mixed directions, soft/mixed/hard rows, no-match fallback, diagonal continuation,
signed readout, cache isolation/reset, fixed-parameter validation, large-score
normalization and vocabulary interpolation/direction replay.
