# Selected DISM model

This branch keeps only the selected DISM-GDN architecture used by the valres
baseline and its scaling variants.

## Architecture

- Post-norm `DismBlock` with RMSNorm, mixer branch and SwiGLU MLP.
- nanochat pretraining uses the first half as pure GDN and the rear half as
  DISM+GDN.
- DISM and GDN share the q/k input linear projections inside a hybrid layer.
- Value residual is enabled and initialized with gate bias 2.0.
- Q/K codebooks are independent per head. The selected vocabulary is dense,
  one q table and one k table per head.
- Soft training uses the vocabulary-temperature relaxation. Hard evaluation
  and deployed inference use integer argmax symbols.
- Direction is fixed true. Hard/soft sampling is one flag per packed training
  row and head.
- The optional SM120 MXFP8 path replaces SwiGLU and q/k/v/sq/sk projections;
  `o_proj`, embedding, LM head and all DISM/GDN kernels remain BF16. See
  [MXFP8 training](MXFP8_TRAINING.md).

## Retained attention types

| Type | Class | Use |
|---|---|---|
| `dism` | `DismAttention` | DISM-only diagnostics and operator tests |
| `hybrid_gdn` | `DismGdnAttention` | selected rear-half pretraining layer |
| `pure_gdn` | `PureGdnAttention` | selected first-half pretraining layer |

All training tensors use BNHD vectors and BHN labels/LSE. Training lengths and
packed boundaries must be multiples of 256. The fixed and packed operators use
the same CUDA core; no padding or Python fallback is inserted for misaligned
training inputs.

## Forward and backward

The ordinary CUDA core evaluates the soft recurrence. Hard rows additionally
run the integer-label forward. During training, backward may use either:

- the default quadratic hard recomputation path;
- an explicitly enabled asynchronous planner/event path.

`DISM_HARD_PLANNER` defaults to `0` (quadratic). Set it to `1` to opt into the
planner. `DISM_HARD_BACKWARD=ready|event|quadratic` selects the planner
consumer policy. Both routes must produce the same gradients.

## Decoding

Hard inference lives under `flash_dism.inference` and includes arbitrary-length
prefill, a complete-history cache prime, CPU/GPU planner backends and
single-token append. `nanochat.hard_inference` provides the GDN-hybrid
reference integration.

## Removed variants

This branch does not support DISM+SWA, SDN/PaTH, GQA-DISM, shared or paired
codebooks, TransVQ, product-quantized vocabularies, quantile balancing,
selector normalization, per-depth optimizer groups, alternative layer layouts,
or categorical REINFORCE/dual-forward training.

## Legacy migration

The runtime intentionally rejects historical `DismConfig` metadata. A separate
offline migration must produce a clean config before building the model.

The selected runtime stores convolution filters under
`conv.weight` in `[channel, 1, tap]` order. Historical fused checkpoints store
`conv_weight` in `[lag, channel]` order. Migration must map

```text
conv_weight -> conv.weight
[lag, channel] -> flip(lag).T[:, None, :]
```

and apply the same permutation to matching AdamW moment tensors. It must not
mutate the source checkpoint in place.
