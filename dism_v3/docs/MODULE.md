# DISM module contract

## Public layers

`flash_dism.DismAttention` is the DISM mixer with no additional recurrent
branch. `flash_dism.hybrid_gdn.DismGdnAttention` adds a GDN branch and aliases
the DISM/GDN q and k input linear weights. `flash_dism.pure_gdn.PureGdnAttention`
uses the same GDN projection and gated output path without DISM.

The layer input/output is `[B,N,width]`. CUDA training requires BF16 autocast,
`N % 256 == 0`, and packed execution with `batch=1` plus contiguous int32
`cu_seqlens`.

## DISM options

The selected module construction keeps:

- `head_dim`, `value_dim`, `readout_dim`;
- `vocab_size` as the number of integer Q/K symbols;
- short-convolution width;
- scalar `vocab_temperature`;
- optional value residual;
- optional host-only `tau_scale` for the selected tau schedule.

`hard` is bool `[rows,H]`. A packed training row is the hard-sampling unit;
document boundaries are used only by loss normalization. `direction` is bool
`[B,H]`, and the selected nanochat path supplies all-true direction with
`direction_true=True`.

The parameters are:

| Parameter | Shape / meaning |
|---|---|
| `q_conv.weight`, `k_conv.weight`, `v_conv.weight` | input projections |
| `*.conv_weight` | depthwise short-convolution filters |
| `sq_proj`, `sk_proj` | signed readout projections |
| `q_vocab`, `k_vocab` | `[H,V,D]` codebooks |
| `log_sel_tau` | `[H]`, rtau = softplus |
| gate and output projections | shared post-mixer output path |

## Soft and hard math

The soft path uses scaled codebook logits and an LSE relaxation. The hard path
uses integer argmax labels:

```text
H_ij = 1[q_symbol_i == k_symbol_j]
```

The suffix recurrence and exact-zero no-match fallback remain in the CUDA core.
Hard backward is selected at dispatch time without changing the forward
objective.

## Plane minimization

The selected training path uses one shared sequence boundary snapshot per
forward. Embedding interpolation stays in the Triton operator and consumes the
same BNHD q/k and codebook buffers as the core.

## Removed module interfaces

No SWA, SDN, PaTH, GQA, TransVQ, PQ, quantile-balance, selector-normalization,
cross-head/layer vocabulary sharing, generic categorical REINFORCE, or alternate
readout interfaces remain in this branch.

## MXFP8 boundary

`DismMLP` can use `flash_mxfp8.MXFP8SwiGLU`. The q/k/v short-convolution
frontend has an MXFP8 linear variant, while its depthwise convolution and the
downstream DISM/GDN math remain BF16. `sq_proj`/`sk_proj` can use
`flash_mxfp8.MXFP8Linear`. `o_proj` remains the common BF16 output path.

## Opt-in top-four hard training

Nanochat model specs may explicitly set `hard_query_topk=4` for local weighted
query alternatives on the hard path, while retaining native v3 soft and a single
committed argmax history. The default remains 1. This baseline supports aligned
CUDA eager/graph training with planner disabled; cached/SAM inference and
Dynamo are not supported. See [implementation and measured limits](TOPK_HARD_TRAINING.md).

With top-four enabled, `hard_query_sampled=true` explicitly selects a sampled
local Q read with a zero-history control and surrogate router VJP. It retains
the same argmax history and uses the existing model RNG counter. This is the
narrow opt-in interface described in [sampled hard training](SAMPLED_HARD_TRAINING.md),
not the removed generic sequence-level REINFORCE interface. The separate
[event planner prototype](PLANNER_DECOUPLED_READOUT.md) is not model dispatch.
