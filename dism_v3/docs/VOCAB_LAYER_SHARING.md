# Cross-layer Q/K vocabulary sharing

`vocab_layer_group_size=2` aliases q/k codebooks across every two consecutive
DISM layers. In the restored valres layout (`6  pure GDN + 6 DISM hybrid`),
the groups are:

- layers 6 and 7
- layers 8 and 9
- layers 10 and 11

Within each layer, q and k remain distinct tensors. "Same-head" sharing means
layer 7 head `i` uses exactly the q tensor and k tensor of layer 6 head `i`;
there is no q/k collapse and no sharing across head indices.

The implementation is a Parameter alias applied by
`DismLM._tie_vocabularies()`. Embeddings, interpolation, hard labels, DISM
core, planner and output path are unchanged.

## Cost

The baseline stores six per-layer q tables and six k tables. Pair sharing stores
three q tables and three k tables:

- valres baseline: `72,035,808`
- layer-paired variant: `70,856,160`
- savings: `1,179,648`

## Requirements and interaction

- `vocab_layer_group_size` is a positive integer.
- It is mutually exclusive with the older all-layer `vocab_share_layers`.
- Head identities remain independent.
- It may be combined with head-level compact layouts only when their table
  shapes match.

## Validation

- Pair identity is preserved through device moves and strict state-dict load.
- Compiled forward/backward passes; the paired table receives the aggregate
  gradient from both layers.
- The q/k utilization probe reads the shared table from either layer.
- The full valres model config differs from A only by
  `vocab_layer_group_size=2` and explicit default recurrent-mixer fields.
