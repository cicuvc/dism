# Sliding-window DeltaNet recurrent mixer

This is an opt-in replacement for the gated delta rule in the DISM hybrid
architecture. The original GDN path and checkpoint layout remain unchanged
when `recurrent_mixer="gdn"` (the default).

## Configuration

```json
{
  "recurrent_mixer": "sliding_delta",
  "sliding_delta_chunk_size": 32,
  "window_size": 128,
  "head_dim": 64,
  "value_dim": 64
}
```

`rear_half_dism` and `layer_layout` retain their existing meanings:

- `g` is a pure recurrent layer, now sliding DeltaNet.
- `h` is DISM plus the recurrent branch, now DISM plus sliding DeltaNet.
- `d` remains DISM only.

The initial integration requires `gqa_query_channels=1` and
`kv_group_size=1`. Value residual, packed document boundaries, the shared
Q/K linear in DISM hybrid layers, and the common output gate are supported.

## Operator integration

The operator and layer are vendored under
`python/flash_dism/ops/sliding_delta_net/` from the standalone `deltastar`
implementation. Package imports are relative; operator math is unchanged.

The pure mixer uses the vendored `SlidingWindowDeltaNet` layer directly. The
DISM+SDN mixer calls the same Triton training operator but keeps DISM's shared
Q/K input projections and shared post-sum output path.

The operator uses a 128-token sliding overwrite window with chunk size 32.
Q/K/V head dimensions are 64. The implementation is gate-free DeltaNet: there
is no decay projection, `A_log`, or `dt_bias`; the output SiLU gate remains.

## Validation

The implementation has been checked with:

- strict loading of an existing 72M GDN checkpoint with no state-key changes;
- pure and DISM-hybrid forward/backward on CUDA, including value residual;
- packed two-document isolation for the hybrid branch;
- a full 72M packed 16x2048 smoke step: 19.2 GiB peak allocated and
  approximately 171k tokens/s forward+backward on the local RTX 5090.

For scale, the concurrent existing GDN run was logging approximately
210k tokens/s for the same micro-batch shape before compile/optimizer effects.
The first SDN run should therefore be judged primarily on capability, not on
raw throughput.
