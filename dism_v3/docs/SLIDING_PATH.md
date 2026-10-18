# Sliding-window PaTH recurrent mixer

`recurrent_mixer="sliding_path"` replaces GDN with the tied-key/reflector
sliding-window PaTH operator. It is an opt-in alternative to
`sliding_delta`; the default `gdn` path remains unchanged.

## Configuration

```json
{
  "recurrent_mixer": "sliding_path",
  "sliding_delta_chunk_size": 32,
  "window_size": 128,
  "head_dim": 64,
  "value_dim": 64
}
```

The existing `sliding_delta_chunk_size` field is shared by the two sliding
operators so an SDN/PaTH comparison differs only in `recurrent_mixer`.
The current integration requires `gqa_query_channels=1` and
`kv_group_size=1`.

## Operator semantics

PaTH keeps the Q/K/V projections, short convolutions, Q/K L2 normalization,
optional value residual, and head-wise gated output norm of sliding DeltaNet.
Its recurrent readout is

\[
s_{ij}=\mathrm{scale}\,q_i^\top H_i\cdots H_{j+1}k_j,\qquad
H_t=I-\beta_t k_t k_t^\top,
\]

followed by a softmax over exactly
`max(document_start, i-window_size+1) <= j <= i`. Values remain raw `v_j`;
keys also serve as Householder reflector vectors. The default scale is
`head_dim**-0.5`.

Pure layers use the vendored `SlidingWindowPaTH`; DISM hybrid layers keep the
DISM Q/K projection sharing and shared post-sum output path, replacing only
the recurrent raw branch. Pure SDN and PaTH layers have identical parameter
names and shapes and support strict state-dict exchange.

## Validation

- Pure SDN and PaTH state dicts load into each other with `strict=True`.
- Tiny `PureSlidingPathAttention` and `DismSlidingPathAttention` pass CUDA
  forward/backward, including value residual.
- The hybrid branch preserves packed document isolation.
- A two-layer compiled model passes forward/backward.
- `path-lf` model and CLI configs differ from `sdn-lf` only in
  `recurrent_mixer` and run/model names.
- The full 72M compiled 16x2048 smoke step used 16.8 GiB peak allocated
  memory and reached about 178k tokens/s for forward+backward after warmup.

The prepared run is
`dism-fwedu-valres-72m-2500m-path-lf-20261019`; it is intentionally not
launched yet.
