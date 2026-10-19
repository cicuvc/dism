# Parameter-matched pure-GDN valres control

This control removes every DISM branch while retaining the valres width, depth,
GDN frontend, post-norm, value residual and training objective.

## Parameter matching

With the same 12 layers, width 384, six heads and head dimension 64, pure GDN
at the valres FFN width 2576 has `67,865,496` parameters. Restored valres has
`72,035,808`, a gap of `4,170,312`.

The control matches that gap using:

- eight layers with intermediate size 2878;
- four layers with intermediate size 2877;
- one six-element bias in each layer's GDN beta projection.

The resulting count is exactly:

```text
72,035,808
```

The 2877/2878 split is used because a uniform integer FFN width cannot satisfy
the remainder exactly. The 72 total beta-bias parameters (12 layers x 6 heads)
fill the final remainder.

Defaults remain unchanged: `gdn_b_bias=False`, empty
`layer_intermediate_sizes`, and the existing source branch without per-layer
FFN customization.

## Configuration

```json
{
  "layer_layout": "gggggggggggg",
  "rear_half_dism": false,
  "value_residual": true,
  "gdn_b_bias": true,
  "layer_intermediate_sizes": [
    2878, 2878, 2878, 2878, 2878, 2878, 2878, 2878,
    2877, 2877, 2877, 2877
  ]
}
```

All attention mixers instantiate as `PureGdnAttention`; there is no DISM
selector, q/k codebook, DISM core or DISM NIAH branch.
