# Q/K vocabulary coupling and width scaling

Status: diagnostic note. No production behavior changes.

## 1. Current model width

The restored valres 72M baseline uses:

```text
n_embd = 384
n_head = 6
head_dim = 64
Q/K selector width = 6 * 64 = 384
```

Thus the Q and K projections are both square `384 -> 384` transforms of the
same residual stream. They use independent linear projections and depthwise
short convolutions, followed by SiLU, but their output widths equal the full
residual width.

## 2. Measured Q/K feature coupling

On a 1024-token PG19 sample, baseline per-layer/per-head Q and K features show:

- direct Q/K cosine: `0.118`
- centered Q/K cosine: `0.015`
- CCA top-1 / top-4 / top-8:
  `0.942 / 0.910 / 0.874`
- held-out Q -> K affine map:
  - R2: `0.416`
  - prediction/target cosine: `0.824`

The Q/K codebooks themselves still use nearly independent code addresses after
centering; this is why the same code's Q and K vectors are nearly orthogonal.
The overlap is in the feature subspaces, not in the per-code embedding vectors.

## 3. What changes under shareqk

Compared with the restored baseline:

| metric | baseline | shareqk |
|---|---:|---:|
| direct Q/K feature cosine | 0.118 | 0.431 |
| centered Q/K cosine | 0.015 | 0.213 |
| CCA top-1 | 0.942 | 0.949 |
| Q -> K affine R2 | 0.416 | 0.453 |
| same-position symbol agreement | 5.63% | 8.27% |
| lag-1 symbol agreement | 4.13% | 6.21% |
| lag-16 symbol agreement | 3.71% | 4.61% |
| lag-256 symbol agreement | 2.73% | 3.39% |

Active-code counts and usage entropy do not show a simple low-cardinality
collapse. Instead, shareqk moves Q and K feature geometry closer together and
increases symbol agreement across all tested lags.

Working interpretation:

> Tying Q/K codebooks removes the freedom for two distinct but input-correlated
> selector coordinate systems. Training responds by aligning Q/K features more
> strongly and increasing spurious cross-lag symbol matches. The result is role
> and address coupling rather than a conventional codebook-rank collapse.

## 4. Width-scaling hypothesis

At `n_embd=384`, Q/K selector width is also 384. The two projections are square
transforms of the same residual stream, so substantial linear overlap is
structurally easy to learn. The Q/K feature measurements above are consistent
with this.

Hypothesis: increasing residual width may reduce the shareqk failure if Q/K
selector width stays smaller than the residual width:

```text
Q/K projection width / residual width < 1
```

In that regime, Q and K become lossy, non-invertible views of the residual.
Independent projections can discard different portions of the residual and
retain more role-specific selector geometry.

The hypothesis does **not** automatically hold if the number of heads or
head dimension grows proportionally. A square `d -> d` Q/K projection at larger
hidden width can preserve the same coupling problem.

Other factors that can break the coupling without increasing width:

- short-convolution and SiLU nonlinearity;
- Q/K projection orthogonality or decorrelation penalties;
- separate Q/K code role adapters;
- keeping independent Q/K tables while sharing only across heads or layers.

## 5. Required scaling test

To test the hypothesis cleanly, compare at least:

1. residual width increased, Q/K selector width fixed;
2. residual and Q/K selector widths scaled together;
3. matched Q/K table-sharing and non-sharing controls.

Record:

- Q/K feature CCA and held-out affine R2/cosine;
- same-position and cross-lag symbol agreement;
- active Q/K codes and usage entropy;
- q/k coverage and max load;
- NIAH present/absent/margin at 8k and 16k;
- PG19 long-context bpb.

The prediction is not merely "larger is better": the effect should depend on
the ratio of selector width to residual width and on whether the two projection
families are forced into the same square coordinate system.

## 6. Diagnostic artifacts

Generated under the restored valres run:

- `diagnostics/qk_svd.py`, `diagnostics/qk_svd.json`
- `diagnostics/qk_mapping.py`, `diagnostics/qk_mapping.json`
- `diagnostics/qk_feature_geometry.py`,
  `diagnostics/qk_feature_geometry.json`
