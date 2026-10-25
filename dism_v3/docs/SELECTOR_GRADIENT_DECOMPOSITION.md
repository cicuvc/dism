# Where selective-STE gradient amplification comes from

2026-10-25. Follow-up to SELECTOR_SURROGATE_GRADIENTS.md. No training changes.
Four selected outlier heads and two relatively aligned controls were inspected
at 8K on the same checkpoints/documents as the preceding probe. Indices below
are zero-based. This is a targeted mechanism check, not a population estimate.

## Method

`probe_selector_surrogate_gradients.py --dump-operands --layers ...` records
fixed soft operands, gS/gH and the actual hybrid inputs to output normalization.
Baseline document 2: layers 7 and 9; scheduled LN/sigmoid document 1: layers 6
and 9. Exact upstream equality is checked before switching each layer to hard.

`decompose_selector_vjp.py` computes c=<gS,gH>/||gS||² per head and splits
`gH = c*gS + residual`. Each cotangent is passed through a fresh production
soft forward/backward. Gradients are also recomputed after projecting gH onto
the tangent space of the soft hybrid normalization input, independently at each
token/head. RMSNorm removes the radial component along uS; LayerNorm first
centers uS and the cotangent, then removes the centered radial component.
This idealized projection ignores epsilon; it is a diagnostic intervention,
not the actual norm Jacobian or a proposed production gradient rule.

`decompose_selector_dense.py` independently reconstructs single-head full-8K
soft DISM in FP64 from BF16 token/codebook values, with smooth interpolation
(no intermediate BF16 rounding). It separates the local recurrence-score
cotangent into numerator and denominator-correction terms:

    T_num = P * S * (g @ V.T)
    T_den = -P * <g, O>

Each term is propagated through the reverse diagonal recurrence and through
query-LSE / key-interpolation derivatives. Per-token vocabulary contributions
are accounted for as outer products, avoiding a token × vocab × feature buffer.
Token cancellation is sum_i ||gradient_i|| / ||sum_i gradient_i||; this is a
cancellation index, not a multiplier between soft and surrogate gradients.

## Cotangent geometry

| Case | q original ratio | q projected ratio | q_vocab original ratio | q_vocab projected ratio |
| --- | ---: | ---: | ---: | ---: |
| Baseline L7 H1 | 1360.3 | 23.05 | 1074.6 | 12.62 |
| Baseline L9 H0 | 149.3 | 27.23 | 187.9 | 33.24 |
| Scheduled LN L6 H0 | 814.0 | 18.18 | 777.7 | 21.77 |
| Scheduled LN L9 H0 | 15.97 | 4.16 | 11.67 | 3.11 |
| Baseline L7 H4 (control) | 1.38 | 1.04 | 1.34 | 1.02 |
| Scheduled LN L6 H1 (control) | 1.29 | 1.06 | 1.15 | 1.10 |

Ratios are relative to the original soft gradient of the same parameter/head.
The parallel component contributes roughly c times the original gradient; the
large outlier contribution is in the residual. BF16 VJP linearity error
||B - J^T(c*gS) - J^T residual|| / ||B|| ranges from roughly .2% to .7% for the
reported q/q_vocab cases, nowhere near the observed amplification.

For baseline L7 H1, the removed cotangent is only 3.40% of ||gH||; for scheduled
LN L6 H0 it is 5.29%. Nevertheless its image under J_soft^T has almost the same
norm as the complete surrogate q gradient. These fractions are NOT additive
shares of gradient energy after J: components need not remain orthogonal.
Removing it reduces the q ratio by about 59x and 45x respectively. Large
residual amplification remains, so tangent projection is not a complete fix.

Projecting the genuine soft cotangent also changes q norms (e.g. 1.26x/1.28x
in these two outliers), because epsilon, casting and the true normalization
Jacobian are not identical to ideal tangent projection. A production intervention
must not silently discard these effects.

## FP64 audit: not primarily numerator/denominator cancellation

The independent dense oracle reproduces the key amplification:

| Case | q surrogate/soft ratio, FP64 | q_vocab ratio, FP64 |
| --- | ---: | ---: |
| Baseline L7 H1 | 1358.6 | 1073.0 |
| Scheduled LN L6 H0 | 812.9 | 777.8 |
| Baseline L7 H4 control | 1.38 | 1.34 |

Across these three cases and both cotangents, production-vs-FP64 gradient
cosines exceed .99949 and norm ratios differ by less than .8%. Forward output
cosines exceed .99999. Thus BF16 production rounding does not explain the
hundreds/thousands-fold effect in these particular cases. This is not a blanket
certification of every production backward case.

The earlier hypothesis of a large numerator/denominator cancellation breaking
is **not supported for the two outliers**:

| Outlier | Soft ||T_den||/||T_num|| | Local hard/soft ||T|| ratio | After reverse recurrence |
| --- | ---: | ---: | ---: |
| Baseline L7 H1 | 3.90e-5 | 927.94 | 927.94 |
| Scheduled LN L6 H0 | 3.19e-5 | 218.14 | 218.14 |

The denominator correction is tiny on the soft cotangent. The numerator term
alone already increases 952x and 222x. There is no large opposing denominator
term that previously hid it, and the recurrence is not adding a large ratio
at this stage. Combined with the projection intervention, the evidence points
to cotangent/value alignment and norm geometry as the principal mechanism for
these outliers. It does not show that internal normalization or recurrence is
irrelevant for every other head.

## Vocabulary token accumulation adds another effect

| Case | q_vocab cancellation soft → surrogate | k_vocab cancellation soft → surrogate |
| --- | ---: | ---: |
| Baseline L7 H1 | 22.81 → 2.57 | 5.10 → 1.15 |
| Scheduled LN L6 H0 | 14.21 → 4.01 | 5.02 → 1.30 |
| Baseline L7 H4 control | 27.54 → 21.64 | 18.48 → 18.76 |

The surrogate's per-token vocabulary contributions are more coherent in the
outliers. Lost cross-token cancellation further increases their aggregate
parameter gradients. It is an additional mechanism, not an explanation of the
already large q-input gradients or the entire amplification by itself.

## Implications and boundaries

The strongest supported interpretation is: the hard cotangent has components
that are small in output space but strongly excite soft selector sensitivity,
particularly along the soft hybrid output's radial direction. Shared vocabulary
accumulation then removes some cancellation across tokens. Large soft/hard
output-gradient cosine alone does not constrain either effect.

A future STE experiment should test norm-aware cotangent transport and/or
per-head magnitude limiting against a small-alpha baseline, while preserving
real soft samples. Projection alone still leaves 10–30x outliers and can remove
legitimate epsilon-dependent components. No STE or projection training was
started. Repeating at the intended step-14000 switch point remains necessary.

## Artifacts and execution notes

Generated dumps/results live at
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/diagnostics/selector_decomposition_20261025`,
linked from ignored `build/selector_decomposition`. Each selected case has
`results.json` and production gradients; the three oracle cases also have
`dense.json` and dense gradients. `norm_projection.png` and `summary.csv` are
produced by `summarize_selector_decomposition.py`.

Saving the first large operand dump exhausted available system-disk space.
Only this diagnostic's generated directory was moved to the data disk, and
failed dumps were regenerated. The running training continued normally; no
checkpoint or training process was changed. No CUDA kernels were edited and no
sanitizer run was performed. FP64 validation covers only the listed heads.
