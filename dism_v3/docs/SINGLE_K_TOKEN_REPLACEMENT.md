# Single K token replacement and control variates

2026-10-26. CPU-only diagnostic while sampled-CV pretraining continues. No model,
optimizer, training configuration, checkpoint, or production kernel changes.

## Result

Single K replacements have a discernible **frozen-cotangent** signal on part of
the panel. Direct categorical estimates are noisy. An exact zero-history
counterfactual is a much better control than the native soft-path tangent.
Importantly, this control admits linear symbol scans, including the effect of
changing the normalization denominator. This does not establish an unbiased LM
gradient, an LM loss improvement, or a training throughput improvement.

On held-out positions, 40.2% of baseline-C decisions and 69.0% of LN/sigmoid
decisions have a nonzero alternative reward (absolute threshold 1e-12 in the
rescaled reward units below). At least one of the three alternative symbols
improves the linearized loss for 28.0% and 50.3%, respectively. Fifteen of 24
baseline heads and 19 of 24 LN heads have nonzero held-out signals. Missing
signals are important: K top-four support can have little overlap with symbols
read by later argmax Q positions. These are distinct documents/layers, not a
controlled model comparison.

Exact categorical variance, evaluated on 32 held-out positions per head:

| Model / layer | Nonzero decisions | Direct SNR, pooled | Zero-history CV variance / direct, pooled | Same ratio, median active head |
|---|---:|---:|---:|---:|
| Baseline C / 7 | 36.7% | 0.369 | 4.621% | 0.0106% |
| Baseline C / 9 | 43.8% | 0.210 | 0.0301% | 0.0152% |
| LN/sigmoid / 6 | 56.8% | 0.472 | 7.517% | 1.844% |
| LN/sigmoid / 9 | 81.2% | 0.341 | 0.0384% | 11.327% |

Do not extrapolate the pooled reductions to every head: one head contributes
91.8% of baseline and 99.7% of LN squared target-gradient signal. The median
per-head direct SNR is about 0.39–0.43 in all four layer panels. LN layer 9's
median per-head SNR increases from 0.395 to 1.342 with the zero-history control;
some heads still remain below one. Near-exact zero-history heads produce
roundoff-sized residual variance, not evidence of universally noiseless learning.
SNR here is sqrt(sum target squared norm / sum categorical variance) in local
top-four logit coordinates. It excludes variance from choosing positions,
heads, layers, documents, or minibatches and from the full nonlinear network.

The native soft probability tangent is not a consistently useful proxy. Its
pooled centered correlation with hard replacement rewards is 0.349 in baseline
and 0.012 in LN; nonzero alternative sign agreement is 58.8% and 62.3%.
A coefficient fitted on separate calibration positions retains 189.5% and
100.0% of direct variance, respectively. In LN layer 6 it raises variance to
1058%, showing that a fitted coefficient need not transfer across positions.

Cotangent sensitivity is substantial. Repeating with the all-soft cotangent
still finds improving alternatives for 27.2% / 49.1% of decisions, but pooled
zero-history residual variance is 0.436% / 36.24% of direct variance. At layer 9
alone it is 27.48% / 36.92%, much higher than with the local-hard cotangent.
Thus the control remains useful in aggregate here, but its dramatic main-panel
reduction is not a cotangent-independent property. This further motivates
actual loss checks before making claims about a sampled training network.

## Intervention and estimator

All Q labels stay argmax. All K labels except one position j stay argmax. The
support consists of the four largest K logits at j, normalized within that
support; slot zero is argmax. For each candidate c, change K[j] to c in the
actual committed string and recompute every affected suffix match and readout.
This includes the symbol's later historical uses, unlike an endpoint-only K
replacement. Each action is an independent single-position intervention, not
simultaneous changes to the whole string.

Let H_c denote the sequence of full-history outputs, and B_c the corresponding
zero-history outputs. Both use their own exact unit-fallback normalization.
With fixed cotangent g, define

```
r_c = sum_i <g_i, H_c[i] - H_argmax[i]>
s_c = sum_i <g_i, B_c[i] - B_argmax[i]>
score_c = e_c - pi
G = sum_c pi_c * r_c * score_c

C ~ categorical(pi)
G_hat = (r_C - s_C) * score_C + sum_c pi_c*s_c*score_c
```

The second term is essential: an action-dependent proxy cannot simply be
subtracted as a scalar baseline. All four outcomes are enumerated here, so both
the expectation and categorical variance are exact for this local objective.
The main control uses coefficient one without fitting. Fitted controls use
alternating selected positions for calibration and the other half for testing.
The direct estimator already subtracts the paired argmax baseline; it is not
the particularly noisy estimator based on an uncentered full-sequence loss.

The soft proxy differentiates the native direction=true Jensen soft recurrence
through its entire future diagonal history and readout normalization. Its
candidate value is d<g,Osoft>/dpK_j(c) minus the value for the argmax candidate.
It is a probability-space tangent at the original soft distribution, not an
exact finite one-hot intervention or a full-soft-logit gradient comparison.

## History and normalization

For one K replacement, changed pairs begin at (i,j), where Q[i] equals the old
or new label. Their differences propagate to (i+1,j+1), etc. while the unchanged
successor symbols match, and vanish at the next mismatch. Maximum observed
propagation is 18 hops in baseline and 19 in LN; this is an observation, not an
algorithmic bound. Long repetitive strings can propagate much farther.

An endpoint-only intervention retains the original predecessor history and
changes only pairs whose key endpoint is j. Its weighted reward error relative
to full replacement is 18.4% / 1.72% for baseline layers 7 / 9 and 10.3% / 1.35%
for LN layers 6 / 9. Pooled late-layer errors hide larger per-head differences:
the median active LN layer-9 head has 14.9% endpoint error. Historical effects
are therefore often small on this panel, but not generally negligible.

For baseline output O_i and denominator Z_i, accumulate all affected pair
weights in a row before evaluating its exact counterfactual:

```
O'_i - O_i = sum_j delta_w_ij * (x_ij - O_i)
             / (Z_i + sum_j delta_w_ij)
x_ij = (sq_i dot sk_j) * v_j
```

Omitting denominator changes yields large errors (pooled relative errors 1.91
and 2.51), despite high correlation in some panels. The control and primary
intervention both include these changes exactly.

## Linear zero-history control

Zero-history readout has committed-symbol prefix states

```
M_i(c) = sum_{j<=i, K[j]=c} sk_j outer v_j
n_i(c) = count{j<=i : K[j]=c}
b_i = exp(-tau) + n_i(Q[i])
B_i = sq_i^T M_i(Q[i]) / b_i
```

After that forward scan, build reverse states for each symbol c:

```
A_c^+(j) = sum_{i>=j,Q[i]=c} sq_i outer g_i / (b_i + 1)
t_c^+(j) = sum_{i>=j,Q[i]=c} <g_i,B_i> / (b_i + 1)
A_c^-(j), t_c^-(j): same with (b_i - 1), including only n_i(c)>0
```

For c different from original symbol a=K[j], the exact zero-history reward is

```
s_j(c) = sk_j^T A_c^+(j) v_j - t_c^+(j)
         - sk_j^T A_a^-(j) v_j + t_a^-(j)
s_j(a) = 0
```

Each query contributes to one symbol's states; a selected key reads its few
candidate states. For fixed vocabulary/rank/value width/top-k this is linear
in sequence length, with O(C*R*DV) working symbol state plus saved per-token
outputs/counts. The prototype matches the dense zero-history counterfactual
within 1.3e-15 relative L2 on four real layer/head cases. These scans calculate
the control for a single changed K with all other labels fixed. They do NOT
calculate the expected normalized output of many independently sampled K's,
or make the full-history residual free. No GPU implementation or speedup is
claimed; the scalar Python CPU prototype takes about 0.85 s per 8K head here.

## Inputs, validation, and limits

Reuse the immutable captures described in
[LOCAL_SAMPLING_CONTROL_VARIATES.md](LOCAL_SAMPLING_CONTROL_VARIATES.md): final
step-22888 baseline C, PG19 document 2, layers 7/9; final LN/sigmoid scheduled,
document 1, layers 6/9. All 12 heads each: 48 head cases. There are 64 evenly
spaced K positions from N/16 to 15N/16, N=8192, with 32 calibration and 32 held-out
positions per head: 1536 held-out decisions and 4608 non-argmax interventions.

Inputs come from the all-soft upstream network. Main g comes from switching
only the selected layer, all its heads, to original top-one hard, leaving other
layers soft. Results with the all-soft g are also saved. Neither is a cotangent
from the current top-four or sampled-CV training. Saved mean-loss cotangents
are multiplied by N; reward magnitudes are not actual mean LM loss changes.

Arithmetic is FP64 at captured vectors and BF16-cast codebooks. Labels are
reconstructed from CPU FP64 dot products: versus CPU FP32, zero Q labels and
one K label differ across the panel. Soft output relative L2 discrepancy versus
saved production BF16 outputs ranges from 0.128% to 1.192%. Rounding the full
counterfactual outputs to BF16 retains high pooled reward correlation, but
small-sign agreement is only 91.9% / 83.9%; this check is not a native CUDA
replacement comparison and does not validate native reduction-rounding errors.

Independent small FP64 tests compare full recomputation and endpoint-only
recomputation, including long identical-symbol chains, first/last positions,
empty symbol buckets, and tau=-0.4/0/0.3. Full soft tangents match autograd;
categorical control expectations match enumeration; linear control matches
dense zero-history replacement. No tolerances were relaxed. No GPU kernels
were changed, so no new CUDA sanitizer claims are made.

Tools: `tools/probe_single_k_replacement.py` (`--self-test` for independent
checks) and `tools/check_single_k_prefix_control.py`. Ignored artifacts are
under `build/single_k_replacement`, a data-disk symlink: per-head `.pt`/`.json`,
`summary.json`, reproducible aggregation `summarize.py`, and `validation.log`.
Extraction source and immutable input hashes/checkpoint paths are embedded in
each artifact. The later script change only extends the self-test to include
the separate linear-control prototype.

Next meaningful validation is paired actual LM loss under sparse K mutations,
with other random choices identical, before changing training. The present
evidence supports investigating a K endpoint/control path; it does not by
itself establish that full historical REINFORCE is required or that long-range
retrieval improves. The running sampled-CV experiment remains unchanged.
