# Interpolation top1 margin and vector norms, three3000-step checkpoints

Measured64 identical validation sequences x2048, every8th position,16384 sampled
rows per head/side.15 layers x4 heads,1,966,080 sampled choices/model. Pure-hard
model inference, but observe the embedding interpolation softmax computed before
the hard core. Actual BF16 token Q/K and effective codebooks are replayed in FP32,
using actual scale1 (no extra1/sqrt64). Q uses Q codebook; K uses K codebook.

Definitions: probability margin=p_top1-p_top2; logit margin=z_top1-z_top2 in nats,
so p_top1/p_top2=exp(logit margin). L2 norms are over each64D head, not residual256.

## Softmax confidence

Numbers are pooled across equal-sized heads/tokens, not softmax of mean logits.

| Arm/side | Mean p1 | Median p1 | Mean probability margin | Median margin | Mean logit margin | Mean entropy |
|---|---:|---:|---:|---:|---:|---:|
| baseline Q | .6842 | .7575 | .5814 | .6466 | 3.1814 | 1.2882 |
| baseline K | .4888 | .4384 | .3778 | .2695 | 1.7243 | 2.3433 |
| vocab_silu Q | .6647 | .7349 | .5673 | .6197 | 3.1261 | 1.4962 |
| vocab_silu K | .4915 | .4486 | .3798 | .2735 | 1.7431 | 2.3946 |
| no_qk_silu Q | .8380 | .9717 | .7506 | .9535 | 6.2337 | .5116 |
| no_qk_silu K | .7057 | .7657 | .5909 | .6497 | 3.3157 | 1.0730 |

Fractions p1>.9 (Q/K): baseline37.64%/13.77%, vocab_silu35.73%/14.15%,
no_qk_silu60.42%/37.66%. Probability margin<.01:2.30%/5.06%,2.79%/5.31%,
.73%/1.62%. No-activation model is substantially more confident per input even
though its language-model NLL is worse. Confidence is not correctness.
Marginal codeword utilization entropy and per-input softmax entropy describe
different axes; more concentrated marginal use need not mean sharper row softmax.

## Token L2 norm distribution

| Arm/side | Mean | P10 | Median | P90 |
|---|---:|---:|---:|---:|
| baseline Q | 3.8038 | 2.5992 | 3.6786 | 5.1949 |
| baseline K | 2.8436 | 2.0686 | 2.7737 | 3.7092 |
| vocab_silu Q | 4.6271 | 3.1467 | 4.4157 | 6.4726 |
| vocab_silu K | 4.0168 | 2.7929 | 3.8782 | 5.4383 |
| no_qk_silu Q | 5.8887 | 4.6616 | 5.8336 | 7.2029 |
| no_qk_silu K | 4.3924 | 3.4882 | 4.3281 | 5.3770 |

## Effective codebook L2 norm distribution

All512 entries per head, equal weighting, not usage weighted. Vocabulary SiLU
applied in FP32 before BF16 cast; baseline/no_qk_silu effective codebook is just
the BF16 master cast.

| Arm/side | Mean | P10 | Median | P90 | Mean norm of selected entry |
|---|---:|---:|---:|---:|---:|
| baseline Q | 7.9113 | 7.0013 | 7.9027 | 8.8309 | 8.7979 |
| baseline K | 7.9193 | 7.0215 | 7.9075 | 8.8300 | 8.5532 |
| vocab_silu Q | 4.6541 | 3.7217 | 4.6244 | 5.6135 | 5.9165 |
| vocab_silu K | 4.6912 | 3.7741 | 4.6666 | 5.6534 | 5.7098 |
| no_qk_silu Q | 7.9342 | 7.0294 | 7.9284 | 8.8490 | 8.5503 |
| no_qk_silu K | 7.9199 | 7.0227 | 7.9108 | 8.8292 | 8.4972 |

Raw master mean norms: baseline7.9114/7.9193, vocab_silu7.9267/7.9532,
no_qk_silu7.9343/7.9199. Thus applying SiLU, not a large raw-parameter norm change,
accounts for most of the vocabulary_silu effective-codebook norm difference.

## Interpretation and limits

No_qk_silu increases token norms by about55% while codebook norms stay nearly
unchanged; logit gaps grow about96%Q/92%K. At fixed vector direction, positive
token scaling leaves argmax unchanged but increases logit gaps/sharpness.
However mean(logit gap/token norm) is0.7963/0.6020 baseline versus1.0107/0.7216
no_qk_silu: scaling is not the entire difference. Mean selected cosine also changes
from.4349/.4186 to.4843/.4404. These are trained-model observational comparisons,
not a controlled proof of a mechanism.

Within-head token norm/logit-margin Pearson correlations averaged over heads:
baseline.4091/.2878, vocab_silu.4655/.2866, no_qk_silu.4135/.3860. Positive association
is present, not perfect. Codeword SiLU reduces codebook norms, increases token norms
and selected cosine (.5408/.5099), while maintaining roughly baseline margins.
Selected entries tend to have larger norms than the uniform codebook average in
all arms; selection itself induces bias, so this alone does not prove collapse.

Sharper interpolation in no_qk_silu is consistent with its smaller measured
soft-to-hard NLL penalty. It does not explain why overall NLL is worse without
additional training-time/causal diagnostics. Per-input confidence is not semantic
retrieval accuracy, and raw norm magnitude alone is not a quality objective.

Validation: zero argmax disagreements against CUDA in all5,898,240 sampled choices;
max p1 absolute error3.16e-6/3.46e-6/4.23e-6. Traced first-batch model outputs
bitwise equal untraced outputs; all recorded metrics finite. Uniform-softmax and
margin-ratio/shift helper tests2 passed. No parameters/training state modified.
Sampling every8th position can carry positional bias; no all-token census claim.

Artifacts: `dism-lm-runs/activation-study-3k-20260910/embedding-margin-norm/`:
report.json includes mean/std/P01/P10/median/P90/P99 and per-layer/head statistics;
each mode.pt stores[L,side,H,sample,metric], side Q/K, metric names in report;
*_codebook_norms.pt stores raw/effective[L,H,512] norm arrays. The earlier
embedding-margin directory contains probability-only results on identical samples.
Reproduce with `python -m experiments.embedding_margin_3k --root STUDY --bundle
BUNDLE --output NEW_OUTPUT`.
