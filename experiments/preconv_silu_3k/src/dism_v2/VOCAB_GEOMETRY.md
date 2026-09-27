# Shortconv SiLU / codebook geometry diagnostics

2026-09-10, local RTX 5090, conda `blkw`. Inference only: no optimizer steps,
checkpoint edits, production model changes, or kernel changes.

## Setup

Original hybrid checkpoint `dism-swa50m-softcap30-20260909-offline/latest.pt`,
step30000, 15 layers, 4 heads, D64, vocabulary512, context2048, pure hard,
BF16 CUDA embedding/core and model. NLL uses BF16 LM head followed by accurate
FP32 torch softcap30 and cross entropy, not the approximate training CE.

Inputs are the frozen `swa-eval-bundle-20260910.pt`. Calibration uses sequence
indices [0,32), evaluation [32,256), first2048 tokens of each8192-token sequence.
Thus calibration and evaluation do not share sequence indices. Earlier64-sequence
pilot is a subset of the224-sequence evaluation, not an independent replication.
The evaluation contains458752 target tokens. All variants use exactly these tokens.

Calibration moments use65536 tokens per head and side. Fixed-input label experiments
sample every16th token (4096/head). Global frequency correlations use the previously
measured13,107,200 labels/head/side. That larger corpus overlaps calibration, so
the correlations are descriptive, not independent causal evidence.

## Geometry on unchanged baseline activations

Let mu be the dataset activation mean for a head, E_i its actual BF16 codeword.
The logit decomposes as z E_i = mu E_i + (z-mu) E_i. Correlations below are Pearson
correlations across512 codewords between log1p(global count) and the named statistic,
then averaged across60 heads.

| Statistic | Q | K |
|---|---:|---:|
| Frequency correlation with mu dot E_i | 0.698 | 0.666 |
| Frequency correlation with codeword norm | 0.391 | 0.354 |
| Frequency correlation with codeword coordinate mean | 0.192 | 0.094 |
| Usage mass on the8 codewords with largest mu dot E_i | 60.5% | 65.8% |
| Mean energy / total energy, before SiLU | 47.2% | 64.1% |
| Mean energy / total energy, after SiLU | 43.3% | 54.2% |
| Mean coordinate, before SiLU | -0.1833 | -0.1882 |
| Mean coordinate, after SiLU | 0.1302 | 0.0828 |
| Negative coordinate fraction, after SiLU | 55.5% | 54.9% |

Mean-energy fraction is ||E[z]||²/E[||z||²], calculated per head before averaging.
The mean-bias RMS / residual-content RMS across vocabulary is0.943Q and1.300K;
the common-across-codeword logit component is removed for both RMS quantities.
This ratio is not an explained-variance R².

For pre-SiLU measurements, replay the same stateless convolution with activation=None
at the original baseline layer input. Do not invert SiLU or use activations from a
globally modified model. SiLU lowers the mean-energy fraction in36/60 Q heads and
55/60 K heads. Thus these trained activations do not support the claim that SiLU
alone introduced the anisotropy. Their learned projection/convolution already
has a large mean component. This does not identify how it arose during training.

FP32 GEMM replay using actual BF16 activations/codewords reproduces every sampled
CUDA argmax (zero changed labels). At fixed original layer inputs:

| Intervention | Q effective vocab | K effective vocab | Changed Q/K labels |
|---|---:|---:|---:|
| Baseline | 19.88 | 16.40 | 0 / 0 |
| Subtract full calibration mean | 92.66 | 80.87 | 67.2% / 73.8% |
| Apply SiLU to codewords | 19.46 | 16.48 | 36.3% / 42.0% |
| Disable Q/K SiLU | 16.06 | 10.06 | 48.5% / 58.1% |

Effective vocab is exp(entropy), averaged per head; it is not the count of used IDs.
These local label experiments do not propagate changed outputs into subsequent layers.

## End-to-end reversible checkpoint surgery

Each intervention is applied to all15 layers, allowing changes to propagate.
Codeword SiLU is evaluated on FP32 master parameters before casting to BF16 and is
used consistently for both Q/K addressing and interpolation. The norm-matched variant
rescales each transformed codeword to its original individual L2 norm. It does not
mean-center or normalize Q/K.

Centering subtracts a fraction of the separately calibrated per-head mean vector
from Q/K after the original SiLU, then casts to BF16. V, tau, SWA and all weights
remain unchanged. With pure-hard core, interpolation values themselves are unused;
the codeword intervention affects the output through labels, not interpolated values.

| Variant | NLL | Delta vs baseline | Paired SE | Effective Q/K vocab |
|---|---:|---:|---:|---:|
| Baseline | 3.389215 | 0 | 0 | 19.75 / 16.19 |
| Codeword SiLU | 3.441231 | +0.052016 | 0.002293 | 19.38 / 15.89 |
| Codeword SiLU, original norm preserved | 3.416848 | +0.027633 | 0.001642 | 20.95 / 16.50 |
| Disable Q/K SiLU | 3.429387 | +0.040172 | 0.001707 | 15.95 / 9.76 |
| Subtract25% mean | 3.389736 | +0.000521 | 0.000373 | 26.02 / 21.05 |
| Subtract100% mean | 3.607987 | +0.218772 | 0.002539 | 98.75 / 79.42 |

SE is computed from paired per-sequence mean NLL differences, not individual tokens.
Packed documents remain correlated; error bars are descriptive. The25% intervention
does not show a statistically clear benefit or loss at a nominal95% interval.
More uniform codeword usage is not by itself a language-model improvement.
The absolute baseline differs from earlier validation because the sample differs.

Interpretation: mean-dependent codeword preference is strongly present, but removing
SiLU or matching activations on both sides is not an immediate fix for this checkpoint.
It is still possible that a symmetrically parameterized model trained from scratch
would learn differently. These experiments cannot decide that question.

## Reproduction and checks

```bash
RUNS=/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs
RUN=$RUNS/dism-swa50m-softcap30-20260909-offline
OMP_NUM_THREADS=4 /home/cicuvc/miniconda3/envs/blkw/bin/python \
  -m dism_v2.check_vocab_geometry \
  --checkpoint "$RUN/latest.pt" \
  --bundle "$RUNS/swa-eval-bundle-20260910.pt" \
  --global-counts "$RUN/vocab-load-validation/counts.pt" \
  --output "$RUN/vocab-geometry-rerun" --sequences 224
```

Output must be a new directory. Existing artifacts:
`$RUN/vocab-geometry-confirm-20260910/{report.json,geometry.json,calibration.pt,losses.pt}`
plus six per-variant count tensors. Calibration includes pre/post SiLU sampled
activations, means, and second moments for further CPU-only analysis.

Runtime checks: original features equal uninstrumented baseline bitwise; fixed-input
CUDA argmax equals FP32 replay; all evaluated losses finite; expected call/count totals;
all persistent tensor bytes unchanged; restored baseline per-token NLL bitwise equal.
Helper/cleanup tests: `tests/test_vocab_geometry.py` and `tests/test_vocab_load.py`,
5 passed. No training or checkpoint save was performed.
