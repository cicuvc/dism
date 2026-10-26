# Local sampled Q reads and control variates

2026-10-26. Offline checkpoint diagnostic. No model/optimizer updates, CUDA
execution, checkpoint writes, or changes to the running top-four training.

## Result

A control variate works well for the **frozen-cotangent local router gradient**
on this panel, but ordinary v3 soft output is not a useful scalar baseline.
Its current-read categorical tangent is also an unreliable proxy. A different
construction—soft history followed by the same current hard candidate read—is
effective. Removing that proxy's history altogether works almost identically
and admits a simpler prefix-state implementation.

The impressive pooled variance reductions are dominated by a few heads. On
layer 9 alone, the parameter-free, zero-history control retains 2.81% of raw
sampling variance in baseline C and 0.94% in LN/sigmoid. This is promising
evidence for a prototype, not evidence of better training or a measured speedup.

## Fixed-history sampling and its target

Retain argmax Q history and argmax K labels. At position i, let H_c denote the
complete hard read for candidate c, including its own unit-fallback denominator.
The support is the four largest Q logits, with normalized probabilities pi.
Only the current read is sampled; the draw never changes later Q history.

For fixed input, support, and output cotangent g:

```
r_c = <g, H_c>
score_c = e_c - pi
G = sum_c pi_c * r_c * score_c
```

For an arbitrary candidate proxy s_c and scalar baseline b, frozen for purposes
of the router VJP, the following estimator has exactly expectation G:

```
C ~ categorical(pi)
G_hat = (r_C - alpha*s_C - b) * score_C
        + alpha * sum_c pi_c*s_c*score_c
```

The final term is essential for an action-dependent control. Coefficients must
not depend on the current draw. No score-function gradient of the whole LM
objective is claimed. In a sampled network forward, g itself depends on the
sample: the fixed-g expectation identity does not establish an unbiased LM
gradient. History and support changes remain nondifferentiable, as in existing
top-four training. This experiment does not estimate K-selector gradients.

## Proxies tested

1. **Soft scalar baseline:** b = <g,O_soft>. Also fit one multiplicative
   coefficient on calibration queries.
2. **Soft local tangent:** s_c = d<g,O_soft_i>/dpQ_i(c), with the prior soft
   history frozen. For native direction=true Jensen matching, local read weights
   P_ij and signed readout S_ij give

   ```
   t_ij = P_ij * (S_ij * <g_i,V_j> - <g_i,O_soft_i>)
   s_c = sum_j t_ij*pK_j(c) / pQ_i(c)
   ```

   This intentionally excludes future-history derivatives. Including them would
   target a different action from the local read being sampled.
3. **Soft-history candidate read:** use exp(tau)*(1+H_soft_previous) as the
   continuation weight, but retain the hard current K-symbol filter and each
   candidate's own denominator. This is a new proxy, not native v3 soft output.
4. **Zero-history candidate read:** replace that continuation weight by exp(tau).
   No soft recurrence or fitted model is needed.
5. **Oracle scalar baseline:** variance-minimizing scalar derived from all four
   hard values at each test query. Diagnostic bound only, not a deployable
   single-read baseline.

For candidate-dependent proxies, test both fitted coefficients (plus fitted soft
scalar baseline) and the parameter-free alpha=1,b=0 estimator. Fitting minimizes
logit-gradient variance on earlier queries, never on the later evaluation
queries. All candidate expectations and variances are enumerated exactly.

## Data and numerical protocol

Reuse the immutable operand captures from `SELECTOR_GRADIENT_DECOMPOSITION.md`:

| Model | Final checkpoint | PG19 document index | Layers | Heads |
|---|---:|---:|---|---|
| Original RMSNorm/gate baseline C | 22888 | 2 | 7, 9 | all 12 each |
| LayerNorm/sigmoid scheduled | 22888 | 1 | 6, 9 | all 12 each |

Both use the old `ggggggdddddd` layout and 8192-token inputs. The two models'
documents/layer panels differ; this is not a controlled comparison of models.
These are final checkpoints, not early-training checkpoints or the running
top-four checkpoint. The fixed-.5 model has no operand capture in this panel.

Inputs come from an all-soft forward. Main results use the saved raw-DISM output
cotangent from switching the selected layer, all its heads, to original top-one
hard; other layers remain soft. Repeat with the all-soft cotangent as a
sensitivity check. Neither cotangent comes from a top-four or sampled forward.
Mean-loss cotangents are multiplied by N, uniformly across all estimators.

Each of 48 layer/head cases has 256 evenly spaced queries: 128 in positions
[N/4,N/2) for calibration, 128 in [N/2,N) for evaluation. Thus 6144 held-out
query/head decisions are evaluated. Context remains full 8K; no history is
truncated at the calibration/evaluation boundary. Only one document per model
is represented and adjacent queries are correlated; no confidence-interval or
cross-document generalization claim is made.

All arithmetic is CPU FP64 at the saved BF16 vectors and BF16-cast codebooks.
Q/K argmax is reconstructed from these FP64 dot products, not captured native
labels. Compared with CPU FP32 logits, no Q label changes occurred; one K label
changed across the LN panel, none across baseline. This does not prove identity
with native CUDA label reduction. The independent soft output has relative L2
error 0.10–1.37% versus captured production BF16 output across the heads.

## Gradient results

Variance means sum of per-query estimator covariance traces in selected-logit
coordinates. Each row below is divided by raw single-sample variance. Target
gradient is the same across estimators; lower is better.

| Estimator | Baseline C | LN/sigmoid |
|---|---:|---:|
| Raw single sample | 1 | 1 |
| Soft scalar baseline | 1.00107 | 0.99898 |
| Fitted soft scalar baseline | 0.90119 | 1.00455 |
| Fitted soft local tangent + scalar baseline | 0.88553 | 2.30061 |
| Unit soft local tangent | 0.99995 | 0.99998 |
| Oracle scalar baseline | 0.46808 | 0.53728 |
| Unit soft-history candidate control | 2.05665e-5 | 4.74968e-6 |
| **Unit zero-history candidate control** | **2.02079e-5** | **4.76190e-6** |
| Fitted zero-history control + scalar baseline | 3.53014e-5 | 3.94861e-4 |

The parameter-free structural proxy outperforms the fitted version here. This
also illustrates calibration drift: fitting more coefficients can hurt later
queries. The poor LN soft-tangent result is reported without retuning on test
positions.

Pooled numbers need the following qualifications:

| Unit zero-history control | Baseline C | LN/sigmoid |
|---|---:|---:|
| Earlier selected layer variance/raw | 7.29e-11 (L7) | 1.88e-10 (L6) |
| **L9 variance/raw** | **0.02814** | **0.00943** |
| Median head variance/raw, nonzero raw variance | 2.19e-5 | 0.00327 |
| Worst head variance/raw | 0.50019 | 0.47071 |
| Heads with nonzero raw variance | 20/24 | 22/24 |

Baseline L7 heads 6 and 8 contribute 97.0% of pooled raw variance; LN L6 head 0
alone contributes 98.9%. Several early heads have effectively no useful hard
continuation for these reads, making zero-history nearly exact. The overall
factor is therefore not representative of every head. No active head became
worse with the unit zero-history control on this panel.

The improvement survives mapping router gradients into Q input space and into
the shared Q codebook. For the latter, per-token independent sampling variances
are added and the mean codebook gradients are summed before computing signal
energy. Codebook variance/raw is 3.35e-5 / 2.13e-5, rather than the logit-space
ratios above. This does not include downstream projection-parameter covariance.
With the all-soft cotangent, logit variance/raw is 1.21e-5 / 8.98e-6.

## A cheap structural control, and forward correction

The zero-history read is

```
M_i(c) = sum_{j<=i, k_j=c} sk_j outer v_j
n_i(c) = count{j<=i : k_j=c}
B_i(c) = sq_i^T M_i(c) / (exp(-tau) + n_i(c))
```

This is a per-symbol causal prefix aggregate. It does not require SAM matching
or query-dependent suffix events. With fixed vocabulary/rank/value dimensions,
its work is linear in sequence length: update one symbol state per key and read
four states per query. Prefill can use a grouped/segmented prefix scan; decode
can maintain the matrices/counts incrementally. Efficient training backward,
state storage, scans, and bandwidth still need implementation and profiling.
Four cheap proxy reads are required; this is one expensive history read, not
one total matrix read. Running full native soft on every hard row just to obtain
a control would instead reintroduce quadratic work.

This proxy also permits an unbiased **output** estimator:

```
B_bar = sum_c pi_c * B_c
O_hat = B_bar + (H_C - B_C),   C ~ pi
E[O_hat | fixed inputs/history] = sum_c pi_c * H_c
```

For fixed g its router estimator is

```
G_hat = sum_c pi_c * <g,B_c> * score_c
        + <g,H_C-B_C> * score_C
```

These identities are exact for any B. The first term handles the easy part
analytically, while only the history-dependent residual is sampled. Ordinary
SQ/SK/V/tau gradients and proxy terms would also need consistent treatment in
a training implementation; this diagnostic does not install such an autograd
operator.

Held-out output RMS deviation, normalized by the RMS of the complete hard
top-four mixture, is:

| Output estimator | Baseline C | LN/sigmoid |
|---|---:|---:|
| Plain sampled H_C | 78.19% | 86.19% |
| Zero-history corrected output | 0.156% | 0.161% |

The variance reduction is again pooled and head-concentrated. Simply discarding
the residual would change the operator and could discard rare retrieval events;
it is not justified by low average residual energy. This panel contains ordinary
PG19 text, not a targeted long-match/retrieval challenge.

Low output variance is encouraging but does not establish unbiased downstream
loss/gradients through LayerNorm, gating, and later discrete selectors. Frozen
cotangents can miss precisely these nonlinear effects. This is the next
full-model experiment to perform, with a separately measured cost budget.

## Checks, artifacts, and reproduction

`tools/probe_local_sampling_cv.py --self-test` passes independent FP64 checks:
soft categorical derivative against autograd; corrected estimator expectation;
router softmax derivative against autograd; hard geometric weights against
ordinary recurrence; and full candidate extraction/soft outputs against a
separate scalar recurrence. The complete panel also checks each corrected
estimator's categorical mean and each corrected output's categorical mean.

An additional output-rounding sensitivity check rounds both H and B to BF16
before FP64 moment calculations. Unit zero-history gradient variance/raw remains
1.94e-5 / 4.59e-6. This is not a CUDA numerical test: accumulation error, native
labels, actual kernel output rounding, and runtime cost remain unmeasured.
No CUDA kernels changed and no sanitizer run was relevant to this CPU analysis.

Generated tensors, input hashes, exact checkpoint paths, logs, and
`summary.json` / `rounding_audit.json` are under ignored
`build/local_sampling_cv`, linked to data-disk diagnostic storage. Extraction
uses two pinned CPU threads, nice 10, with CUDA disabled. Captures are read-only.
The extraction stage took about 2–3 seconds per head on this host; that is
diagnostic CPU runtime, not planner or training performance.

Example (conda blkw; no downloads):

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  taskset -c 20-21 nice -n 10 python tools/probe_local_sampling_cv.py \
  --input build/selector_decomposition/ln_scheduled/operands_8192_1_6.pt \
  --model ln_scheduled --output build/local_sampling_cv
CUDA_VISIBLE_DEVICES='' python tools/probe_local_sampling_cv.py \
  --output build/local_sampling_cv --analyze
```

Current evidence favors testing a **prefix-state control plus one sampled hard
residual**, while preserving the existing soft training signal. It does not
support using the original soft output as a drop-in baseline, replacing the
running training, or claiming a subquadratic implementation is already faster.
