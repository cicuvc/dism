# Soft preferences versus local hard costs across training

Date: 2026-10-25. Diagnostic only; no training or checkpoint changes.

## Question and conclusion

Following [the hard candidate probe](HARD_SYMBOL_COUNTERFACTUALS.md), test whether
soft loss gradients prefer beneficial discrete symbol replacements, and whether
this preference disappears as the hard-path probability increases.

For the LN + sigmoid scheduled run, the available steps 12000, 14000, 22000 and
22888 do **not** show a strong preference followed by monotonic deterioration.
Candidate ordering is already weak at 12000/14000. The final checkpoint actually
has the clearest positive signal on the sampled 8K document. The 2K results remain
weak. This is evidence of an imperfect, context-dependent surrogate for local
hard coordinate changes, not evidence that soft training is useless or that
hardening has no effect on learning.

The earliest checkpoint already has hard probability about 0.50. This experiment
cannot establish what happened in the first half of training or causally separate
hardening from other changes during training.

## Protocol

- Same final-checkpoint candidate panel as the preceding probe: PG19 documents
  0 and 1 at 2048 tokens, document 0 at 8192 tokens, with BOS; layers 6 and 9,
  heads 0 and 1, Q and K, midpoint and a final-checkpoint small-margin position.
  There are 32 nodes at 2K and 16 at 8K per checkpoint. Indices are zero-based.
- Keep four symbol IDs per node fixed across checkpoints: the final argmax and
  three final alternatives. Include the current checkpoint's argmax if absent.
  This is a retrospective panel, not a random population sample.
- Measure full-model hard loss after changing one symbol. Freeze **all other
  hard labels at every layer** to that checkpoint's baseline. Continuous
  activations still respond to the intervention. Costs are summed token NLL
  differences; negative means improvement.
- Compute soft loss cotangents in two backgrounds: `all_soft` (all DISM layers
  soft) and `local_soft` (only the target layer soft, others hard with frozen
  labels). In the latter, target Q/K/SQ/SK/V/codebooks/tau are asserted exactly
  equal to the hard baseline inputs. These backgrounds bracket the state
  mismatch; neither reproduces every mixed training mask.
- Reconstruct the dense soft operator's categorical derivatives in FP64 at the
  captured BF16 operands and codebook values, using the actual model loss
  cotangent at raw DISM output. Multiply mean-loss cotangents by sequence length.
  This is an ideal soft-operator Jacobian diagnostic, not a claim of bitwise
  identity with the production BF16 backward.
- Main score: `dL/dp[c] - dL/dp[a]`, the directional derivative of transferring
  probability from current argmax `a` to candidate `c`. Also record
  `dL/dz[c] - dL/dz[a]`, the logit-margin directional derivative. Lower predicts
  a better candidate. Neither is the result of an Adam update or a realizable
  shared Q/K/codebook parameter update.
- Pair accuracy compares soft ordering with actual hard cost ordering within
  the fixed four-symbol panel, excluding exact ties. Selection utility chooses
  the minimum predicted score among the panel plus current argmax (score zero).
  Uniform and hindsight-best comparisons use that same set, including no-op.

For Q, with `A = T @ p_k`, `dL/dp_q = A/p_q` and
`dL/dz_q = A - p_q sum(A)`. For K, `B = T.T @ log(p_q)`,
`dL/dp_k = B` and `dL/dz_k = p_k (B - sum(p_k B))`.
Here T includes the reverse diagonal softplus recurrence, not just the final
normalized readout derivative.

## Results

Pair ordering agreement, probability-transfer score:

| Step | Hard probability | 2K all-soft | 2K local-soft | 8K all-soft | 8K local-soft |
|---|---:|---:|---:|---:|---:|
| 12000 | 49.8% | 50.0% | 55.1% | 41.1% | 32.1% |
| 14000 | 58.1% | 47.9% | 53.7% | 49.2% | 53.8% |
| 22000 | 91.3% | 46.5% | 55.3% | 46.4% | 51.8% |
| 22888 | 95.0% | 46.0% | 45.1% | 60.7% | 62.5% |

Each 2K cell has 113–121 non-tied pairs; each 8K cell has 56–65. These are
correlated pairs from very few documents, not independent statistical trials.
Filtering to hard cost differences greater than 0.1 summed NLL gives final
agreements 46.4%/44.6% at 2K and 60.4%/62.3% at 8K, with no change in the
qualitative temporal pattern.

Mean actual summed-NLL change per node after selecting a candidate:

| Step | Context | All-soft selection | Local-soft selection | Uniform expectation | Hindsight best |
|---|---:|---:|---:|---:|---:|
| 12000 | 2K | +0.201 | +0.154 | +0.199 | -0.212 |
| 14000 | 2K | +0.150 | +0.179 | +0.232 | -0.301 |
| 22000 | 2K | +0.251 | +0.151 | +0.070 | -0.462 |
| 22888 | 2K | -0.031 | -0.085 | -0.152 | -0.612 |
| 12000 | 8K | +0.136 | +0.302 | +0.081 | -0.684 |
| 14000 | 8K | +0.242 | +0.236 | +0.187 | -0.432 |
| 22000 | 8K | +0.560 | +0.595 | +0.776 | -0.173 |
| 22888 | 8K | -1.531 | -1.561 | -0.800 | -1.894 |

At final 8K, all-soft selection improves 10/16 nodes and leaves 6 unchanged;
local-soft improves 9/16 and leaves 7 unchanged. There are no worsened selections
in this small slice. At final 2K, all-soft improves/worsens/leaves unchanged
12/7/13 nodes; local-soft 10/7/15. Their aggregate gains are smaller than the
uniform expectation. Isolated gains cannot be added to predict a joint update.

The secondary logit-margin metric gives final pair agreements 41.6%/35.4% at
2K and 58.9%/60.7% at 8K. Final 8K selected changes are -1.234/-1.332 summed
NLL. Thus the positive final-8K result and weak 2K result are not specific to
using probability-transfer scores. Q/K breakdowns also do not reveal a stable
one-sided temporal trend: final all-soft 8K agreement is 60.0% for Q and 61.5%
for K, versus 44.6% and 47.4% at 2K.

## Direction versus frequency of receiving soft gradients

The all-soft median categorical-logit gradient norm at 2K is
3.73e-4, 2.48e-4, 2.69e-4, 3.14e-4 across the four checkpoints. It does not
collapse monotonically. However, the scheduled soft-path fraction falls from
50.2% to 5.0%, a roughly tenfold reduction in opportunities to receive direct
soft selector gradients. Multiplying the conditional median norms by that
fraction would give about a twelvefold reduction from step 12000 to final.
This is only an exposure-scale illustration, **not** a measured optimizer update:
mixed masks, gradient correlations, shared Q/K use by GDN, learning rate and
optimizer moments all matter.

Consequently, less frequent soft correction remains a plausible hardening
effect even though the present data do not show a reliably useful direction
being erased. A finite categorical jump also differs from an infinitesimal
surrogate direction; soft can learn useful features without reliably ranking
every single-symbol hard intervention. Holding target-layer inputs equal does
not consistently fix the disagreement, so upstream state mismatch alone is
not a sufficient explanation here.

To separate causes further, the next useful diagnostic is a same-input finite
soft probability-transfer curve toward each hard candidate: compare its
initial derivative, finite soft loss change, and actual hard loss change.
This can distinguish local-linearization failure from soft/hard objective
mismatch. Establishing a causal hardening effect would additionally require
matched continuation experiments from an earlier shared checkpoint with held
versus ramped hard probability; no such training was launched here.

## Validation and artifacts

- FP64 independent-logit derivative checked against autograd on a small dense
  reference, atol 1e-10 and rtol 1e-9, at the start of every checkpoint job.
- Hard no-op replay and causal-prefix equality exact; model RNG counter unchanged.
- Final checkpoint's 144 alternative hard costs exactly reproduce the preceding
  hard-candidate probe.
- Dense soft forward relative error versus captured production output is at
  most 1.14% across these heads/backgrounds. This is a numerical limitation;
  the derivative unit check does not certify every BF16 production gradient.
- No CUDA kernels changed; no sanitizer run. All four jobs have completion
  markers and 48 records each. Existing training was left running.

Tools: `tools/probe_soft_candidate_history.py` and
`tools/summarize_soft_candidate_history.py`. Invoke from the repository root
with conda `blkw` and this worktree's `dism_v3/python:nanochat` on `PYTHONPATH`.
The probe takes `--checkpoint`, `--panel` (preceding LN scheduled probe's
`results.json`), and `--output`. The summarizer reads
`dism_v3/build/soft_candidate_history/{step}/results.json`.

Generated data are outside Git under the ignored symlink
`dism_v3/build/soft_candidate_history`: exact checkpoint manifest, per-step
results/logs, `summary.json`, `candidate_scores.csv`,
`pair_threshold_sensitivity.json`, and `history.png`. Its target is
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/diagnostics/soft_candidate_history_20261025`.

## Follow-up: reported historical output alignment peak

The user reports an earlier training experiment in which hard/soft DISM output
cosine rose to about 0.89 during early/middle training, then fell to 0.6–0.75.
The records are not on this machine; at the user's request no further search
was made. Treat this as a supplied historical observation, not a measurement
from the present LN + sigmoid run. Its exact aggregation/input protocol is
unavailable here.

That observation supports considering late forward-path divergence. It does
not directly establish that a correct soft preference over discrete hard
symbols existed and was subsequently lost: output alignment and local hard
credit assignment are different measurements. Increasing hard-path sampling
probability also does not itself anneal the symbol softmax temperature or force
its distributions to become one-hot. Shared V/SQ/SK/tau increasingly receive
hard-path updates while direct soft selector feedback becomes less frequent;
branch specialization is a plausible mechanism, not a demonstrated cause.

Added `tools/probe_output_alignment_history.py`, a forward-only probe of all six
DISM layers and all heads on the same three document/context cases. It measures
raw DISM output **before** GDN addition, normalization and gating. Each actual
operator call is paired with the opposite path on exactly the same operands,
under both all-soft and all-hard upstream backgrounds; full-model all-soft
versus all-hard outputs are also compared. Per-head cosine flattens tokens and
channels, then the table takes the median across document/layer/head cells.

| Step | 2K same hard inputs | 8K same hard inputs | 2K end-to-end | 8K end-to-end |
|---|---:|---:|---:|---:|
| 12000 | 0.283 | 0.332 | 0.277 | 0.327 |
| 14000 | 0.378 | 0.384 | 0.358 | 0.352 |
| 22000 | 0.467 | 0.473 | 0.466 | 0.471 |
| 22888 | 0.462 | 0.478 | 0.463 | 0.473 |

The sampled second half of **this** run shows improving output alignment then
a plateau, rather than the reported historical peak/drop. Same-soft-input
cosines show the same broad trend. Thus the historical observation cannot be
generalized to this configuration from these checkpoints, nor can this run
invalidate that observation. Earlier checkpoints, different training designs,
data, and aggregation definitions remain relevant differences. Notably, output
alignment here improves while local hard-candidate gradient ordering remains
weak; the two measurements need not move together.

Validation: paired calls leave the original operator output bitwise unchanged;
instrumented and plain full-model mean losses match exactly in all 24 passes;
the model RNG counter stays unchanged. The mean-loss CE route differs from the
unreduced CE route used for candidate costs, so baseline loss values from the
two probes should not be treated as bitwise-equivalent. No backward or training
update was performed. Results are in per-step `output_alignment.json`,
`output_alignment_summary.json` and `output_alignment.png` under the same
artifact directory. The historical series is deliberately not plotted.
