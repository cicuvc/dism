# Local hard-symbol candidate replacement

2026-10-25. Small diagnostic of whether individual Q/K symbol changes have
measurable task-loss consequences. No optimizer updates, categorical-training
changes, or checkpoint mutation. The running RMSNorm/sigmoid training is retained.

## Protocol

Use the same final checkpoints as SELECTOR_SURROGATE_GRADIENTS.md: baseline C,
LN/sigmoid scheduled, and LN/sigmoid fixed .5. Evaluate the first two local PG19
prefixes at 2K and the first at 8K, with BOS and identical targets, batch 1,
all-hard inference, MXFP8 FFN and BF16 CE. Per-token CE is accumulated in FP64.

For layers 6 and 9, heads 0 and 1 (zero-based), inspect Q and K separately at:

- the midpoint token;
- the smallest top-1/top-2 logit-margin token in [N/4,N/2).

For each node, test three alternatives from its top-four FP32 logits (using
BF16 token/codebook values), excluding the actual baseline kernel argmax.
Candidate probabilities are selector confidence, NOT a soft-loss gradient.
No whole-vocabulary enumeration, paired Q/K switch, or learned selector update.

Only one token's Q or K label in one head/layer changes. Every other hard label
in EVERY layer is frozen to the baseline assignment; continuous downstream
activations still respond normally. The hard summary/scan/readout is recomputed
for the modified labels, rather than reusing incompatible hard state. Each
candidate is measured independently against the same baseline.

There are 48 nodes and 144 candidates per model, 144 nodes / 432 candidates total.
At 2K there are 32 nodes/model (two documents); at 8K 16 (one document). Locations
selected by low margin differ by model. This is not a representative population
estimate or a controlled comparison of identical candidate interventions across
models. PG19 remains diagnostic data, not training supervision.

## Results

| Model | Candidates improve / worsen / unchanged | Nodes with a better alternative | Nodes with total-NLL gain > .1 |
| --- | ---: | ---: | ---: |
| Baseline C | 40 / 51 / 53 | 17/48 | 15/48 |
| LN/sigmoid scheduled | 68 / 20 / 56 | 30/48 | 30/48 |
| LN/sigmoid fixed .5 | 9 / 13 / 122 | 6/48 | 6/48 |

| Model | Better-alternative nodes at 2K | At 8K |
| --- | ---: | ---: |
| Baseline C | 17/32 | 0/16 |
| LN/sigmoid scheduled | 19/32 | 11/16 |
| LN/sigmoid fixed .5 | 3/32 | 3/16 |

Among nodes with a beneficial alternative, median best total-NLL gains are
2.064 / 1.296 / 1.563 nats respectively. Maximum gains are 4.334 / 4.850 / 5.223.
These are sums over all scored target tokens, NOT per-token NLL gains. All
candidates retain the same normalization within each document/context.
For example, scheduled LN/sigmoid at 8K, document 0, L6 H1, midpoint Q:
changing one symbol reduces total NLL by 4.850 (mean NLL by 0.0005921), and
increases its direct causal matches from 0 to 51.

Benefits are not solely deletion of matching: among the 30 improved scheduled
LN/sigmoid nodes, the best candidate adds direct matches in 15 and removes them
in 15; only four best candidates have zero direct matches. Baseline has six
adding and eleven removing matches, with two zero-match best candidates.
These counts measure direct equality edges incident on the changed node, not
recursive chain length or independent useful retrieval events.

Fixed-.5 has 122/144 unchanged candidate losses, consistent with the earlier
observation of many suppressed DISM heads. That observation alone does not prove
why every unchanged candidate is inactive: some substitutions also leave the
relevant match structure unchanged.

## Confidence ranking is not a reliable utility ranking here

Across pairs of alternative candidates with distinct probability and loss,
higher confidence agrees with lower loss in 34/81 (42.0%) baseline pairs,
37/81 (45.7%) scheduled pairs, and 7/21 (33.3%) fixed-.5 pairs. Correlated pairs
from three candidates at a node are not independent trials; no significance
claim is made. Among nodes whose alternatives have differing losses, the
highest-probability alternative attains the best loss in 8/31, 10/30, and 3/9
nodes respectively (ties for best count as hits).

Thus merely sampling current categorical probabilities is not equivalent to
selecting beneficial local changes. The test establishes that task-based
counterfactual labels contain measurable information beyond the current
confidence ranking; it does not test alignment of a soft LM-loss gradient.

## Interpretation and next boundary

There is repeatable local hard-symbol utility on these texts, particularly in
the scheduled LN/sigmoid model. This supports investigating counterfactual-cost
labels for a KL-constrained selector target, rather than assuming that a noisy
sequence-level REINFORCE estimate is the only available supervision.

However, candidate selection observes the same future targets used to score it.
These are hindsight local-oracle gains, not evidence that a causal selector can
predict or generalize the better assignment. Improvements from separately tested
nodes are not additive. Other latent assignments are held fixed, so a parameter
update that changes many symbols need not realize the same gain. Baseline's
0/16 improved 8K nodes also cautions against a universal claim.

A subsequent test should construct targets on training text, fit a small
KL-limited selector update, and evaluate on distinct text with ordinary recomputed
argmax assignments. It should retain the original soft training signal and
separate local target quality from parameter-fitting/generalization failure.
No such training has been started.

## Checks and artifacts

All three fixed-label sweeps completed. Repeated original evaluation, a no-op
label override at every node, and a repeat of the best alternative were exact.
Every candidate left losses before the modified token exactly unchanged.
Losses were finite and model RNG counters unchanged. These checks establish
repeatability/causality in this BF16 implementation, not immunity to systematic
quantization error or statistical generalization. No CUDA kernel modifications
or sanitizer run.

Tools: `probe_hard_symbol_candidates.py`, `summarize_hard_symbol_candidates.py`.
Ignored `build/hard_symbol_candidates` links to data-disk diagnostic storage,
so new artifacts do not consume the nearly full system disk. It contains
per-model results, `summary.json`, `candidates.csv`, `candidate_effects.png`, and
`status.json`. The initial exploratory run allowed downstream argmax to adapt;
it was stopped and preserved under `*_adaptive_initial` and is excluded from
all tables above. Its interrupted log text may remain in `sweep.log`; the three
fixed-label `complete` markers and status file identify the completed protocol.
