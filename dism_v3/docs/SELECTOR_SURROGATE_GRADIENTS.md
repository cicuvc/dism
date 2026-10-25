# Selective STE: same-input q/k and vocabulary gradient disagreement

2026-10-25. Follow-up to SOFT_HARD_HEAD_DIAGNOSTIC.md, using the same three
step-22888 checkpoints, first four PG19 test prefixes, BOS, 2K/8K lengths and
BF16 CE. The running RMSNorm/sigmoid experiment remains unchanged.

## Controlled intervention

For each document/context, run an all-soft reference and obtain gS=dL/dO at
all six raw DISM outputs. Then independently switch one whole DISM layer's heads
to hard, leaving all other layers soft, and obtain gH at that layer. Assert exact
equality of q,k,sq,sk,v,eq,ek,tau against the baseline at the switched layer.
The GDN branch, normalization, gate and downstream model run normally.

Reconstruct a local soft DISM computation at those same operands and compare:

- A = J_soft(z)^T gS, the original soft selector gradient;
- B = J_soft(z)^T gH, the proposed selective-STE surrogate gradient.

z consists of q, k, q_vocab and k_vocab. Both VJPs use the same soft graph and
its own saved forward buffers. B is **not** the true derivative through hard
argmax. q/k measurements are at DISM operator inputs after projection/convolution,
not projection-weight gradients; vocabulary measurements are parameter gradients.
Gradients are multiplied by document length to remove mean-loss normalization.
No updates, RNG-counter changes, hard-probability changes or checkpoint writes.

Tools: `probe_selector_surrogate_gradients.py`, `summarize_selector_surrogate.py`.
Artifacts: ignored `build/selector_surrogate/{baseline,ln_scheduled,ln_fixed05}/results.json`,
`summary.json`, `heads.csv`, `selector_cosine_8k.png`,
`selector_direction_scale_8k.png`. Generic `hard_*` column names inherited from
the metrics helper refer to B, not a true hard derivative.

## Main results

Each median covers 288 document/layer/head cells. Pooled cosine concatenates
all cells' gradient vectors, weighting directions by their actual magnitudes.
It does not sum document gradients into a minibatch or apply Adam preconditioning.
Norm ratio is ||B||/||A|| for that same concatenation. Parameter categories are
kept separate because their dimensions and meanings differ.

| 8K model | Gradient | Head cosine median | Pooled cosine | Pooled norm ratio |
| --- | --- | ---: | ---: | ---: |
| Baseline C | q | .905 | .039 | 45.58 |
| Baseline C | k | .966 | .152 | 14.95 |
| Baseline C | q_vocab | .930 | .116 | 93.17 |
| Baseline C | k_vocab | .949 | .064 | 32.58 |
| LN/sigmoid scheduled | q | .936 | .136 | 13.50 |
| LN/sigmoid scheduled | k | .960 | .350 | 3.64 |
| LN/sigmoid scheduled | q_vocab | .958 | .154 | 13.52 |
| LN/sigmoid scheduled | k_vocab | .955 | .419 | 6.20 |
| LN/sigmoid fixed .5 | q | .999 | .968 | 1.14 |
| LN/sigmoid fixed .5 | k | .999 | .969 | 1.00 |
| LN/sigmoid fixed .5 | q_vocab | .999 | .876 | .77 |
| LN/sigmoid fixed .5 | k_vocab | .999 | .831 | .51 |

The effect also exists at the training context length:

| 2K model | q cosine / ratio | k cosine / ratio | q_vocab cosine / ratio | k_vocab cosine / ratio |
| --- | ---: | ---: | ---: | ---: |
| Baseline C | .057 / 22.00 | .228 / 8.88 | .102 / 32.69 | .196 / 15.43 |
| LN/sigmoid scheduled | .339 / 3.53 | .443 / 2.46 | .434 / 2.92 | .629 / 1.98 |
| LN/sigmoid fixed .5 | .949 / 1.24 | .961 / 1.02 | .787 / .83 | .334 / .57 |

Example (zero-based indices): baseline, PG19 document 2, 8K, layer 7 head 1.
Output-gradient cosine is .9619, but q-gradient cosine is .0161 and norm ratio
1369.7. This is direct evidence that aligned output cotangents do not imply
aligned or similarly scaled selector gradients under the soft Jacobian.

At 8K, vocabulary head cosines are negative in 14/288 (q_vocab) and 18/288
(k_vocab) baseline cells, and 8/288 and 15/288 scheduled LN/sigmoid cells.
Sign reversal is not the only or principal aggregate issue: large, nearly
orthogonal surrogate vectors can dominate while the median head remains aligned.
In scheduled LN/sigmoid, cells with cosine < .5 account for 22.5%, 27.3%, 28.2%,
and 44.2% of reference squared-gradient norm in q, k, q_vocab, k_vocab respectively.

Fixed-.5 has much closer directions for most cells, but the previous diagnostic
found strong output suppression in many heads. It is not a positive control for
learning useful matching through STE. Even here, 2K k_vocab has median cosine
.999 but pooled cosine .334, showing why medians alone are insufficient.

## Implication for an experiment

The endpoint evidence argues against an unscaled alpha=1 selector surrogate as
the first experiment. Retain actual soft examples and the original hard schedule;
start from a trained scheduled checkpoint. First measure that checkpoint (step
14000 is available), because final-checkpoint results cannot establish early or
mid-training compatibility. Start with a small ramped coefficient and consider
per-layer/head surrogate norm limits using reference RMS statistics from genuine
soft samples. A global coefficient alone does not eliminate headwise outliers;
global gradient clipping alone can let them suppress unrelated useful updates.
No adaptive clipping or STE training is implemented by this diagnostic.

This still does not prove that a differing surrogate is harmful: gH represents
a different objective state, so exact agreement with A is not itself the goal.
Training and held-out soft/hard loss must ultimately decide. Four document
prefixes, one checkpoint per recipe, and whole-layer interventions cannot establish
training-wide safety or single-head switch effects.

## Validation and limitations

All three sweeps completed: 144 layer interventions and 288 local selector VJPs,
plus 24 all-soft reference forwards. Each layer input comparison was exact;
loss and selector gradients were finite; RNG counters were unchanged.
A separate audit reconstructed a fresh soft graph and applied gH as its first
backward, instead of its second backward. Across six layers of baseline document
2 at 8K, maximum relative gradient difference was 1.525e-5, far below the observed
amplification. Results are in `audit/results.json`.

The initial probe exposed accumulated MXFP8 workspaces because partial VJPs
skip upstream FFN backward. It hit the diagnostic process's allocator cap;
training was unaffected. Final probes explicitly release those caches between
independent graphs; peak allocated memory was about 8.13 GB. CPU packed-layout
metadata is retained on CPU for local replay. The successful sweeps replaced
partial diagnostic output, not evaluation or training artifacts.

Production relaxed BF16 backward is used. There was no CUDA kernel change,
finite-difference/reference backward validation or sanitizer run. The fresh-graph
check establishes repeatability, not correctness of the underlying derivative.
