# Early training dynamics: rear DISM vs middle DISM

2026-10-24. User selected LayerNorm + calibrated sigmoid low-rank gate as the
tentative next-run candidate. At the time of this analysis, no new full pretraining run had been launched.
The subsequent run is documented in LAYERNORM_SIGMOID_PRETRAIN.md.
The initialization/SQ/norm/gate screen sources and reports were committed as
`43e284d` before this analysis. This report concerns the earlier production
RMSNorm/original-gate models, not the proposed next model.

## Sources and interpretation boundaries

Read-only local W&B event files, no remote API: original old layout run
`ew5yujgg`, middle-layout planner run `ldu8rz57`, and old-layout planner run
`xfi42yvt`. Extractor: `tools/analyze_early_layout_history.py`; ignored scalar
records: `build/early_layout_history.json`. The three full runs use the same
document-balanced recipe and131,072 packed tokens/update. The first old run
does not use graph/planner; the other two do. Code versions also differ, so
the third run is a useful backend control, not a perfect factorial experiment.

The logged gradient norm is global pre-clipping L2. `update_rms_max/layerXX`
is the maximum RMS applied update among that layer's parameter tensors, **not
a gradient norm or a pooled layer update norm**. `hidden_rms_max/layerXX` is
the maximum token RMS of the residual entering final block postnorm, not the
mean activation RMS. Q/K usage metrics are measured by a separate all-hard
eager probe, not on the training soft forward. Entropies are normalized.

Early model/optimizer checkpoints were removed by retention; only late22,000
and22,888 checkpoints remain. Historical per-parameter gradients at500/1000
cannot be reconstructed from these scalar records. No historical checkpoint
was mutated or resumed to approximate them.

## Divergence appears during learning, not at step zero

| Step | Old soft | Middle soft | Old+planner soft | Middle minus old+planner |
| --- | ---: | ---: | ---: | ---: |
| 0 | 10.526791 | 10.526788 | 10.526791 | -.000003 |
| 500 | 4.672307 | 4.698193 | 4.664771 | +.033421 |
| 1000 | 4.016951 | 4.160902 | 4.009223 | +.151679 |
| 1500 | 3.802748 | 3.878477 | 3.792413 | +.086064 |
| 2000 | 3.678488 | 3.712888 | 3.676066 | +.036822 |
| 4000 | 3.445220 | 3.477176 | 3.442373 | +.034803 |

Old and old+planner early soft curves track much more closely than either
tracks the middle layout. This argues against planner/backend alone being
the main explanation. At step1000 the middle layout has **better hard loss**:
4.224970 versus4.370192 for old+planner. Its hard-minus-soft gap is .064067
versus .360969. The corresponding gaps at500 are .039669 versus .147501;
at1500, .075943 versus .391040. A smaller hard/soft gap is therefore not
automatically good: here it accompanies a substantial soft-loss disadvantage.

Hard sampling probability is only about2.08% at500 and4.15% at1000. The largest
early soft deficit develops while the training path remains overwhelmingly
soft, although this does not eliminate effects from occasional hard steps.

## Global gradients and updates

| Window | Old grad median | Middle grad median | Old+planner grad median |
| --- | ---: | ---: | ---: |
| 0–99 | 1.85687 | 1.86056 | 1.85687 |
| 100–499 | .66469 | .56396 | .66295 |
| 500–999 | .49439 | .43056 | .49645 |
| 1000–1999 | .41459 | .38792 | .41025 |

All three clip62% of the first100 updates. In updates100–499 the clipped
fractions are5%,2.25%,5.25%; none clips in500–1999. The middle model has smaller
global gradients in the divergence window, not a global gradient explosion.
This scalar measurement does not establish gradient direction or noise variance.

Nor do the applied updates show uniform starvation of middle DISM layers.
In1000–1999, the average over six DISM layers of each layer's median max-update
RMS is2.2708e-4 for middle versus1.9552e-4 for old+planner. For the last DISM
layer specifically, middle L8 is2.7586e-4 versus old+planner L11 at1.9683e-4,
about40% larger. The maximum may come from different parameter tensors, so
this is not proof of stronger codebook learning.

Middle hidden residual maxima grow near the last DISM and GDN tail, but show
no obvious catastrophic scale blow-up in these windows. In1000–1999, median
max-token RMS at middle L8/L9/L10/L11 is2.384/1.870/1.673/2.523. This should
not be confused with an average hidden RMS or an amplification Jacobian.

## Selector specialization differs, particularly near the last DISM

At step1000, the last DISM query normalized entropy is .707 in middle L8,
versus .577 in old+planner L11 (.561 in the original old run). Query mean top
probability is .178 versus .276 (.290 old); key entropy is .587 versus .465
(.505 old). Initial entropies were all about .945–.948.

This supports a different learned selector distribution, rather than an
initialization that already starts collapsed. The contrast is not uniform
across every DISM layer; compare ordinal/depth carefully. A sharper selector
is not intrinsically better, and these logged probes run all-hard contexts.
The relevant observation is that the middle tail DISM does not specialize in
the same way as the final rear-DISM layers during the soft-loss divergence.

## Fresh initial-gradient control with identical block weights

`tools/probe_early_layout_gradients.py` measures a fresh seed42 full-size model
on the first fixed8x2048 training batch using eager all-soft backward. It tests:

1. Native old layout.
2. Native middle layout (same seed but construction order can change draws).
3. The **same old-layout initialized blocks**, reordered as
   `[0,1,2,6,7,8,9,10,11,3,4,5]`, retaining all weights and first-layer GDN.

No optimizer update occurs. The third case isolates block placement from
different initial random draws; it is not a historical checkpoint probe.

Global gradient norms are2.27990/2.27677/2.27509 respectively. First mixer
output-bias gradient norms are2.08677/2.08325/2.08067. The initial first-layer
scale/bias phenomenon is shared by both layouts and does not explain their
different later behavior by itself.

For identical DISM blocks moved into the middle, mean ratios of gradient L2
norms relative to their old positions are:

| Quantity | Middle-reordered / old |
| --- | ---: |
| DISM output | 1.0236 |
| q_vocab | 1.0283 |
| k_vocab | 1.0232 |
| sq projection weight | 1.0231 |
| sk projection weight | 1.0312 |

Thus this controlled step-zero probe finds no initial gradient attenuation
caused by the three GDN tail layers. A small increase in gradient magnitude
does not prove equally useful gradient direction. Learned tail Jacobians and
gradient noise after hundreds of updates remain unmeasured here.

## Working interpretation and next-run implications

Evidence favors a **learned allocation of work**: middle DISM plus a GDN tail
learns a different soft-selector/readout solution, with less final soft-loss
benefit and a smaller hard/soft discrepancy. This fits the prior final-checkpoint
ablation showing smaller DISM LM benefit, but remains a hypothesis: neither
selector entropy nor global gradient norms prove that the GDN tail discards
DISM information or identify a unique causal mechanism.

Do not treat this as “initial gradients vanish”, “larger global gradient noise”,
or “planner alone is broken”. Also do not equate the lower middle hard loss
at1000 with superior language modeling overall; the user prioritizes soft loss.

LayerNorm+sigmoid remains the tentative next candidate, not an established
fix for layout regression: all recent300-update screens used the **old layout**
and1/8 production global batch (about4.9M tokens), while the historical main
divergence is around500–1500 production updates (65.5–196.6M tokens).
For a middle-layout follow-up, the informative measurements are fixed-batch
per-DISM codebook/readout gradients and update-to-weight ratios, branch RMS
and gate statistics, and DISM ablation loss before/after the GDN tail at these
early milestones. Preserve checkpoint snapshots if these causal probes are
needed later. These measurements can distinguish learned underuse from poor
gradient directions; the current historical records cannot.

Ignored artifacts: `build/early_layout_gradients/{old,middle_native,middle_reordered}.json`,
`build/early_layout_history.json`, and `build/early_layout_dynamics.png`.
