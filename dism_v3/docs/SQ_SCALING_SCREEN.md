# Post-SiLU SQ scaling screen

The user requested SQ scaling only. Production output normalization and gate
order and math remain unchanged: FLA RMSNorm(DISM + GDN), then multiply by the
existing SiLU low-rank output gate, then output projection. No sigmoid-gate or
embedding-scale intervention is included.

`dism_v3/tools/probe_sq_scaling.py` temporarily wraps `_activate_readout` in fresh
models, multiplying only the query readout after SiLU by 1, .5, or 1/sqrt(32).
SK, vocabulary selector, tau, GDN, parameters and optimizer groups are unchanged.
R=32 is explicitly validated. The scale is a constant captured in CUDA Graph.
In exact arithmetic this rescales DISM's numerator/output, not its matching
distribution or denominator; BF16 rounding makes exact bitwise equivalence to
output-side scaling unnecessary and generally unavailable.

## Protocol

- Same 123M `ggggggdddddd` architecture and frozen local train/validation batches
  as `INITIALIZATION_TRAINING_SCREEN.md`.
- 300 updates, 8 x 2048 tokens per update, 1/8 production global batch;
  LR 8e-4, original 500-update warmup and 22,888-update hard annealing schedule,
  clip 1, MXFP8 FFN/optimizer, BF16 CE, CUDA Graph on, planner off.
- Primary criterion: soft document-balanced validation loss, measured every
  50 updates on fixed 131,072 packed validation tokens. Hard loss is secondary.
- Each scale has reference seed 42 and a paired small embedding perturbation
  trajectory, retaining all other random streams. These are not independent
  seed replicates or a statistical estimate of final training variance.
- All six trajectories run in isolated processes. Scale 1 is rerun here using
  the production fused norm: the prior gate screen used an unfused norm across
  its arms, so its baseline is not substituted into this comparison.
- No production defaults or existing checkpoints are modified. This screen
  performs fresh diagnostic training and does not save resumable checkpoints.

Ignored artifacts: `dism_v3/build/sq_scaling_screen/` and
`dism_v3/build/sq_scaling_screen.log`. The protocol records source and frozen-batch
hashes. Per-trajectory records include every training loss/gradient norm, first
bias gradient share, validation checkpoints and final RNG counter.

## Results

All six trajectories completed 300 updates with finite gradients and RNG counter
300. Final losses are the mean of each reference/perturbed pair:

| SQ scale | Soft loss | Soft difference vs 1 | Hard loss (secondary) | Absolute soft pair difference |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5.723166 | 0 | 5.757574 | .002430 |
| .5 | 5.722434 | -.000732 | 5.751198 | .005756 |
| 1/sqrt(32) | 5.722200 | -.000966 | 5.735810 | .001018 |

Half scaling improves one trajectory and worsens the other at update 300:
reference 5.719556 vs baseline 5.724381; perturbed 5.725312 vs 5.721952.
Inverse-root scaling improves both slightly: reference 5.722709, perturbed
5.721692. Its pair-mean soft differences at updates 50/100/150/200/250/300 are
+.016355/+.000390/-.003422/-.003379/-.004349/-.000966. Thus its initial slowdown
turns into a small advantage, which shrinks by update 300.

The primary soft-loss result is weakly positive for inverse-root scaling, not
evidence of a necessary normalization or a robust full-training improvement.
Paired trajectories are correlated and cannot yield independent-seed confidence
intervals. Reducing hard loss or the hard-soft gap does not override the user's
soft-loss screening criterion. The previous sigmoid-gate screen had a larger
observed final soft improvement (~.0044), though the screens used different
norm implementations and did not test gate/scaling interaction.

Results, source and batch hashes are retained in the ignored output directory;
`comparison.json` and `soft_loss.png` summarize the curves. Production SQ scaling
remains 1 and output norm/gate order remains untouched.
