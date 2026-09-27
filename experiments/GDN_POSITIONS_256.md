# GDN vs GDN/DISM per-position NLL

Completed 2026-09-11. Reproduce with conda blkw:
`python -m experiments.compare_gdn_positions` (choose a new output directory on rerun).

Both checkpoints completed 3000 updates. Mixed: 9 GDN + 3 DISM, FFN1428,
72,188,142 parameters. Pure: 12 GDN, FFN1392, 72,184,080 parameters.
Width384, context2048, identical checked training settings, no SWA.
Frozen `swa-eval-bundle-20260910.pt`, first256 sequences/2048 positions,
524,288 tokens total; same inputs/targets for both models. Mixed hard_prob1.
BF16 autocast with FP32 softcap30 and PyTorch per-token CE; TF32 disabled.
Saved per-token losses and complete provenance/configs in report.json.

Output directory under dism-lm-runs:
`activation-study-3k-20260910/gdn-position-256-20260911/`.

| Positions (zero-based) | Mixed NLL | Pure GDN NLL | Mixed - pure |
|---|---:|---:|---:|
| 0–127 | 3.895661 | 3.901437 | -0.005776 |
| 128–255 | 3.704144 | 3.722499 | -0.018355 |
| 256–511 | 3.676127 | 3.692742 | -0.016615 |
| 512–1023 | 3.669226 | 3.690766 | -0.021540 |
| 1024–2047 | 3.673295 | 3.687956 | -0.014662 |
| All | 3.688457 | 3.704759 | -0.016301 |

Overall paired-sequence SE0.001258; approximate95% interval
[-0.018767,-0.013835]. First128 interval includes zero. Other broad segments
favor mixed. No monotonic growth of benefit with position: last1024 vs first512
delta difference -0.000322, SE0.003130. Plot uses non-overlapping64-token bins
and pointwise paired-sequence intervals.

This frozen sample is not the training logger's100-batch validation stream;
absolute NLL differs. Original final validation was3.774441 mixed vs3.792573 pure,
consistent in direction. Sampling intervals do not cover training-seed variation;
packed sequences can be correlated. One seed, slightly different FFN widths.
DISM improves this matched-budget run but these curves alone do not establish
a specifically long-range retrieval mechanism.
