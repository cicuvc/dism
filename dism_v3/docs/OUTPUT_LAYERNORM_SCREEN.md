# Mixer output LayerNorm screen

User-requested experiment: replace the mixer output RMSNorm with LayerNorm.
All twelve layers change their **per-head 64-dimensional** normalization,
including both pure GDN layers and GDN+DISM layers. Residual-stream postnorms
and final LM norm remain unchanged. The output expression is
`LayerNorm(DISM + GDN) * SiLU(low_rank_gate(x))`, then output projection.
For pure GDN the norm input is its single branch output.

FLA `FusedLayerNormGated` replaces `FusedRMSNormGated`, with the same epsilon
1e-5, activation, dtype and original weight Parameter. No bias is added: the
intervention isolates subtracting the per-head mean, with identical trainable
parameter identities, counts and optimizer grouping. SQ scaling remains 1;
neither sigmoid low-rank gating nor embedding activation scaling is included.

## Protocol

`dism_v3/tools/probe_output_layernorm.py` builds isolated fresh models and reuses
the short-screen training loop. Two LayerNorm trajectories run for 300 updates:
seed 42 reference and a small deliberate BF16 embedding perturbation. The two
production fused-RMSNorm baselines are reused from `SQ_SCALING_SCREEN.md`; the
script verifies identical batch-cache hash, training-loop source hash and model
spec before reuse. This is not an independent initialization-seed sweep.

Full 123M original layout `ggggggdddddd`, 8x2048 packed tokens/update (1/8 full
production global batch), original LR8e-4 with 500-update warmup, hard annealing
over 22,888 updates, clipping at1, MXFP8 FFN/optimizer and BF16 CE. CUDA Graph
enabled, planner disabled. Soft document-balanced loss on the same fixed
131,072-token packed validation cache is the primary screening metric; hard
loss is secondary. Validation occurs every50 updates. Early hard probability
is at most about1.24%, so this largely tests early soft training.

Artifacts: ignored `dism_v3/build/output_layernorm_screen/` and
`dism_v3/build/output_layernorm_screen.log`. Production model defaults and
existing checkpoints are untouched; the runner does not save resumable models.

## Results

Both LayerNorm trajectories completed 300 updates, finite gradients and RNG
counter 300. Full forward/backward CUDA Graph replay succeeded with the FLA
LayerNorm replacement. Existing dependency/Triton deprecation warnings remain
visible in logs; no kernel changes or new sanitizer coverage are claimed.

| Output norm | Reference soft loss | Perturbed soft loss | Pair mean | Difference vs RMSNorm |
| --- | ---: | ---: | ---: | ---: |
| RMSNorm | 5.724381 | 5.721952 | 5.723166 | 0 |
| LayerNorm | 5.720753 | 5.720908 | **5.720831** | **-.002336** |

LayerNorm improves soft loss in both matched trajectories at update300. Its
pair-mean differences at updates50/100/150/200/250/300 are
+.005243/-.000373/-.001161/-.000895/-.002470/-.002336: initially slightly slower,
then a small advantage. Final hard means are almost identical:
RMSNorm5.757574, LayerNorm5.757474.

The final soft difference between the two LayerNorm trajectories is .000155,
versus .002430 for the reused RMSNorm pair. This is one deliberate perturbation
pair, not independent-seed evidence of lower training variance. Initial first
bias gradient concentration is not removed (about83.5% of squared gradient norm),
and clipping frequency is similar (41.17% vs41.67%).

This is a weak positive early-soft signal, larger than SQ inverse-root scaling
in the preceding screen (~.00097) but smaller than the sigmoid-gate screen's
observed improvement (~.0044). The latter used an unfused norm and is not a
strict joint factorial comparison. Full-budget experiments and interaction
tests are still required. Production defaults remain RMSNorm.

`comparison.json` and `soft_loss.png` summarize the fixed-validation curves
beside the per-trajectory JSON artifacts.
