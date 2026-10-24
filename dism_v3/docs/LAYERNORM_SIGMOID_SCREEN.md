# LayerNorm and sigmoid low-rank gate interaction

User request: test the combination of mixer output LayerNorm and the calibrated
sigmoid low-rank gate. SQ scale stays1, embedding scale stays1; residual-stream
postnorm and final norm remain unchanged.

## Matched 2x2 protocol

All arms normalize per-head64-channel mixer output in FP32, apply the gate in
FP32, then cast to BF16 before the output projection. LayerNorm adds mean
subtraction; its weight and epsilon1e-5 are unchanged and no bias is added.
The original effective gate is SiLU(W2(W1(x))); the alternative is
c W2(sigmoid(W1(x))), with the same fixed per-layer unit-input-RMS calibration
from `INITIALIZATION_TRAINING_SCREEN.md`. This is not a bounded output sigmoid.

FLA's fused gated normalization does not expose a raw identity output-gate
activation. Therefore all four arms use matched explicit norm expressions,
avoiding fused/unfused rounding as a gate interaction confounder:

- RMSNorm + original gate: reuse original initialization screen pair.
- RMSNorm + sigmoid gate: reuse original initialization screen pair.
- LayerNorm + original gate: rerun with explicit expression (the earlier
  `OUTPUT_LAYERNORM_SCREEN.md` tested a fused implementation).
- LayerNorm + sigmoid gate: new paired runs with the same expression.

Runner: `dism_v3/tools/probe_layernorm_sigmoid.py`. The underlying build/training
loop and fixed cached inputs are shared with the initialization screen. Each
new trajectory is isolated in a subprocess. Four new trajectories each perform
300 updates; four existing trajectories provide the RMSNorm controls.

Same123M original `ggggggdddddd` model, seed42, original weights, 8x2048 tokens
per update (1/8 production global batch), LR8e-4 with500-update warmup, original
22,888-update hard annealing, clip1, MXFP8 FFN/optimizer, BF16 CE, CUDA Graph on,
planner off. Fixed131,072 packed validation tokens every50 updates; soft loss
is primary. Reference/perturbed trajectories share initialization except one
small deliberate embedding perturbation; they are not independent seed runs.

Ignored output: `dism_v3/build/layernorm_sigmoid_screen/` and matching `.log`.
No production defaults or existing checkpoints are changed. The models are
fresh short diagnostic training runs without resumable checkpoint output.

## Results: all four new trajectories complete

Every new trajectory completed300 updates with finite gradients and RNG counter
300; full forward/backward graph replay succeeded. Table means average one
reference and one deliberately perturbed trajectory, not independent seeds.

| Output norm / gate | Reference soft | Perturbed soft | Mean soft | Difference vs baseline |
| --- | ---: | ---: | ---: | ---: |
| RMSNorm / original | 5.724382 | 5.721855 | 5.723118 | 0 |
| RMSNorm / sigmoid | 5.718292 | 5.719144 | 5.718718 | -.004401 |
| LayerNorm / original | 5.720166 | 5.718262 | 5.719214 | -.003905 |
| **LayerNorm / sigmoid** | **5.717230** | **5.716473** | **5.716851** | **-.006267** |

The combination is best in both matched trajectories at update300. Compared
with sigmoid alone, its mean soft loss improves by .001867; compared with
LayerNorm alone, by .002363. The gains overlap rather than adding fully:
combination-minus-sigmoid-minus-LayerNorm-plus-baseline is +.002038. This is a
descriptive interaction on this screen, not a significance test.

The combination is not superior at every earlier checkpoint: at50 and100 it
trails sigmoid alone. Its final hard mean is5.753625, versus5.752990 for sigmoid
alone; soft loss remains the user-selected primary metric. This result favors
advancing the combination to a longer/full-budget test, but does not establish
full-training improvement or reduced random-seed variance. Production defaults
remain untouched. No new CUDA kernel or sanitizer claim is involved.

`comparison.json` and `soft_loss.png` in the ignored output directory contain
the full validation curves. The rerun LayerNorm-only explicit implementation
scores5.719214, distinct from the earlier fused implementation's5.720831;
the matched explicit controls above are used for the interaction comparison.
