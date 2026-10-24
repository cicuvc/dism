# Initialization short training screen

This is a fresh-model diagnostic following `INITIALIZATION_DIAGNOSIS.md`, not
a replacement for a full-budget architecture comparison. Production defaults,
existing checkpoints and completed training runs remain unchanged.

The user selected early **soft validation loss** as the primary screening
criterion, based on prior experience that improvement often predicts better
later performance. This is a heuristic, not a guarantee; full training remains
necessary. Hard loss and gradient concentration are secondary diagnostics.

## Protocol

- Full 123M original `ggggggdddddd` architecture, postnorm, MXFP8 FFN and
  optimizer, BF16 CE; fresh seed 42 models, hard RNG seed 1729.
- 300 updates, 8 x 2048 packed tokens per update (one microbatch), **one eighth
  of the production global batch**. Original LR 8e-4, 500-update warmup, AdamW
  betas (.9,.95), epsilon 1e-8, weight decay .01 and global clipping at 1.
- Original 22,888-update hard annealing schedule; hard probability only reaches
  about 1.24% by this screen's end. This mostly screens early soft training.
- Cache the first 300 local SlimPajama training batches, shuffle seed 1337;
  all arms consume exactly the same cached inputs/targets/document boundaries.
- Fixed eight validation batches (131,072 packed tokens), separately score
  soft and hard document-balanced loss at 0/50/100/150/200/250/300 updates.
- Same CUDA Graph runner for all arms, planner off. All arms use the same
  unfused float RMSNorm/output-gate expression followed by BF16 cast, so the
  baseline differs in rounding from production's fused FLA norm.
- Each arm has a reference run and one deliberate embedding perturbation run:
  add Gaussian std 2e-5 to BF16 embedding weights and round back to BF16,
  approximately 0.08% actual relative RMS. All other initial tensors and hard
  RNG streams match. This is **not** an independent initialization seed sweep
  or an estimate of training variance.

## Arms

1. `baseline`: existing SiLU(W2(W1(x))) gate and original input scale.
2. `input_scale`: multiply embedding **activations** by 50 before the first
   layer; keep embedding parameters, optimizer groups and postnorm intact.
3. `sigmoid`: replace the effective gate by c W2(sigmoid(W1(x))). Calibrate
   each fixed c on unit-RMS first-batch layer inputs to match the RMS of the
   original SiLU(W2(W1(x))) gate. Unit normalization intentionally avoids
   preserving the first layer's small-input suppression. Original weights and
   biases remain trainable, including both low-rank projection biases.
4. `combined`: input scaling plus the calibrated sigmoid gate.

The sigmoid candidate is inspired by the suggested RWKV-7-style low-rank
nonlinearity placement; it does not import RWKV-7's full architecture or init.
Gate calibration is initialization only, not a runtime normalization.

## Execution and artifacts

`dism_v3/tools/probe_initialization_training.py` uses local tokenizer/data only.
Ignored artifacts: `dism_v3/build/initialization_screen/` and
`dism_v3/build/initialization_screen.log`. Per-arm JSON contains every update's
loss, gradient norm and first bias squared-gradient share, validation losses,
gate scales, actual perturbation, elapsed time, final RNG counter and completion.
`protocol.json` records source hash and model spec; `batches.pt` freezes inputs.

The first launcher retained multiple models in one process. After three complete
trajectories, CUDA storage accumulated and the fourth ran out of memory. The
launcher was changed to isolate each remaining trajectory in a subprocess;
completed trajectories were preserved. No math, schedule or tolerance changed
in that launcher fix. An incomplete trajectory restarts fresh on the same data;
these short probes do not write resumable training checkpoints.

## Results: all eight trajectories complete

All runs have 300 updates and RNG counter 300, finite gradients and complete
validation records. Table values average each reference/perturbed pair; these
are correlated trajectories, not two independent random-seed replicates.

| Arm | Soft validation at 300 | Difference vs baseline | Absolute pair soft difference | Initial bias squared-gradient share | Fraction of clipped updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 5.723118 | 0 | .002527 | 83.27% | 42.17% |
| input_scale | 5.746528 | +.023410 | .000160 | .205% | 6.67% |
| sigmoid | **5.718718** | **-.004401** | .000852 | 84.15% | 40.67% |
| combined | 5.736259 | +.013140 | .001238 | .208% | 10.33% |

Sigmoid soft validation improves at all measured updates from 50 onward in both
paired trajectories. Pair means: baseline/sigmoid are 9.367934/8.754758 at 50,
7.003514/6.985042 at 100, 6.487590/6.471905 at 150,
6.124435/6.119614 at 200, 5.885619/5.881311 at 250,
and 5.723118/5.718718 at 300. The strong early advantage shrinks substantially.
Calibrated sigmoid multipliers are approximately .522–.568 across layers.

Input scaling removes the initial bias-gradient concentration and greatly reduces
clipping, but worsens the primary soft-loss metric. Gradient balance alone is
therefore not a sufficient selection criterion. The sigmoid candidate is the
only positive soft-loss candidate in this screen; it should advance before
input scaling or the combination. Full-budget, production-batch training and
multiple independent seeds are still needed to establish final quality and
variance. These paired scalar losses do not prove that weight/hidden trajectories
remain close, and do not identify the cause of earlier full-run differences.

Generated `comparison.json` and `soft_loss.png` are beside the per-run results.
No production architecture defaults were changed. The experiment runner and
this report remain reviewable separately from the committed initial diagnosis.
