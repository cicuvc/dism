# Initialization scale diagnosis, 2026-10-24

Scope: fresh 123,473,088-parameter `ggggggdddddd` models, seeds 42/43/44,
original postnorm recipe, MXFP8 FFN, BF16 CE. Fixed two 2048-token packed rows
from the old run's final saved prefetched batch; these are real data, not the
original training step-zero batch. No checkpoint or running training is modified.
Script: `dism_v3/tools/diagnose_initialization.py`. Results and logs are ignored
under `dism_v3/build/initialization_diagnosis.{json,log}`.

Run with conda blkw Python; the script selects this worktree's Python packages.
Run the script normally for six cases, then with `--embedding-unit` to append
the seed-42 embedding-scale intervention. No optimizer updates are performed.

## Main observation: first-layer scale and bias gradient

Seed 42 baseline:

- Embedding RMS 0.02007; later layer inputs approximately 1.
- First raw GDN output RMS 0.00008549, far below sqrt(norm epsilon)=0.003162.
  Thus the internal RMSNorm is epsilon-dominated, rather than producing unit RMS.
- First effective SiLU gate RMS 0.001099; subsequent gates approximately 0.054.
- First mixer output RMS 0.00001628; later mixer outputs approximately 0.018–0.026.
- Full gradient norm 2.5406. First attention output-projection bias alone has
  gradient norm 2.1009, approximately 68.4% of total squared gradient norm.
  Seeds 43/44 reproduce the pattern: bias norms 2.0492/1.9338, total 2.5009/2.4095.

The bias bypasses the tiny gated branch and enters the first postnorm directly.
The small residual-stream scale makes its sensitivity large. A temporary paired
intervention multiplying embedding weights by 50 raises first input RMS to 1.0036,
effective gate RMS to 0.05514 and mixer output RMS to 0.02092; total gradient norm
falls to about 1.0785. This supports a scale mechanism, not a proposed final recipe:
embedding weight rescaling changes its parameterization and optimizer behavior.
Embedding normalization or explicit activation scaling needs separate evaluation.
At matched nominal relative perturbation, this intervention still gives 2.160%
soft and 2.783% hard final hidden differences (actual input perturbation 0.0820%).
Fixing the initial gradient imbalance therefore does not fix the measured forward
perturbation propagation; these are distinct findings.

## Other hypotheses

- Baseline hybrid DISM RMS is typically roughly twice GDN RMS. Global branch
  cosines are small positive values (examples 0.025–0.037), not broad cancellation.
  GDN is reconstructed by subtracting DISM from the BF16 sum, so its statistics
  are approximate; this does not rule out individual token/head cancellation.
- Sampled selector entropy is 5.83–5.93 nats, versus log(512)=6.238; mean top-one
  probability 0.021–0.034 and mean top-two score margin 0.275–0.332. No initial
  extreme softmax concentration is apparent in these averages. Statistics use
  detached float score recomputation, not the exact BF16 interpolation scores.
- Narrowing log_sel_tau from [-4,4] to [-0.5,0.5] changes the nonlinear tau mean
  as well as its spread. It barely changes total gradient norm or the measured
  perturbation response; it is not the leading initial sensitivity explanation.

## Gate candidates and perturbation limits

The existing gate is SiLU(W2(W1(x))). Two temporary variants retain identical
weights: SiLU(W2(sigmoid(W1(x)))) and W2(sigmoid(W1(x))). The latter uses an
unfused float RMSNorm expression before BF16 cast and therefore also changes
rounding. Neither variant claims to reproduce RWKV-7's whole initialization.

| Seed 42 variant | First effective gate RMS | First mixer RMS | Full gradient norm |
| --- | ---: | ---: | ---: |
| Baseline | .00110 | .0000163 | 2.5406 |
| Sigmoid inside, SiLU outside | .04982 | .0007358 | 2.5611 |
| Sigmoid inside, raw output gate | .09867 | .0014730 | 2.6077 |

Sigmoid supplies a nonzero bottleneck baseline and removes much of the first
gate's input-scale suppression, but does not remove the first postnorm bias
gradient concentration. Raw output gating also roughly doubles later effective
gate strength; this comparison is not scale-matched.

Fixed embedding perturbations, rounded back to BF16, have actual baseline RMS
relative magnitude about 0.0815%. Last-layer relative hidden differences are
2.146% in soft mode and 2.749% in hard mode. Same-input no-grad repeats have zero
hidden difference in these probes. Across the three seeds the soft responses
are 2.139–2.148%, suggesting a systematic propagation/precision effect rather
than one exceptional initialization draw. These finite differences include
BF16/MXFP8 rounding and hard routing; they are not infinitesimal Jacobian norms.
Raw sigmoid gating gives 2.233% soft and 4.199% hard, so simply swapping the gate
does not demonstrate better perturbation robustness at the existing weight scale.

No short training or multi-seed convergence test has been performed here. These
observations do not establish the cause of final downstream differences. Prior
two original-layout runs have identical step-zero hard/soft validation losses,
near-identical first-50-step losses, and begin visibly diverging around steps
100–200; graph/backend differences remain confounders. Production defaults and
kernel math are unchanged. No sanitizer claim is made for these Python probes.
