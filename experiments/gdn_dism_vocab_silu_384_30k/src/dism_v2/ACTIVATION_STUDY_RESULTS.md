# First three 3000-step activation arms: results

All three finished3000 updates /393,216,000 training tokens with the same49,678,876
parameters, raw initialization seed777, token order, optimizer and0→1 hard schedule.
No new optimizer updates/checkpoint surgery performed for this analysis. The new
preconv_silu training is a separate ongoing arm and is not included.

## Original validation:100 effective batches (13,107,200 tokens)

| Arm | Step1000 (p=.3331) | Step2000 (p=.6666) | Step3000 (p=1) | Final delta |
|---|---:|---:|---:|---:|
| baseline | 4.424839 | 4.050162 | 3.943258 | 0 |
| vocab_silu | 4.412390 | 4.044229 | 3.939904 | -0.003354 |
| no_qk_silu | 4.435487 | 4.058697 | 3.952286 | +0.009027 |

Codeword SiLU has a small consistent advantage; its gap shrinks from0.01245 at1000
to0.00335 at3000. Removing Q/K SiLU is behind at all three checkpoints, not only at
the final hard endpoint. Logged mean training losses over updates2510–3000 are
3.963661 /3.960170 /3.972339 respectively. No obvious late loss/gradient blow-up.
These are means of logged effective-batch losses, not all training steps.

## Additional paired GPU validation

256 identical sequences x2048, drawn as the first2048 tokens of each sequence in
the previously frozen8192-token validation bundle. This is a different sample from
the100-batch training validation above, so absolute NLLs must not be compared across
the two tables. BF16 model/head and accurate FP32 torch softcap30/CE; microbatch2.
All losses finite. Checkpoint step/parameter counts and common training settings
were verified. No training was stopped; evaluations coexist with the new arm.

| Arm | p=1 NLL | Paired delta | Approx95% interval | p=.5 NLL | p=0 NLL |
|---|---:|---:|---:|---:|---:|
| baseline | 3.853690 | 0 | — | 3.851578 | 3.849620 |
| vocab_silu | 3.850578 | -0.003112 | [-0.005283,-0.000941] | 3.848993 | 3.847587 |
| no_qk_silu | 3.859816 | +0.006126 | [0.003917,0.008335] | 3.858741 | 3.857670 |

Intervals use mean±1.96*SE of paired per-sequence differences. They do not account
for training-seed variation or all correlations in packed documents; this is only
one training seed. Per-position64-token curves have pointwise, not simultaneous,
bands. No_qk_silu is worse in all eight256-position bins; there is no convincing
evidence of a deficit confined to long-context positions. Most individual64-position
intervals overlap zero. No NIAH or branch-contribution ablation performed here.

Pure-soft minus pure-hard NLL: baseline-0.004070, vocab_silu-0.002991,
no_qk_silu-0.002147. Thus softening helps all slightly, but no_qk_silu has the
smallest hard penalty and remains worse even when soft. This argues against a
larger final soft→hard penalty being the principal explanation of its gap.
Probability changes also change RNG consumption/global directions; RNG is aligned
across models within each probability, not across different probabilities.
This is a final-checkpoint inference intervention, not a new training schedule.

## Final hard vocabulary load

Each arm evaluated on the same256 packed sequences,524288 labels per head/side;
input SHA d6e9d24f7cbb224cfe6f37fac7ea5f183492c49e88dc872f585e8864fa3a6dc4.
Below are means across60 heads. Effective vocabulary=exp(entropy), out of512.

| Arm | Effective Q/K | Unused Q/K IDs | Q/K marginal overlap | Q mass on unused K IDs |
|---|---:|---:|---:|---:|
| baseline | 15.54 /24.40 | 160.53 /157.63 | .33277 | .36194 |
| vocab_silu | 13.01 /19.35 | 242.57 /241.58 | .30423 | .36788 |
| no_qk_silu | 15.75 /33.08 | 144.95 /88.23 | .35623 | .38345 |

The best-loss arm actually uses fewer effective codewords; the no-activation arm
uses more K codewords yet has worse NLL. Thus this study does not support the simple
causal chain “one-sided SiLU causes codeword collapse, which causes the loss gap.”
It also does not establish that concentration is beneficial: the three interventions
change feature geometry/optimization as well as usage, and the hybrid can compensate
through SWA/FFN. Marginal overlap is not causal per-query match coverage. Unused means
unused in this sample; counts cannot be directly compared with the earlier30k model's
much larger6400-sequence histogram as if the sampling budgets matched.

Takeaway: retain SiLU as baseline; codeword SiLU is a mildly positive candidate,
not a decisive win. Removing Q/K SiLU is not supported by this single-seed3000-step
study. Wait for preconv_silu and parameter-matched attention controls before making
a larger architecture decision. Better loss here is not proof of better long-range
retrieval. No mechanism-level causal conclusion is claimed.

Artifacts at `dism-lm-runs/activation-study-3k-20260910/analysis-final/`:
report.json,paired_losses.pt,comparison.png. The plot's training panel uses an11-point
trailing mean of every10-step logs; original console/metrics losses are unsmoothed.
Reproduce with `python -m experiments.analyze_activation_3k --root STUDY --bundle
BUNDLE --output NEW_OUTPUT`. Source is read-only with respect to checkpoints.
