# Delta gate copy-task comparison

Uses a snapshot of /home/cicuvc/cs/projects/rl/dism.py; original untouched.
Original single-layer copy task: width128,H4,D32,V256,token vocabulary128,
B64,N128, first half duplicated, CE on predicting second half. Report accuracy
including copy startup and separately excluding first8 predictions.
1000 updates, seeds0/1, identical data seed12345, AdamW lr.005/WD.01,
warmup50, cosine to.1peak, clip1. Original no_decay grouping retained.
No v3 continuous readout or vocabulary-SiLU changes: isolate delta parameterization.

All arms add token/head Linear(128,4) gate, weight std.02. Initial study uses
bias2; the neutral-initialization repeat and ABBA use bias0. Delta=-logsigmoid(b).
1. temperature: T=0.1**progress, delta=softplus(-b/T).
2. random: row-wise stochastic choice of hard vs soft delta, independent mask.
3. shared: reuse actual logM row mask, no second draw.
Hard branch is deterministic threshold b>=0 => delta0, else +inf; no Bernoulli
sampling of sigmoid(b), no STE. Soft branch in random/shared is softplus(-b).
Both logM and delta hard-fraction schedules progress0→1 over updates0…999.
Temperature arm has the same logM schedule. All share initial weights per seed.
Exact -inf predecessor boundaries; original score code's finite hard sentinel
remains unchanged. Gate freeze at fully-hard endpoint is intentional.

Checkpoint evaluations250/500/750/1000,10 fixed batches (seed424242), five cells:
hard-QK+hard-delta; hard-QK+method-soft-delta; hard-QK+unit-temperature-soft-delta;
mixed-QK+native-delta; mixed-QK+hard-delta. Report absolute CE/accuracy alongside
gaps; shared native becoming hard at end is not evidence alone of good learning.
Evaluation resets its random streams and restores training RNG state.
Initial trial exited before step1 due legacy implicit CPU randint with CUDA
generator; snapshot now explicitly sets device. Empty results/ preserved;
actual study is results-v2/. User service dism-delta-copy-20260926.
Progress console.log, per-arm metrics/evaluation JSONL, final.pt. No claims of
convergence until runs finish. CUDA runs use conda blkw.

## Neutral initialization and double copy

`continue_study.py` sequentially runs `results-copy-bias0/` and
`results-abba-bias0/`, each with all three methods and both seeds. ABBA draws
independent A/B of length32, concatenates A,B,B,A, and supervises only the final
64 targets (BA). Logits at positions63…126 predict tokens64…127. The two segment
startup regions are reported separately; excluding the first8 of *both* B and A
is different from excluding only the first8 of the entire supervised region.

Reproduce summary/plot and the evaluation-only always-open gate ablation:

```bash
/home/cicuvc/miniconda3/envs/blkw/bin/python -m experiments.delta_copy_study.summarize
/home/cicuvc/miniconda3/envs/blkw/bin/python -m experiments.delta_copy_study.ablate_open
/home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest experiments/delta_copy_study/test_gate.py -q
```

`delta_only_gap` fixes Q/K to hard and subtracts method-soft gate CE from hard
gate CE. Positive means degradation from hardening. Temperature's comparison
uses its final T=0.1; random/shared use T=1, so these gaps are not equal-temperature
comparisons (the extra `unit_soft_gap` provides that comparison). `total_gap`
also changes the legacy training/evaluation score implementation. At the final
step random/shared native gates are already all hard: a tiny total gap alone
does not establish that soft and hard gates agree. Intermediate checkpoints
have a one-update offset between legacy logM probability and gate probability;
both schedules are exactly1 at final evaluation.

The ABBA open-gate ablation changes only inference on the same checkpoint and
same evaluation batches. It measures learned-model reliance on reset, not the
benefit relative to a separately trained no-gate baseline.
