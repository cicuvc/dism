# NIAH long-range circuit probe — handoff

训练 eval step 的轻量配对指标、配置与状态隔离见 [NIAH_EVAL.md](NIAH_EVAL.md)。

Status: first P1/P2/P3 pass and L8/L10 head ablations completed 2026-10-16.
Results, paired absent controls, implementation corrections and limitations:
`SOFT_HARD_RELAXATION.md` §47. Local symbol streams: §48; single-query
background-collision interventions and short-chain filtering: §49.
Whole-forward tau scaling and SAM complexity boundaries: §50.
P4/P5/P6 and codebook/cross-model circuit rescue remain open.
Written 2026-10-16 for a circuit-interpretability pass on three 72M checkpoints
that differ in how well they recall a needle at 16k.

Read `AGENTS.md` and `docs/AGENTS.md` first. This document is a plan; the
measured facts in §3 are already established and must not be redone.

## 1. The question

All three models are the same 72M `dism_gdn_hybrid` architecture (12 layers,
0–5 pure GDN, 6–11 DISM+GDN hybrid, direction=true, value residual) trained on
2.5B fwedu-100B tokens with the same seed and data order. They differ in one
training-pipeline decision each, and they differ measurably in NIAH present NLL
at long context:

| ctx | old | restored | gqa2 |
|---|---|---|---|
| 2048 | **1.724** | 1.761 | 1.743 |
| 8192 | **2.230** | 2.567 | 2.359 |
| 16384 | **2.459** | 2.859 | 2.713 |

(present NLL, mean over depths 0.05/0.5/0.95, 16 trials; lower is better)

Question: **which part of the DISM computation carries the needle from the
insertion position to the query position, and what specifically is lost in
`restored`?** Secondary: why is `gqa2` intermediate at 16k but slightly better
than `restored` at 8k.

The deliverable is a statement of the form "at layer L, head h, the query reads
the needle through <mechanism>, and in `restored` <quantity> is <how> different",
backed by numbers.

## 2. The three checkpoints

All under `/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/`.
Load any of them with `load_ckpt()` from `diagnostics/niah.py` (it reads the
sibling `meta_*.json` and builds from `meta['model_spec']`, so each model gets
its own architecture defaults automatically).

This matters more than it looks: `old`'s spec has no `conv_impl` key, and
`DismConfig.conv_impl` defaults to `"fused"`, which is exactly the conv frontend
`old` was trained with; `restored` and `gqa2` carry `"linear_fla"` explicitly.
So building from each spec reproduces each model's own frontend. All three have
already been loaded and evaluated under the current tree (the numbers in §1 were
produced that way), so this is known-good rather than assumed.

| label | run dir | checkpoint | launched | objective |
|---|---|---|---|---|
| `old` | `dism-fwedu-valres-72m-2500m` | `base_checkpoints/seqtrue72m-fwedu-valres/model_019074.pt` | 2026-10-09 (code `68e1975`) | legacy microbatch mean |
| `restored` | `dism-fwedu-valres-72m-2500m-tokenbal-quadforce-20261015` | `base_checkpoints/seqtrue72m-fwedu-valres-tokenbal-quadforce/model_019074.pt` | 2026-10-15 (worktree `8bf415e`) | global valid-token mean |
| `gqa2` | `dism-fwedu-valres-72m-2500m-tokenbal-gqa2-20261015` | `base_checkpoints/seqtrue72m-fwedu-valres-tokenbal-gqa2/model_019074.pt` | 2026-10-15 (worktree `69d7106`) | global valid-token mean, `gqa_query_channels: 2` |

`old` and `restored` have byte-identical specs up to fields that did not exist
on Oct 9 (`conv_impl`, `conv_backend`, `loss_balance`). `gqa2` is `restored` plus
one key.

Two more checkpoints are useful as references, and a third pair is coming:

- `valres-norec` (`dism-fwedu-valres-norec-72m-2500m`): same layout but
  `diag_recurrence=false`. Gains 1.075 / 0.478 / 0.366 — it shows what the
  suffix-chain recurrence is worth. Use it as the "long-range path removed"
  upper bound.
- `kvg2` (`...-tokenbal-kvg2-20261016`, running) and `legacy`
  (`...-tokenbal-legacy-20261016`, queued behind it) are the same isolation pair
  for the shared-KV head grouping and for the loss-normalization revert. Their
  NIAH will tell you whether `restored`'s deficit tracks the objective or the
  architecture. Add them if they have finished.

## 3. Already established — do not redo

1. **DISM carries the far needle; SWA/GDN does not.** Zeroing the DISM core
   output (`niah.py --mode dism_core`) collapses the far-needle gain:
   4k/d0.05 +0.55 → −0.07, 8k +0.68 → +0.01, 16k +0.44 → +0.10. Near-needle
   (d0.95) only drops 0.2–0.35. Zeroing the whole DISM layer (`--mode
   dism_layer`) gives 3.1601 at 16k.
2. **Scrambling all symbols ≈ zeroing** (`--mode flip`), so the discrete symbol
   identity, not the continuous embeddings, is what does the work.
3. **The suffix recurrence is the long-range mechanism.** `valres-norec` loses
   most of the 8k/16k gain (1.075/0.478/0.366).
4. **`absent` NLL differs across arms** (old 2.94–3.19, restored 3.27–3.58,
   gqa2 3.23–3.37), so `gain = absent − present` is not comparable across arms.
   Use present NLL, absent NLL and present token accuracy separately. §4 of
   `SOFT_HARD_RELAXATION.md` has the full per-cell table.
5. **The wandb `qk/*` probe is not usable for gqa2**: it calls `attn.q_conv` /
   `attn.k_conv` directly, bypassing `_forward_cuda`, so on GQA2 it measures
   only the base query channel (6 of 12 slots). For circuit work hook the real
   kernel inputs instead.
6. Codebook concentration is a *pre-existing* drift, not a regression: on the
   repaired probe `old`'s final checkpoint has q coverage 0.406/0.386/0.337/
   0.240/0.134/0.236 for layers 06–11. See §44 of `SOFT_HARD_RELAXATION.md`.

## 4. The circuit, written out

The whole DISM layer is small enough to read in full. The authoritative
implementation is the FP64 oracle
`dism_v3/python/flash_dism/reference/dism_v3_ref.py::dism_ref` (about 40 lines,
lines ~30–65). Reproduced here because every probe below is a query against one
of its tensors:

```
raw_scores = einsum('bnhc,bmhc->bhnm', q_vec, k_vec)          # [B,H,N,M]
norm_score = raw_scores - (q_lse or k_lse)                    # direction selects
hard_score = 0 masked_fill(idx_q != idx_k, -inf)              # discrete gate
logM       = where(hard, hard_score, norm_score) + rtau
rows[i]    = logM[i] + softplus(shift_left(rows[i-1]))        # suffix chain
scores     = causal_mask(rows[1:])
zmax       = scores.amax(-1).clamp_min(0).detach()
weight     = exp(scores - zmax)          # <-- "attention map", N x M
fallback   = exp(-zmax)                  # exact-zero on no match
s_weight   = einsum('bnhc,bmhc->bhnm', sq_vec, sk_vec)        # signed readout
numerator  = einsum('bhnm,bmhc->bhnc', weight * s_weight, v)
output     = numerator / (weight.sum(-1) + fallback)
```

Key structural facts:

- `idx_q[i] == idx_k[j]` is a **hard discrete gate**. Only positions whose key
  symbol equals the query symbol can contribute at all.
- `weight` is the read map. `logM` is accumulated along the diagonal
  (`softplus(shift_left(...))`), so a pair (i, j) contributes in proportion to
  the length of the *matching suffix chain ending at (i, j)*, not to a plain
  dot product. This is the "suffix matching" that `--mode flip` destroys.
- The readout is **signed** (`sq · sk`, SiLU is not nonnegative) and the
  denominator excludes `s_weight`, so the output is not a convex combination
  of `v`. Both signs matter.
- `zmax.clamp_min(0)` plus `fallback` is what produce the *exact zero* output
  when a query has no match at all; that "no match → zero, not noise" property
  is why the far needle can be dropped silently instead of degraded smoothly.

Where the pieces live:

| stage | code |
|---|---|
| q/k/sq/sk/v/eq/ek/tau for a layer | `dism_v3/python/flash_dism/module.py::DismAttention._forward_cuda` |
| GQA2 second query channel + mix | same file, `self.gqa_query_channels == 2` block and `_mix_gqa` |
| kernel entry points | `flash_dism/voc.py::voc_dism`, `flash_dism/compiler.py::voc_forward` |
| FP64 oracle with all intermediates | `reference/dism_v3_ref.py::dism_ref` |
| hard-path summary | `flash_dism/hard_forward.py`, `summary.py`, `include/summary/*.cuh` |
| branch composition | `flash_dism/hybrid_gdn.py::_combine_cuda` → `voc_dism_out + gdn_raw(...)` |
| layer loop, `v_first`, eval path | `nanochat/nanochat/models/dism/model.py::DismLM.forward` |
| value residual | `dism_v3/python/flash_dism/kernels/value_residual.py` (fused) and `value_residual_mix_reference` |

## 5. Probe plan, in priority order

Each probe must be run on **all three** checkpoints with identical inputs, and
must report per layer 06–11 and per head. `gqa2` has 12 head slots (slot 2h is
the base channel, 2h+1 the added one); report both and say which.

### P1 — per-layer necessity (cheapest, do first)

`niah.py::patch` currently has `dism_core` which zeroes *every* DISM layer.
Extend it to zero exactly one layer at a time and sweep L = 6..11, at 8k and 16k
with depths 0.05/0.95. That gives a per-layer contribution curve.

Discriminator: if `restored` loses the needle because one specific layer stopped
carrying it, the curve will show a single layer whose removal is much cheaper
than in `old`. If all layers are uniformly less useful, the cause is upstream of
the layer (symbols, values, or readout).

### P2 — where the read mass lands

On a single NIAH sequence (the same one for all three models — build it with the
fixed seed from `niah.py`, or write it out and reuse it), run `dism_ref` per
layer and extract `weight`, `idx_q`, `idx_k`, `s_weight`. For the query position
i (the token before the number) look at the row `weight[i, :]`:

- total mass on the needle span vs everywhere else,
- the rank of the needle position in that row,
- `idx_q[i]` vs `idx_k[j]` at the needle (does the gate even open?),
- the sign and magnitude of `s_weight[i, j]` at the needle.

Cost note: `raw_scores` is `[1,H,N,N]` fp32, so N=16384 is ~6.4 GB for 6 heads.
Run 2048 and 4096 in the fp64 path, and 16384 one head at a time (1 GB), or
instrument the varlen CUDA kernel to dump only the query row.

This is the single most informative probe: it separates "the gate never opens"
(symbol problem) from "the gate opens but weight is tiny" (chain/recurrence
problem) from "weight is fine but `s_weight` cancels" (readout problem).

### P3 — symbol chain alignment

For the needle position j and the query position i, print `idx_q`/`idx_k` per
layer and head for all three models, plus the resulting match length. The suffix
chain needs `idx_q[i] == idx_k[j]` **and** a long enough matching run ending
there. Check whether `restored` fails at the equality, at the run length, or
neither.

Cheap version: `flash_dism/emb_kernel.py::interp_embeddings` already returns
`idx_k`/`idx_q` (the module currently discards `top1_q`/`top1_k`); hook it.

### P4 — is the value still there

The readout is `Σ_j weight·(sq·sk)·v_j`. Even with perfect `weight`, `restored`
differs only if `v_j` no longer encodes the number. Probe: at the needle
position, train a one-line linear map (or just take the top principal
directions) from `v_j` to the number's token identity, per layer, and compare
cross-entropy across the three models. Also compare `sk_j` (the key readout) the
same way, since `sq·sk` is the actual per-pair gain.

### P5 — does it survive to the residual stream

§30 already measured that flipping all symbols changes the DISM core output by
53–82% (relative) but the residual stream by only 2–3% per layer, 13% at the
last layer, and the loss by 0.9% — dense averaging plus RMSNorm/gate/o_proj
compression. So it is possible that `restored` computes the right needle read
and then attenuates it. Measure, per DISM layer and per model: `|core output|`,
`|o_proj output|`, and the cosine between the layer's DISM contribution at the
query position and the direction that reduces the number's loss.

If P2 and P4 come out equal across models but P5 does not, the fix is in the
output path, not the matcher.

### P6 — codebook health at the operating point

`nanochat/nanochat/metrics.py::probe_qk_vocab_usage` (now kv-head aware) plus
`k_max_share` / `coverage_pooled` per layer at the final step for all three
models. This is descriptive, not causal, but it is cheap and it is the thing
the `kvg2` arm was designed to move.

## 6. How to run things

```bash
cd /home/cicuvc/cs/project/dism-core
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
export FLA_FLASH_KDA=0 CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DISM_HARD_PLANNER=0          # no planner workers; quadratic fallback
PY=/home/cicuvc/miniconda3/envs/blkw/bin/python
```

Existing harnesses to reuse rather than rewrite:

- `dism-runs/dism-fwedu-valres-72m-2500m/diagnostics/niah.py` — needle builder,
  `--mode {none,dism_core,dism_layer,flip}`, final-norm hook, `--trial-chunk`.
- `.../diagnostics/run_niah_pair.sh` — paired runner (in the gqa2 run dir).
- `.../diagnostics/niah_results/*.json` — all stored NIAH numbers.
- `.../diagnostics/pg19_ablate.py`, `sym_prop.py`, `sym_sens.py`,
  `soft_hard_diag_reach_layer6.py` — earlier ablation machinery.
- `dism_v3/tools/probe_old_valres_coverage.py` — codebook coverage probe.

Environment facts that will bite:

- The GPU is **shared with live training** (`kvg2` now, `legacy` queued behind
  it, each ~18 GB of 32 GB). Check `nvidia-smi` before every run and keep the
  probe peak under ~10 GB. 16 trials x 16384 in one forward needs >15 GB and
  will OOM; use `--trial-chunk 4`, which is numerically equivalent (verified:
  gain ±0.012, present NLL ±0.011).
- `/` is at 99% (6 GB free) and `~/.triton` is 14 GB. Point
  `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` at `/media/...` for anything
  that compiles.
- `nanochat/tests/test_dism_document_ce.py` has 4 pre-existing failures
  (`InductorError` in a Triton compile subprocess), reproducible at HEAD. The
  repo baseline is ~56 failures from stale tests.

## 7. Working agreements

- **Astra's sandbox is `workspace-write` on the worktree and `/tmp` only**, so
  it cannot write into `/media/...` run dirs. Put new probe scripts in
  `dism_v3/tools/` and outputs in `dism_v3/build/` (gitignored) or `/tmp`.
- Never write into a checkpoint directory. Never mutate a checkpoint.
- Keep probes monkeypatched / opt-in. Do not change default training or eval
  behaviour; `kv_group_size=1` and `legacy_loss_normalization=False` must stay
  bit-identical.
- Record findings as a new numbered section appended to
  `dism_v3/docs/SOFT_HARD_RELAXATION.md` (the running lab notebook, now past
  §44), with the raw JSON paths next to the numbers.
- Every claim needs a paired, same-input comparison across the three models.
  A single-model number is not evidence about the difference.

## 8. What would count as an answer

## 9. Update 2026-10-16: `rtau` is the long-vs-short contrast knob

Astra found that **raising `rtau` on the located retrieval head improves NIAH
broadly**, and that the 16k deficit is not "no signal" but the needle's
exponential weight being diluted by many short matches in the longer context.
Both follow directly from §4, so here is the derivation to work from.

`rtau` is added to `logM` at **every step of the softplus chain**:

```
rows[i][j] = mix[i][j] + rtau + softplus(rows[i-1][j-1])
```

For a matching run of length k ending at (i, j) the inner values are large, so
`softplus(x) ~= x` and the accumulated score is approximately
`sum(mix) + k * rtau`. For a broken/short run the inner value is small,
`softplus(x) ~= e^x`, and the whole contribution collapses toward 0. So rtau is
multiplied by the **chain length**: it is the knob that sets how much a long
match dominates a short one.

The read is normalised by the whole row, which is where dilution enters:

```
weight[i][j] = exp(core[i][j] - zmax[i])
output[i]    = sum_j weight[i][j] * (sq_i . sk_j) * v_j
               / (sum_j weight[i][j] + exp(-zmax[i]))
```

The row max is 1 by construction, so the needle's *share* of the numerator is
`1 / sum_j exp(core[i][j] - zmax[i])`. Every additional short match in the row
adds a positive term to that sum. At 16k there are simply more of them, so the
needle's share shrinks even though the needle's own term is unchanged. Raising
rtau increases the exponent gap between the needle and every short match, drives
those terms to 0, and restores the share toward 1. `exp(-zmax)` also shrinks,
so the denominator approaches `sum_j weight = 1`.

### What the trained `rtau` values say (checked, negative result)

Endpoint `softplus(log_sel_tau)` at step 19074, per DISM layer, all heads:
`old` max 7.13 mean 0.745; `restored` max 6.02 mean 0.752; `gqa2` max 5.01 mean
1.058; `valres-norec` max 9.95 mean 1.836. The large values sit on the **last**
DISM layer (`old` L11 has three heads >= 2.85, `restored` one, `gqa2` none).

But across the eight checkpoints that have both a checkpoint and a stored NIAH
result, the trained tau does **not** order 16k recall:

| correlation vs 16k present NLL | r |
|---|---|
| tau max over heads | +0.295 |
| tau mean over heads | +0.423 |
| max tau on layer 11 | +0.197 |
| tau max vs 16k gain | -0.296 |

So the endpoint value is not the mediator. That is consistent with the
sensitivity being real: what matters is the **trajectory and the gradient**, not
where tau ended up. Two consequences worth pursuing:

1. **The known tau-gradient defect is now first-class, not a rounding detail.**
   `8961651` pins the planner event route's tau gradient at 3.8e-4..1.6e-1 off
   FP64 while the quadratic fallback is <=1.3e-5. If NIAH is this sensitive to
   rtau, any run or resume that used the event route has a corrupted tau
   trajectory. `restored` deliberately avoids it; `old` predates the planner
   entirely and computed hard rows inside `dism_core` itself, so `old` and
   `restored` differ in the tau-gradient path as well as in the objective. That
   makes the tau path a co-equal suspect to the loss normalization (#1) for the
   old-vs-restored NIAH gap.
2. **Log tau, not just its effect.** `log_sel_tau` per layer/head plus the tau
   gradient norm are not currently in the training logs, so the trajectories
   cannot be compared after the fact. Adding them is cheap and would settle
   whether `restored` grew tau more slowly.

### Cheap tests to run next

- **Test-time tau sweep on all three models.** Astra's sweep was one model; run
  it on `old` / `restored` / `gqa2` at 8k and 16k. If raising rtau on the
  retrieval head closes the old-vs-restored gap, the difference is calibration,
  not a missing circuit, and the fix is a knob rather than an architecture.
- **Direct dilution measurement.** At 4096 (fp64 reference is affordable; 16k
  needs one head at a time) take the last DISM layer, extract `weight` and
  compute, for the query row, the needle's share `1 / sum_j exp(core - zmax)`
  and the number of contributing short matches. Sweep rtau and plot the share.
  That turns Astra's statement into a measured curve per model.
- **Interaction with `gqa_mix`.** `gqa2`'s `gqa_mix` collapsed from 0.5 to
  0.004-0.13 by step 4000 (see the gqa2 README), so for that arm the DISM output
  scale is partly set by a weight-decayed free parameter rather than by rtau.
  The tau sweep may therefore behave differently on `gqa2` than on the two
  GQA1 models; report them separately rather than pooling.

## 10. Candidate fix: stop rtau's gradient on the hard path

Proposal: detach `rtau` on hard rows so only the soft rows train it. Analysis
before running anything.

### Both paths share one scalar that does different jobs

On a hard row `hard_score` is 0 for matching pairs and -inf elsewhere, so
`core ~= k*rtau` and rtau has exactly two effects:

- `weight = exp((k_j - k_max)*rtau)` — winner-take-all between the matching
  chains, i.e. the long-vs-short contrast;
- `fallback = exp(-k_max*rtau)` — the **no-match gate**, because the
  denominator is `sum_j weight + fallback`. rtau is therefore the soft threshold
  between "output ~= 0" and "normalised read".

On a soft row `norm_score = raw - lse` is a real score, so `core ~= sum(norm) +
k*rtau` and rtau instead trades chain length against match quality. Same scalar,
structurally different role. That is the reason to suspect the shared value is
being pulled two ways.

### Two hypotheses, two different fixes

- **(a) the hard-row rtau gradient is noise-dominated.** Detaching denoises it.
  Support: §29 of `SOFT_HARD_RELAXATION.md` measured the hard path's
  signal-to-noise at ~1% (`|g_R| ~= |g_k|/sqrt(K)`, `E[g_k] ~= 0`), and §30
  measured a single symbol flip moving the hard loss by 0.0015%. A scalar that
  Astra's sweep shows to be very consequential for NIAH being co-trained by a
  1%-SNR signal is a real hazard.
- **(b) the two paths simply want different rtau.** Then the fix is a **separate
  `log_sel_tau_hard`** (6 extra parameters per layer) used on hard rows and at
  inference, not a detach. Detaching is the special case that constrains
  `tau_hard = tau_soft`; it can only win if the hard gradient is harmful rather
  than merely different.

Both are cheap. **A (detach) and B (separate tau) answer different questions;
run both.** B strictly dominates A in expressiveness, so B >= A in achievable
loss; if B beats A the conflict is about the value, if they tie it was noise.

### Two cautions

- **Do not judge on the gap alone.** The deployed path is 100% hard, so the
  target is hard val bpb and NIAH; the gap is a diagnostic. Detaching removes the
  hard path's only direct control of its own gate threshold, so the natural
  failure mode is hard loss getting *worse* while the gap narrows -- and a gap
  can also narrow by both paths getting worse. Report hard bpb, gap and NIAH, in
  that order of authority.
- **Schedule interaction.** `hard_prob` ramps to 0.95, so late in training only
  ~5% of rows are soft. Detaching therefore cuts rtau's gradient sample count by
  ~20x exactly when the run is converging, which may freeze rtau early. Consider
  detaching only after the ramp, or keeping a soft fraction throughout.

### Order of work

1. **Eval-time per-head rtau sweep on `old` and `restored` at 8k/16k** (Astra did
   one model). This gives the headroom and names the heads; hours, no training.
   If the optimal rtau is several times the trained value, that both sizes the
   expected gain and shows the hard-path gradient is fighting it.
2. **Log `softplus(log_sel_tau)` per layer/head plus its gradient norm during
   training.** Neither is logged today, so trajectories cannot be compared after
   the fact.
3. **Then A and B as short arms** (2-3k steps), judged on hard bpb and 8k NIAH.

Directly measurable right now on real batches: `dL_hard/drtau` versus
`dL_soft/drtau` (magnitude, sign, and the angle between them). If the hard term
is negligible, A is a no-op and only B is interesting. Blocked only on GPU
memory: on 2026-10-16 18:30 the box was running the `legacy` training (17.7 GB)
plus a user transcription job (13.0 GB), leaving 1 GB free.

One of these, with numbers:

1. **Symbol alignment.** At 16k the needle is lost because `idx_q[i] ==
   idx_k[j]` fails (or the matching run is too short) in `restored` but holds in
   `old`. Mechanism: codebook drift. Fix direction: the `kvg2` / normalization
   levers.
2. **Chain decay.** The gate opens, but `weight[i, needle]` is orders of
   magnitude smaller because the recurrence (`softplus` accumulation) decays
   faster. Mechanism: `rtau` and the chain arithmetic. Fix direction:
   temperature / `log_sel_tau`.
3. **Readout cancellation.** `weight` and `v` are fine, but `sq · sk` at the
   needle is near zero or of the wrong sign in `restored`.
4. **Output attenuation.** The matcher is fine at every layer, but the DISM
   contribution reaching the residual stream is smaller, or `o_proj` /
   RMSNorm-gate compresses it.
5. **Uniform degradation** — no localized culprit, so the deficit is a
   distributed calibration shift rather than a circuit change. That is also a
   valid and useful answer, and it would point back at the objective rather
   than at any specific matcher component.
