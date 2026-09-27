# Trained DISM attention inspection

Local original49.68M hybrid, step30000, pure-hard inference. Three deliberately
small examples: first1024 held-out packed tokens; the original NIAH pilot's
2048/depth90% silver-telescope success and wooden-compass failure. The NIAH
pair is selected by outcome, not a representative performance sample.

## What was measured

In an isolated evaluation process, wrap the existing autograd module's core
forward call to capture its actual CUDA-selected qi/ki, tau, V, O and log2
normalizer. Return the original outputs unchanged and restore the function
after each pass. Captured-versus-ordinary model features are bitwise identical
on all three examples. No installed library, model weight, training code or
CUDA kernel was changed.

Offline CPU reconstruction uses exact FP32 natural-log recurrence:

```
logM[i,j] = tau if qi[i]==ki[j], else -inf
W[i,j] = logM[i,j] + softplus(W[i-1,j-1])
P[i,j] = exp(W[i,j]) / (1 + sum_j exp(W[i,j]))
fallback[i] = 1 / (1 + sum_j exp(W[i,j]))
```

Plots show P, not raw logM: raw pure-hard scores have only two values per
head and do not reveal diagonal accumulation. The production path uses
tanh_finite approximation/BF16 output; this is NOT an exact dump of its
internal approximate W. Reconstructed last16 O rows have relative L2
0.001282–0.002067 versus actual CUDA O over45 layer/sample pairs. Maximum
natural-log normalizer difference is0.001265. Two recurrence/fallback tests
pass against the reference. No FP32 re-labelization was substituted.

## Observations

1. Heads differ substantially: broad historical mixing, vertical columns,
   isolated matches and many entirely inactive rows all appear. There is no
   universal diagonal/local-attention pattern.
2. On the natural1024 sample, query positions512–1023 have mean fallback
   mass0.4279 averaged equally over60 heads;11 heads exceed0.8. L2H2 is
   especially selective: fallback0.9749, mean matching keys0.0586 per query,
   despite tau3.9494. This is not simply all tau values being too small.
   It is NOT proof these heads are dead: their rare matches may matter, and
   output norm/gate changes the relationship between probability mass and
   contribution to the residual stream.
3. Q/K label usage on this sample is concentrated: effective vocabulary size
   exp(entropy) has median13.21/11.81, max46.96/37.86, out of512 labels.
   This is sample-local utilization, not proof of globally collapsed codebooks;
   finite sample size, text distribution and intentional specialization matter.
4. The NIAH success predicts ` yellow`. Its final query has L8H4 weight0.73997
   on the actual ` yellow` token. L9H4 puts effectively1.0 on the following
   double newline; L13H4/L14H3 put0.94721/0.99473 on the period after yellow.
5. The NIAH failure predicts ` a`, yet L9H4 puts0.99903 on the corresponding
   double newline after yellow (total needle mass0.99904). Other heads do
   not show the success case's strong answer/period concentration. At least
   this error cannot be described as every head missing the needle location.
   Attention to punctuation is not intrinsically wrong: causal shortconv and
   previous layers can encode the preceding answer there. Whether that V
   actually contains useful answer information remains unmeasured.

Before architecture changes, these observations motivate checking Q/K address
alignment and how retrieved values affect answer logits, rather than assuming
more position bias or stronger retrieval gates fixes the issue. Attention
weights alone do not establish causal feature attribution. No ablation,
activation patching, head pruning or model change was performed here.

## Figures and artifacts

Directory:
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/dism-swa50m-softcap30-20260909-offline/attention-inspection`

- `natural_attention.png`: layers1/5/10/15, all4 heads. Common log10 scale;
  each256×256 pixel sums key-bin mass and averages query-bin rows. Exact zero
  and values below1e-5 are both shown at the color floor. Not raw per-token P.
- `natural_heads.png`: all60 heads, local/far/fallback mass, effective labels,
  maximum weight. Local means distance0..127; far means>=512; intermediate
  distance mass is not shown. Mass summaries use queries>=512 only.
- Corresponding attention/head/last-query figures for both NIAH cases.
- `needle_comparison.png`: all60 heads' final-query probability on each needle
  token, success versus failure, linear0..1 scale. No binning in this figure.
- `report.json`: numeric statistics, top-key token contexts, reconstruction
  errors, model predictions. Layer/head labels in reporting start at1;
  token positions in data start at0.
- Per-sample `.pt`: exact selected Q/K labels,tau,token IDs,binned P and final
  query's unbinned P. Full dense matrices are reconstructed transiently one
  layer at a time, not permanently stored. Text samples also saved.

```bash
RUN=/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/dism-swa50m-softcap30-20260909-offline
/home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.visualize_lm_attention \
  --checkpoint "$RUN/latest.pt" --output "$RUN/attention-inspection-new"
```

Output must be a new directory. Main natural heatmaps and needle detail figure
were visually inspected. This diagnostic does not change production's
no-materialized-attention policy.
