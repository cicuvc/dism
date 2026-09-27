# Validation Q/K vocabulary load

Local original49.68M hybrid checkpoint step30000. Pure-hard production CUDA
prefill, 6400 held-out packed sequences×2048 =13,107,200 input tokens per
layer/head/side. 15layers×4heads,512 distinct addresses per head. These are
separate codebooks; equal IDs in different heads/layers are not shared entries.

An evaluation-only wrapper records actual CUDA embedding outputs[7]/[6]
(Q/K top1 indices) in int64 histograms and returns outputs unchanged. No
FP32 relabeling or attention matrices. First batch features are bitwise equal
with/without recording. All6400 sequences' final features finite. Histogram
totals exactly equal13,107,200 for every head and side; expected call count
also checked. Two histogram/alignment tests pass. No model, kernel or running
training modifications. Measurement took~45.4seconds excluding loading/plots.

## Load results

The table averages the60 head-specific statistics equally. Uniform expected
count is25,600 per entry; <1%uniform means fewer than256 selections.

| Statistic per512-entry codebook | Q | K |
|---|---:|---:|
|Zero selections, mean entries|235.32|345.38|
|Zero selections, fraction of all slots|45.96%|67.46%|
|At most10 selections, mean entries|295.75|398.92|
|Below1%uniform, mean entries|374.22|446.37|
|Below1%uniform, fraction of all slots|73.09%|87.18%|
|Effective count exp(entropy), mean|20.40|16.58|
|Effective count, median|17.64|14.60|
|Most-used entry mass, mean|30.06%|25.78%|
|Top8 entries mass, mean|71.58%|73.08%|
|Entries covering90% of selections, mean|24.75|16.22|
|Gini, mean|0.96276|0.97437|

Across30,720 slots per side:14,119 Q and20,723 K entries unused. Both sides
unused at11,585 corresponding addresses. In9 Q heads/7 K heads, one entry
receives more than half the selections. The most extreme slot fractions are
82.56% Q/86.55% K. L1H2 used only9 Q labels and4 K labels throughout this
sample; its K effective count is1.93, largest entry81.63%.

## Q/K address alignment

For each head, compute normalized marginals pQ,pK over the same512 address
IDs. Mean overlap sum(min(pQ,pK))=0.34344; median0.33117. This is a distribution
comparison, NOT per-query causal matching rate.

A stronger coverage warning: mean Q mass on addresses with **zero K usage
anywhere in this validation sample** is32.42%; on addresses with K usage
<1%uniform it is36.66%. Since all heads see the same number of tokens, the
former is also the fraction of layer/head/query selections guaranteed to
have no matching K in their evaluated sequence. It is a lower bound on total
unmatched queries, not an equality. It is not a claim about unseen data.

Examples:

| Head | Q/K marginal overlap | Q mass on sample-unused K addresses |
|---|---:|---:|
|L4H4|1.91%|60.95%|
|L2H2|4.11%|77.64%|
|L4H2|4.20%|94.80%|
|L5H4|4.77%|94.72%|

Thus substantial concentration persists beyond the original short text, and
address alignment is a separate issue from making each marginal uniform.
These measurements do not establish that uniform routing is desirable or
that rare addresses should be deleted. They measure **hard top1 usage**,
not soft interpolation probability mass, training gradients, or global
permanent inactivity. Some heads may intentionally specialize or abstain.

## Artifacts / reproduction

Directory:
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/dism-swa50m-softcap30-20260909-offline/vocab-load-validation`

- `report.json`: every head's load/alignment statistics, dataset identity/hash.
- `counts.csv`: layer,head,side,label,count,frequency for all61,440 entries.
- `counts.pt`: int64[15,2,4,512], side0=Q,side1=K.
- `label_load.png`: common actual label-ID axes; color is log10 frequency.
- `ranked_load.png`: within-head sorted frequencies and cumulative load,
  median across heads. Shading is10–90th percentile across heads, not a
  confidence interval. Both figures visually inspected.

```bash
RUN=/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/dism-swa50m-softcap30-20260909-offline
/home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.eval_vocab_load \
  --checkpoint "$RUN/latest.pt" --output "$RUN/vocab-load-new" \
  --sequences 6400 --micro-batch 8
```

Output must be a new directory. This task only measured/recorded behavior;
no balancing regularizer, codebook sharing or architecture change was applied.
