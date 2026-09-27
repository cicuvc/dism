# Targeted DISM readout-head ablation

Completed2026-09-11 after parameter-matched all-GDN training had started.
experiments/head_ablation.py,64 same frozen sequences as suffix diagnostic.
Pure hard,FP32 masters,BF16 model/head,FP32 softcap CE. Each intact per-token
NLL was checked bitwise against the earlier diagnostic. Batch1 with0.1s pause;
active GDN training not stopped. All interventions finite and hook call counts
checked. Hook removed after each condition.

Remove selected head's64 coordinates after output gated RMSNorm and before
o_proj. Norm statistics and projection bias retained. Other head contributions
in this block are unchanged; downstream activations/routing may change. This
tests readout importance,not computation savings or recurrence-specific benefit.

Select each model's largest absolute real-minus-shuffle distant>=4 match-rate
excess. Same-layer comparison head has the closest real match rate among other
heads (can still have very different activity). Selected using this SAME sample;
results exploratory,not held-out confirmatory tests.

| Model | Target | NLL delta ± sequence SE | Control | NLL delta ± sequence SE |
|---|---|---|---|---|
| baseline |L10H2|+0.00307452 ±0.00085174|L10H3|+0.00036606 ±0.00008201|
| width384 |L8H2|+0.00070644 ±0.00014806|L8H6|+0.00035538 ±0.00011956|
| GDN/DISM |L8H6|+0.00069415 ±0.00016989|L8H5|+0.00001945 ±0.00006265|

For target heads,conditional delta on tokens with original distant>=4 match vs
other tokens (both exclude first128 positions): baseline+0.012219/+0.001520;
width384+0.000911/+0.000499;GDN+0.007919/+0.000329. Matching token counts
19,954/58,728/6,585. These conditional groups are not randomized and the
intervention also removes short-match contributions. Thus supports useful head
readout,especially on matching examples,but does not prove long-chain causality.
Sequence SE not training-seed uncertainty; no multiple-comparison correction.

Artifacts activation-study-3k-20260910/head-ablation-64-20260911/report.json,
per-model .pt with intact/target/control per-token losses. Parent script completed.
All-GDN training last checked at step150,finite loss/gradient,0.652s/update.
