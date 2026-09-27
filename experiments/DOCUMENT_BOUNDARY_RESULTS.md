# Frozen-checkpoint document-boundary probe

Model: latest30000-step 72M vocab-SiLU GDN/DISM checkpoint. Local RTX5090;
no training/weight/kernel edits, no A100 job changes. All inference pure hard.

## Design and numerical controls

32 target validation documents B; two distinct other validation documents A
per B; prefix lengths128/512/1536 including final EOS; B's first512 input tokens
predict B tokens1…512. Max sequence2048. Documents are first96 distinct held-out
documents with at least1537 tokens, not a random sample of all document lengths.
Distinct-document prefixes are not checked for topical independence. No separate
semantic-neighbor/adversarial codeword-selected prefix experiment yet.

Five arms: intact; split only voc_dism core (conv remains continuous); split full
DISM module (conv reset too); split GDN modules (recurrent and conv reset);
split both module types. All12 blocks retain their original residual/FFN/norm.
Boundary is immediately after EOS; B-only control contains no leading BOS/EOS.
Because GDN-reset degradation is concentrated near B's start, it may reflect
lost boundary-marker/warm-state information, not useful old-document content.

First implementation exposed BF16 GEMM-shape differences: all-reset vs B-only
mean absolute per-token NLL difference0.01867. This run remains under
document-boundary-32 but is not the primary intervention result. Canonical rerun
splits tokenwise Linear/LayerNorm operations at the boundary in EVERY arm,
including intact. All-reset then equals B-only **exactly for every token and
every prefix**. Canonical vs original intact mean signed NLL difference only
0.0000482; mean absolute per-token0.01836. No tolerance relaxation or checkpoint
precision changes. Labels captured from actual CUDA embedding output.

## NLL results

Average over32 B documents, two A and three lengths. SE clusters by B document;
the192 contexts are not192 independent samples. Negative delta favors reset.

| Intervention | B NLL | Delta vs intact | Paired95% interval |
|---|---:|---:|---:|
|Intact|3.182639|0|—|
|DISM core only|3.182915|+0.000276|[-0.000861,+0.001413]|
|DISM including conv|3.183243|+0.000603|[-0.000598,+0.001804]|
|GDN including conv|3.192340|+0.009701|[+0.006737,+0.012664]|
|Both / B alone|3.193306|+0.010667|[+0.007487,+0.013847]|

GDN-reset delta first32 positions +0.09707; positions128–511 +0.001685.
DISM-reset positions128–511 -0.000173, interval spans zero. No growing net
DISM penalty with tested prefix length. Replacing A by the other A changes
per-token target NLL by mean absolute0.04562 intact vs0.03032 DISM reset;
this is sensitivity, not proof of harmfulness or a logit/KL measurement.

## Cross-document attention exposure

First8 B documents, first A, prefix1536. Reconstruct exact natural-log hard
recurrence from actual CUDA labels and learned tau. Linear scratch, no production
N² tensor. Small dense oracle selftest passed. These are exact-reference weights,
not bitwise tanh-approximate production probabilities. Include fallback in denominator.

| DISM layer | Probability on prior document | Prior-doc probability with matching chain>=4 |
|---|---:|---:|
|4|61.99%|0.192%|
|8|68.31%|3.079%|
|12|18.13%|2.206%|

The chain>=4 statistic means distant source in A with >=4 matching suffix tokens;
it does not specifically mean the chain itself crossed the A/B boundary.
Prefix is three times B's length, so large probability alone is not unusually
strong per-token preference. Most cross-document probability is short-chain.
This measures the exposure directly, not the contribution after gate/output projection.

## Small NIAH follow-up

32 paired cases, B=2048 synthetic repeated-prefix key/color retrieval,
eight balanced colors at each of four depths. Prepend1536 tokens from another
held-out packed block plus EOS (prefix may itself contain multiple documents).
No sampling, full-vocab greedy and answer CE. All-reset logits equal B-alone
bitwise for every case. Same canonical pointwise control.

| Arm | Exact | Eight-way | Answer NLL |
|---|---:|---:|---:|
|B alone|8/32|9/32|4.08684|
|A+B intact|8/32|9/32|4.14418|
|DISM core reset|9/32|9/32|4.06196|
|DISM including conv reset|9/32|9/32|4.06520|
|GDN reset|8/32|11/32|4.19151|
|Both reset|8/32|9/32|4.08684|

DISM-reset paired answer NLL delta -0.07898, SE0.03271 (approx95% interval
[-0.14309,-0.01487]); core-only -0.08222, SE0.03463. Exploratory32-case sample,
multiple arms, not a corrected significance test. One extra exact success is
not compelling evidence of substantially restored retrieval.

## Interpretation and limits

Cross-document codeword interference/exposure demonstrably exists. Removing it
reduces prefix sensitivity and mildly improves synthetic retrieval answer NLL,
but does not improve average ordinary NTP on this sample or rescue long retrieval.
This weakens the claim of a large *immediate* inference-NLL penalty, not the
training-time hypothesis that pollution inhibits development of useful circuits.
No boundary-aware retraining, soft/mixed-state evaluation, or full-attention
comparison was performed. Those remain necessary to test that stronger hypothesis.

Reproduction: conda blkw, `python -m experiments.document_boundary_probe`,
then `python -m experiments.document_boundary_niah`,
`python -m experiments.summarize_document_boundary` (new output directories
required for evaluation reruns). Primary artifacts under checkpoint directory:
document-boundary-32-canonical/{results.pt,data.json,summary.json,niah.json,boundary.png}.
