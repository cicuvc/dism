# Matched suffix diagnostic,64 frozen sequences

Completed2026-09-11 by experiments/suffix_diagnostic.py. Models baseline,
width384_72m,gdn_dism_384_matched at step3000. Same64 sequences from frozen
swa-eval-bundle,first2048 tokens,not the4096 packed-prefix sample. FP32 parameters,
BF16 model/head,FP32 softcap30 CE. First traced/untraced NLL bitwise checked.
Numba exact causal suffix DP self-tested against brute force; no full attention
matrix stored. Query maxima and length histograms over all j<=i,plus distant
i-j>=128. Query fractions exclude first128 positions. Three K shuffles preserve
each sequence/layer/head histogram,with one permutation shared across heads/layers
of a sequence; Q unchanged. No shuffling is fed into model inference.

NLL baseline3.8622913,width3843.7362504,GDN3.6984782. Paired per-sequence
GDN-minus384 mean -0.0377720,SE0.00347999 (64 sequences,not training-seed error).
Original training validation GDN3.7744408 vs3843.8201692.

Distant query match probability,real / mean shuffled:

| Model | length>=4 | length>=8 | length>=16 |
|---|---|---|---|
| baseline |4.7082% /4.5589%|0.93061% /0.86510%|0.027710% /0.020666%|
| width384 |4.6465% /4.5391%|0.44493% /0.22413%|0.014185% /0.001515%|
| GDN |2.0887% /1.6952%|0.29912% /0.11161%|0.055339% /0.002954%|

Early-layer long matches can be explained by frequency: baseline layer1 >=4
33.05% vs shuffled35.68%,width384 layer1 27.43% vs31.09%. Later heads show
enrichment,e.g.width384 L9H2 15.13% vs2.86%,GDN L8H6 5.36% vs1.36%.
GDN DISM layers are physical4/8/12,not1/2/3. GDN Q/K entropy-effective vocabulary
averaged over its18 heads is28.55/25.30 on this sample.

Pearson correlation between NLL and fraction of heads with distant length>=4:
baseline0.02742,width3840.01938,GDN0.000824. After subtracting each sequence's
256-position-block means:0.01922,0.00365,0.00121. No clear pooled negative
association. These are observational,not causal retrieval-benefit estimates.

Shuffle removes ordering,local/position structure and cross-string alignment;
enrichment alone does not establish semantic usefulness. Rare >=16 events have
small denominators; three shuffles are a pilot,not robust uncertainty estimates.
Different layer/head counts prevent comparing unnormalized total match counts.
No per-head significance testing or causal DISM ablation performed.

Artifacts: activation-study-3k-20260910/suffix-diagnostic-64-20260911/report.json,
three model .npz files with labels,per-token NLL and query max suffix lengths.
