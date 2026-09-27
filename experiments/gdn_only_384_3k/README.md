# Parameter-matched all-GDN control

User authorized2026-09-11: replace all3 DISM layers in the GDN/DISM hybrid
with GatedDeltaNet.12 GDN layers,width384,H6,K64,V128,shortconv4/output gate,
no SWA,no DISM,no GLA. FFN1392 yields72,184,080 parameters vs mixed model
72,188,142 (-4,062,0.0056%). Untied GPT2 input/head,LayerNorm PreNorm retained.
CLI --gdn-only; architecture field remains hybrid for model dispatch,while
hard_schedule reports not applicable and hard_prob stays0 (no row RNG consumed).

Same3000 updates,batch64/micro8,seed777,LR1e-3,WD.01,warmup100,softcap30,
100-batch validation every1000 updates,checkpoints every500,offline W&B.
Independent source snapshot/run/Triton cache; no old training stopped.
Full12-layer microbatch8,N2048 forward/backward/AdamW/eval passed in115.92s.
Training confirmed step10: CE10.8013334,finite preclip grad norm1.99765,
0.656s/update,peak allocated10.109GiB. Not a completed experiment.

Service dism-gdn-only-384-3k-20260911.
Output activation-study-3k-20260910/gdn_only_384_matched:
queue-status.json,preflight.log,console.log,metrics.jsonl,latest.pt,offline W&B.
No vocabulary evaluation because this model contains no codebooks.
