# 384-wide GDN/DISM + vocabulary SiLU, 30k

User authorized 2026-09-11, corrected target from 3k to **30k** optimizer steps.
Fresh seed777 run, not a continuation of baseline. Main LMConfig and trainer now
default to vocab_silu; explicit baseline remains available. This transforms the
FP32 Q/K codebook parameters with SiLU before BF16 interpolation; Q/K shortconv
keeps its original SiLU. No vocabulary normalization or sharing added.

12 layers [GDN,GDN,GDN,DISM] x3, width384, H6, DISM D=DV64, Q/K vocab512,
FFN1428, 72,188,142 parameters. No SWA. Untied GPT2 input/head, PreNorm LayerNorm.
Local FineWebEdu stream and cached GPT2 tokenizer; context2048, batch64/micro8.
AdamW lr1e-3, WD.01 with no_decay groups, warmup1000 then cosine to0.1 peak,
hard_prob linearly0→1 over30k. Softcap30 CE. Validation100 same-size batches
every1000 updates; save every1000. W&B offline. Total3,932,160,000 train tokens.

Independent source snapshot src/, preflight full microbatch forward/backward,
finite gradients, optimizer update, evaluation, effective-codebook assertions.
Runner launches training only after preflight passes.

Service: dism-gdn-vocab-silu-384-30k-20260911.
Under dism-lm-runs/gdn-dism-vocab-silu-384-72m-30k-20260911:
queue-status.json, preflight.log, console.log, metrics.jsonl, latest.pt, wandb/.

Launch verified: preflight passed in329.99s including first-build overhead;
loss10.906336, finite gradients (preclip norm3.762527), peak9.301GiB.
Training PID833192, runner829911. Step1 completed: loss10.904495,
grad_norm1.347339, lr1e-6, hard_prob0, peak9.608GiB. Initialized config confirms
steps30000, hard schedule update0=0/update29999=1, activation=vocab_silu.
