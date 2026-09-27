# Parameter-matched GDN/DISM restart

User explicitly authorized stopping FFN1024 training and restarting from scratch
with FFN1428 on2026-09-11. Original trainer PID785502 received SIGTERM, finished
its current update and saved latest.pt; old logs/checkpoints/W&B remain untouched
under activation-study-3k-20260910/gdn_dism_384. Old launcher's final3000-step
assertion reports failed after this intentional early stop, not numerical failure.

New count72,188,142 vs previous384 DISM+SWA reference72,188,040: +102 parameters.
Structure unchanged: [GDN,GDN,GDN,DISM]x3,zero SWA,width384,6heads,DISM D/DV64,
GDN K/V64/128,V512,untied GPT2 weights,original LayerNorm PreNorm. Only FFN
1024→1428. Same3000 steps,batch64/micro8,seed777,LR1e-3,WD.01,warmup100,
softcap30,hard_prob0→1,offline W&B,validation100 batches each1000 updates,
save each500. New optimizer,RNG,data stream,annealing; no --resume.

Reuse old frozen source and compiled CUDA cache without modifying them;
new output/Triton cache/W&B. Fresh full12-layer microbatch8,N2048 preflight
checks exact count,finite loss/all gradients,AdamW and eval before training.
Original5-test soft/mixed/hard suite passed before the previous run.

Service dism-gdn-dism-384-matched-3k-20260911; run_study.py.
Output:
/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910/gdn_dism_384_matched

queue-status.json,preflight.log,console.log,metrics.jsonl,latest.pt,offline W&B.
No automatic post-training vocabulary evaluation (mixed-layer evaluator pending).

Old checkpoint verified at step811 (66,593,550 parameters). New full-microbatch
preflight passed in139.19s. New training confirmed step30: CE8.78042984,finite
pre-clipping gradient norm1.60042,0.706s/update,peak allocated10.148GiB.
Current state training, not completed.
