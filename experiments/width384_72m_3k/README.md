# 384-wide approximately72M hybrid, queued after vocab128

User authorized widened residual-stream experiment with72M budget and untied
input/output token weights. Use original DISM+parallel RoPE SWA128, NOT GLA,
vocabulary normalization, balance, cross-layer/head sharing or codeword SiLU.

Configuration: width384,12 layers,6 heads,D=DV64,Q/K vocab512 per head,
FFN1024,context2048,GPT2 vocabulary50257,LayerNorm PreNorm,original postconv
Q/K SiLU. Exact parameter count72,188,040. Untied token embedding/head account
for38,597,376; remaining parameters33,590,664. Compared with original49.68M,
width/head count/budget increase and depth decreases15→12; cannot isolate width
as the sole causal factor.12x1024 is chosen over15x512 to retain FFN capacity.

Same3000 steps,batch64/micro8,seed777,LR1e-3,WD.01,warmup100,softcap30,
hard_prob0→1,tanh_finite,OPT13,offline W&B.100 effective validation batches
every1000 updates,save every500,final256-sequence hard vocabulary evaluation.
Total393,216,000 training tokens (~5.45 tokens/parameter), not20 tokens/parameter.

Main trainer adds --width and --layers; heads derived as width/64 and width
must be a positive multiple of64. Defaults256/15 preserve previous invocations.
Independent src/ snapshot and CUDA/Triton caches preserve all active jobs.
run_study.py waits for vocab128 phase=complete, including final vocabulary
evaluation. Predecessor/preflight failure stops without retry or restarting jobs.

tests/test_lm_width384.py checks parameter count, untied weights and head/FFN
shapes on CPU. Deferred GPU tests cover soft/mixed/hard one-layer6-head
forward/backward/AdamW/eval plus actual12-layer,microbatch8,N2048 mixed training
update and finite gradients. Logs loss,gradient norm and peak allocated memory.
All tests must pass before the3000-step trainer launches. No automatic batch
reduction or numerical-tolerance change on failure.

User service: dism-width384-72m-3k-20260910.
Output:
/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910/width384_72m

queue-status.json,preflight.log,console.log,metrics.jsonl,latest.pt,best.pt,
offline W&B,vocab-load-final/. Initial phase: waiting_for_vocab128.
