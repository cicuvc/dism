# Codeword RMS normalization, queued after balance

User authorized2026-09-10. Independent original-hybrid arm, not combined with
balance, GLA, codeword SiLU, sharing, or token Q/K normalization. Each Q/K
codeword is transformed by `E * rsqrt(mean(E**2, dim=-1) + 1e-6)` in FP32,
before per-head expansion and independent BF16 casts. No learned gain/bias,
no centering. This keeps norm approximately sqrt(D)=8 instead of changing the
initial logit scale by8x as unit-L2 normalization would. Does not remove the
directional bias of postconv token SiLU or change hard matching semantics.
All interpolation outputs use the same effective codebooks; autograd differentiates
through normalization back to the master weights. CUDA kernels are unchanged.

LMConfig.dism_vocab_norm / --dism-vocab-norm, default false. Code supports hybrid
and hybrid_shared; scheduled arm is original hybrid only. Existing initialization
RNG and all master weights are exactly baseline-identical at seed777. No new
parameters:49,678,876. Masters retain original no_decay treatment.

15 layers,width256,H4,D=DV64,V512,FFN1024,SWA128,context2048,untied GPT2 head.
3000 steps,batch64/micro8,seed777,LR1e-3,WD.01,warmup100,softcap30,
hard_prob0→1,tanh_finite,OPT13,offline W&B.100 effective validation batches
every1000 updates; save every500;256-sequence final vocabulary-load evaluation.
train/loss remains CE only. No auxiliary balance objective.

Isolated src/ snapshot and separate CUDA/Triton caches preserve active balance
and cloud GLA jobs. Launcher run_study.py avoids stdlib queue shadowing,
refuses duplicate status creation, waits for balance phase=complete INCLUDING
its final vocabulary evaluation. On predecessor failure it stops without
training; no automatic retry or stopping/restarting prior jobs.

Tests: tests/test_lm_vocab_norm.py. CPU3 passed (norm/init/count and grouped
tied/untied normalization gradient vs F.rms_norm); GPU3 skipped deliberately
until predecessor finishes. Scheduled preflight runs all6, checking real
CUDA embedding/core backward, AdamW and eval for probabilities0/.5/1.
GPU tests must pass before training starts; no GPU success is claimed yet.

Service: dism-vocab-norm-3k-20260910 (user systemd service).
Output:
/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910/vocab_norm

Files: queue-status.json,preflight.log,console.log,metrics.jsonl,latest.pt,
best.pt,offline W&B,vocab-load-final/. Current state: waiting_for_balance.
