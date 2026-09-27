# Sampled soft vocabulary balance, 3000-step training

User authorized2026-09-10. Original baseline parameterization (post-conv Q/K
SiLU, no codebook SiLU, no sharing),49,678,876 parameters. From scratch3000 steps,
same seed777, data, batch64/micro8,context2048,FFN1024,LR1e-3/WD.01,warmup100,
softcap30,hard_prob0→1 and offline W&B as the other local arms.

Objective: CE +0.01*L_balance. For each layer, head and Q/K side, use actual
BF16 token vectors/effective codebooks with FP32 auxiliary GEMM and softmax,
scale1, and compute KL(mean_token p || uniform512). Average losses over heads,
Q/K sides, then layers. No per-token entropy minimization, normalizing vectors,
frequency-EMA correction or primary CUDA kernel changes.

Sampling:128 positions per2048-token sequence at stride16, offset b%16 for
microbatch row b.1024 sampled token vectors per head/side at microbatch8.
No extra RNG consumption. Marginal is per microbatch, NOT across all8 accumulated
microbatches: mean(KL(microbatch marginal)) differs from KL(effective-batch marginal).
Systematic sampling can have positional bias; this is an initial low-overhead test.

Auxiliary gradients stay enabled even for a hard row/core endpoint. Thus vocabulary
and Q/K can still receive auxiliary gradients at hard_prob1 while hard argmax remains
nondifferentiable in the original objective. Validation computes CE only; no auxiliary
loss or batch statistics affect model outputs. train/loss remains unsmoothed CE,
train/objective records augmented loss, train/balance_loss records unweighted KL.
Full-scale weighted penalty is bounded by0.01*ln512 (~0.0624), apart from rounding.

Isolated src/ snapshot avoids modifying active experiments. CPU tests3 passed:
uniform zero KL, explicit sampled formula and gradient equivalence, identical
initial weights/parameter count. GPU tests3 must pass before
training: finite model gradients at0/.5/1, nonzero full-hard auxiliary vocabulary
gradient, CE/objective decomposition, evaluation excludes regularizer.

Original user service `dism-vocab-balance-3k-20260910` waited for
`activation-study-3k-20260910/preconv_silu/queue-status.json` phase=complete,
including its final vocabulary evaluation. Preconv completed training at step3000,
but its launcher named queue.py shadowed Python's standard queue module during
the post-training torch import. Its failed status stopped balance before preflight.

User explicitly authorized direct balance start on2026-09-10. Renamed the balance
launcher to run_study.py to remove the same import collision; --start-now bypasses
the preconv post-evaluation dependency, not the GPU tests. The old failure record
is preserved as queue-status-before-direct-start.json. No preconv retraining.
New service `dism-vocab-balance-direct-3k-20260910` started at20:51 CST:

```bash
/home/cicuvc/miniconda3/envs/blkw/bin/python -u experiments/vocab_balance_3k/run_study.py --start-now
```

Preflight completed:6 passed in212.96s, including all three GPU gradient modes.
Training reached step10: CE10.4068356, unweighted balance0.6648491,
objective10.4134836, gradient norm1.4157492, step time1.015s. These are startup
checks, not evidence of a training-quality improvement. Sequence remains
GPU preflight -> training ->256-sequence final vocabulary load evaluation.
No automatic retry, no stopping/restarting prior work. Separate CUDA/Triton caches.

Output directory:
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910/vocab_balance_001`

queue-status.json,preflight.log,console.log,metrics.jsonl,latest.pt,best.pt,
offline W&B and vocab-load-final/. Every1000 updates runs100 effective validation
batches; latest saved every500 and at completion. Current state is training,
not a completed experiment. W&B is offline.
