# Three-arm activation training study (3000 updates)

All three arms completed3000 updates and final vocabulary evaluation. Original
validation and additional paired GPU diagnostics: [ACTIVATION_STUDY_RESULTS.md](ACTIVATION_STUDY_RESULTS.md).
Launch-status notes below are historical.

User-authorized 2026-09-10. Train from scratch, not checkpoint surgery/fine-tuning.
Only baseline, codeword SiLU, and no Q/K SiLU; no centering or norm-matched arm.

## Controlled settings

- All three: original hybrid, 15 layers, residual256, 4 heads, D=DV64,
  Q/K vocabulary512, FFN1024, parallel RoPE SWA128, untied GPT2 embedding/head.
-49,678,876 trainable parameters each. Same raw initial weights (seed777),
  same deterministic FineWeb-Edu stream order, GPT2 tokenizer and EOS packing.
- Effective batch64, microbatch8, context2048, 3000 optimizer updates:
  393,216,000 training tokens/arm, about7.92 tokens/parameter, not20.
- AdamW peak1e-3, wd.01, betas(.9,.95), grad clip1; existing no_decay grouping.
  Warmup100 updates (same fraction as1000/30000), cosine to10% peak at end.
- Hard probability linear update0=0 to update2999=1; not the old30k schedule.
- Softcap30 Triton CE, CUDA embedding/core, tanh_finite, BWD_OPT13;
  hard_bits0, OUTPUT_Q_ALIAS=kv, OUTPUT_TMA0. BF16 autocast, FP32 parameters.
- Every1000 updates,100 effective validation batches of the same held-out prefix,
  current hard_prob and reset eval RNG. Final3000-step validation is pure hard.
- Save latest every500 updates and on clean stop; save best validation checkpoint.
  Raw effective-batch train loss logged every10 updates (no EMA smoothing).
  W&B offline project `dism-activation-study`.

`LMConfig.dism_activation`:

- `baseline`: original swish Q/K shortconv, untransformed codebook.
- `vocab_silu`: retain Q/K swish; effective Q/K codebooks are SiLU(FP32 master),
  then separately expanded/cast to BF16. The effective codebooks are used for
  both addressing and interpolation. PyTorch applies the SiLU gradient chain rule.
- `no_qk_silu`: Q/K shortconv activation=None. V shortconv remains SiLU.

No normalization, extra parameters, tying, changed initialization variance, or
EMA centering. The codeword activation intentionally changes effective initial
norms; it is not the previously examined norm-preserving surgery.
Default remains baseline and old checkpoints load with the default config field.

## Execution and artifacts

Local5090 sequential queue, user service:
`dism-activation-3k-20260910.service`.

Study root:
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910`

Subdirectories: `baseline`, `vocab_silu`, `no_qk_silu`.
Each contains `console.log`, `metrics.jsonl`, `config.json`, `latest.pt`, eventually
`best.pt`, offline W&B data and source hashes. Root `manifest.json` tracks queue
phase, child PID, settings and completed arms; `sources/` archives Python/CUDA files.

Queue first runs3 full-size updates per arm, saving complete model/optimizer/data/RNG
state. After all three pass, resume baseline to3000, then vocab_silu, then no_qk_silu.
Those3 updates are included in3000, not extra or reset. On completion of each arm,
run256-sequence pure-hard vocabulary utilization evaluation into `vocab-load-final`.
No automatic retry, restart, GPU reservation or manipulation of other tasks.
Any child failure, incomplete checkpoint, or changed snapshotted source halts queue.
Do not edit study Python/CUDA sources during the queue without planning a controlled
stop; source consistency checks reject silently running later arms with changed code.

```bash
systemctl --user status dism-activation-3k-20260910 --no-pager
journalctl --user -u dism-activation-3k-20260910 -n 10 --no-pager
```

Runner: `python -m dism_v2.run_activation_study --output NEW_DIRECTORY`.
Existing manifest is deliberately not auto-resumed; inspect individual checkpoints
and use explicit trainer `--resume` if recovery is needed.

## Preflight

84 tests passed across activation, vocabulary sharing, shared architecture and
generation regression suites. New checks establish identical raw initialization
and parameter counts, exact effective-codebook SiLU chain rule, and finite whole-model
gradients at hard_prob0/.5/1 for all three modes. Hard label/codebook gradients remain
zero at probability1 by design. The default architecture/interface is unchanged.

No convergence/validation outcome is claimed before the runs finish. This is a
single-seed early-trend study and cannot establish a robust architecture ranking.

Launch check: all three full-size3-update smoke runs finished with finite loss and
gradients. Baseline resumed successfully and reached update20 (loss9.98499,
step time0.869s). The other two arms remain queued after their smoke checkpoints.
