# Local hybrid prefill length extrapolation

Checkpoint: original49.68M hybrid, step30000, pure hard. BF16 CUDA prefill,
training `tanh_finite`, softcap30; diagnostic NLL uses FP32 torch tanh/CE on
BF16 head logits. No CPU incremental path in this experiment.

RoPE frequencies/theta and SWA128 are unchanged. Only the evaluation instance's
nonpersistent RoPE table capacity and context guard are enlarged to8192.
Strict checkpoint load succeeded; all parameter shapes are unchanged.

256 held-out packed sequences ×8192 =2,097,152 identical target tokens for
each configuration. Each8192-token sequence is independently evaluated in
nonoverlapping512/1024/2048/4096/8192 blocks, resetting all layer state and
position at each block. Thus the aggregate comparison has identical targets;
the first block additionally supplies a same-prefix per-position comparison.
EOS packing follows training, without inserted BOS. All token NLLs finite.

| Prefill/reset block length | Mean NLL | Delta versus2048 |
|---|---:|---:|
|512|3.444778|+0.068630|
|1024|3.400321|+0.024173|
|2048|3.376149|0|
|4096|3.365591|−0.010558|
|8192|3.367980|−0.008169|

On targets at positions2049–8192 only, paired deltas are−0.014040 for4096
and−0.010856 for8192. However, much of this benefit is avoiding reset cold
starts. Excluding the first512 target positions after each2048 boundary,
on targets beyond2048:

-4096 minus reset2048: **+0.003107 ±0.001289** nats.
-8192 minus reset2048: **+0.014113 ±0.002032** nats.

The ± quantities are1.96×standard error of paired per-sequence averages.
This shows a small degradation away from reset boundaries, not evidence
that arbitrarily longer context is unconditionally better. There is no
catastrophic NLL deterioration at4×training length on this sample. It does
not establish long-range retrieval ability: documents are packed, resets
also affect SWA/shortconv, and sequence independence is only approximate.

First2048-position mean NLL differs by~0.0001 between2048 and8192 execution;
shape-dependent numerical/discrete-label differences remain rather than
claiming bitwise causal prefix invariance. The previous100-batch hard eval
NLL3.4484 used a different/larger sample and packing; don't compare that
absolute number as a context effect.

Plots use64-position bins, with within-sequence averages before computing
descriptive95% bands. Raw unbinned per-token losses and position means are
retained. Evaluation took~83seconds excluding checkpoint loading/plotting.
Existing data-stream/position-moment/config checks:5 passed; script compiled;
both generated figures visually inspected.

## Reproduce

```bash
RUN=/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/dism-swa50m-softcap30-20260909-offline
/home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.eval_lm_length \
  --checkpoint "$RUN/latest.pt" --output "$RUN/length-prefill-new" \
  --sequences 256 --micro-batch 4 --lengths 512 1024 2048 4096 8192
```

Initial artifacts in `$RUN/length-prefill-8192/`: `report.json`, `per_token.pt`,
`per_position.png/svg`, `length_generalization.png/svg`. The cold-start-excluded
statistics above were calculated from saved `per_token.pt`; subsequent script
runs also include this statistic directly in the JSON report. Output must be
a new directory; this work did not modify the checkpoint or training code.
