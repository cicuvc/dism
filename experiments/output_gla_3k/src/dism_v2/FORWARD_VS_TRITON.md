# Three forward kernels versus legacy Triton (2026-09-09)

CUDA baseline is commit82f9305, containing the accepted persistent summary
and newly verified persistent OUTPUT. No kernel changes in this comparison.

## Configuration / measurement

- RTX5090, sm120a, Torch2.13+cu130, CUDA13.1 build, conda blkw.
- Both:B64/H4/N1024, BF16, value dimension64, tau3.
- CUDA:D64/DV64, embedding vocabulary512, scale1, actual CUDA embedding
  outputs, int32 labels, no bitset, saved vertical/horizontal backward boundaries.
  tanh_finite with lineinfo, both directions and hard_prob0/0.5.
- Legacy tt_dism.py:N_HEADDIM=N_VOCAB64, original soft path, Q/D chunk32,
  SEPS1e-4, random FP32 beta. Original softmax input conversion is untimed.
- Each worker:20 warmups and30 profiled forward calls on an ordinary stream,
  no per-launch synchronization or CUDA Graph. Three independent processes
  per variant, reverse variant order in round2, no concurrent GPU benchmark,
  unlocked clocks/cache. Both wrappers execute their original allocation and
  initialization, but only the three named kernel GPU durations are counted.
- CUDA embedding and Triton input softmax are outside timing. No backward,
  optimizer, CPU launch gaps or buffer-initialization kernels are included.
  This is not end-to-end training throughput or isolated continuous-fill timing.
- Different score definitions, fallback and approximations: comparable-shape
  performance only, not equivalent mathematical work or numerical validation.
  All workers assert finite output and the exact summary/passing/output launch
  sequence; prior CUDA numerical/sanitizer evidence is in OUTPUT_OPTIMIZATION.md.

## Results

Units:us. Each entry is median of the three round medians. For total, first
sum the three GPU durations belonging to each invocation, then take medians;
it need not equal the sum of three independently computed stage medians.

| Variant | Summary | Passing | Output | Per-call total | Relative total throughput |
|---|---:|---:|---:|---:|---:|
| Triton soft |617.293|55.199|1129.066|1933.013|1.00x|
| CUDA soft q_from_k |199.023|31.376|472.413|704.221|2.74x|
| CUDA soft k_from_q |227.487|31.872|470.925|730.476|2.65x|
| CUDA mixed0.5 q_from_k |223.023|31.552|454.957|709.979|2.72x|
| CUDA mixed0.5 k_from_q |234.734|32.080|459.021|725.660|2.66x|

Stage-relative throughput:summary2.63–3.10x, passing1.72–1.76x,
output2.39–2.48x. All numbers describe these launches, not the embedding cost.

There is visible run/sample variation, particularly in Triton. Its per-round
total medians are1944.710/1933.013/1799.670us; do not present2.74x as a locked-
clock invariant. No slow samples were removed. The pooled90-sample means are:

| Variant | Mean total us | Relative throughput from mean time |
|---|---:|---:|
| Triton |1906.803|1.00x|
| CUDA soft q |728.261|2.62x|
| CUDA soft k |744.351|2.56x|
| CUDA mixed q |739.580|2.58x|
| CUDA mixed k |752.921|2.53x|

The mean-time comparison therefore supports approximately2.5–2.6x aggregate
forward-kernel throughput for this measurement protocol. Individual-stage
medians suggest about2.5x when their typical times are simply summed; that is
a separate statistic from the per-invocation total median above. No cause
for the intermittent longer launches was established by this timing run.

## Reproduce

```bash
DISM_TILE_LSE=tanh_finite DISM_LINEINFO=1 /home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.benchmark_forward_comparison --rounds 3 --output dism_v2/benchmarks/forward_vs_triton_sm120a.json
```

All stage/per-call samples, round medians, pooled means and version metadata:
benchmarks/forward_vs_triton_sm120a.json. The script does not trim outliers.
