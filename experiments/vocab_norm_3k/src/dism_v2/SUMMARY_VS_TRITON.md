# Current CUDA summary versus legacy Triton (2026-09-09)

No kernel changes in this comparison. CUDA is the current leader-only producer
with three K slots for D64 and independent warpgroup exit synchronization.

## Configuration and scope

- RTX5090/sm120a, conda blkw, B64/H4/N1024, BF16.
- Legacy tt_dism.py: N_HEADDIM=N_VOCAB=64, original soft preprocessing
  perprocess_kernel_hh, Q_CHUNK_SIZE=D_CHUNK_SIZE=32, tau3/SEPS1e-4.
  Original softmax conversion is performed outside timing; the unchanged
  parallel_attn_fwd wrapper launches summary/passing/output.
- CUDA: D=DV64, embedding vocabulary512, scale1/rtau3, int32 labels,
  tanh_finite, lineinfo enabled, no bitset. Actual CUDA embedding operands are
  prepared before timing. Both fixed directions and hard_prob0/0.5 are tested.
- Both methods: ordinary forward streams,20 untimed warmups,30 CUPTI samples,
  no per-launch synchronization or CUDA Graph. Only the GPU duration of the
  summary launch is extracted; no embedding/passing/output time is included.
  Their surrounding forward launches still affect cache/stream context.
- Three rounds in separate processes, reverse variant order in round2,
  no concurrent GPU test. Clocks are not locked. Tables use the median of the
  three per-round medians (90 samples per variant retained in the JSON).
- The algorithms and numerical approximations differ. This is a comparable
  shape/summary-stage performance comparison, not equivalent-math verification,
  isolated-back-to-back-summary throughput, or end-to-end training throughput.

## Results

| Summary | Median us | Relative throughput (inverse GPU time) |
|---|---:|---:|
| Legacy Triton soft |607.3085|1.00x|
| CUDA soft, q_from_k |196.4790|3.09x|
| CUDA soft, k_from_q |225.6310|2.69x|
| CUDA mixed0.5, q_from_k |220.7185|2.75x|
| CUDA mixed0.5, k_from_q |233.5025|2.60x|

Per-round medians (us):

- Triton:606.9570,607.3085,609.4050.
- CUDA soft q:196.2710,196.4790,197.1830.
- CUDA soft k:225.6630,225.6310,225.2785.
- CUDA mixed q:221.2305,220.7185,220.2390.
- CUDA mixed k:233.6945,233.5025,233.3265.

All benchmark workers checked finite output; this is not a new numerical
accuracy suite. The CUDA implementation's preceding correctness/sanitizer
acceptance is documented in experiments/summary_leader_only/README.md.

## Reproduce

```sh
DISM_TILE_LSE=tanh_finite DISM_LINEINFO=1 \
 /home/cicuvc/miniconda3/envs/blkw/bin/python \
 -m dism_v2.benchmark_summary_comparison --rounds 3 \
 --output dism_v2/benchmarks/summary_vs_triton_sm120a.json
```

Raw samples/configuration: benchmarks/summary_vs_triton_sm120a.json.
