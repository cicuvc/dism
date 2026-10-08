# Initial prefill throughput

RTX 5090, 450 W power limit; B=H=1, R32/DV64, BF16 inputs/GEMMs,
FP32 accumulation/atomic output, chunk16, tau=ln64, no resets.
No atomic redesign was applied. CPU planner is single-threaded.

Labels are synthetic: independent uniform vocab512, independent Zipf vocab512
(exponent 1.2), or all-repeat. These are workload brackets, not checkpoint label
distributions. Vectors start resident on GPU. Compilation and first-use warmup
are excluded. Full pipeline medians use five fresh plans on the same labels:
GPU label D2H -> C++ SAM/stream planning -> native chunk packing/upload -> GPU
output initialization/vector execution -> synchronization. Final plan teardown
is outside timing. Model projections, vocabulary interpolation, cache prime,
multi-head orchestration and whole-model inference are not included.

Cached vector time uses Triton do_bench, includes output initialization, excludes
planning/transfer, and is not a full prefill throughput figure. CPU/pipeline
timers are wall clock; component medians need not sum to the median total.

| Labels | N | CPU plan ms | Pack/upload ms | Cached GPU ms | Pipeline ms | Pipeline tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| uniform512 | 1024 | 1.486 | 0.127 | 0.0114 | 1.693 | 604,976 |
| uniform512 | 4096 | 5.618 | 0.334 | 0.0189 | 6.057 | 676,282 |
| uniform512 | 16384 | 23.026 | 1.101 | 0.0582 | 24.298 | 674,308 |
| uniform512 | 65536 | 130.145 | 9.659 | 0.3483 | 140.340 | 466,981 |
| zipf512 | 1024 | 1.885 | 0.267 | 0.0279 | 2.339 | 437,883 |
| zipf512 | 4096 | 8.496 | 0.806 | 0.0743 | 9.472 | 432,451 |
| zipf512 | 16384 | 38.389 | 3.717 | 0.2869 | 42.530 | 385,235 |
| zipf512 | 65536 | 191.775 | 22.288 | 0.9871 | 215.251 | 304,464 |
| repeat | 1024 | 2.592 | 0.303 | 0.0655 | 3.054 | 335,275 |
| repeat | 4096 | 11.310 | 1.196 | 0.2415 | 12.879 | 318,028 |
| repeat | 16384 | 50.833 | 4.736 | 0.9448 | 56.526 | 289,850 |
| repeat | 65536 | 238.255 | 27.878 | 3.7621 | 269.619 | 243,069 |

All outputs were finite. Numerical correctness relies on the separate oracle
suite; this benchmark does not compare dense oracles at 64K. The measured full
pipeline is host-bound; atomic removal alone cannot remove that bottleneck.
No claim is made that binary-lifting LCA specifically dominates CPU time.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 /home/cicuvc/miniconda3/envs/blkw/bin/python dism_v4/decoding/benchmark_prefill.py --output dism_v4/decoding/results/prefill_throughput.json
```

JSON includes raw samples, GPU metadata/program sizes and both throughput
denominators. This measures the existing implementation, without switching
defaults, rewriting the atomic path or starting a training/evaluation run.
