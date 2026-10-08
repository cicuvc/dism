# Default batch/head prefill pipeline

`dism_v4.decoding.HardDismPrefill` is the integrated full-hard prefill API.
Default: reusable **8-worker CPU pool**, chunk16, BF16 GEMMs, FP32 state and
output. `workers=1` is the explicit serial fallback; `mma_precision="tf32x3"`
remains the numerical diagnostic. Existing single-head helpers remain available.

```python
from dism_v4.decoding import HardDismPrefill

with HardDismPrefill(device="cuda") as engine:
    # q_label/k_label: int32 [B,H,N], on CPU or the engine CUDA device.
    # tau: [H] or [B,H]; reset: optional bool [B,H,N], same device as labels.
    # sq/sk: contiguous [B,N,H,R], v: contiguous [B,N,H,DV], CUDA FP32/BF16.
    out = engine(q_label, k_label, sq, sk, v, tau, reset=reset)

    # Reusable uploaded plan, when labels/tau/reset remain unchanged:
    prepared = engine.prepare(q_label, k_label, tau, reset=reset)
    out = prepared.execute(sq, sk, v)  # FP32 [B,N,H,DV]
```

Reuse one engine across calls/layers to avoid recreating pools. Close it after
use. Prepared plans own their uploaded metadata and survive engine closure.
They may execute only on their upload CUDA stream; cross-stream calls fail
explicitly. Preparation requires CPU participation and is not graph-capturable.
No backward, finite soft delta, varlen, implicit model monkey-patching or
decoding-cache initialization is added. Raw hard-core prefill is now connected;
vocabulary selection/projections and final decode-cache `prime()` remain caller
responsibilities. Existing model dispatch has not been silently changed.

## Data flow

1. Stack all label/reset tensors and perform one bulk D2H for all batch/heads.
   GPU-resident tau, if supplied, has a separate small D2H.
2. CPU pool independently plans and packs chunks per `(batch,head)`.
3. Merge packed scalar arrays without packing chunks again. Translate row j to
   `(b*N+j)*H+h`, preserving the negative query encoding and per-stream order.
   This directly addresses flattened BNHD vectors; no vector transposes/copies.
4. Release individual native programs/packed arrays. Upload six merged metadata
   arrays, one copy per array, not per head.
5. Zero FP32 output and issue **one Triton vector-kernel launch** for every
   stream across all heads. Heads have disjoint output addresses; stream
   contributions inside each head still use FP32 global atomic add.

There is still a serial NumPy metadata-merge step. All-head planning completes
before upload; no CPU/GPU overlap or streaming waves are claimed. Metadata uses
int32 global rows, so B*H*N must fit signed int32. No per-chunk matrix snapshots
are stored. An entirely no-match batch skips the vector launch and returns zero.

## Verification and timing

`test_prefill_batch.py` passed 12 batch cases (BF16 and TF32x3), N1/65/257,
R16/DV32 and R32/DV64. It compares combined-grid output with individually
executed heads, and the TF32x3 path with native FP64. Covers independent batch
resets/tau, [H]/[B,H] tau, CPU/GPU labels, inactive heads among active heads,
an entirely inactive batch, one-shot/reusable interfaces and rejected
cross-stream execution. Parallel CPU's separate 48-head regression remains.
Combined-grid smoke also passed compute-sanitizer memcheck with zero errors;
the log is `decoding/results/prefill_batch_memcheck.log`.

Initial local RTX5090 end-to-end timing, B2/H8/N4096/R32/DV64, synthetic Zipf512
labels, no reset, tau=ln64, default BF16/C16; median of three warmed runs:

| CPU workers | D2H + planning + packing + merge/upload + GPU ms | Cached GPU ms |
| --- | --- | --- |
| 1 | 166.76 | 0.515 |
| 8 | 44.98 | 0.449 |

About **3.71x** full-pipeline acceleration. Both use the same one-launch GPU
batch grid; small cached GPU variation is timing variation, not CPU worker
acceleration of the kernel. Retained uploaded metadata is 16.80 MB. Excludes
JIT compilation, model projections/vocabulary interpolation and final cache
prime. This is not complete-model throughput.

```bash
OPENBLAS_NUM_THREADS=1 /home/cicuvc/miniconda3/envs/blkw/bin/python dism_v4/decoding/test_prefill_batch.py --output dism_v4/decoding/results/prefill_batch.json
```
