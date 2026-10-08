# Chunked Triton prefill vector executor

The default batched entry point is now [HardDismPrefill](PREFILL_BATCH.md):
eight CPU workers, direct consumption of packed chunks, one combined GPU grid.
This document describes the underlying single-head/chunk arithmetic.

Opt-in inference prototype in `decoding/prefill_triton.py`; does not replace
model prefill, change decoding caches or modify training. CPU SAM planning
still uses binary-lifting LCA. Chunk packing is C++, not a Python event loop.

## Concrete chunk algebra

Chunk size C counts **events**, not tokens or query rows. An event is either
a key update or a query readout. Give query events decay 1. Let

```
p[t] = sum(log(decay[e]), e <= t within this chunk)
```

All zero-decay key events start a new chunk. Their decay is omitted from p,
and a chunk-reset flag discards M_in. Thus p is always finite and
non-increasing. Unlike global pivot normalization, this remains valid when a
query precedes a much larger later key coefficient. CPU uses FP64 to compute
p, then uploads FP32 chunk-local values.

Let beta[k] be key event weight, gamma[q] normalized query weight. Define
event-sized SQ/SK/V blocks with zeros in non-query/non-key slots. Then:

```
history[q] = gamma[q] * exp(p[q]) * (SQ @ M_in)[q]
pair[q,k] = 1[k precedes q] * gamma[q] * beta[k] * exp(p[q]-p[k])
local = ((SQ @ SK.T) * pair) @ V
M_out = exp(p[last]) * M_in
        + SK.T @ (V * (beta * exp(p[last]-p))[:,None])
```

If reset is set, the incoming-state terms in history/M_out are zero. The
masked pair weights and end-state key weights never require exponentiating
positive log differences. Invalid/padded slots are explicitly masked.
Key/query chronological order already enforces causality; same-position key
events precede their query events.

## GPU mapping and memory

One CTA per stream, eight warps, `num_stages=1`. CTA loops sequentially over
its chunks with FP32 R*DV state. The default four GEMMs use BF16 operands and
FP32 accumulators, including BF16 rounding of weighted intermediates.
The explicit `mma_precision="tf32x3"` diagnostic keeps GEMM operands FP32.
All stream outputs
use global FP32 atomic add. No shared FP32 atomics.

No global per-chunk matrix snapshots or per-event DV output arrays are
allocated. Temporary C*C scores and R*DV state are CTA-local, but ptxas can
spill them to local memory; resource reports must not be confused with a
guarantee of register-only physical storage. Compiler-generated shared memory
serves tensor-core operands and layout exchanges. No explicit additional
shared-memory algorithmic intermediate is introduced in this implementation.

Total explicit device storage is padded scalar chunk metadata plus N*DV output
and input vectors. With fixed C the metadata remains O(N log N), but short
streams have considerable padding overhead. Within-stream chunks are **not**
parallelized in this version. Long-stream critical paths, tiny-stream
inefficiency, and atomic contention remain performance limitations.

## Usage

```python
from dism_v4.decoding import HardPrefillPlan
from dism_v4.decoding.prefill_triton import TritonPrefillPlan

cpu_plan = HardPrefillPlan(q_labels, k_labels, tau, reset=reset)
gpu_plan = TritonPrefillPlan(cpu_plan, device="cuda", chunk_size=16)
out = gpu_plan.execute(sq, sk, v)  # contiguous CUDA [N,R], [N,R], [N,DV]
```

Single sequence/head; no autograd or finite soft delta. FP32/BF16 inputs and
FP32 output. R/DV in [1,128] can be padded for GEMMs, but only the tested shapes
below are validated; large combinations may exceed resources. Constructor
performs native CPU packing and synchronous metadata uploads. `execute` has
no host transfers and launches output zeroing plus one vector kernel. Reuse
the uploaded plan for timing; do not mistake cached-plan timings for one-shot
end-to-end prefill. Inputs must remain finite. Concurrent stream handling and
CUDA Graph use are not validated.

## Validation and performance

```bash
OPENBLAS_NUM_THREADS=1 /home/cicuvc/miniconda3/envs/blkw/bin/python dism_v4/decoding/test_prefill_triton.py --output dism_v4/decoding/results/prefill_triton.json
/usr/local/cuda-13.4/bin/compute-sanitizer --tool memcheck --error-exitcode 1 /home/cicuvc/miniconda3/envs/blkw/bin/python dism_v4/decoding/test_prefill_triton.py --smoke
```

Tests cover C16/32/64, R/DV=(3,5)/(16,32)/(32,64)/(64,32)/(32,128),
FP32/BF16 inputs, reset, no-match, random/repeated labels, tau=0/negative/ln64,
partial chunks, and N4096 repeat chains with tau=.7/ln64/1000. Comparisons use
the native FP64 event executor and independent dense recurrence; BF16 tests
compare against the same rounded input vectors. FP32 tolerance is explicitly
3e-4 absolute/relative, separate from the FP64 planner tests. Atomic reduction
order is nondeterministic.

Initial four-warp/default-stage build exceeded shared capacity in one larger
configuration and had spills even at R=DV32. Eight warps/one stage fits the
tested configurations. The R32/DV64 FP32 baseline uses 171 registers/no spills
at C16, 236/no spills at C32, and 255/136 spills at C64 (Triton-reported values,
not byte counts). Other shapes also have retained spills. Default C16 follows
the measured smaller working set; C32/64 remain explicit experiments.

Machine-readable results include packing/upload time separately from cached
execution (including output zeroing). See `decoding/results/prefill_triton.json`.
No model-level throughput claim, multi-head batching or cache handoff yet.

The primary path passed 60 shape/chunk/input cases plus three N4096 long-chain
stress cases; maximum absolute error against native FP64 was 2.43e-5 in the
recorded run. Memcheck smoke passed with zero errors. C16/R32/DV64 FP32 SASS
has no CALL, 171 registers, zero spills and 8192 bytes shared memory.
Native C++ chunk packing plus upload takes approximately 1.8 ms random /
2.5 ms repeat at N4096/C16, excluding SAM planning. The earlier Python packing
took 38/73 ms. Cached vector execution takes approximately 0.14/0.56 ms;
the repeat C32 result is similar (0.54 ms), so C16 is not universally fastest.

## BF16 GEMM experiment

User-approved default is `mma_precision="bf16"`. This changes more than input
storage: history-readout state operands, weighted local scores and weighted
V operands are also rounded to BF16 before GEMM. The carried state, dot
accumulators, rescale math and global atomic output remain FP32. Explicit
`tf32x3` remains available for numerical comparison; no residual second MMA
is used by the BF16 path. Strict FP32 tests explicitly select `tf32x3`.

```bash
OPENBLAS_NUM_THREADS=1 /home/cicuvc/miniconda3/envs/blkw/bin/python dism_v4/decoding/test_prefill_bf16.py --output dism_v4/decoding/results/prefill_bf16.json
```

96 comparisons: 32 cases at each C16/32/64, N257/4096,
R16/32 × DV32/64, random/repeat labels, resets, tau=0/tiny/.7/ln64,
signed Gaussian or SiLU features. Inputs are BF16 and the FP64 oracle consumes
those same rounded inputs, isolating intermediate GEMM rounding. The TF32x3
path is checked on every case. BF16 metrics are reported, not marked as meeting
the strict FP32 tolerance and not used to relax existing tests.

| C | Worst cosine | Max relative L2 | Norm ratio range | Mean signed projection bias |
| --- | --- | --- | --- | --- |
| 16 | 0.99999769 | 0.2149% | 0.999728–1.000270 | -1.96e-5 |
| 32 | 0.99999776 | 0.2117% | 0.999731–1.000277 | -2.07e-5 |
| 64 | 0.99999785 | 0.2072% | 0.999731–1.000277 | -2.02e-5 |

Signed projection bias is `<O-Oref,Oref>/||Oref||^2`. Per-case biases span
approximately -2.73e-4 to +2.75e-4. Worst individual nonzero-row cosine is
0.9999891. Maximum absolute error is 0.164 on unnormalized synthetic features;
relative metrics should not obscure this scale dependence. These observations
do not establish absence of long-term autoregressive bias or checkpoint quality
regressions. No checkpoint generation has been run with this executor.

Example cached execution (N4096, BF16 source inputs, tau=1e-9, includes zeroing;
excludes planning/transfer):

| Pattern / R / DV | C | BF16 ms | TF32x3 ms | BF16 registers / spills |
| --- | --- | --- | --- | --- |
| random / 32 / 64 | 16 | 0.064 | 0.145 | 115 / 0 |
| random / 32 / 64 | 32 | 0.099 | 0.181 | 183 / 0 |
| random / 32 / 64 | 64 | 0.137 | 0.325 | 230 / 0 |
| repeat / 32 / 32 | 16 | 0.489 | 1.165 | 134 / 0 |
| repeat / 32 / 32 | 32 | 0.276 | 0.720 | 172 / 0 |
| repeat / 32 / 32 | 64 | 0.245 | 0.707 | 234 / 0 |

Chunk choice is workload dependent; no automatic label-dependent dispatch is
introduced. Default is C16/BF16; the C16/TF32x3 bring-up baseline is retained
as an explicit diagnostic. BF16 smoke also checks the constructor default.
BF16 C16/32/64 smoke passed both memcheck and synccheck with zero errors;
logs are retained in `decoding/results/prefill_bf16_*check.log`.
