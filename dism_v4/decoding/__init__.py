"""CPU SAM planning + single-launch CUDA hard DISM decoding.

Public entry point: NativeDecodeCache. Flatten batch/head to BH. A step takes
int32 labels [3,BH] (key, query, reset-before-current-query), BF16 or FP32
sk/sq [BH,R], and v [BH,DV]; returns FP32 [BH,DV]. Model weights and tau are
fixed over the cache lifetime. No autograd, finite soft gates, beam reordering,
CUDA Graph capture, or cross-stream use. Instantiate a new cache per document.

Tuning:
  rebuild_interval: larger reduces rebuilding, increases raw tail scan cost.
  sample_interval: larger reduces ancestor matrices, increases CPU query work.
  materialize_threshold (>R): larger reduces subtree matrices, increases work.
  rebuild_chunk: larger spends more temporary GPU memory to reduce launches.
  cache_dtype: BF16 raw payload by default, FP32 diagnostic option.

Snapshots use FP32 matrix summaries, not a full N*R*DV tensor. GPU rebuild
uses FP32 Euler prefix sums and ancestor doubling in bounded channel
slabs. Unlike the demo's serial O(N*R*DV) rebuild, this parallel implementation
does O(N*R*DV*log(N)) work when samples exist, to reduce critical path latency;
it keeps O(N*chunk) vector workspace plus O(N*log(N)) temporary integer topology.
Rebuilds are synchronous and can cause latency spikes. Include them in timings.
Ordinary queries are one DISM kernel plus one bulk label D2H and task H2D.
The denominator and its reciprocal are evaluated in FP64 on CPU from scalar
SAM summaries; GPU queries only evaluate signed numerators and multiply by the
uploaded reciprocal. Rebuild geometric/ancestor coefficients are also CPU-made.
DISM_DECODE_FP64_PREFIX=1 builds a separate FP64-prefix diagnostic extension.

Build / validate / benchmark (conda blkw, CUDA_HOME configured):
  python dism_v4/decoding/build.py
  python dism_v4/decoding/test_planner.py
  python dism_v4/decoding/test_native.py
  python dism_v4/decoding/benchmark.py --pattern zipf --n 2048

The existing training forward computes prefill output. prime() initializes the
cache from the full per-document label/sk/v history without returning outputs.

Experimental GPU control plane: HardDismDecoder(..., planner_backend='gpu')
keeps CPU rebuilds but moves ordinary steps entirely to GPU. Default stays CPU.
For CUDA Graph use low-level GpuPlannerCache: step() returns a reused output,
advances device position, and performs no host transfer. Explicitly rebuild()
outside the graph before rebuild_interval new tokens. Rebuild replaces topology
and scratch addresses: discard/recapture old graphs, do not replay them afterward.
check_status() synchronizes and reports capacity/horizon/invalid-reset errors.
The BNHD adapter schedules rebuilds eagerly and is not itself graph compatible.
GPU task metadata uses O(N) storage per head (no N*R*DV persistent expansion).
Per-step coefficients/denominator are FP32 in this variant; CPU scalar snapshot
construction stays FP64, uploaded counts/weights and vector summaries are FP32.
"""
from .runtime import NativeDecodeCache
from .api import HardDismDecoder
from .gpu_runtime import GpuPlannerCache

__all__=['NativeDecodeCache','HardDismDecoder','GpuPlannerCache']
