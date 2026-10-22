# MXFP8 training path

This branch adds an opt-in SM120 MXFP8 path for the selected DISM-GDN model.
The default BF16/VALRES path remains unchanged when `mxfp8=False`.

## Replaced operations

MXFP8 is used for:

- SwiGLU gate/up/down projections;
- DISM/GDN/Pure-GDN q/k/v input linear projections before the depthwise
  short convolution;
- DISM `sq_proj` and `sk_proj`.

The following remain BF16:

- token embedding;
- untied LM head and document-balanced fused cross-entropy;
- `o_proj`;
- GDN recurrence;
- DISM suffix-matching core, embedding interpolation and hard planner;
- q/k codebooks and all small head/gate projections.

The MXFP8 modules keep FP32 matrix masters. E4M3 weights and E8M0 block scales
are non-persistent buffers rebuilt after model loading and updated after every
optimizer step.

## Configuration

The selected experiment config is
`nanochat/configs/sequence_true_72m_valres_mxfp8.json`.

The FFN width is rounded from 2576 to 2560 so all reduction and output
dimensions meet the persistent GEMM swizzle constraints. The resulting model
has 71,814,624 parameters.

`mxfp8_optimizer=True` selects `FusedAdamWMXFP8`: FP32 master parameters,
BF16 Adam moments, and fused MXFP8 cache updates. A standard AdamW remains
usable for diagnostics; in that mode caches are refreshed before the next
forward after a parameter update.

## Build

`dism_v3/build.py` builds the MXFP8 libraries by default. Disable that optional
work with `DISM_BUILD_MXFP8=0`.

The standalone build entrypoint is:

```bash
PYTHONPATH=dism_v3/python python -m flash_mxfp8.build
```

Generated libraries go under `dism_v3/python/flash_mxfp8/_libs` and are not
part of Git. `build.py` includes them in platform wheels.

## Compilation boundary

The MXFP8 layers hold ctypes/TMA plans and are marked as Dynamo graph
boundaries. `torch.compile(model, dynamic=True)` remains supported, but plans
are created outside the compiled graph. `fullgraph=True` is not supported for
the MXFP8 modules.

## Initial RTX 5090 results

The fixed resident benchmark uses four 32,768-token microbatches per step,
document-balanced backward, and a real optimizer step.

| Path | Step | Throughput | Peak allocated |
|---|---:|---:|---:|
| VALRES BF16 | ~668 ms | ~196k tok/s | 16.37 GiB |
| MXFP8, shape-tuned configs | ~608 ms | ~215.7k tok/s | 17.58 GiB |

Steady-state forward drops from about 248 ms to 243 ms; backward drops from
about 418 ms to 361 ms. Kernel time in the profiled step drops from 629 ms to
529 ms.

Profiled GPU-kernel share after tuning:

| Family | BF16 | MXFP8 |
|---|---:|---:|
| BF16 GEMM | 240.5 ms (38.2%) | 71.7 ms (13.5%) |
| MXFP8 kernels + activation quantization | - | 164.2 ms (31.0%) |
| DISM | 112.9 ms (17.9%) | 110.5 ms (20.9%) |
| GDN | 50.4 ms (8.0%) | 49.9 ms (9.4%) |
| causal short conv | 18.7 ms (3.0%) | 18.5 ms (3.5%) |
| elementwise/reduction | 77.6 ms (12.3%) | 51.2 ms (9.7%) |
| fused CE | 8.2 ms (1.3%) | 9.1 ms (1.7%) |

Dense sublayer timings at one 32,768-token microbatch:

| Part | BF16 fwd+bwd | MXFP8 fwd+bwd |
|---|---:|---:|
| SwiGLU FFN | 4.86 ms | 2.69 ms |
| Pure-GDN Q/K/V | 1.45 ms | 1.83 ms |
| Hybrid Q/K/V | 1.90 ms | 2.37 ms |
| DISM sq/sk | 0.93 ms | 1.06 ms |
| `o_proj` | 0.75 ms | 0.68 ms |

At width 384, Q/K/V and sq/sk are too small for MXFP8 because activation
quantization, dgrad/wgrad and launch/plan overhead dominate. A standalone
1024-to-1024 projection, representative of the planned 400M width, improves
from about 2.22 ms to 1.85 ms (17%).

## 125M candidate

`nanochat/configs/sequence_true_125m_valres_mxfp8.json` uses:

```text
n_layer=12, n_embd=512, n_head=8, head_dim=64,
value_dim=64, readout_dim=32, intermediate_size=3968
```

It has 125,876,608 parameters. Shape-specific MXFP8 configs are selected for
512-wide QKV/sq/sk and the 512/3968 SwiGLU.

Resident training at the same four 32,768-token microbatches:

| Path | Step | Throughput | Peak allocated |
|---|---:|---:|---:|
| BF16 | ~916 ms | ~143.1k tok/s | 23.43 GiB |
| MXFP8 full | ~752 ms | ~174.3k tok/s | 25.51 GiB |
| MXFP8 FFN-only | ~719 ms | ~182.3k tok/s | 23.76 GiB |

For the selected FFN-only path, forward drops from about 325 ms to 253 ms and
backward from 586 ms to 461 ms, for an overall 27.4% throughput gain. At
182.3k tok/s, 2.5B tokens take about 3.8 hours before dataloader/logging
overhead.

`mxfp8_projections=False` keeps QKV and sq/sk in BF16 while retaining MXFP8
SwiGLU, BF16 `o_proj`, BF16 embedding and BF16 LM head. At 512 width this is
4.6% faster and uses about 1.75 GiB less peak memory than putting all
projections in MXFP8.

## Kernel-fragmentation reductions

The production 125M path now combines four adjacent boundaries without changing
its objective:

- `MXFP8Linear` preserves arbitrary leading input dimensions inside its
  autograd function. The sq/sk callers no longer create token-plane views around
  separate custom-op boundaries.
- The hard-flag Triton sampler also emits the fixed-True direction buffer and
  `softplus(log_sel_tau)`. Its hard flags are bitwise identical to the previous
  head-major sampler, and the custom-op autograd applies the softplus chain rule.
- `FusedAddRMSNormPlan` has output modes `bf16`, `mxfp8`, and `bf16_mxfp8`.
  The last mode is used at the post-attention RMSNorm -> SwiGLU edge: one kernel
  writes normalized BF16 for the residual path plus row-wise and token-major
  column-wise MXFP8 layouts. `MXFP8SwiGLU.forward_quantized` consumes both and
  skips its former direct input quantizer.
- Quadratic packed hard fallback now passes document tasks to
  `hard_backward`; SQ/SK/V/dO are consumed in place. The equal-length
  `torch.stack` copies and output scattering path are gone. The packed tau
  reduction uses a compact per-document table and the shared-LUT stride rather
  than the packed token length.

One 131,072-token CUDA-only step on the same 125M model:

| Metric | Before | After |
|---|---:|---:|
| CUDA kernels | 26,036 | 25,220 |
| D2D/memcpy events | 1,043 | 721 |
| Memcpy bytes | 4.03 GB | 3.22 GB |
| GPU busy union | 707.3 ms | 699.7 ms |
| First-to-last activity | 875.3 ms | 868.3 ms |
| `<20 us` kernels | 19,084 | 18,281 |
| `softplus` kernels | 288 | 192 |
| fixed-direction `ones_like` | 48 | 0 |
| 2 MiB fallback stack copies | 384 | 0 |

An unprofiled ten-step replay of the same step medians at 813.4 ms, vs roughly
830 ms before these changes. Packed quadratic fallback passes memcheck (0
errors) and racecheck (0 hazards). Main CUDA hard-backward spills remain the
documented pre-existing limitation: 8/112/80/192-byte stacks for
R16-DV32/R16-DV64/R32-DV32/R32-DV64; this change does not claim a spill-free
backend.
