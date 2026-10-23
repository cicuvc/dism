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
- untied LM head and document-balanced fused cross-entropy by default
  (the independent `mxfp8_ce=True` option below quantizes their training GEMMs);
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

### Nsight Systems capture

A separate `cuda,nvtx,osrt` capture used the 125M torch-AdamW checkpoint at
step 500 with `DISM_HARD_PLANNER=0`, quadratic hard backward and no training
probes. Artifacts:

- `build/nsys_opt125m_torchadamw_20261022.nsys-rep`
- `build/nsys_opt125m_torchadamw_20261022.sqlite`
- `build/nsys_opt125m_torchadamw_20261022_analysis.json`

The capture contains one optimizer step and all eight accumulation
microbatches. Nsys raises the step scope to 902.3 ms versus an uninstrumented
818.6 ms warmup, so only delta and topology conclusions use this trace.
GPU activity union is 706.4 ms and no-activity time is 195.9 ms. All 26,163
kernel/copy/memset activities are on one CUDA stream. There are 4,322 gaps of
at least 10 us totaling 182.8 ms; the largest individual gap is 0.91 ms. This
is launch/dependency fragmentation, not a planner tail.

Largest repeated inter-kernel gaps:

| Predecessor -> successor | Count | Total | Max |
|---|---:|---:|---:|
| `_add_rmsnorm_output` -> MXFP8 SwiGLU | 96 | 15.16 ms | 0.213 ms |
| BF16 projection GEMM -> value-residual fwd | 138 | 11.80 ms | 0.142 ms |
| multiply -> l2norm fwd | 92 | 9.69 ms | 0.174 ms |
| BF16 projection GEMM -> `_inverse_rms_add` | 96 | 8.68 ms | 0.133 ms |
| BF16 projection GEMM -> causal conv fwd | 194 | 8.18 ms | 0.356 ms |
| GDN chunk output -> cast | 48 | 7.03 ms | 0.199 ms |

Runtime API time is 42.5 ms for `cudaLaunchKernel`, 18.5 ms for
`cuLaunchKernel`, 12.1 ms for `cuLaunchKernelEx`, and 23.9 ms for
`cudaMemcpyAsync`; these enqueue costs overlap GPU execution and are not added
to the device budget. Device copies remain 595 D2D events / 3.22 GB, with only
58 D2H and 139 H2D events. The next useful targets are therefore the new
RMSNorm-to-SwiGLU boundary and forward attention/projection serialization, not
hard-backward fallback traffic.


## Optional MXFP8 fused linear cross entropy (2026-10-23)

`DismConfig.mxfp8_ce=True` selects the imported `fp8gemm` fused CE for
**differentiable training losses**. It is independent of the FFN/projection
switches. Omitted/false retains the existing BF16 loss, including when loading
an older checkpoint. No environment variable silently changes this choice.
The opt-in belongs to `ModelSpec` and therefore its checkpoint fingerprint;
changing it is not an exact resume of an older run.

The ready-to-use config is
`nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32_ce.json`.
It differs from the existing `gggddddddggg` 3B config only in `mxfp8_ce: true`.
Use the usual `--model-spec` option; graph/planner still require the separate
`--cuda-graph --cuda-graph-planner` flags. This integration did not restart or
modify the already-running pretraining job or its configuration/checkpoints.

The implementation is vendored under `flash_mxfp8`, with no runtime import from
`~/cs/project/fp8gemm`. `FUSED_CE_SOURCE.json` records the base commit and exact
source hashes: the snapshot includes the upstream, then-uncommitted softcap
implementation. Local adaptations are package/build paths, a Dynamo boundary,
and the document-sum interface. CUDA device sources remain Torch-free.

### Objective and state contracts

- MXFP8 computes the LM projection, dX and dW; the raw projection output is BF16.
  CUDA computes partial LSE from `softcap * tanh(raw / softcap)` in FP32.
  Triton applies the same transform and its derivative before independent
  row/column MXFP8 quantization of dlogits. The existing `softcap=30` is retained.
- Targets use `ignore_index=-1`. Empty and ignored-only documents contribute
  zero numerator/count/gradients. Document weights are based on valid target
  tokens, not sequence-level hard-sampling flags. Documents may cross CE chunks.
- `balance="document_sum"` returns `(sum_of_document_means, valid_document_count)`.
  The count is nondifferentiable and remains on the GPU. The existing
  `DocumentLossAccumulator` still performs the single global normalization
  across microbatches/ranks, before clipping and optimizer updates.
- The criterion owns no parameters or persistent buffers. LM-head parameter
  identity/dtype, optimizer grouping/schema, RNG counters, annealing state and
  dataloader state are unchanged. Weight quantization runs on each invocation,
  so eager and captured training see optimizer updates without a stale cache.
- Inference, no-grad validation and unreduced evaluation retain BF16 LM-head/CE.
  Only first-order differentiation is supported. Configured model/ vocabulary
  widths must be multiples of 128 for the selected upstream GEMM configs.
- The default upstream chunk/config choices remain 1024 tokens, fwd=8,
  dX=12, dW=12, raster_group=13. The criterion is a Dynamo boundary; concurrent
  use of one criterion on multiple streams is unsupported. Graph setup clears
  its `_states` cache and retains all captured workspace tensors.

### Build without replacing libraries used by a running job

```bash
conda activate blkw
PYTHONPATH=dism_v3/python python -m flash_mxfp8.build --only lse
```

This builds only `_libs/tk_mxfp8_lse/liblse.so`. The ordinary version build and
`python -m flash_mxfp8.build` also include the new library. Do not run a full
in-place rebuild over the libraries mapped by a running training process.
No existing loaded library was rebuilt during this integration.

### Validation and numerical limits

The CUDA graph tests cover planner on/off, changing packed boundaries,
ignored targets, optimizer updates, intervening evaluation, and restored
optimizer/RNG/annealing state. Ten replay/resume combinations passed, including
four using the new CE. A separate metadata test verifies unchanged legacy
model identity, state keys and parameter/no-decay schema.

The imported softcap tests keep the upstream input distribution (`W ~ N(0,.05)`)
and tolerances; integration tests add document-sum accumulation and saved-gradient
reuse. The initial 29 operator cases passed normally and under memcheck
(0 errors). The final 32 operator cases, including H=768/V=32000 BF16
accuracy and the Dynamo boundary, passed under synccheck (0 errors). Together
with the ten graph/resume cases and one identity test, 43 checks passed.
The BF16 accuracy tests use the upstream loss-error <.01 and gradient
relative-L2 <.08 budgets, distinct from the stricter quantized-component
comparisons. No whole-training sanitizer claim is made.

A stronger exploratory input (`W ~ N(0,1/sqrt(H))`, activations doubled on the
second replay, softcap=30) **fails** the upstream `atol=rtol=2e-6` partial-LSE
comparison: the largest difference was `5.44e-5` against PyTorch tanh. The
separate FP64 comparison reproduces this; an instruction-matched Triton probe
agrees with the CUDA LSE within `9.54e-7` on the inspected case. Thus the
`tanh.approx.f32` approximation is visible at larger logits. The original
acceptance tolerance was not widened, and the instruction-matched probe is
only diagnostic, not a replacement oracle. Logs/probes are under ignored
`dism_v3/build/mxfp8_ce_*` and `diagnose_mxfp8_ce.py`.

All eight new CUDA specializations use 168 registers, zero stack and zero
spill loads/stores. SASS inspection found zero CALL instructions in uncapped
kernels, but 16/64/64/256 static CALL sites in the softcap configs 0/8/70/71.
These occur in the generated division slow paths around the upstream
`div.rn.f32`; zero-spill does not mean zero-CALL. No math/launch tuning was done
in this port. The standalone upstream graph test emits PyTorch's existing
AccumulateGrad stream-mismatch warning, while replay checks pass; the warning
has not been suppressed. Throughput was not measured while the pretraining
job shared the GPU.


### Idle-GPU training-step throughput (2026-10-23 afternoon)

Measured on the RTX 5090 after the GPU became free, with the existing
`bench_training_cuda_graph.py`. Both variants use `--mode graph --planner`,
`gggddddddggg`, changing packed boundaries, 8 rows × 2048 tokens × 8 microbatches
(131,072 tokens/optimizer step), `--warmup 8 --steps 32`, one Torch CPU thread,
16 planner workers, and separate fresh processes with the same seed. Only the
model-spec `mxfp8_ce` switch differs. These are synthetic resident batches;
H2D, forward/backward, planner drain, global document normalization, clipping,
and optimizer are timed. Capture/warmup, dataset reading, evaluation, saving,
and training diagnostics are excluded. Clocks were not locked.

| Hard probability | BF16 CE ms/step | MXFP8 CE ms/step | BF16 tokens/s | MXFP8 tokens/s | Throughput gain |
|---|---:|---:|---:|---:|---:|
| 0.000 | 901.1 | 878.2 | 145,451 | 149,256 | +2.6% |
| 0.475 | 862.1 | 827.7 | 152,034 | 158,358 | +4.2% |
| 0.950 | 836.8 | 811.7 | 156,636 | 161,485 | +3.1% |

The measured allocator peak increases by about 175–177 MiB. At p=0 and p≈.475,
both variants take 1536 event routes and zero quadratic routes. At p=.95,
BF16 CE records e=1452/q=84 (5.5% fallback), while MXFP8 CE records e=1312/q=224
(14.6% fallback); the measured end-to-end gain already includes this difference.
This comparison does not isolate planner timing from model trajectory changes
caused by quantization, and does not establish long-run convergence equivalence.
Logs, exact commands and result JSON are in ignored
`dism_v3/build/ce_throughput/`. No checkpoint was loaded or modified, and no
pretraining job was started for this measurement.

### Final RMSNorm → MXFP8 CE handoff (2026-10-23)

With `mxfp8_ce=True`, BF16 training activations now use `RMSNormMXFP8` for
the final norm. It reuses `RMSNormDualQuantPlan`: one reduction kernel produces
inverse RMS, then the fused normalization/quantization kernel writes BF16 y
and both MXFP8 activation layouts. Row scales are written in CE chunk order,
so each chunk receives contiguous views without a repacking kernel. At the
current 16,384-token microbatch and 1,024-token chunk size this removes 16
separate activation-quantization launches per microbatch (128 per optimizer
step). The existing RMSNorm backward kernels are reused.

`MXFP8FusedLinearCrossEntropy.forward(..., quantized_input=chunks)` accepts one
`(row, row_scale, column, column_scale)` tuple per chunk. Metadata validation
does not read GPU data or synchronize; callers must ensure the bytes represent
the supplied BF16 y. CE still returns its STE gradient through y. Each norm
invocation owns its output/inverse buffers, and CE rebinds both forward GEMM
and wgrad descriptors. Multiple forwards before backward, retained backward
and graph replay do not depend on a mutable cached norm plan. The low-level
plan's default output/layout remains unchanged for existing callers.

The module uses the existing `norm.weight`, adds no parameters or checkpoint
buffers, and leaves the model fingerprint, optimizer groups, RNG/annealing and
dataloader state unchanged. Evaluation, logits, no-grad calls and FP32 input
retain the original final norm. No new configuration switch is required;
diagnostic A/B can set `model.mxfp8_ce_norm = None` to use the previous path.

Validation covers 85 distinct operator/model cases: 74 in the combined CE,
MXFP8 training, document-loss and graph/resume suite, one new direct
model/checkpoint-schema comparison, and ten additional autocast norm cases.
Quantized bytes, CE losses and gradients are bitwise equal to direct
quantization **of the same BF16 y**. RMSNorm's different FP32 reduction order
can flip BF16 rounding boundaries; comparisons against PyTorch retain the
existing fused-norm output budget (`rtol=.01, atol=.008`) and require gradient
relative L2 below .001. Model comparisons retain the existing model tolerances.
Both FP32/BF16 norm factors and autocast on/off are covered, including tail
chunks, ignored targets, changing weights/boundaries and multiple forwards.

Memcheck and synccheck each report zero errors on the targeted norm/CE probe;
this is not a whole-training sanitizer claim. The five inspected norm kernels
have zero spills and zero CALLs at M=16384/H=768; their register counts are
14/80/30/72/26 (inverse, quantization, dot, backward, factor reduction).
The standalone graph regression still emits PyTorch's AccumulateGrad stream
warning; it is not suppressed. An exploratory eager-model fixture with
duplicate document boundaries produced NaNs **before final norm in both
implementations**. The model comparison therefore uses nonempty segments;
empty-document CE semantics are tested independently. This existing upstream
eager GDN limitation remains unresolved, with the failure and probe preserved
in `build/norm_ce/eager_empty_document_failure.log` and `model_nan_probe.log`.

The idle-GPU whole-step comparison uses the updated CPU planner, the same
123M `gggddddddggg` config, CUDA Graph, hard probability .95, 16 planner workers,
8 × 2048 tokens × 8 microbatches, eight warmup steps and 48 measured steps.
Four fresh processes run separate/fused/fused/separate, all with e=2304/q=0.
These synthetic resident-batch timings include the same work as the training
benchmark above; GPU clocks are not locked.

| Final norm path | Run times (ms/step) | Pooled ms/step | Pooled tokens/s |
|---|---:|---:|---:|
| Separate norm/quantization | 788.317, 796.970 | 792.643 | 165,361 |
| Fused norm/dual quantization | 787.764, 793.678 | 790.721 | 165,763 |

The observed +0.24% throughput difference is smaller than between-run
variation, so this does not establish a stable whole-step speedup. Peak
allocated memory increases by approximately 4–5 MiB.

`tools/bench_mxfp8_norm_ce.py` additionally measures captured resident-input
components at M=16384/H=768/V=32000, BF16 activations/head and FP32 norm factor.
It matches the model's BF16 autocast around RMSNorm and disables autocast for
CE. Medians from nine alternating rounds of 200 replays are:

| Component | Separate (ms) | Fused (ms) |
|---|---:|---:|
| Norm + dual quantization, forward | 0.13334 | 0.03065 |
| Norm + dual quantization + norm backward | 0.32831 | 0.13066 |
| Norm + CE + backward | 11.67655 | 11.45752 |

The shorter five-round CE-inclusive measurement went the opposite direction
(11.43009 → 11.64483 ms), so the CE-inclusive difference is not a stable
isolated speedup claim. The direct norm/quantization saving is clear, while
the large CE workload and unlocked-clock variation obscure its small effect
on the total. Preliminary `component*.json` runs without autocast are retained
for audit but are **not** representative of the model's norm execution;
`component_amp*.json` are the corrected measurements.

Logs, exact commands,
resource inspection and result JSON are under ignored `dism_v3/build/norm_ce/`.
No checkpoint was loaded/modified and no pretraining job was started.
