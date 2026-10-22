# Interpolation dQ fused into dKV with TMA reduction

## Status

The CUDA Graph baseline was committed as `8344b7d`. This optional experiment
does not replace the default interpolation backward. On the RTX 5090 (SM120),
the measured implementation gives no throughput benefit and fails the full
model's strict gradient diagnostic. Do not promote it to the training default.

`emb_bwd_split(..., fuse_dq_tma=True)` replaces the separate `_dq` and `_dkv`
launches with `_dkv_dq_tma`. Inside each dKV token-tile iteration, it reuses
scores, probabilities and `ds` to compute `ds @ key * beta`. A tensor descriptor
atomically reduces the FP32 token-gradient tile into a zeroed BNHD output.
Vocabulary gradients retain the existing global FP32 atomics. Shared storage
is used for the TMA output layout; there is no shared FP32 atomic add.

Both token gradients must now be zeroed on every invocation/replay. Autotuning
restores token and vocabulary outputs, preserving contributions from the other
interpolation side. The true-direction LSE-only branch remains specialized.
The formula is unchanged, but vocabulary tiles now reduce token gradients
through global atomics, changing floating-point reduction order.

## Measurements

All timings below enable CUDA Graph replay, exclude compilation/autotuning,
and use the worktree's modules in conda `blkw` (PyTorch 2.13.0+cu130,
Triton 3.7.1). Artifacts are ignored under
`dism_v3/build/interp_tma/`.

| Measurement | Separate dQ/dKV | Fused TMA | Change |
|---|---:|---:|---:|
| Complete interpolation backward, median | 0.77744 ms | 0.78685 ms | +1.21% time |
| Complete training step, mean | 960.096 ms | 961.507 ms | +0.15% time |
| Complete training step, median | 959.227 ms | 960.482 ms | +0.13% time |
| Training tokens/s | 136,520 | 136,319 | -0.15% |

The isolated test uses 16,384 tokens, 12 heads, D=64, vocabulary=512,
true direction, beta=1 and tau=0.5. It measures both interpolation sides,
delta/preparation and output initialization. Seven alternating-order rounds
each contain 100 measured replays after 10 warmup replays. The final result is
`bench_16384_final.json`; earlier candidate sweeps also showed about 1% slowdown.
The fused autotuner tried 32 tile/split/stage/warp configurations per branch;
configurations exceeding the device shared-memory limit were rejected.

The whole-model comparison uses two identically initialized scratch models,
each with its own optimizer and captured graph. It alternates backend order
for 12 measured steps per backend after four warmup steps. The model has
123,473,088 parameters, 12 layers, H=768, FFN=1152, R=32, MXFP8 FFN only,
sequence length 2048, microbatch 8 and accumulation 8. Packed document counts
are 8/16/32/64, repeated twice per optimizer step; the annealing step is fixed
at 11,444. Timing includes input copies, metadata handling, normalization,
clipping and optimizer work, with GPU synchronization at the step boundary.
The fused path was slower in 10 of 12 pairs. The small aggregate difference
does not establish a meaningful regression, but provides no evidence of gain.
See `model_paired.json`. These are synthetic inputs, without checkpoint access.

## Compiler and numerical checks

The selected fused LSE-only kernel uses M=64, K=64, SPLITS=32, 4 warps,
2 stages: 188 registers, zero spills, 49,668 shared bytes. The selected full
branch uses M=64, K=32, SPLITS=16, 4 warps, 2 stages: 164 registers, zero spills,
49,924 shared bytes. Both contain `UTMAREDG` and no `CALL` in SASS. Other sweep
candidates have spills or exceed shared capacity; these are not the selected
kernels. Cubins/PTX/SASS and resource reports accompany the final microbenchmark.

Reusing score computations saves work, but the fused path also introduces
token-output initialization, TMA reductions for each vocabulary tile and an
additional live matrix product in the dKV loop. These are plausible costs
offsetting reuse; the current measurements do not isolate a single cause.
The selected kernels do not spill.

Local tests cover FP64 oracle comparisons, both direction modes, partial
vocabulary tiles, strided inputs, beta/tau variants, token-tail masking with
three splits, autotuned/fixed launches and graph replay with changed/zero
gradients. The isolated production-shape comparison has relative L2 error
below 1e-7 against the original path in the final run.

The combined interpolation regression run passes 108 tests, including 17 new
fusion tests. CUDA 13.1 Compute Sanitizer memcheck and synccheck each pass four
selected tests with zero reported errors: fixed-launch oracle cases 0/1/2
(both direction modes, D32/64, strided and partial-vocabulary inputs) and
autotuned graph replay. These checks do not cover every autotuning candidate
at every model shape.

Targeted racecheck passes six fixed-configuration cases, including graph
replay and zero-gradient replay, with zero errors/hazards. It instruments
`_dkv_dq_tma` only (`--kernel-name kns=_dkv_dq_tma`), covering both directions,
D32/64, vocabulary 37/512 and strided inputs at M=64, K=64, SPLITS=3,
4 warps, 2 stages. The initial unfiltered pytest racecheck was stopped after
its first completed test because instrumentation of reference/autotuning work
was slow; it is not counted as a completed check. Its log is retained, alongside
`racecheck_fixed.py` and the completed targeted log, under the artifact directory.

The full-model check deliberately uses original eager execution as its
reference, enabling fusion only for the graph being checked. The original
split graph passes all three reference steps. The fused graph fails the
unchanged `atol=0.001, rtol=0.03` diagnostic on
`layers.0.attn.o_proj.bias` at two steps:

| Reference step | Maximum absolute error | Relative L2 | Cosine |
|---|---:|---:|---:|
| 0 | 0.0010833740 | 0.0138489 | 0.99990416 |
| 1 | 0.0019531250 | 0.00747297 | 0.99997216 |

Step 2 passes. All three losses match the reference exactly, and RNG/annealing
buffers pass exact checks. The diagnostic still exits with failure after
reporting all steps. Local operator agreement and matching losses do not
override this full-model failure; no tolerances were loosened.

## Reproduction

From the worktree root, activate conda `blkw` and set only this worktree's
packages on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
export DISM_HARD_PLANNER=0 OMP_NUM_THREADS=1
python dism_v3/tools/bench_interp_tma.py \
  --output dism_v3/build/interp_tma/local_new.json
python dism_v3/tools/bench_interp_tma_model.py \
  --model-spec nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32.json \
  --output dism_v3/build/interp_tma/model_new.json
python dism_v3/tools/check_training_cuda_graph.py \
  --model-spec nanochat/configs/sequence_true_125m_valres_mxfp8_ffnonly_slimpajama_3b_h768_h12_i1152_r32.json \
  --interp-backend tma
python -m pytest dism_v3/tests/test_interp_bwd_tma.py \
  dism_v3/tests/test_interp_bwd_split.py dism_v3/tests/test_interp_true.py -q
```

Choose fresh benchmark output paths. Omit `--interp-backend tma` for the
original full-model diagnostic control. `bench_training_cuda_graph.py` also
accepts this explicit backend option for component profiling.
