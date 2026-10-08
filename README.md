# DISM v3 + nanochat

Maintained CUDA/Triton operators, Torch models, references, tests and nanochat
pretraining. Historical experiments and v4 remain on `main`.

- `dism_v3/src`, `include`: operators and customized ThunderKittens headers.
- `dism_v3/python/flash_dism`: interpolation, fused kernels, autograd, models
  and Torch cached decoding references.
- `dism_v3/tests`, `src/probe`: regression tests and opt-in diagnostic kernels.
- `dism_v3/docs`: contracts and known numerical limitations.
- `nanochat`: training, evaluation, checkpoint/resume and selected baseline.

## Build

Use conda `blkw` locally. Requires Torch, CUDA-capable clang, CUDA/CCCL under
`/usr/local/cuda`, and fatbinary there or on PATH. Current target is sm120a;
this cleanup adds no Hopper support. Models require FLA, Transformers,
causal-conv1d and (for SWA) FlashAttention. Nothing downloads implicitly.

```bash
cd dism_v3
DISM_BUILD_JOBS=4 python build.py
PYTHONPATH="$PWD/python" python -m pytest tests
```

Default build: R16/32 × D32/64 × DV32/64, fixed and varlen, BF16 output.
`DISM_BUILD_CONFIGS='32,64,64'` selects a focused build.
`DISM_ENABLE_FP32=1` enables diagnostics; `DISM_BUILD_PROBES=1` builds probes
needed by low-level tests. Tests require the extension during collection.
Strict gradient diagnostics remain separate from default relaxed acceptance.
Fixed length and all packed boundaries must be **256-aligned**.

## Training

See [nanochat/README.md](nanochat/README.md). Selected baseline:
`randomness-sequence_true-786m`, 40,467,834 parameters, width384, three GDN
layers then three GDN+DISM layers, postnorm, fixed true direction, single-forward
loss and sequence-level hard sampling capped at .95. It is not the old SWA
hybrid; 786M refers to training tokens, not parameters.

No weights, datasets, W&B runs or compiled binaries are included. v3 is ordinary
tracked source instead of a legacy gitlink. See [CLEANUP.md](CLEANUP.md).
