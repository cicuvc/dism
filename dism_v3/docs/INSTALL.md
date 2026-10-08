# flash-dism

Distribution name: `flash-dism`; import: `flash_dism`. The wheel also contains
the native `cu_flash_dism`, `_dism_prefill`, and `dism_decode_cuda` extensions.
Current release: 0.1.0, Linux/sm120a.
No v4 operators or nanochat framework are bundled in this package.

The SAM inference path is included independently of v4 training. Its native
build also requires NVCC, a host C++ compiler and Ninja. `CUDA_HOME` selects the
NVCC toolkit (locally validated with `/usr/local/cuda-13.4`); `MAX_JOBS` defaults
to 2 for these extensions. `TORCH_CUDA_ARCH_LIST` defaults to the visible GPU's
architecture and must be set explicitly on builders without a visible GPU.
These controls do not change the training clang build's sm120a target.
Installed wheels include the native binaries. Direct source-checkout use or
opt-in FP64/timing diagnostics builds in the user's Torch extension cache,
never inside site-packages. Native sources are included for those diagnostics.

## Installation

Install a CUDA-enabled PyTorch in the target environment first. The compiler
requirements are unchanged: CUDA-capable clang supporting sm120a, CUDA toolkit
and CCCL at `/usr/local/cuda`, and fatbinary on PATH or in that toolkit.
Locally use conda `blkw`. From repository root:

```bash
python -m pip install --no-build-isolation ./dism_v3
# Development install (Python edits are live):
python -m pip install --no-build-isolation -e ./dism_v3
```

`--no-build-isolation` builds against the already installed Torch/CUDA stack
instead of downloading a second Torch into an isolated build environment.
Use `--no-deps` only when all runtime dependencies are already installed.
The core requires Torch, Triton, Transformers and flash-linear-attention.
SWA additionally requires FlashAttention (`flash-dism[swa]`). Install native
dependencies suitable for your Torch/CUDA versions before installing offline.

The installer invokes `build.py`, retaining its incremental cache and default
eight R/D/DV configurations with fixed/varlen instances. Existing controls work:

```bash
DISM_BUILD_JOBS=4 python -m pip install --no-build-isolation -e ./dism_v3
DISM_BUILD_CONFIGS='32,64,64' python -m pip install --no-build-isolation ./dism_v3
```

Changing C++/CUDA, build configuration or Torch requires rebuilding/reinstalling,
including in editable mode. FP32 diagnostics and probes remain opt-in through
`DISM_ENABLE_FP32=1` / `DISM_BUILD_PROBES=1`. There is no CPU-only build fallback.

```bash
python -m pip wheel --no-build-isolation --no-deps ./dism_v3 -w /tmp/wheels
python -c 'import flash_dism; print(flash_dism.supported_configs())'
```

Wheels are native CPython/platform wheels, not pure-Python or manylinux wheels.
They are built for the local Torch/CUDA ABI and sm120a; do not assume portability
to another Torch version or GPU architecture. Source distributions contain the
CUDA sources and customized headers, never precompiled binaries or checkpoints.

After installation nanochat needs only its own directory on PYTHONPATH;
adding `dism_v3/python` manually is no longer necessary.

## Packaging validation

Validated locally with Python 3.12, Torch 2.13.0, Triton 3.8.0+git5cd2dd4b,
Transformers 5.15.0 and flash-linear-attention 0.4.2. These are the tested
versions, not a guarantee that every version allowed by metadata works.
Normal and editable installs were checked outside the source directory;
the installed wheel passed the training/varlen README examples and the multi-config
operator suite (93 tests). A source archive was independently rebuilt with
R32/D64/DV64, without reusing the repository's compiled objects.

After adding SAM inference, a fresh wheel installed outside the repository
passed 115 SAM/README/multi-config tests. Both SAM binaries were verified to
load from that wheel with JIT building disabled; the source archive was checked
for the complete native SAM sources. This supersedes the earlier 93-test wheel
check above for package contents.
