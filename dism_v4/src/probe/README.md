# Optional diagnostic sources

These CUDA probes and their Torch host wrappers are not compiled or linked by
default. `varlen_pack`/`varlen_unpack` are historical layout diagnostics; native
varlen production now directly uses global head-major scalar metadata.

From dism_v3, using conda blkw:

```bash
DISM_BUILD_PROBES=1 python build.py
PYTHONPATH=python:.. python -m pytest -q
```

Restore the production-only extension with `DISM_BUILD_PROBES=0 python build.py`.
Probe-dependent tests explicitly skip when disabled; the multi-configuration
oracle/autograd/ragged tests remain active. Dense gradient diagnostics additionally
require `DISM_BACKWARD_DEBUG=1`; FP32 output instances require `DISM_ENABLE_FP32=1`.
Neither is enabled by the probe switch itself. Device CU files remain Torch-free.

Production sources, phase entrypoints and delta stay outside this directory.
