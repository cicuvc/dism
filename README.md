# DISM: Discrete Attention via Suffix Matching

DISM is an efficient alternative to attention for long-range dependencies,
based on discrete suffix matching.

Inspired by ROSA.

## Installation

Install CUDA-enabled PyTorch and the required CUDA/C++ toolchain first, then
install the `flash-dism` package from this repository:

```bash
python -m pip install --no-build-isolation ./dism_v3
```

For an editable development installation:

```bash
python -m pip install --no-build-isolation -e ./dism_v3
```

The Python import name is `flash_dism`. See the
[installation guide](dism_v3/docs/INSTALL.md) for build requirements and the
[operator and module guide](dism_v3/README.md) for usage.
