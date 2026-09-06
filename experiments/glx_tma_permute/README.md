# GLX column permutation through the B TMA load

Tested 2026-09-06 on RTX 5090 (sm120), CUDA 13.1.115. The probe verifies the
data path intended for the Dism score GEMM:

```text
global BF16 B -> permuting TMA -> TK swizzled shared tile
              -> TK ldmatrix/warp MMA: C = A B^T
              -> reinterpret FP32 accumulator registers as GLX MMABuffer
```

No accumulator shuffle or global/shared output permutation is performed
between the MMA and the GLX interpretation.

## Permutation

Let `N` be `warp_k_size`, `U=N/8`, and let a ThunderKittens row-layout
accumulator's physical MMA column be

```text
p = 8*c + 2*g + e
```

where `c` is the local column-register index, `g=lane%4`, and `e` selects the
two scalar elements of the packed accumulator value. The same register slot in
GLX represents logical column

```text
q = c + U*g + 4*U*e.
```

The B load therefore materializes physical shared row `p` from global logical
row `q`. A 5D TMA tensor map expresses this without a data-moving kernel:

| TMA dimension | box extent | global row contribution |
|---|---:|---:|
| contiguous D segment | 32 or 64 | 0 |
| `e` | 2 | `4*U*e` |
| `g` | 4 | `U*g` |
| `c` | `U` | `c` plus runtime key-row base |
| D outer segment | `D/segment` | 0, changes D offset |

The runtime base is supplied as the coordinate of the `c` dimension, so the
same descriptor supports arbitrary flattened `[B*H*N,D]` key-block offsets.
The test includes an unaligned row base of 7. D=128 is loaded with two 64-wide
TMA transactions into the two outer shared-memory segments.

This transform is a uniform permutation of C's columns and can be absorbed in
B. It is separate from GLX `roll()`: roll is a row-dependent shear and still
runs in registers before the diagonal scan.

## Results

Inputs use exactly representable small integer BF16 values, so the host dot
product and Tensor Core result can be compared bit-exactly.

| warp_k_size | D | row base | TMA B errors | MMA max abs | result |
|---:|---:|---:|---:|---:|---|
| 32 | 32 | 0 | 0 | 0 | PASS |
| 32 | 64 | 7 | 0 | 0 | PASS |
| 32 | 128 | 0 | 0 | 0 | PASS |
| 64 | 32 | 7 | 0 | 0 | PASS |
| 64 | 64 | 0 | 0 | 0 | PASS |
| 64 | 128 | 7 | 0 | 0 | PASS |

Compute Sanitizer memcheck reported zero errors. The one-warp probe uses
38–64 registers/thread depending on N and D, with zero spill loads/stores and
zero local memory. These counts include TMA setup, operand loads, the full
score GEMM, a diagnostic copy of the loaded B tile, and output materialization;
they are not estimates for the fused Dism kernel.

## Reproduction

Run from the repository root:

```bash
bash experiments/glx_tma_permute/run.sh
```

The script builds in a fresh `/tmp` directory, runs all six cases, runs
memcheck, and writes ptxas/resource logs there. `CUDA_ROOT` and `GLX_ROOT` may
override the dependency paths.
