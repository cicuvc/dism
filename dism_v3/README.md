# flash-dism

CUDA/Triton operators and PyTorch modules for DISM v3: causal retrieval through
matching sequences of learned Q/K symbols, with differentiable vocabulary
interpolation and a soft Q/K readout. Includes forward/backward kernels,
packed-sequence training, and SAM-based full-hard prefill/decoding.

## Operators and Modules

| API | Role | Input |
|---|---|---|
| `voc_dism` | Vocabulary interpolation and DISM with autograd | Fixed or packed Q/K/readout/V tensors |
| `dism_core` | DISM without vocabulary interpolation | Prepared operands, labels and LSE |
| `DismAttention` | Projections, short convolutions, DISM and output gate | `[B,N,width]` |
| `DismSwaAttention` | DISM plus RoPE sliding-window attention | `[B,N,width]` |
| `DismForCausalLM` | Decoder with SwiGLU and fused cross-entropy | Token IDs or embeddings |
| `inference.HardDismPrefill` | CPU SAM planning + Triton prefill | Hard labels and BNHD readout/V |
| `inference.HardDismDecoder` | SAM snapshot/tail decoding | Hard labels and BNHD readout/V |

DISM accumulates matches along diagonals. For matching score `logM`, its causal
recurrence and output are:

```text
W[i,j] = logM[i,j] + softplus(W[i-1,j-1])
O[i]   = sum_{j<=i} exp(W[i,j]) * dot(sq[i], sk[j]) * V[j]
         / (1 + sum_{j<=i} exp(W[i,j]))
```

Hard rows use Q/K label equality; soft rows use vocabulary-interpolated scores.
The denominator includes a zero-value fallback. The readout `sq·sk` affects only
the numerator and may be negative: the result is not necessarily a convex
combination of V. The operator expects already-parameterized `sq/sk`; it adds
neither another SiLU nor an implicit `1/sqrt(D)` or `1/sqrt(R)` scale.

## Installation

Requires Linux, Python 3.10+, CUDA-enabled PyTorch, a clang build supporting
sm120a, and a CUDA toolkit with CCCL. The current build targets **sm120a** and
expects the toolkit at `/usr/local/cuda`. Install PyTorch first, then from this
directory:

```bash
python -m pip install --no-build-isolation .
# Editable development install:
python -m pip install --no-build-isolation -e .
```

The distribution is `flash-dism`; import it as `flash_dism`. Installation builds
`cu_flash_dism` through `build.py`, plus the native SAM extensions using NVCC/C++.
Installed binaries are loaded without compiling; source-checkout inference can
JIT-build into the user's Torch cache. Runtime dependencies
include Triton, Transformers and flash-linear-attention. The SWA module also
needs a compatible FlashAttention installation (`.[swa]`). There is no CPU-only
build mode. See [installation](docs/INSTALL.md) for toolchain requirements,
configuration selection and wheel compatibility.

## Training

```python
import torch
from flash_dism import voc_dism

B, N, H, D, DV, R, V = 1, 256, 4, 64, 64, 32, 512
def vectors(channels):
    return (torch.randn(B, N, H, channels, device="cuda", dtype=torch.bfloat16)
            * 0.1).requires_grad_()

q, k, v, sq, sk = [vectors(c) for c in (D, D, DV, R, R)]
eq = (torch.randn(H, V, D, device="cuda") * 0.1).requires_grad_()
ek = (torch.randn(H, V, D, device="cuda") * 0.1).requires_grad_()
tau = torch.full((H,), 0.5, device="cuda", requires_grad=True)
direction = torch.ones(B, H, device="cuda", dtype=torch.bool)
hard = torch.rand(B, H, N, device="cuda") < 0.5

out = voc_dism(q, k, sq, sk, v, eq, ek, tau,
               direction=direction, hard=hard)  # [B,N,H,DV]
out.float().square().mean().backward()
```

Q/K use BF16 `[B,N,H,D]`, V uses BF16 `[B,N,H,DV]`, and sq/sk use BF16
`[B,N,H,R]`. Supported widths are independently **D,DV ∈ {32,64}** and
**R ∈ {16,32}**. Vocabularies are `[H,V,D]` or shared `[V,D]`; FP32 master
parameters are recommended. `tau` is FP32 `[H]` in natural-log units.
`direction` is bool `[B,H]` (true selects query-LSE normalization), and `hard`
is bool `[B,H,N]`, shared across all keys of a query row.

Fixed N must be positive and **256-aligned**. The kernels return BF16 outputs,
use FP32 accumulation, and replay the forward decisions in backward. Labels
are not differentiable. Tile softplus uses a finite sentinel and an approximate
formula; default numerical acceptance is separate from strict gradient checks.
See [numerical validation](docs/BACKWARD_DIMS.md), rather than assuming FP64
equivalence or an exact derivative through hard label selection.

### Model Integration

```python
import torch
from flash_dism import DismAttention

attn = DismAttention(256, 4, head_dim=64, value_dim=64,
                     readout_dim=32, vocab_size=512, layer_idx=0).cuda().train()
x = torch.randn(1, 256, 256, device="cuda", requires_grad=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    y, _, _ = attn(x, hard_prob=0.5, hard_seed=1234)
y.float().square().mean().backward()
```

The FLA-style interface returns `(output, None, past_key_values)`. Q/K/V use
linear + short convolution + SiLU; sq/sk use linear + SiLU. Codebooks are
initialized with SiLU(Gaussian), without a runtime codebook activation.
Per-head normalization/gating and output projection are inside the module;
residual connections and block normalization belong to the caller.

Keep master parameters in FP32 and use BF16 autocast. Respect `_no_weight_decay`
markers when building optimizer groups. Training defaults to all-soft and
evaluation to all-hard; the caller owns the `hard_prob` schedule. `hard_seed`
is salted by `layer_idx`; explicit flags override sampling. Save any external
schedule and RNG counters when checkpointing training.

`DismSwaAttention(..., window_size=128)` adds a local branch with separate
linear + RMSNorm + RoPE Q/K and shared V. The window includes the current token.
Both branches share the output gate after summation; DISM itself has no RoPE.
GDN variants are available in `flash_dism.hybrid_gdn` and `flash_dism.pure_gdn`.
See [modules](docs/MODULE.md), [language models](docs/MODEL.md), and the repository's
[nanochat training framework](../nanochat/README.md), which is not part of the wheel.

### Packed Sequences

```python
# Reuse attn above. Documents have aligned lengths 256 and 512.
cu = torch.tensor([0, 256, 768], device="cuda", dtype=torch.int32)
packed = torch.randn(1, 768, 256, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    y, _, _ = attn(packed, cu_seqlens=cu, max_seqlen=512,
                   hard_prob=0.5, hard_seed=1234)
```

Packed execution requires B=1 and every boundary, including the total token
count, to be 256-aligned. Convolution and attention state do not cross document
boundaries. Inputs are not automatically padded. `voc_dism` also accepts
`cu_seqlens`, or a reusable `VarlenLayout` through `layout=`.

Modules support `torch.compile(..., dynamic=True)` and internally handle the
dynamic length of cu_seqlens. Other shape/control changes can still specialize
graphs. The GDN branch permits graph breaks; raw varlen metadata uploads do not
support direct CUDA Graph capture. See [compiler integration](docs/DYNAMO_ASSESSMENT.md).

## Memory and Scaling

Let B be batch size, H the number of heads, N the sequence length, V the
codebook size (not the value tensor), D the matching dimension, R the soft
readout dimension, and DV the value dimension. Write `A = D + R + DV`.
The bounds below count arithmetic work and tensor storage, not measured latency;
they exclude model weights, optimizer state, FFNs and the language-model head
unless explicitly stated. Tile sizes are fixed implementation constants.

**CUDA training (forward and backward):**

- Vocabulary interpolation: `O(B H N V D)` work. Its tiled implementation
  does not retain a full `[B,H,N,V]` matrix.
- DISM core: `O(B H N² A)` work, including score/readout GEMMs, scans and
  backward recomputation. Causal masking changes constants, not the order.
- Combined: `O(B H [N V D + N² A])` work and
  `O(B H [N A + N²/32 + N²/16] + H V D)` tensor storage, including
  codebooks/gradients. The quadratic terms represent checkpoint rectangles;
  several buffers of those sizes coexist. This is **not linear-space training**.

For packed documents of lengths `n_s`, replace `B N` by `sum_s n_s` and
`B N²` by `sum_s n_s²`: no cross-document rectangle is allocated.
For the current packed layout, define per-head element counts
`F = sum_s n_s floor((n_s-1)/32)`,
`W = sum_s n_s floor((n_s-1)/16)`, and
`G = sum_s n_s ceil(n_s/32)` (empty documents contribute zero).
`VarlenLayout.checkpoint_bytes(H)` reports `4 H (F + W + 3 G)` bytes for its
checkpoint budget, **not** total peak memory: it excludes input/output tensors,
interpolation, gradients, temporary summary buffers and allocator overhead.
See [packed layout](docs/VARLEN_IMPLEMENTATION.md) and
[backward implementation](docs/BACKWARD.md).

For full attention modules, additionally account for the projections and short
convolutions. With residual width C and convolution width K, their per-token
work is `O(C H A + C² + K H (D + DV))`, independent of history length.
Multiply by `B N` for a full sequence or `B M` for M new tokens.

## Inference

Full-hard inference uses a **suffix automaton (SAM)** over the key symbols.
It replaces dense diagonal scans with matching queries and aggregated vector
readouts. Only Q/K **labels** are discrete: the signed soft readout
`dot(sq, sk)` is still evaluated. No N×N attention matrix is materialized.

The interfaces below consume prepared labels and vectors, not hidden states.
Projections, vocabulary argmax, short-convolution state, SWA/GDN state and the
output gate remain the model caller's responsibility. They do not automatically
replace the module's FLA cache. Use one cache per document and layer, with fixed
weights and tau; no autograd, mixed hard/soft rows, beam reordering or packed
`cu_seqlens` interface. Arbitrary positive sequence lengths are accepted;
the training kernel's 256-token alignment is not required.

### Prefill

`HardDismPrefill` builds a SAM and causal event streams on CPU, parallel across
batch/head (eight workers by default), then executes a combined Triton grid.
Chunks of 16 events use BF16 GEMMs by default, with FP32 running matrix state,
output accumulation and normalization. `mma_precision="tf32x3"` is the
higher-precision diagnostic option. Preparation includes a bulk label D2H copy
and plan upload; it is not CUDA-graph capturable. Output accumulation uses atomic
adds and is not bitwise deterministic.

For one head and N tokens, let E be the number of scalar events and E_C the
number of slots after padding each stream to chunk size C (default 16):

| Part | Time/work | Space |
|---|---|---|
| CPU planning, including current binary-lifting LCA | `O(N log² N)` upper bound | `O(N log N)` |
| Scalar event program | `E = O(N log N)` | `O(E_C)` |
| Triton vector execution | `O(E_C [R DV + C(R+DV)])` | `O(E_C + N(R+DV))` global storage |

For fixed C, R and DV, vector work is `O(N log N)`; the current CPU planner
has the additional logarithmic factor. Each resident stream CTA also needs
an `R×DV` accumulator and chunk scratch, but there is **no persistent
N×R×DV matrix cache**. Multiply work and storage by B×H for equal-length
batches. Parallel workers reduce latency, not total work or memory.

Prefill returns FP32 `[B,N,H,DV]`. It does not initialize the decoding cache;
call `prime` separately with the **complete** prompt history:

```python
from flash_dism.inference import HardDismPrefill, HardDismDecoder

B, N, H, R, DV = 1, 65, 4, 32, 64
iq = torch.randint(0, 16, (B, H, N + 1), device="cuda", dtype=torch.int32)
ik = torch.randint(0, 16, (B, H, N + 1), device="cuda", dtype=torch.int32)
def payload(c):
    return torch.randn(B, N + 1, H, c, device="cuda", dtype=torch.bfloat16) * 0.1
sq, sk, value = payload(R), payload(R), payload(DV)
tau = [0.5] * H

with HardDismPrefill(workers=8) as prefill:
    prompt_out = prefill(iq[..., :N], ik[..., :N],
                         sq[:, :N].contiguous(), sk[:, :N].contiguous(),
                         value[:, :N].contiguous(), tau)

decoder = HardDismDecoder(B, H, R, DV, capacity=4096, tau=tau,
                          planner_backend="cpu")
decoder.prime(iq[..., :N], ik[..., :N], sk[:, :N], value[:, :N])
token_out = decoder.append(iq[..., N:], ik[..., N:],
                           sq[:, N:], sk[:, N:], value[:, N:])
# prompt_out: [1,65,4,64]; token_out: [1,1,4,64], both FP32.
```

### Decoding

`HardDismDecoder` combines frozen SAM snapshots with a short raw tail.
Ordinary CPU-planned steps batch all heads' labels/tasks into one D2H/H2D
exchange and launch one CUDA DISM readout kernel. The alternative
`planner_backend="gpu"` handles ordinary planning and readout in one kernel,
without CPU handshakes; snapshot rebuilds still use CPU planning plus GPU work.

Per head, let T be history length, b the rebuild interval (default 128),
k the ancestor sample interval (default 4R), and t the subtree materialization
threshold (default 5R, required t>R). If a query emits z raw-token tasks and
m matrix tasks, its vector work is
`O(z(R+DV) + m R DV)`. Sampling/materialization bound snapshot traversal work;
the raw tail adds at most b tokens. This is **not** a scan of all T keys on
every ordinary step, nor a worst-case constant-time end-to-end decoder.

The retained state contains raw SK/V history, scalar SAM topology, and
`S = O(T/(t-R) + T/k)` matrix summaries. Its space is
`O(T(R+DV) + S R DV)`, or `O(T(R+DV))` with the default k/t proportional
to R. Raw GPU buffers are preallocated to the requested **capacity**, not just
current T; summaries use FP32 and raw payloads default to BF16.

The current parallel snapshot rebuild costs up to `O(T R DV log T)` vector
work, with `O(T c + T log T)` additional workspace (c = `rebuild_chunk`,
default 32). Rebuilding every b tokens adds amortized
`O(T R DV log T / b)` work per token. Thus **fixed b=128 is not an
asymptotic subquadratic guarantee**: it trades occasional rebuild spikes for
cheap ordinary steps. Larger/adaptive b trades more tail work for fewer
rebuilds; no automatic square-root schedule is implemented. `prime` also
builds these summaries and recovers the query tail; it is not a free handoff
from prefill (CPU tail recovery can cost `O(T b)`).

The low-level `GpuPlannerCache.step` can be captured/replayed in CUDA Graphs
**between rebuilds**; its output buffer is reused. Rebuild outside the graph
before exhausting the horizon, then discard and recapture the graph because
addresses change. The high-level BNHD adapter above is not graph-capturable.
Tau must be finite and nonnegative for decoding. Checkpoint/model integration
must preserve the same hard labels, readout parameterization and tau as training.

See [SAM implementation and validation](docs/SAM_INFERENCE.md) for provenance,
build details and correctness tests.

## Documentation and Tests

- [Installation](docs/INSTALL.md): source builds, editable installs and ABI constraints.
- [Native frontend](docs/NATIVE_FRONTEND.md): low-level inputs, LSE conventions and dispatch.
- [Modules](docs/MODULE.md) / [language models](docs/MODEL.md): model interfaces and loss conventions.
- [Backward validation](docs/BACKWARD_TESTING.md): acceptance tests and stricter diagnostics.
- [Torch oracle](python/flash_dism/reference/dism_v3_ref.py): mathematical forward and explicit backward.

After installing from a source checkout, run:

```bash
python -m pytest tests/test_readme.py tests/test_multi_config.py tests/test_native_frontend.py
python -m pytest tests/test_sam_inference.py
```

Low-level probe tests require `DISM_BUILD_PROBES=1`; FP32 diagnostic instances
require `DISM_ENABLE_FP32=1`. The current validated production build is sm120a;
support for other GPU architectures is not implied by the Torch reference.
