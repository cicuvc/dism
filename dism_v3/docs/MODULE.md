# Torch module

`flash_dism.DismAttention` accepts `[B,N,width]` and returns
`(output [B,N,width], None, past_key_values)`, following FLA GatedDeltaNet.
It contains no residual addition or input PreNorm; the enclosing block owns them.
Training uses the CUDA kernels; cached inference uses the Torch reference.

```python
import torch
from flash_dism import DismAttention

attention = DismAttention(
    width=256, heads=4, head_dim=64, value_dim=64,
    readout_dim=32, vocab_size=512,
).cuda()  # retain FP32 master parameters
x = torch.randn(2, 256, 256, device="cuda", requires_grad=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    y, _, _ = attention(x, hard_prob=0.5)
y.float().square().mean().backward()
```

## Parameterization

- Q/K/V: local `kernels.conv1d.CausalShortConv1d` fuses bias-free linear
  projection, causal depthwise shortconv (default size 4), and SiLU. Projection
  weights are `q_conv.weight` (and K/V equivalents), convolution coefficients
  are `q_conv.conv_weight[lag,channel]`, where lag 0 is the current token.
  This replaces the initial module's separate Linear/FLA ShortConvolution;
  old module state_dict keys are not automatically migrated.
- SQ/SK: independent bias-free Linear + SiLU, without RMSNorm, RoPE or convolution.
  Optional `readout_l2_norm=True` adds per-head unit L2 normalization after SiLU,
  along R, with eps1e-6 and FP32 reduction for BF16. DefaultFalse. No learnable
  norm gain or sqrt(R) rescaling. Available in DismConfig and cached decoding.
  `qknorm_eps` and `rope_theta` remain constructor options for hybrid SWA only.
  No additional 1/sqrt(R) scaling; their signed dot product is the v3 numerator
  readout. Matching Q/K, vocabulary selection, W and denominator are unchanged.
  Linear weight names/shapes remain, but old RMS parameters are removed;
  this is not a function-preserving migration from RMSNorm/RoPE checkpoints.
- Each head has separate FP32 Q/K vocabularies initialized as `SiLU(N(0,1))`.
  Parameters are the effective codewords: no runtime vocabulary activation.
  Interpolation converts them to BF16. Initial effective values match the
  former Gaussian-parameter-plus-SiLU scheme, but gradients no longer include
  its SiLU Jacobian and trained values are no longer restricted to its range.
  Token Q/K shortconv remains SiLU; SQ/SK use the parameterization above.
  Old raw-Gaussian vocabulary checkpoints require an explicit one-time SiLU
  conversion to preserve their forward function; no automatic conversion is
  performed, and old optimizer states are not equivalent under this change.
- `rtau = softplus(log_sel_tau)` with v2's uniform `[-4,4]` initialization.
  It is not clamped. `q_vocab`, `k_vocab`, `log_sel_tau` carry the v2
  `_no_weight_decay=True` marker; the optimizer must honor it. Biases and norm
  parameters should also be excluded according to the training policy.
- Output: gated RMSNorm on `[B,N,H,DV]` independently along each head's DV
  dimension, then concatenate heads and apply the output projection. The gate
  has a separate value per head/channel; RMS affine weight `[DV]` is shared
  across heads, as in FLA GatedDeltaNet. Low-rank linear gate, SiLU in the norm.
  Plain DismAttention has no SWA. The old concatenated-head normalization is
  intentionally removed; its norm weight shape is not checkpoint-compatible.

D/DV independently support 32/64; R supports 16/32. Width need not equal H*DV.
Use BF16 autocast, rather than casting the whole module and its master
vocabularies to BF16. FLA still provides FusedRMSNormGated. Pure DISM has no RoPE;
the optional hybrid SWA branch retains it.
The supplied CE kernels belong to a training loss, not this attention module.

## Randomness and packed documents

For `torch.compile` training, see [DYNAMO_ASSESSMENT.md](DYNAMO_ASSESSMENT.md).
Packed graphs require `torch._dynamo.config.capture_dynamic_output_shape_ops=True`
for exact-sized checkpoints and explicit dynamic marking of cu_seqlens axis0
outside the compiled call. A fixed max_seqlen upper bound avoids specializing
on document lengths; compiled default is total tokens. Changing document count
does not require recompilation under this recipe, including one/empty documents.
CUDA Graph capture and dynamically annealed Python-float probabilities are
not covered by this contract.
Alternatively, on the tested Torch version, set
`torch.fx.experimental._config.use_duck_shape=False` for the compilation and
execution scope to avoid per-input marking. This private configuration was
tested with single-document and head-count-collision first inputs; plain
`dynamic=True` alone still recompiles in those cases. The library does not
change this setting globally. See the assessment for a scoped example.

Pass `hard_prob` explicitly from the optimizer-step schedule; forward has no
internal step counter. If omitted, training defaults to soft and evaluation
defaults to hard. Evaluation still samples direction unless explicitly given.
A CUDA `generator` controls the convenience sampler: one global direction per
call, plus `[B,H]` sequence decisions from local `triton_rand_bool`. A scalar int64
seed is drawn with Torch using the supplied generator; the Triton kernel writes
boolean flags directly, avoiding a full-sized float random tensor. Probability
endpoints consume no row RNG. Masks differ from the previous Torch sampler for
the same generator state; resetting a generator replays the new sampler.
Flags remain global-memory tensors matching the v3 core API; this is not
warp-generated RNG inside the DISM kernels.

Pass `hard_seed=123` (or a scalar int32/int64 tensor on the input device) through
forward kwargs to control row sampling explicitly. Priority is explicit `hard`
> `hard_seed` > seed drawn from `generator`/global RNG. Probability endpoints
ignore the seed and do not consume row RNG. Explicit seeds do not advance the
caller's generator for row sampling, but direction still consumes RNG unless
explicitly supplied or reused from cache. The seed is per call, not a cached
stream offset: repeat it with the same shape to replay, or vary it per step.
For padding masks, sampling indexes the unpadded packed layout.
An explicit hard_seed is automatically XORed with a deterministic SplitMix64
salt of layer_idx (including layer0), so all layers may receive the same base
seed. Give each layer a distinct nonnegative layer_idx. None leaves the seed
unsalted for standalone use. Integer and scalar tensor seeds mix identically;
CUDA tensor salting stays on-device, without consuming RNG or a host read.
This applies to plain/hybrid, fixed/varlen and decoding; explicit hard masks
and probability endpoints bypass salting. Change the base seed between steps
if new masks are desired. Direction sampling is independent and unchanged.

Torch decoding also accepts hard_seed, using a private Torch generator; a device
seed tensor incurs a host read in this reference path. Torch and Triton RNGs
are different, so the same seed does not promise identical masks across these
backends or different chunkings. Use explicit hard flags for such comparisons.

For deterministic comparisons pass `direction` (bool `[B,H]`) and `hard`
(bool `[B,H]` for CUDA); explicit hard flags take precedence over hard_prob. These
flags are reused in backward, with no new sampling.

The module automatically routes soft rows through CUDA O(N²) forward/backward
and hard rows through integer CUDA O(N²) forward plus asynchronous CPU planning
and event-stream backward. Eager and fullgraph Inductor use the same numerical
paths. No manual planner argument, parameter, persistent buffer or checkpoint
migration is required. No-grad hard evaluation runs the dedicated CUDA forward
without starting a planner. The low-level soft-only CUDA helpers remain
separate building blocks. See [runtime ownership and limits](NATIVE_TRAINING_PLANNER.md#default-training-operator-and-module-integration).
Packed CUDA accepts `[training_rows,H]` flags for equal-length original rows,
not per-document flags. Unequal attention-mask-unpadded rows with explicit
hard flags are rejected. Torch/reference decoding retains its token-mask API.

`cu_seqlens=` selects packed batch-one execution. All boundaries and total N
must be 256-aligned. The same boundaries are passed to convolution so that
local state cannot leak between documents. A native layout is constructed per
call. CPU or same-device CUDA int32 boundaries are accepted. Validated lengths
provide the convolution's maximum length without another GPU-to-CPU read per
branch. Optional `max_seqlen` must be at least the longest document; smaller
values are rejected rather than silently leaving output tokens unwritten.
Empty documents within a nonempty pack are supported.
SQ/SK are token-local Linear + SiLU; they require no packed position tables.
Empty total packs are rejected by the module (the low-level API remains
available). No input padding, vocabulary sharing or checkpoint migration is
introduced.

```python
# x: [1,768,width], documents of length 256 and 512, plus an empty document
cu = torch.tensor([0, 256, 256, 768], device="cuda", dtype=torch.int32)
with torch.autocast("cuda", dtype=torch.bfloat16):
    y, _, _ = attention(x, hard_prob=0.0, cu_seqlens=cu)
```

## FLA forward and decoding

```python
forward(hidden_states, attention_mask=None, past_key_values=None,
        use_cache=False, output_attentions=False, **kwargs)
```

DISM-specific arguments (`hard_prob`, `hard`, `hard_seed`, `direction`, `generator`,
`cu_seqlens`, `max_seqlen`) are keywords. Attention matrices are not materialized;
`output_attentions=True` still returns None, as GatedDeltaNet does.

```python
from flash_dism import DismAttention, DismCache

layer = DismAttention(256, 4, layer_idx=0).cuda().eval()
cache = DismCache()  # one shared object for all layers, each with unique layer_idx
prefix = torch.randn(1, 37, 256, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    y, _, cache = layer(prefix, past_key_values=cache, use_cache=True)
    token = torch.randn(1, 1, 256, device="cuda")
    y_next, _, cache = layer(token, past_key_values=cache, use_cache=True)
```

Only NEW tokens are supplied when continuing. All cached calls require eval().
`use_cache=True` selects Torch prefill even for aligned lengths. Eval calls with
nonaligned lengths or CPU inputs also use Torch. Aligned CUDA calls without a
cache retain the original CUDA forward. No implicit Cache is allocated: with
`past_key_values=None`, the third return remains None even when use_cache=True.
A supplied cache requires constructor `layer_idx`. With use_cache=False an
existing cache is read but not updated. Reset by supplying a fresh cache.

The cache stores three pre-convolution projection histories (`conv_state`) and
a tensor tuple of K/SK/V, raw K LSE, key labels, last W row, direction and tau
(`recurrent_state`). All entries are batch-major for beam reordering. Direction
is sampled once and reused; supplying a different direction or changing tau is
rejected. All model weights must remain fixed during the cache lifetime; no
automatic weight fingerprint/migration is performed.
Cached SK contains the SiLU-activated linear features, without rotary positions.

Torch inference runs projections, convolution, per-head vocabulary interpolation,
DISM recurrence and gated output in FP32 (FP64 for a double model), with autocast
disabled internally. It returns the input dtype, or the enclosing autocast dtype.
It uses exact softplus/infinity semantics, not the CUDA tile approximation.
Consequently it is a numerical reference, not bitwise-identical CUDA decoding;
hard labels can differ near BF16 argmax ties. Default is full hard; soft/mixed
diagnostics are available. Mixed sampling uses the Torch reference RNG, not the
training Triton sampler; pass explicit flags for chunk-independent comparisons.
Space is linear in history; each new token scans its history, so full generation
is quadratic. This is not the subquadratic hard-label SAM implementation.

Training `attention_mask` supports 2D 0/1 padding masks via unpadding/re-padding;
every resulting document must still have length divisible by256. Outputs at
padding positions are zero. It cannot be combined with explicit cu_seqlens.
Explicit direction must agree across batch when packing. Cached Torch inference
currently supports only equal-length, unpadded batches: all-one masks are accepted
(the newest N columns are consumed); padding masks and cu_seqlens are explicitly
rejected instead of leaking state across documents. Use independent caches for
unequal-length documents. Empty total inputs remain unsupported.

The installed FLA `Cache` fails on this environment's Transformers API because
`FLALayer` lacks `get_max_length`. `DismCache` subclasses FLA Cache and supplies a
local compatible layer class plus tuple-aware beam reordering, without modifying
site-packages. Ordinary functional FLA caches are accepted by the same forward
protocol; cache adaptation is not hidden inside the attention layer.

## DISM + SWA variant

`DismSwaAttention` is a separately exported subclass, retaining the same FLA
forward interface and DISM parameterization; the plain DismAttention remains
available. It adds independent `swa_q_proj` and `swa_k_proj` fused
Linear + per-head RMSNorm + RoPE branches. It uses **the exact same V tensor**
from the one `v_conv` call for both branches: no separate V projection or conv.
SWA head dimension equals value_dim (32 or64), and its head count equals DISM's.
This matches FlashAttention2's Q/K/V dimension requirements without padding V.

```python
from flash_dism import DismSwaAttention
layer = DismSwaAttention(256, 4, value_dim=64, window_size=128, layer_idx=0).cuda()
```

Window size counts the current token:128 means keys `[i-127,i]`. Fixed CUDA
uses `flash_attn_func`, packed CUDA uses `flash_attn_varlen_func`, both causal,
dropout0, scale1/sqrt(DV), window_size=(127,0). Document-local rotary positions
restart on both SWA Q/K branches. `rope_theta`/`qknorm_eps` affect SWA only;
DISM SQ/SK have no normalization or positional rotation.

Raw SWA and DISM outputs are added, then pass through the single shared
per-head gated RMSNorm and output projection. Both losses backpropagate to V.
FlashAttention is imported only when the CUDA SWA branch executes.

Cached/CPU inference retains the Torch reference path for both branches.
An extra ninth recurrent-state tensor caches at most window_size-1 rotated SWA
keys; V reuses DISM's full history without an additional persistent copy. New
SWA queries/keys use the DISM history length as rotary offset. Multi-token
continuations mask future keys and enforce the window for each query. Keep
window_size and all model parameters fixed while using a cache. Plain/hybrid
caches are intentionally not interchangeable. Existing padded/packed decoding
limitations still apply. No training run is started by this implementation.

## Validation

`PYTHONPATH=python:.. python -m pytest -q tests/test_module.py tests/test_module_decode.py tests/test_decode_reference.py`

Coverage includes all eight D/DV/R configurations with finite nonzero gradients
for every parameter, mixed-head directions and row decisions, packed-versus-
separate unequal-length documents for outputs, input and parameter gradients,
empty documents, exact cross-document output/gradient isolation, generator
replay, and unaligned-length/max-length rejection. Core oracle tests remain
separate and unchanged.

## Opt-in quantile balancing (fixed true direction)

For the optimization and soft/hard derivation, see
[QUANTILE_BALANCING.md](QUANTILE_BALANCING.md).

Set `qb_q_strength` and `qb_k_strength` independently in the nanochat ModelSpec
`config` (or `DismConfig` / `DismAttention`). Each is in `[0,1]`: zero disables
that side; both zero retain the original path and state-dict schema. Example:

```json
{
  "direction_policy": "true",
  "direction_antithetic": false,
  "qb_q_strength": 0.25,
  "qb_k_strength": 0.5,
  "qb_iterations": 4,
  "qb_update_interval": 16,
  "qb_samples": 2048,
  "qb_update_rate": 0.1
}
```

The other four values shown are defaults. Prices are per-layer/head/code,
non-parameter FP32 buffers in logit units: `beta * dot - strength * price`.
`qb_update_interval` counts gradient-enabled training forwards/microbatches,
not optimizer steps. After each interval, detached current scores warm-start
`qb_iterations` quantile coordinate updates; `qb_update_rate` blends the fitted
prices into the saved prices. Each rank samples `qb_samples` evenly spaced
valid positions (repeats when fewer positions are available), and all ranks
concatenate their samples before fitting. Nanochat excludes `targets < 0`;
callers without targets use all input positions, or may pass `qb_valid_mask`.
Empty ranks contribute no samples. Finite iterations and ties do not guarantee
exact equal argmax counts, and equal hard counts need not imply equal soft mass.

A forward uses an owned snapshot of the **previous** prices for both soft and
hard selection. Updates affect subsequent forwards, not the current sequence;
backward does not refit or update prices. `eval()` freezes all QB state.
The soft true-direction score includes the required extra key bias
`-E_{priced K}[Q price]`, including its derivative through K probabilities.
Tau is absorbed in Q LSE once; the key bias does not absorb another tau.
See [SOFT_HARD_RELAXATION.md §53](SOFT_HARD_RELAXATION.md#53-fixed-true-qb-训练实现2026-10-16).

Enabled QB requires `direction_policy="true"`, no antithetic direction, and no
`direction_flip`. The low-level attention caller must explicitly provide
`direction_true=True`; supplied direction tensors must also be all true.
Unsupported modes fail rather than fall back to unpriced execution. This first
implementation supports aligned fixed/packed CUDA training and evaluation,
but rejects cached/unaligned Torch decoding, categorical REINFORCE, and
`gqa_query_channels > 1`. Shared-KV groups retain identical K prices within a
group. The QB interpolation/update intentionally forms an eager graph break;
use `torch.compile(..., fullgraph=False)` as in the hybrid training loop.
Interpolation uses bounded token chunks and recomputes probabilities in
backward. It is not yet a fused QB interpolation kernel or a throughput claim.
Activation-checkpoint recomputation of the stateful QB forward is unsupported;
the current nanochat training path does not use it.

QB checkpoints persist `attn.qb.{q_price,k_price,forward_count,update_count}`.
Strict resume must restore these buffers and the existing RNG, schedule,
optimizer and dataloader states. An old checkpoint has no QB state; enabling
QB on it is an explicit model-spec/state migration, **not** an automatic exact
resume. The training entry continues to reject changed specs on resume.
No new parameters or optimizer groups are introduced.
