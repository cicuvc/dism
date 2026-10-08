# Full DISM causal language model

Following `flame/custom_models/sba`, the package exports `DismConfig`,
`DismModel`, and `DismForCausalLM`, with Transformers AutoConfig/AutoModel/
AutoModelForCausalLM registration (`model_type="dism_v3"`). All implementation
lives here; the reference SBA tree and installed packages are unchanged.

## Architecture and defaults

Token embedding -> PreNorm decoder blocks -> final RMSNorm -> untied LM head.
Each block is residual hybrid DISM+SWA attention, then residual SwiGLU MLP.
Both branch inputs use RMSNorm. Attention retains per-head output gated RMSNorm.
MLP is standard PyTorch `down_proj(SiLU(gate_proj(x))*up_proj(x))`.

Defaults are width256,12 layers,4 heads,D=DV64,R32,QK vocab512,SWA128,external
vocab50257. `attention_type="hybrid"` is default; `"dism"` omits SWA.
`intermediate_size` is configurable; if omitted, use the SBA/FLA parameter-budget
rule `ceil_to_256(hidden_size*hidden_ratio*2/3)`, hidden_ratio4 (default FFN768).
These are configurable construction defaults, not a claim of any fixed parameter
budget or a new approved training run. Input/output embedding tying is rejected.

Linear, fused projection and embedding weights use normal std0.02 by default.
Keep the attention-specific SiLU-Gaussian effective vocabularies, softplus-tau
initialization and shortconv coefficient initialization; HF post_init must not
replace them with generic Gaussian codebooks. RMS weights start at one.

## Training

```python
import torch
from flash_dism import DismConfig, DismForCausalLM

model = DismForCausalLM(DismConfig()).cuda()  # FP32 master parameters
optimizer = torch.optim.AdamW(model.optimizer_param_groups(0.01), lr=1e-3)
ids = torch.randint(0, model.config.vocab_size, (2, 256), device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    result = model(input_ids=ids, labels=ids, hard_prob=0.5, hard_seed=123)
result.loss.backward()
```

Pass the optimizer-step hard_prob schedule externally. One hard_seed is forwarded
to all layers, which salt it by their layer_idx. Change the base seed per step if
new masks are desired. Explicit direction/hard, generator, cu_seqlens and
max_seqlen are forwarded to attention. Labels are unshifted token IDs: the LM
shifts exactly once. Final tokens and packed document transitions are ignored;
padding transitions are also excluded. Caller-supplied -100 labels are preserved.

CUDA loss uses the local fused **linear + cross entropy** kernel, BF16 hidden/head
operands and FP32 mean loss, with configurable `ce_softcap` (default30), optional
`ce_chunk_size`, `ignore_index=-100`. Packed loss is averaged over valid tokens,
not equally weighted document means. The existing CE kernel's varlen mode uses
document means, so the model deliberately passes boundary-masked labels through
its token-mean path. All-ignored labels return zero loss.

With labels and fused CE, logits default to None. `return_logits=True` explicitly
materializes them too. Without labels, logits are always computed by default;
`logits_to_keep=N` selects the final N positions (0 means all). With labels,
logits_to_keep must be0. Returned logits include the same softcap as the loss.
`fuse_cross_entropy=False` or CPU evaluation uses Torch CE for verification.
No dense attention scores are materialized by the CUDA training path.

`optimizer_param_groups(wd)` excludes all biases/1D parameters, vocabularies,
tau and QK RMS weights from decay. Names back up Python no-decay markers so HF
checkpoint loading cannot accidentally change this grouping.

`gradient_checkpointing_enable()` uses non-reentrant recomputation. Missing
hard seeds and directions are sampled outside the checkpoint closure, then
replayed; this also handles an explicit generator without resampling in backward.
Original attention/CE gradient tolerances are not changed.

## Varlen, cache and serialization

Training accepts packed `[1,T]` IDs with int32 cu_seqlens; all document boundaries
must be256-aligned. Empty documents inside a nonempty pack are allowed. Padding
masks are supported where each unpadded document has aligned length. Packed
RoPE and all convolution/attention state reset at document boundaries.

Eval defaults to use_cache=True and constructs DismCache when absent; that means
prefill/decoding uses the current Torch reference, not the aligned CUDA path.
For fast aligned evaluation without caching (including packed input), pass
use_cache=False. Training rejects decoding caches. Return dictionaries follow
BaseModelOutputWithPast/CausalLMOutputWithPast, with optional hidden states;
return_dict=False returns tuples. Attention matrices are not returned.

```python
model.eval()
tokens = model.generate(ids[:1, :16], max_new_tokens=8, do_sample=True,
                        temperature=0.8, top_p=0.9)
model.save_pretrained("checkpoint")
# import flash_dism before AutoModelForCausalLM.from_pretrained("checkpoint")
```

Generation uses the local FLA-compatible DismCache, not Transformers DynamicCache.
Each continuation supplies only new tokens, and each layer keeps its own cached
direction. Same inherited restrictions apply: equal-length, unpadded decoding
only; packed caches, unequal-length padded generation and arbitrary position_ids
are not supported. Cache model weights/theta/window must stay fixed. No SAM or
subquadratic decoding is claimed. The model-level API is tested with the locally
installed Transformers/FLA versions; static/quantized cache modes are rejected.

## Verification

`PYTHONPATH=python:.. python -m pytest -q tests/test_modeling.py`

Tests cover default hybrid construction, untied weights, packed label boundaries,
CPU dense-vs-cached logits, generate, HF/Auto round trip, parameter no-decay,
fixed/varlen CUDA fused loss and all-parameter backward, all-ignored loss, and
checkpointed-vs-normal loss/gradient replay. These are smoke/correctness tests,
not a training-convergence or throughput result.
