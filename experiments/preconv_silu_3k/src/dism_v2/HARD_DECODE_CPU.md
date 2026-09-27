# Pure-hard incremental CPU core

`hard_decode_cpu.py` is an independent inference-only implementation of the
pure-hard DISM recurrence. No training or existing CUDA path is modified.

For new query label q_i, enumerate only key positions with k_j=q_i (including
the newly appended j=i). A label→positions inverted index avoids scanning the
whole key string. Each matching position gets a cursor:

```
W_i[j] = tau + softplus(W_previous[j-1])   if the predecessor is active
         tau                            otherwise (a fresh match)
```

All other positions are exactly unreachable and have no cursor. Fresh matches
must be created at *every* occurrence of the current label, not just by moving
old cursors forward. Finite negative W is retained; unreachable means −∞,
not W<0. There is no pruning,beam approximation or finite negative sentinel.
The normalized output includes the fixed score0/value0 fallback. Stable
FP64 natural-log scores and shifted exponentials avoid exp(W) overflow;
output defaults to FP32. Ordinary floating-point underflow remains possible
for negligible weights. This is the full recurrence,not tile tanh LSE.

## Cache and API

```python
import torch
from dism_v2.hard_decode_cpu import HardDismCPUCache

# All tensors on CPU. Four heads,64-dimensional values; D can differ from DV.
cache = HardDismCPUCache(rtau, batch_size=1, value_dim=64)
# One self-attention step: q_label/k_label[1,4],v_t[1,4,64].
out_t = cache.step(q_label, k_label, v_t)
# Or labelize CPU Q/K via FP32 vocabulary logits,without interpolation:
out_next = cache.step_qkv(q_t, k_t, v_next, q_vocab, k_vocab)
```

`prefill(q_labels,k_labels,values)` accepts[B,H,N],[B,H,N,DV],including empty
chunks and continuation of a nonempty cache. `clone()` produces independent
continuations; `reset()` clears the sequence. `active_cursors` exposes copied
position→natural-log-score dictionaries for diagnostics. Batch items advance
in lockstep; use separate caches for independent/variable-length sequences.
Tau is fixed when the cache is created. Each successful step appends one KV
and one query; prefill does not load future keys ahead of causal queries.

Storage: copied key IDs and original-dtype V,derived inverted position index,
and one sparse previous-row cursor map per batch/head. No past Q/K vectors,
query string,embedding interpolation,W/logM matrix or dense per-row score
array is stored. Input value buffers can be reused without changing the cache.
For M current matching keys,cursor update is O(M),value aggregation O(M*DV),
apart from constant-per-step validation/copying. Worst-case identical labels
give M=N and linear decode work; this is not an unconditional constant-time
decoder. Python/Torch overhead can dominate small inputs; no wall-clock
speedup over CUDA or vectorized dense CPU code is claimed.

## Scope and numerical checks

Full-hard outputs do not depend on the soft direction; this CPU implementation
draws no RNG. It does **not** reproduce the training wrapper's redundant
direction-generator consumption. A model-level integration must keep this
separate from any mixed-mode/RNG contract.

`step_qkv` uses FP32 vocabulary logits/top1 with first-index tie breaking.
For an exact comparison using CUDA-selected labels,pass those integer IDs
directly: BF16 GEMM/argmax and softmax rounding near ties can change labels.
FP64 normalization also deliberately does not reproduce BF16 probability-MMA
rounding or approximate CUDA tile LSE.

CPU tests compare each cursor and output with `dism_ref.dism_recurrence` plus
zero-fallback normalization, and compare the Q/K labelization path with both
directions of `voc_dism_ref(hard_prob=1)`. Coverage includes all nine D/DV
32/64/128 combinations,BF16/FP32/FP64 cached values,multi-batch/head,causal
self keys,no matches,new starts,broken chains,negative tau,1024-step long
chains,all-repeated labels,chunk equivalence,cloning and buffer ownership.

This is the accelerated DISM **core**, not a complete autoregressive LM
implementation. Per-layer shortconv states,RoPE positions,SWA's own KV cache,
token sampling and the residual/FFN loop still need a separate model-level
generation integration. No active training or remote deployment is touched.
