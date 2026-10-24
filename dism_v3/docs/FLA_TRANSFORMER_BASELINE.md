# FLA Transformer baseline (2026-10-24)

The `fla_transformer@1` nanochat architecture provides the requested postnorm
Transformer baseline. It replaces every GDN/DISM mixer with full causal
multi-head attention plus RoPE, retaining the document-balanced training recipe.
The user explicitly selected postnorm to minimize changes beyond the mixer.

## Architecture and parameter budget

| Setting | DISM reference | Transformer |
| --- | --- | --- |
| Layers / residual width / heads | 12 / 768 / 12 | 12 / 768 / 12 |
| FFN intermediate | 1152 | 1664 |
| Vocabulary / tied embeddings | 32000 / no | 32000 / no |
| Normalization | postnorm | postnorm |
| Parameters | 123,473,088 | 123,489,024 |
| FFN / CE / optimizer | MXFP8 | MXFP8 |

The difference is 15,936 parameters (0.0129%). Attention uses FLA `Attention`
projection modules, FLA `RotaryEmbedding`, and the FlashAttention varlen entry
point used by FLA. Q/K/V/O arithmetic is BF16 under autocast with FP32 master
weights, like the reference's non-MXFP8 projections. There is no GQA, qk_norm,
attention bias, attention gate, convolution, DISM selector, or GDN value-residual.
RoPE uses base 10000, head dimension 64; attention is causal, no dropout/window.

Block math: `u = RMSNorm(x + AttentionRoPE(x))`, then
`out = RMSNorm(u + SwiGLU(u))`. The existing fused residual RMSNorm produces
MXFP8 inputs for the FFN. Final RMSNorm/CE reuse the existing MXFP8 CE path and
softcap=30. Optimizer grouping, betas=(.9,.95), eps=1e-8, initialization std=.02,
and document-balanced loss follow the existing recipe. Independent construction
and different parameter shapes do not guarantee identical per-layer initial draws.

The maintained spec is
`nanochat/configs/fla_transformer_postnorm_123m_mxfp8_slimpajama_3b.json`.

## CUDA Graph support

Full packed microbatch forward and backward were captured successfully, including
attention, RoPE, MXFP8 FFN, final norm and CE. Optimizer, clipping, document-count
normalization and dataloader remain outside capture, as in the DISM run.
There is no DISM planner in this architecture; `--cuda-graph-planner` is rejected
by its graph runner rather than silently ignored.

The stock FLA Attention wrapper does not forward precomputed RoPE chunk indices.
Its ordinary variable-length index preparation cannot be blindly captured.
`PackedAttention` changes metadata plumbing, not the attention/RoPE kernels:

- pass precomputed fixed-storage RoPE chunk indices into `RotaryEmbedding`;
- supply the host-known maximum document length to FlashAttention;
- prepare rotary tables during warmup;
- stage changing document boundaries and chunk indices before each replay.

`PackedAttentionTrainingGraph` validates 256-aligned, bounded documents, compacts
empty intervals, pads metadata capacity with trailing empty intervals, and keeps
input/parameter/gradient storage stable. It preserves capture-time RNG and
preexisting gradients; replay accumulates into graph-owned gradients. Graph-owned
MXFP8 workspaces and RoPE tables stay alive across intervening eager validation.
Only the new architecture selects this runner; DISM graph dispatch is unchanged.

## Validation

- Initial focused tests: 4 passed; then registry/NIAH/document-loss suite including
  the focused tests: 30 passed.
- Final focused rerun: 4 passed, including upstream FLA attention forward/backward
  parity, document isolation, changing boundaries/empty documents, all-ignored
  targets, gradient accumulation, optimizer updates, longer-context eval followed
  by replay, and model/optimizer restore followed by the next update.
- Both BF16 CE and MXFP8 CE graph paths were tested. Gradient comparisons use
  atol=.001/rtol=.03; loss uses atol=.005/rtol=.005. These do not certify bitwise
  equality of floating-point FlashAttention reductions. No tolerances were loosened
  after a failure, and no production kernel math was changed.
- Full 123.49M configuration: `[8,2048]`, 8 accumulated microbatches, two optimizer
  steps, local 8K NIAH, and subsequent replay passed. Peak allocated memory was
  about 9.04 GB, with concurrent downstream evaluation still active. The measured
  step timings are not isolated throughput measurements.
- Existing PyTorch/TorchScript deprecation and first-cuBLAS-context warnings remain
  visible. An intermediate test mixed eager backward on the captured model with
  graph-owned gradients and emitted an AccumulateGrad stream warning; the final
  resume test uses captured replay versus restored eager execution and no longer
  emits it. No warnings were suppressed. No new sanitizer coverage was run because
  this change adds Python orchestration, not CUDA kernels.

Logs are gitignored: `dism_v3/build/fla_transformer_{tests,regression,tests_final,full_probe}.log`.

## Training launch

Run directory under the existing `dism-runs` storage:
`fla-transformer-postnorm-123m-h768-h12-i1664-mxfp8-ce-3b-docce-niah-clip1-lr8e4-20261024-154944`.
It includes `train.sh`, `supervise.sh`, `model_spec.json`, `launch_manifest.json`,
source snapshots/hashes and `logs/train.log`. The manifest identifies the prior
`ggggddddddgg` + MXFP8 CE run as recipe reference.

Fresh training: 22,888 steps / 2,999,975,936 packed tokens, sequence length 2048,
device batch 8, total batch 131072, shuffle seed 1337, LR 8e-4 for all groups,
warmup 500, warmdown .25, final LR fraction .1, weight decay .01, clip=1.
Save every 2000 steps; validation and local 8K NIAH every 500.
Hard/soft dual evaluation and vocabulary histograms are disabled because neither
exists in this model. Standard validation BPB and NIAH remain enabled. NIAH now
accepts an explicit packed-model capability without inventing hard/soft state;
DISM's existing hard-mode/state-restoration behavior is preserved.

The downstream evaluation was left running. Training and evaluation share the
GPU until evaluation finishes, so early wall time should not be used as an
uncontended performance comparison.
