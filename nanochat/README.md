# nanochat pretraining

Source: `/root/autodl-tmp/dism-randomness-code/nanochat` on
`connect.westd.seetacloud.com:29687`, selected run
`randomness-sequence_true-786m` (2026-10-06). Curated import: v4, hashed-symbol
experiments and chat/RL launchers are excluded. The dataloader, checkpoint
manager and base training loop retain the remote implementation.
Original source hashes are in `source_manifest.json`.

## Baseline

`configs/sequence_true.json` is the unchanged original model specification.
40,467,834 parameters: width384, six heads, D=DV64, R32, Q/K vocab512,
FFN1445, untied 32k embeddings, context2048. First three layers: pure GDN;
last three: GDN+DISM sharing Q/K input-linear weights, with independent
convolution filters and V paths. Postnorm, no SWA. Soft readout is linear+SiLU
without L2 normalization. GDN beta is sigmoid (no negative eigenvalues).

Direction is fixed true. One training forward with one hard flag per packed
training row/head/layer, shared by all 2048 tokens in that row, including
across documents. Hard probability linearly rises from 0 to .95 in 6000 steps.
Validation reports all-hard and all-soft. Original final hard/soft NLL:
3.09840534 / 3.08634381 over 1,068,889 valid / 1,179,648 packed tokens.
These are historical remote results, not fresh evaluation of this tree.

## Current training defaults (2026-10-15)

The DISM runtime now defaults to compact CPU-planned metadata and `ready` hard
backward: use event backward only once both planning and H2D are complete;
otherwise use the dedicated quadratic hard backward. For explicit controls,
`DISM_HARD_BACKWARD=event` restores waiting and `DISM_PLANNER_COMPACT=0` restores
the classic padded layout. Read these settings before creating the first runtime.
Worker count, affinity, convolution configuration and model architecture are unchanged.

For existing `conv_impl="linear_fla", conv_backend="cuda"` configurations, the
convolution frontend now calls CUDA causal_conv1d forward/backward directly.
The historical configuration name and `weight`/`conv.weight` parameter layout
are retained, including initialization RNG, optimizer grouping and shared Q/K
aliases. There is no checkpoint or Adam-moment migration. The explicit Triton
backend remains available; the direct CUDA path does not silently fall back.

One int32 token-to-document map is built from `cu_seqlens` with GPU searchsorted
per model forward and shared across pure GDN, hybrid DISM/GDN and value-residual
frontends. This removes FLA shortconv's per-document GPU-scalar host reads.
Empty documents are handled with right-sided search; no mutable-pointer cache
or change to packed-row hard sampling is introduced. Shortconv custom ops keep
the CUDA channel-last layout inside an opaque functional/autograd boundary for
Dynamo. This does not eliminate DISM's exact-layout validation or GDN metadata
host work, and does not add cached decoding or CUDA Graph support.

DISM and SWA-control training now use chunked fused linear CE with softcap and
the actual `cu_seqlens`. Their default scalar `mean` is an equal-weight mean of
valid document means, **not a token mean**. A document with no valid target
(including an empty boundary interval) is excluded. Targets are already shifted
by the dataloader and are not shifted again. Padding/ignored labels remain -1.
Hard flags are still sampled per packed training row/head, never per document.
The unreduced token NLL/BPB validation interfaces keep their previous semantics.
This change targets the nanochat adapters; the separate Hugging Face LM API's
loss contract is not changed by this integration.

The training loop requests `loss_reduction='document_sum'`, returning a
differentiable numerator S and a non-differentiable count D. For a complete
optimizer step over all ranks and microbatches, the objective is:

```text
S = sum_documents [sum_valid_tokens NLL / number_of_valid_tokens_in_document]
D = number_of_documents_with_at_least_one_valid_target
L_step = S / max(D, 1)
```

Each microbatch backpropagates S_m / accumulation_steps, purely as temporary
gradient scaling. After accumulation, `DocumentLossAccumulator` sums gradients
across ranks and multiplies by accumulation_steps / global_D, **before** AMP
unscaling and gradient clipping. All ranks then hold identical globally
normalized gradients. DISM's plain AdamW performs no hidden averaging; if an
optimizer averages these identical gradients later, that AVG is an identity.
Do not wrap this path in an additional DDP gradient-averaging wrapper.
The logged training loss also uses global S/D, not the last microbatch mean.
Explicit dual-path experiments count documents once, not once per branch.

No whole-step batch prefetch is introduced: the existing one-next-batch protocol,
consumed dataloader position, RNG counters, annealing state, optimizer groups
and state schema are preserved. The accumulator exists only during a step.
Checkpoint metadata records `loss_normalization`; resuming a historical
token-mean checkpoint warns that the objective changes, without resetting state.
This intentionally changes the training objective, so it does not promise the
same optimization trajectory as a historical token-mean run. No existing
checkpoint is rewritten and no training run is automatically restarted.

## Training command

Build the sibling v3 extension first. Environment: `pyproject.toml` dependencies
plus FLA, causal-conv1d, FlashAttention and the v3 build dependencies.
From repository root:

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
export NANOCHAT_BASE_DIR=/path/to/new-run
export HF_ENDPOINT=https://hf-mirror.com
python -m scripts.base_train \
  --model-spec nanochat/configs/sequence_true.json --model-tag random40m \
  --hf-tokenizer /path/to/llama2-tokenizer --data-dir /path/to/fwedu-100B \
  --seq-align 256 --loader-workers 2 --data-shuffle --shuffle-seed 1337 \
  --device-batch-size 64 --total-batch-size 131072 --num-iterations 6000 \
  --warmup-steps 157 --warmdown-ratio .25 --final-lr-frac .1 \
  --matrix-lr .0015 --embedding-lr .0015 --unembedding-lr .0015 \
  --scalar-lr .0015 --weight-decay .01 \
  --eval-every 1500 --eval-tokens 1179648 --dual-eval-valid-tokens 1048576 \
  --eval-device-batch-size 64 --save-every 1000 \
  --core-metric-every -1 --sample-every -1 \
  --wandb-project dism-v3 --run sequence-true
```

6000 × 131072 = 786,432,000 packed tokens. Do not add `--dual-path-loss`.
Batch64 reflects the remote GPU, not a local memory-fit guarantee. Use a new
W&B identity, never overwrite the historical run. This cleanup starts no jobs.

## Resume and testing

Use `--resume-from-step STEP` with the same base directory/model tag and
training arguments. Model state contains `training_step`, `hard_probability`,
`rng_counter` (one increment per microbatch). Restore these together with
optimizer/schema and rank-local RNG/dataloader state. The loader tracks
consumed batches rather than ahead-of-consumption prefetch. Original guards
reject incompatible model/tokenizer fingerprints and training settings.
Checkpoint metadata is published last; retention removes only complete sets.

```bash
PYTHONPATH="$PWD/nanochat:$PWD/dism_v3/python" python -m pytest nanochat/tests
```

GDN hybrid supports packed prefill/recomputed generation, not cached decoding.
The explicit SAM prefill/decoding backend is available in flash_dism.inference;
it does not automatically replace nanochat's model/cache dispatch. V4 training
operators remain excluded.
