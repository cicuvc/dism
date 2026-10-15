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
The host-device synchronization audit in
[training benchmark](../dism_v3/docs/TRAINING_BENCHMARK.md) measured what is
left of that host work and found it hidden under GPU execution: the production
step is GPU-occupied about 92%, FLA's once-per-microbatch chunk-index round trip
blocks 117 ms per step but removing it is worth 0.7% +- 1%, and the step absorbs
25-35 ms of extra host work before it costs step time. Further host-side work
here has a hard ceiling of about 5% and is not the live lever.

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

## Host prefetch staging

Session status (2026-10-15): implementation committed as `4726418`; fresh
token-balanced run `dism-fwedu-valres-72m-2500m-tokenbal-20261015-r3` stopped at
the user's request after logged step294. No checkpoint yet (save interval2000),
no automatic restart. Last100 logged steps averaged619.84 ms /211.5k tokens/s.
See [training closeout and benchmark caveats](../dism_v3/docs/TRAINING_BENCHMARK.md).

GDN chunk metadata uses one owned boundary snapshot per model forward. The
pure-GDN and hybrid layers share that tensor identity, while DISM keeps its
separate validated `prepare_pack` outputs. This avoids per-hybrid-layer FLA
cache misses without retaining stale metadata when callers overwrite their
boundary buffers between forwards. The internal `_share_branch_boundaries`
flag is valid only for immutable per-forward snapshots; standalone attention
calls retain their previous behavior. No checkpoint keys or RNG rules change.
`tests/test_gdn_boundary_reuse.py` checks actual compiled execution, cross-layer
cache hits, in-place input-boundary changes and forced-cache-miss gradient parity.
The local 72M checkpoint-4000 replay (document-balanced CE, four microbatches,
only launch threads bound to P cores) reduced chunk-index cache misses from
28 to 4 and associated stream synchronizations from 56 to 8 per step. Three
eight-step windows averaged 619.37 ms before versus 610.75 ms after (about 1.4%
lower latency). This excludes live data loading and W&B, and is not a restarted
training run. Profiles: `dism_v3/build/gdn_conv_diagnostic_20261015.trace.json`
and `dism_v3/build/gdn_boundary_after_20261015.trace.json`. CPU synchronization
duration includes useful queued GPU work; it is not wholly removable idle time.

The parallel loader owns each queued batch independently of the source's reused
buffers. For CUDA training the producer copies directly into pinned host memory;
the training thread only submits nonblocking H2D copies. The bounded queue remains
eight batches. There is no device prefetch stream or training RNG use in the
producer. Restored pending batches are pinned once before consumption; checkpoint
version, consumed progress, pending order and source cursor are unchanged.

`tests/test_data_prefetch.py` covers CPU/CUDA ownership, serialized queue replay
and RNG preservation; `tests/test_checkpoint_resume.py` covers exact training
resume. `../dism_v3/tools/bench_prefetch_staging.py` replays trusted checkpoint
batches without advancing the real loader or writing checkpoints. It measures
ready-queue staging only, not tokenizer starvation or end-to-end throughput.
On the local 5090, the 2026-10-15 consumer/producer/producer/consumer comparison
reduced median host submission from 93–95 us to 32–33 us per microbatch. At four
microbatches per step this is only about 0.24 ms of host work, not an explanation
for the observed GPU utilization gap. Raw results belong in ignored `build/`.

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

DISM/SWA model specs may set `loss_balance` to `document` (default) or `token`.
Both use the chunked fused softcap CE. Training sums the corresponding loss
numerators and valid counts over all accumulation microbatches/ranks before
normalizing gradients, including unequal valid-token counts and ignored-only
microbatches. Token mode does not pass document boundaries to CE; attention and
shortconvolution still receive the original packed boundaries. Checkpoint
metadata records `global_valid_document_mean_v1` or `global_valid_token_mean_v1`.
BPB/unreduced token diagnostics are unchanged. This option changes the objective,
not model weights, optimizer schema, RNG counters or dataloader state.
