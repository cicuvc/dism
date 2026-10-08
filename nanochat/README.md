# nanochat pretraining

Source: `/root/autodl-tmp/dism-randomness-code/nanochat` on
`connect.westd.seetacloud.com:29687`, selected run
`randomness-sequence_true-786m` (2026-10-06). Curated import: v4, hashed-symbol
experiments and chat/RL launchers are excluded. The dataloader, checkpoint
manager and base training loop retain the remote implementation.
Original source hashes are in `source_manifest.json`; import adjustments and
verification are recorded in the repository's `CLEANUP.md`.

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
Pure DISM/SWA Torch cache references remain in flash_dism. No SAM/v4 inference
backend is included on this branch.
