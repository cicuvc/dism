# nanochat integration

- Source baseline: randomness-sequence_true-786m, tracked in configs and README.
  Do not replace it with another local nanochat fork or change model defaults
  to approximate its behavior. No v4/hashed-symbol experiments in this branch.
- Preserve optimizer no-decay groups, shared Parameter aliases, state_dict keys,
  RNG counter and hard-probability/training-step buffers across save/load.
- Sequence hard means a packed batch row, not a document. Direction is true in
  the selected config. Single-forward loss, no implicit dual-path switch.
- Prefetch must not advance the checkpoint's consumed data position. Retain
  saved next batch, loader state and all RNG states. Run resume tests on changes.
- GDN is intentionally outside Dynamo fullgraph; do not force fullgraph=True
  in the training loop. Do not claim cached hybrid decoding is implemented.
- Tests do not authorize training, remote changes or dataset downloads.
