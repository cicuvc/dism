# Soft only CUDA with sequence hard skips

Implemented and measured on 2026-10-14 after planning commit `69eaa5f`.
This is an intentionally incomplete split of soft and hard execution: the
CUDA training core computes soft sequences and skips hard sequences. The
[dedicated hard forward](HARD_FORWARD_PLAN.md) now has a complete standalone
summary/chunk/LUT/readout entry, but it is not integrated into this training
dispatcher. Event-stream hard backward is also not dispatched by this core.

## Breaking interface and output contract

Public hard flags are bool `[B,H]` for fixed execution and bool
`[training_rows,H]` for packed batch-one execution. Staged native flags use
uint8 with the same shape. Token masks, including constant `[B,H,N]` tensors,
are rejected rather than silently sampled at their first token.

Packed rows have equal length `T/training_rows`, divisible by 256. Every
document must be contained in one row; all documents in that row inherit its
head flags. The existing CPU document table validates this mapping without
reading hard flags back from the GPU. Flags remain immutable during execution.

Skipped positions return zero output, zero log2-normalizer and zero vector/LSE
gradients; they contribute nothing to head-shared dtau. These are placeholders,
not hard-attention results. Saved forward checkpoints use the existing finite
LOG_ZERO sentinel; reverse summaries are zero. Hard/mixed CUDA training and
hard CUDA evaluation are not meaningful until the dedicated path is integrated.
The independent hard inference/reference operators are unchanged.

The low-level optional caller-owned vertical buffer is not written for skipped
tasks; the public core initializes its own saved buffer. Labels and tau remain
in the operand API for interpolation/coordination compatibility, but the soft
score kernels no longer load labels or use tau for a hard overwrite. Tau's soft
LSE absorption and gradient remain unchanged.

## Kernel and adapter changes

Fixed persistent F1/F3/B1/B3 consumers and their producers skip the same hard
task before acquiring any input slot, advancing neither pipeline nor mailbox
phase for that task. Varlen CTAs return uniformly before constructing barriers
or starting a producer. Numerical soft scan, MMA/readout, precision, tolerances
and launch defaults remain unchanged.

Removed per-token hard tests from forward score finishing, backward recompute
and B3 gradient selection. Removed Q/K label and token-hard shared metadata
and asynchronous loads from F1/F3. Forward score finishing now only converts
the natural-log MMA result to log2.

Output/summary initialization makes skipped results deterministic, at a real
bandwidth cost. F2/B0/B2 still run on initialized placeholders: this is not
complete task compaction or an all-hard zero-launch fast path. Vocabulary
interpolation and its backward also remain active. No hard-flag host/device
synchronization was added.

The module convenience sampler now emits sequence flags. Nanochat retains the
original row/head RNG samples, counter/seed identity and direction draw offset;
it replaces expansion to token flags with a transpose to compact row flags.
The existing full-length flag call is still used to preserve that direction
draw. Token-granularity nanochat configurations now fail explicitly. Checkpoint
keys, annealing buffers, optimizer schema and consumed dataloader state are
unchanged. No live training or user checkpoint was modified.

## Performance on RTX 5090

Synthetic 72M operator shape B16/H6/N2048/D64/R32/DV64, vocabulary512,
tau=.5, true direction, BF16. Uniform rows contain one document; ragged rows
contain 256/512/256/1024-token documents. Timings below are CUDA-event windows
of eager native stage calls, including allocation and host launch gaps, not
isolated instruction timings. Stages are measured separately and are not
additive. No CPU planner, transfers, full model or optimizer is included.

All-soft baseline versus the final all-soft confirmation, milliseconds:

| Layout | Stage | Before | After | Reduction |
| --- | --- | ---: | ---: | ---: |
| Uniform | F1 summary | 0.2310 | 0.2416 | -4.6% |
| Uniform | F3 forward | 0.6044 | 0.5572 | 7.8% |
| Uniform | B1 summary | 2.1476 | 1.9330 | 10.0% |
| Uniform | B3 QK | 2.0319 | 1.8906 | 7.0% |
| Ragged | F1 summary | 0.1054 | 0.1002 | 4.9% |
| Ragged | F3 forward | 0.2530 | 0.2390 | 5.5% |
| Ragged | B1 summary | 0.9052 | 0.8357 | 7.7% |
| Ragged | B3 QK | 0.8711 | 0.8208 | 5.8% |

Across three after-runs, uniform core backward medians were 3.932–3.983 ms
versus 4.306 ms before. Uniform core forward varied from 0.862 to 0.989 ms
versus 0.894 ms before; F3 stage medians varied 0.546–0.613 ms. Thus backward
improvement is more consistent than end-to-end forward improvement. The new
initialization cost and eager-window variability must not be hidden by reporting
only the fastest forward sample. The final confirmation's vocabulary-plus-core
forward/backward pair was 6.598 ms versus 6.824 ms before, about 3.3% lower.

In the full probability sweep, a realized hard fraction of 94.8% gave uniform
core forward/backward 0.327/0.918 ms versus 0.883/4.724 ms before. All-hard gave
0.208/0.265 ms after. These figures omit the hard computation by design: they
measure the residual soft path and its overhead, not equivalent-operator or
full-training speedups. Dense checkpoint allocation and F2/B0/B2 work remain.

Raw ignored artifacts are `build/soft_only_before_20261014.json`,
`build/soft_only_after_initial_20261014.json`,
`build/soft_only_after_final_20261014.json`, and
`build/soft_only_allsoft_confirm_20261014.json`. Baseline uses 3 warmups,
5 repeats and 3 samples; the final probability sweep uses 5/20/7, and the
all-soft confirmation uses 10/20/7 with 3 seconds of settling. The latter also
contains separate instrumented kernel attribution, not used as latency data
in the table. Reports retain extension/source hashes, clocks and temperatures.

## Validation and limitations

- Selected core, summary, forward, varlen, vocabulary/autograd, eight-config,
  native-frontend and code-generation regression: 315 passed, 41 skipped.
  Skips are existing optional FP32/probe exclusions; the final build contains
  all eight production BF16 configurations, with FP32 diagnostics disabled.
- Four saved pre-change fixed/varlen cases covering both directions had exactly
  matching all-soft outputs and normalizers. Maximum gradient difference was
  5.96e-8, consistent with atomic accumulation ordering. The maintained oracle
  and numerical tolerances were not changed.
- Nanochat structure, compact-row RNG/direction mapping, compiled execution,
  optimizer/checkpoint and loader resume checks: 14 passed. An older schedule
  test fixture needed its missing vocab_temperature buffer added; the actual
  annealing implementation was unchanged.
- Memcheck: initial 19 core tests, zero errors. Synccheck: 7 focused runtime
  tests, zero errors. Backward-filtered racecheck: 7 tests, zero hazards.
- Unfiltered racecheck: 7 numerical passes but 11 displayed error groups in
  fixed F1/F3. Their own SASS contains ARRIVES.LDGSTSBAR with `[UR5+0x8]` and
  `[UR23+0x8]`, respectively. The existing per-kernel policy classifies these as
  accepted suspected-tool warnings, not a clean racecheck result. Raw logs and
  classification evidence remain in `build/soft_only_*check*_20261014.*`.
- All eight configurations' primary device instances have no CALL and no Torch
  header dependency. The modified optional summary probe passed device syntax
  checking but was not included in the production build.
- Spills remain. R32/D64/DV64 F3 fixed/varlen use stack 8/24 bytes; B1 uses
  312/304 bytes; B3 uses 296/176 bytes. These are stack sizes, not dynamic spill
  traffic counts. Resource reports and SASS LDL/STL counts are recorded in
  `build/soft_only_codegen_resources_20261014.json`; ptxas reports spill loads
  and stores in the build logs. No shared FP32 atomics were introduced.

This is a selected regression campaign, not a claim that every historical
token-mask caller/test has migrated to the breaking API. Old CUDA hard-attention
benchmarks cannot be used as equivalent baselines against zero placeholders.
