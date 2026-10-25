# Old-layout LayerNorm + sigmoid gate pretraining

User authorized a fresh complete run on2026-10-24, stopping additional layout
diagnostics: old layout `ggggggdddddd`, LayerNorm + calibrated sigmoid low-rank
gate, planner and CUDA Graph on, MXFP8 CE off. This run is not a resume.

## Durable architecture representation

The existing `dism_gdn_hybrid@1` config now accepts opt-in
`mixer_output="layernorm_sigmoid"` with an explicit positive finite
`mixer_gate_scales` list of one constant per layer. Defaults remain the original
RMSNorm/SiLU gate. Omitted fields do not modify a legacy ModelSpec fingerprint.
New specs contain the variant and scales, so checkpoint fingerprints reject
accidental resume with different architecture semantics.

Each mixer normalizes its64-dimensional per-head output with LayerNorm (same
weight and eps1e-5, no extra bias), then multiplies by
`c * W2(sigmoid(W1(x)))`, then applies the existing output projection/bias.
The sigmoid-input projection retains the original weight/bias names and draw
order. LayerNorm uses the exact explicit FP32 centered expression from the
selected screen, followed by BF16 cast; no new CUDA kernel was added. Both
eager and Dynamo dispatch use the opt-in explicit output module. Residual
postnorms, final norm, SQ scaling1, value residual and original hard sampling
semantics remain unchanged.

The twelve constants (.522–.568) are frozen from the successful paired screen's
unit-RMS gate calibration, not recalibrated at startup/resume or on evaluation
data. They are encoded in the maintained model spec:
`nanochat/configs/sequence_true_123m_layernorm_sigmoid_slimpajama_3b.json`.
State_dict parameter names, parameter count123,473,088, optimizer groups, RNG
counters and annealing buffers retain their schema. Scale values are architecture
metadata, not newly introduced trainable parameters or uninitialized buffers.

## Validation before launch

-19 tests passed: candidate config/identity/initialization math, original registry
 and document loss regressions, candidate planner graph forward/backward,
 changing packed boundaries/empty targets, accumulation, optimizer updates,
 evaluation followed by replay, and resumed optimizer/RNG/gradient state.
- Same-seed old/candidate initial state tensors match exactly in the test.
 Candidate graph comparisons retain existing tolerances; none were loosened.
- Actual123M model,8x2048 tokens,8 accumulated microbatches,16 planner workers,
 two optimizer updates at a nonzero hard probability, and intervening eager
 evaluation passed. Final route counts:90 event-based,6 quadratic; counter16.
 Peak allocated memory14.62GB. This is a validation probe, not a throughput test.
- No sanitizer coverage added; Python orchestration/output expressions changed,
 not CUDA kernels. Existing warnings remain visible.

Logs: ignored `build/layernorm_sigmoid_integration_tests.log` and
`build/layernorm_sigmoid_full_preflight.log`.

## Training recipe and launch

Reference: completed old-layout planner run ending `20261024-030902`.
Run directory under existing `dism-runs` storage:
`dism-slimpajama123m-h768-h12-i1152-r32-mxfp8-ffnonly-3b-docce-niah-clip1-lr8e4-ggggggdddddd-ln-sigmoid-planner-20261024-210059`.
Local pointer: `build/layernorm_sigmoid_training_run.txt`.

22,888 updates /2,999,975,936 packed tokens; batch8x2048 with8 accumulation
microbatches, total131,072 tokens/update. Original shuffle seed1337, LR8e-4,
warmup500, warmdown.25, final LR fraction.1, weight decay.01, clip1. MXFP8 FFN
and optimizer enabled, projections BF16, CE BF16. Planner16 workers and CUDA
Graph enabled. The `DISM_HARD_PLANNER=0` environment setting disables only the
eager planner; the explicit `--cuda-graph-planner` enables the graph planner.

Save every2000 updates; validation, dual hard/soft loss,8K NIAH and Q/K histograms
every500, with prior token budgets unchanged. Existing checkpoint retention
policy remains unchanged. No new diagnostic instrumentation is added to this
run beyond the inherited training metrics.

The run includes exact model_spec, train/supervisor scripts, source snapshots
and SHA256 manifest, and logs. New integration code is not automatically committed
by this launch; the source snapshot preserves the executable versions.

Confirmed running beyond update22 with finite losses/gradients after initial
validation and8K NIAH. Early throughput is about124k packed tokens/s.
W&B: https://wandb.ai/cicuvc-personal/dism-spj-fixed/runs/0ck0de45
Supervisor PID628640, training PID628643 at launch; durable PID/status markers
in the run directory remain the source of truth for subsequent status checks.
# Fixed hard-probability follow-up (2026-10-25)

At the user's request, the annealed run was stopped with SIGTERM (exit 143),
retaining its step 12000 and 14000 checkpoints. The replacement is a fresh run,
not a resume, using `sequence_true_123m_layernorm_sigmoid_hard05_slimpajama_3b.json`.
Its only model-config difference is `hard_prob_fixed: 0.5`. Layout remains
`ggggggdddddd`; every mixer uses LayerNorm plus calibrated sigmoid low-rank gate.
Planner and CUDA Graph remain enabled, with MXFP8 FFN/optimizer and BF16 CE.
Data, seeds, batch, optimizer, LR schedule, token budget and evaluations match
the stopped run.

`hard_prob_fixed=None` retains the original annealing behavior. A finite value
in [0,1] initializes and updates the existing probability buffer to that value,
including step zero. It does not change sequence-row sampling, RNG counters,
the temperature schedule, or checkpoint tensor keys. Explicit hard/soft
validation remains unchanged. The new field participates in ModelSpec identity.

The launch helper checks that the old process stopped, runs fixed-probability
CUDA Graph/planner and resume regression tests, checks the configuration diff,
and records a separate source snapshot and launch manifest before launching.
The new run path is recorded in ignored
`build/layernorm_sigmoid_hard05_training_run.txt`; the old pointer is retained.

## Automatic continuation queue (2026-10-25)

The user requested a serial continuation after the fixed-0.5 run finishes:

1. Resume the preceding LayerNorm/sigmoid annealed experiment from its latest
   complete checkpoint, step 14000, and finish the original 22888-step budget.
   Model, optimizer, rank training state (RNG and dataloader) and metadata were
   copied to a separate output directory; the original checkpoint is preserved.
2. Start a fresh RMSNorm/sigmoid experiment with the original 0-to-0.95 hard
   probability schedule. All 12 mixers use the calibrated sigmoid low-rank gate;
   only output LayerNorm is replaced by RMSNorm. Per-layer gate scales, postnorm,
   layout, planner/graph, BF16 CE, optimizer and data recipe remain unchanged.

The opt-in config is
`nanochat/configs/sequence_true_123m_rmsnorm_sigmoid_slimpajama_3b.json`.
The `rmsnorm_sigmoid` implementation matches the earlier sigmoid-only short
screen's explicit FP32 RMS normalization and raw calibrated gate expression.
Default model behavior remains unchanged.

`tools/run_training_queue.py` waits for exit code zero and the final step 22888
checkpoint before advancing. It uses an exclusive queue lock and stops on
failure rather than skipping a failed training. The durable queue directory is
recorded in ignored `build/sigmoid_training_queue.txt`; its plan, state and log
identify all run directories. The current training is not restarted or modified.
Each future run has a source snapshot and launch manifest.

Validation: 11 tests passed across RMSNorm/sigmoid, LayerNorm/sigmoid and fixed
hard probability, including math, all-layer installation, planner CUDA Graph
forward/backward and checkpoint state restoration. Log:
`build/queued_sigmoid_tests.log`. Existing dependency warnings remain visible.
