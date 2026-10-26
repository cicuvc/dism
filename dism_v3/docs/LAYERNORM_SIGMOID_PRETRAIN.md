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
order. The initial implementation used the explicit FP32 centered expression
from the selected screen, followed by BF16 cast. The fused implementation
described below now serves supported BF16 eager/CUDA Graph calls; Dynamo and
unsupported inputs retain the explicit expression. Residual
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

## Output implementation cost (2026-10-27)

Measured on RTX 5090 after `44b21bd`. Both paired models retain K4, old layout
`ggggggdddddd`, 123,473,088 parameters, hard probability .5, CUDA Graph,
MXFP8 FFN/optimizer and BF16 CE/projections. Only `mixer_output` and its
associated scale metadata differ. Synthetic changing-document batches use
8×2048 rows and eight accumulation microbatches (131,072 tokens/step), as in
the [top-k full-step benchmark](TOPK_HARD_TRAINING.md). Four warmup and ten
measured optimizer steps per model alternate order, from identical initial
parameter values. No checkpoint or dataset is accessed; these are compute-step
timings, excluding corpus loading, evaluation and checkpoint I/O. All losses
and gradient norms remained finite; losses are not expected to match because
the two output formulas differ.

| Planner | Original RMSNorm/SiLU step | LayerNorm/sigmoid step | Extra time | Original token/s | New token/s | Throughput change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | 931.1 ms | 1060.2 ms | +129.0 ms (+13.86%) | 140,765 | 123,634 | -12.17% |
| On | 960.4 ms | 1092.3 ms | +132.0 ms (+13.74%) | 136,482 | 119,992 | -12.08% |

Both planner-on models selected event for all 480 measured layer/microbatch
decisions. The slowdown is present with planner off and is not explained by
readiness misses. Production kernel math, defaults and models were not changed.

`tools/bench_mixer_output.py` isolates the same dimensions: 16,384 tokens,
12 heads × 64 values, low-rank gate 768→96→768, FP32 gate/norm parameters,
BF16 output projection and BF16 autocast. All inputs require gradients.
Forward, backward and combined forward/backward have separate CUDA Graphs;
the table uses median GPU time over ten alternating rounds of 20 replays.
The isolated backward reuses saved forward tensors. Separate timings need
not sum to joint timing due to cache state and scheduling; use joint timing
for the per-step extrapolation.

| Output tail variant | Forward | Backward | Joint |
| --- | ---: | ---: | ---: |
| Original fused RMSNorm/SiLU | .164 ms | .469 ms | .626 ms |
| Original formula, explicit PyTorch diagnostic | .447 ms | 1.335 ms | 1.781 ms |
| Explicit RMSNorm/sigmoid | .452 ms | 1.301 ms | 1.759 ms |
| Current explicit LayerNorm/sigmoid | .544 ms | 1.423 ms | 1.966 ms |

The output tail includes both gate projections, norm/gate and output projection;
it excludes the mixers and their sum. LayerNorm/sigmoid adds 1.340 ms per layer
and microbatch. Multiplying by 12 layers × eight microbatches gives 128.7 ms,
close to the measured 129.0 ms planner-off whole-step difference.

The gate projections alone cost .200 ms originally versus .204 ms with the
bottleneck sigmoid (joint). Norm/gate alone rises from .200 ms to 1.523 ms.
Even preserving the original formula but spelling it out as FP32 PyTorch
operations adds 1.156 ms to the output tail, about 86% of the observed 1.340 ms
delta. This diagnostic preserves the expression, not a bitwise numerical
equivalence claim. Removing fusion and repeatedly traversing full-size FP32
intermediates is the main implementation cost; the bottleneck sigmoid itself
is small. A fused LayerNorm/raw-gate forward and backward is therefore a
useful next optimization, but its achievable speedup is not measured here.
CUDA Graph removes launch submission work without fusing these tensor kernels.

Reproduction: run `tools/bench_topk_model.py` with the maintained top-four spec,
`--compare-output --disable-output-fusion --topk 4 --probability .5 --rounds 10`, and add
`--output-planner` for the on pair. Use a fresh `--output` under `dism_v3/build`.
Run `tools/bench_mixer_output.py --output <fresh-build-path>` for the isolated
screen. Both tools use only this worktree's modules on PYTHONPATH. Ignored raw
records are `build/topk_event_training/output_compare_k4_p50_{off,on}.json`
and `build/topk_event_training/mixer_output_micro.json`.

## Fused output follow-up (2026-10-27)

The user requested two forward kernels and confirmed that the formula remains
`norm(raw) * linear(g)`. `kernels/mixer_output.py` implements:

1. Down projection plus sigmoid, with rank at most 128.
2. Up projection plus per-head LayerNorm (or RMSNorm) and calibrated raw gate,
   with the complete norm dimension, at most 128, in one program.

Output projection remains the existing linear layer. Both pure GDN and hybrid
GDN+DISM use the new path; the explicit SAM inference output path also calls
the same dispatcher. Supported calls require contiguous CUDA BF16 activations
and BF16 projection semantics (BF16 autocast for FP32 projection parameters).
Unsupported dimensions, dtypes, layouts, empty inputs and Dynamo use the
original explicit expression. `norm.enable_fused_output=False` is a runtime
diagnostic switch; it is absent from model metadata and state_dict. Default
RMSNorm/SiLU models are untouched. Parameter objects, names, initialization,
optimizer groups and checkpoint schema are unchanged.

The down/up operands, biases, projection results and sigmoid result retain
their BF16 boundaries. Statistics, affine products and norm-weight reduction
use FP32. Backward fuses norm/raw-gate differentiation, reduces norm-weight
partials, and uses BF16 matrix multiplications plus sigmoid differentiation
for both projections. It supports first derivatives only. It does not promise
bitwise gradients: the old explicit LayerNorm expression has two separate
`raw.float()` autograd edges with intermediate BF16 gradient rounding, whereas
the fused norm derivative combines in FP32 before casting. GEMM/reduction
orders can also differ. Numerical tests compare every input/parameter gradient
against both the original expression and an FP64 reference with BF16 boundaries.

Bounded autotune covers down projection (five BM/BK/warp/stage candidates),
up/norm/gate (four BM/warp/stage candidates) and norm/gate backward (two warp
counts). Backward keeps BM32 and the partial-gradient layout fixed. Every
trial completely overwrites its outputs. Keys include relevant shape, bias,
norm type and operand dtypes; Triton persists winners in its user cache.
Normal warmup tunes before graph capture. Capture uses a cached winner
directly; a cold tuning key uses the original fixed configuration without
benchmarking or pretending that it was tuned. The cold-cache regression clears
all three tuner caches and replaces benchmarking with a throwing function to
verify this behavior. Tuning uses approximately 20 ms of graph replay per
candidate, outside steady-state timing.

The final autotuned isolated output-tail measurement at 16,384 tokens gives:

| Variant | Forward | Backward | Joint |
| --- | ---: | ---: | ---: |
| Original fused RMSNorm/SiLU | .165 ms | .470 ms | .627 ms |
| Explicit LayerNorm/sigmoid | .545 ms | 1.425 ms | 1.968 ms |
| New fused LayerNorm/sigmoid | .168 ms | .524 ms | .697 ms |
| Explicit RMSNorm/sigmoid | .453 ms | 1.303 ms | 1.761 ms |
| New fused RMSNorm/sigmoid | .166 ms | .521 ms | .696 ms |

The LN output tail saves 1.271 ms (64.6%) per layer/microbatch, approximately
122.0 ms across 12 layers and eight microbatches. This is a microbenchmark
extrapolation, not an end-to-end measurement by itself. Raw record:
`build/topk_event_training/mixer_output_autotuned_micro.json`. The pre-autotune
fixed-config implementation measured .725 ms joint for LN and .717 ms for RMS,
recorded separately in `mixer_output_fused_micro.json`.

25 distinct numerical/integration tests passed after autotune: all inputs and
parameter gradients, FP64 reference with BF16 boundaries, tail rows and odd
dimensions, contiguous/non-contiguous upstream gradients, no bias, BF16
parameters without autocast, no-grad dispatch, shape/dtype fallback, warm/cold
graph replay, exact zero gradients, all-layer initialization and state identity,
planner graph and checkpoint/optimizer/RNG resume, fixed hard probability and
SAM state/rebuild regressions. No existing acceptance threshold was relaxed.

`tools/audit_mixer_output.py` checks eight LN/RMS graph cases with changing
inputs and exact zero upstream gradients, including partial token tiles,
rank17/norm33 and the rank128/norm128 limits. Compute Sanitizer memcheck and
synccheck report zero errors; racecheck reports zero hazards/errors/warnings.
The final autotuned SASS inspection covers 76 generated specializations,
including every tuning candidate: zero CALLs and zero spill slots. Maximum
registers are 121 for down/sigmoid, 194 for up/norm/gate and 160
for norm/gate backward; the register pressure remains an optimization target.
This is bounded shape coverage, not an exhaustive claim. Artifacts are ignored
under `build/topk_event_training/mixer_autotuned_{memcheck,synccheck,racecheck}`.
Each tool uses a fresh isolated Triton cache so all candidates execute under
instrumentation and instrumented timings do not pollute normal tuning winners.
The earlier fixed-config audit remains in `mixer_fused_*`.

Full optimizer-step comparison uses `--compare-fusion --topk 4 --probability .5
--rounds 10` with the same top-four training spec and changing-document packing
as above. The two scratch models have identical LayerNorm/sigmoid formulas,
initial parameter values and model metadata; only the runtime fusion switch
differs. Four warmup steps per model include tuning, followed by ten timed
steps in alternating order. The on pair adds `--output-planner`.

| Planner | Explicit step | Fused step | Saved per step | Explicit token/s | Fused token/s | Throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | 1061.9 ms | 935.9 ms | 126.0 ms | 123,429 | 140,049 | +13.46% |
| On | 1093.8 ms | 969.4 ms | 124.4 ms | 119,836 | 135,213 | +12.83% |

Both planner-on models again selected event for 480/480 measured decisions.
All losses and gradient norms stayed finite. The planner-off paired loss
difference stayed below 2.1e-4 across the ten measured optimizer steps; this
short synthetic check is not a long-run convergence guarantee. The roughly
124–126 ms saving accounts for almost all of the earlier 129–132 ms output
implementation penalty. This is compute-step throughput with synthetic input,
not a dataset-loading/evaluation/checkpointing benchmark. Raw results:
`build/topk_event_training/output_fusion_k4_p50_{off,on}.json`.
