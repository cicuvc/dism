# Host-only vocabulary temperature

Vocabulary beta is a non-learned scheduling value. DismAttention and nanochat
DismLM now store it as a Python float, not a Tensor or module buffer. Moving a
model to CUDA does not upload it. The compiled fixed/packed vocabulary entry
passes it through the `voc_forward_host_beta_v1` scalar custom-op schema;
autograd captures the immutable forward-time value and supplies it to the
corresponding scalar backward op. The interpolation kernels receive their
existing scalar launch parameter, with unchanged math, launch defaults and
BF16 rounding. The beta==1 fast path is retained.

The subsequent [fixed-true specialization](DIRECTION_TRUE.md) versions these
host-beta ops to v2 to carry its static direction flag and empty unused output;
the host-only beta contract described here is unchanged.

No state_dict keys change: the previous temperature buffer was non-persistent.
Initialization restores the configured temperature; set_training_step computes
the same cosine schedule on the host. Resuming training must still restore
training_step, hard_probability, RNG counters, optimizer and dataloader state,
then apply the resumed step's schedule as the existing loop does.

The previous tensor-beta custom ops remain for explicitly supplied low-level
tensor inputs and existing compiled-call tests. Explicit CUDA tensor beta still
requires scalar readback in that compatibility path; it is not the model's
production path. No global temperature cache or stale-pointer cache is used.

Compilation limit: the installed Torch/Inductor specializes host scalar custom-op
arguments, even with dynamic=True and specialize_float=False (also tested with
a Scalar schema). Changing beta can therefore recompile a fullgraph; numerical
updates work, but a continuously annealed host-float schedule is not a
no-recompile contract. Current r3 uses beta_start=beta_end=1 and is unaffected.
Fixed-temperature repeated calls are tested with error_on_recompile=True.
The failed dynamic-no-recompile experiments remain in the ignored
`build/host_beta_{schedule,scalar}_tests.log` files, not counted as passing.

An initial device-pointer-beta implementation was tested and discarded: keeping
the scalar on the host removes the round trip at its source and requires no
Triton kernel changes. Its ignored `build/device_beta_*` artifacts are diagnostic
history, not evidence of the final implementation.

The fresh r3 pretraining was stopped at user request after logged step4192.
The latest complete checkpoint is004000; no checkpoint was changed and training
was not restarted. Any future resume starts from that exact saved state, not
from the unsaved step4192 position.

## Validation

85 distinct tests pass: 84 in `build/host_beta_tests.log`, plus the new host
temperature/schedule/state test in `build/host_beta_final_tests.log` (the latter
also reruns three operator tests with constant-temperature no-recompile guards).
Coverage includes fixed/packed actual Inductor forward/backward, non-unit beta,
mixed direction, the existing module shape matrix, diagnostics, sequence routing
and checkpoint resume. New host-op profiles reject scalar reads inside either
vocabulary custom op. Host-only storage survives `.cuda()`, and restoring a
saved training step reproduces the temperature schedule without state-key changes.

No final CUDA/Triton source, kernel math, launch defaults or tolerances changed;
therefore no new kernel sanitizer/SASS campaign is claimed. The discarded
device-pointer experiment's SASS checks do not apply to this host-only change.

## 72M checkpoint4000 replay

Read-only r3 checkpoint replay with four resident saved batches, B16/N2048,
four accumulation microbatches, original diagnostics enabled, three warmups,
two phase samples and two six-step throughput windows. Only scratch model and
optimizer copies advance; no official training or dataloader is resumed.

| Measurement | Original device scalar | Host float |
| --- | ---: | ---: |
| Throughput window1 / window2, ms/step | 658.03 / 660.00 | 663.15 / 663.17 |
| Forward event-window mean, ms | 265.41 | 266.93 |
| Backward event-window mean, ms | 385.53 | 376.01 |
| Profile aten::item count | 113 | 65 |
| Profile cudaStreamSynchronize count | 145 | 97 |

Exactly24 forward and24 backward host-beta ops execute in the final profile,
removing48 scalar reads/synchronizations. No material throughput improvement is
established by these short windows (659.01 vs663.16ms); CPU synchronization wait
mostly overlaps already queued GPU work and is not an additive savings budget.
Scratch repeated-step losses/gradients can vary slightly due to atomic reduction
and ready/event dispatch scheduling; initial loss/norm match at printed precision.

Baseline artifacts: `build/device_beta_before.json`, `.trace.json`, and
`device_beta_before_retry.log` (the original launcher log records a missing
tools PYTHONPATH failure, before any benchmark ran). Baseline emulates original
compiler scalar readback using a process-local helper override; no kernel math
changes. Final artifacts: `build/host_beta_after.json`, `.trace.json`, `.log`.
These generated files remain outside Git.
