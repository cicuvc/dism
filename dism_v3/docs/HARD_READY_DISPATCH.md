# Nonblocking hard backward selection

Initially an opt-in experiment after `fbef43c` (hard quadratic prototype).
As requested on 2026-10-15, `ready` is now the production default, together with
compact planner metadata. `DISM_HARD_BACKWARD=event` retains the explicit
wait-for-plan control, and `DISM_PLANNER_COMPACT=0` selects classic metadata.
The measurements below predate that default change unless explicitly stated.

Set `DISM_HARD_BACKWARD=ready` before the default planner is first created, or
use `AsyncHardPlanner(backward_policy='ready')` as the explicit eager override.
`event` retains the original wait-for-plan policy. `quadratic` is a diagnostic
forced-fallback mode (it still submits planning during forward).

At the hard backward boundary, after queuing soft backward work:

1. If the CPU future is incomplete, run the CUDA quadratic fallback.
2. If complete, propagate a stored planner exception rather than suppressing it.
3. Query the final H2D event without synchronization. Only a completed upload
   selects event backward; otherwise use the fallback.

The last event covers bulk metadata and stream-bucket uploads. `future.done()`
alone is insufficient: it only guarantees the work has been enqueued. Readiness
is sampled on the host, so a plan that would become ready before the queued GPU
backward actually executes can conservatively fall back. There is no grace
period, host spin, device-side dynamic dispatch or wait before choosing.

Fallback marks the invocation abandoned for planning. The coordinator drains
the already-issued D2H copy while retaining pinned buffers, then skips native
planning if not started. A native plan already running is allowed to finish;
if abandoned in the meantime, its upload is omitted. This does not forcibly
interrupt C++ or free live copy buffers. Existing ticket capacity/leases remain
in effect, and shutdown still drains jobs. `Future.cancel()` is not used.
An already-enqueued upload retains its ordinary allocator/event ownership.

Retained graphs can execute again: an abandoned/no-result plan uses fallback;
a completed uploaded plan can use events. Different routes need not be
bitwise-identical (BF16 reductions and analytic tau implementations differ).
No hard/direction resampling occurs. Completed planner failures propagate when
observed; an abandoned background job is not synchronously joined by backward.

## Initial fallback implementation

The fixed kernel is unchanged. `backward_hard_recompute` obtains integer
boundaries, LUT and normalizer by rerunning hard forward, then invokes the
quadratic backward. No additional quadratic forward checkpoint is kept alive
on every layer. Recomputing readout is extra work, included in measurements.

Packed documents are grouped by exact length using the existing CPU layout.
Each bucket is stacked into fixed-length tensors, computed independently,
then scattered back to packed positions. Empty documents are skipped. Hard
flags are selected from the **original equal-length training row**, not sampled
or reassigned per document. No padding or cross-document attention is introduced.
These copies and Python launches are prototype overhead, not an optimized native
packed CUDA implementation. Native packed tasks and forward-saved small
statistics remain possible follow-up optimizations.

Eager and compiler core/vocabulary backward share the same ticket selector.
Custom-op input/output contracts are unchanged, so saved-tensor ownership,
checkpoint version checks and existing AOT handles remain in place. Fallback
uses saved invocation inputs, not hidden ticket-owned model tensors. No changes
to RNG counters, annealing, optimizer grouping/schema or dataloader position.
Async CUDA Graph capture remains unsupported.

## Reproduction and validation

Use conda `blkw`, only this worktree's Python packages, and the existing compiled
eight-configuration CUDA extension. No CUDA kernel bodies/default launches were
changed in this policy experiment; prototype CALL/spill/sanitizer limitations
remain in [HARD_BACKWARD_QUADRATIC.md](HARD_BACKWARD_QUADRATIC.md).

```bash
python -m pytest dism_v3/tests/test_hard_ready_dispatch.py dism_v3/tests/test_hard_training.py -q
```

40 cases passed: pending/ready/upload-pending/failed decisions, packed and fixed
FP64 oracle gradients for soft/hard/mixed masks, a deliberately blocked planner
with successful retained-graph backward before release, compiled core fallback,
and original event-path ownership tests. Existing tolerances are unchanged.

`tools/bench_model_training.py --hard-backward-policy event|ready|quadratic`
selects the policy for scratch checkpoint replay. Diagnostic route recording no
longer waits on incomplete futures; such records report `plan_unavailable` and
omit timing fields that cannot yet be read. GPU/host metrics must distinguish
actual backward waits from later diagnostic/shutdown drains.

## Idle-GPU 72M replay (2026-10-14)

RTX5090, checkpoint step19074, 72,035,808 parameters, B16/N2048,
four accumulated microbatches, six DISM layers, scheduled sequence-hard
probability0.95, true direction, 16 unbound planner workers. Same scratch model,
optimizer state and four resident checkpoint-prefetched batches; linear_fla
frontend and process-local chunked document-balanced CE review wrapper.
Document counts34/30/32/43 are normalized over all139 documents in the step.
No checkpoint writes, dataloader advancement or external training run.

Three sustained windows of eight optimizer steps after four warmup steps:

| Metric | event | ready |
| --- | ---: | ---: |
| Window step times, ms | 937.30 / 927.54 / 922.42 | 792.07 / 797.05 / 791.35 |
| Median step, ms | 927.54 | 792.07 |
| Median tokens/s | 141,312 | 165,480 |
| Phase forward per microbatch, ms | 99.78 | 97.88 |
| Phase backward per microbatch, ms | 130.85 | 98.42 |
| Diagnostic-step planner host wait, ms | 161.35 | 0.004 |
| Diagnostic-step routes | 24 event | 16 event + 8 quadratic |
| Peak allocated CUDA memory, decimal GB | 18.792 | 18.792 |

Step time fell14.6%, sustained throughput rose17.1%. Phase figures are means
over three separately measured steps, divided by four microbatches, not sums
used to reconstruct the sustained-window median. Route/wait statistics are
from a separate diagnostic step, not counters over the full sustained windows.
Each diagnostic microbatch selected quadratic for its first two hard backwards
(the two highest DISM layers in reverse order) and events for the other four.
The fallback's packing, full hard-forward state recomputation and scattering
are included. This supports the intended latency-absorption mechanism even
with the current unoptimized packed fallback; it does not establish long-run
convergence or live-dataloader throughput. Forward-state/gradient semantics,
production default and checkpoint schema remain unchanged.

Reports: `build/hard_dispatch_event_idle.json`,
`build/hard_dispatch_ready_idle.json`, with corresponding `.log` files.
The earlier `hard_dispatch_event_20261014` run overlapped another GPU workload
and is excluded from this comparison. GPU process checks around these idle
runs found no competing compute process. The benchmark retains existing Dynamo
recompile-limit warnings from training-probe hooks; both runs use the same
probe settings, and compilation warmup is outside measured windows.

A subsequent event-policy repeat (three warmups, two eight-step windows) gave
944.07/939.77ms per step, corroborating that the gain was not an improving
clock/order trend. Report: `build/hard_dispatch_event_idle_repeat.json`.
