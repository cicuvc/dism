# Sampled hard read with a zero-history control

2026-10-26. Opt-in training implementation of
[the local sampling/control-variate diagnostic](LOCAL_SAMPLING_CONTROL_VARIATES.md).
Existing top-four and native top-one defaults are unchanged. No long training
was started or restarted; the running top-four job retains its original code
and configuration.

## Operator and gradients

Keep native argmax Q history and argmax K labels. Select the same four current
Q candidates as existing top-four training, preserving native argmax as slot 0.
Normalize probabilities pi over those four candidates and sample one slot C.
The sample affects only the current read, never subsequent Q history.

Let H_c be the complete candidate-normalized suffix read, and B_c the same
candidate read with history removed (continuation weight exp(tau)). Output:

```
O = sum_c pi_c B_c + H_C - B_C
```

Thus E[O] is exactly the full top-four output at fixed input/history/support.
For a pair whose K symbol is candidate s, define

```
b_s = 1 / (exp(-tau) + count_s)
h_ij = A_ij / (1 + sum_{j:k_j=C} A_ij), when s=C; otherwise 0
m_ij = (pi_s - 1[s=C]) * b_s + h_ij
```

The implementation uses this signed pair weight in one shared vector read and
one shared backward traversal. SQ/SK/V gradients differentiate the entire
output, including both control terms. Tau uses the distinct base and hard
normalizer derivatives; the control is not detached from these gradients.

The router uses the explicit surrogate VJP

```
U_s = pi_s * <g,B_s> + 1[s=C] * <g,H_C-B_C>
dz_s = U_s - pi_s * sum_t U_t
```

For fixed output cotangent g this estimates the full top-four router gradient
without bias. It is deliberately not the ordinary derivative obtained by
treating C as a constant: that would omit the residual score term. This does
not establish an unbiased gradient of the complete sampled-network LM loss,
because its cotangent depends on the draw. No convergence/quality claim is made.
As in existing top-four, discrete support/history and hard K labels receive no
gradient; native soft gradients are retained and added normally.

## Implementation and configuration

Enable both fields in a DISM model spec:

```json
{"hard_query_topk": 4, "hard_query_sampled": true}
```

Omitting `hard_query_sampled` preserves existing behavior. Parameters, state_dict
keys, no-decay grouping, optimizer schema, and annealing buffers are unchanged.
The model-spec fingerprint distinguishes the new option. Set
`DISM_HARD_PLANNER=0`; CUDA Graph is supported, planner/cache/Dynamo remain
explicitly unsupported just as for full top-four.

- `python/flash_dism/kernels/sampled_hard.py`: categorical sampling, integer
  prefix counts, one-candidate hard normalizer, signed vector read/backward.
- `python/flash_dism/topk_training.py`: existing native-soft integration with
  an explicit sampled backend; shared vocabulary selection and Q/codebook VJP.
- nanochat passes the existing per-microbatch counter snapshot and layer seed.
  The sampler uses a separate XOR seed domain and token/head/row offsets. It
  does not consume extra model-counter increments or global Torch RNG draws.
  Backward saves the draw rather than resampling. Graph replay reads the new
  counter value; graph construction restores counter/global RNG/optimizer state.
- Evaluation with this option uses the frozen model counter, making the draw
  deterministic for fixed inputs and state. It does not silently switch to full
  top-four or argmax evaluation. Cached decode is not implemented.

The current fast path still uses quadratic history/vector traversal. It does
not yet materialize per-symbol SK/V prefix matrices or use an event planner.
The four base denominators use exact integer prefix counts: histogram each
32-token K block, scan block counts, subtract the prefix at the current document
start, and add the causal partial current block. This removes four-candidate
counting from the expensive history traversal. The hard normalizer processes
only the sampled candidate with a 32x64 tile; backward retains 32x32 tiles.

Temporary histogram storage is O(B H (N/32) nextpow2(C)); saved history storage
remains O(B H N^2/32), as in full top-four. No dense N-by-N-by-vocabulary tensor
is introduced. Global FP32 atomics remain in backward, so this is not a claim
of batch-invariant results. Histogram shared atomics are integer, not FP32.

## Validation

`tools/check_sampled_training.py` checks an independent FP64 recurrence and
autograd oracle for forward, SQ/SK/V/tau, and the corrected router VJP, at
N256/512/1024, R16/32, DV32/64, random and identical-symbol strings, tau=0/.1,
and document boundaries. Representative forward relative L2 is about .0024,
vector gradients about .0016–.0024; the same tolerances as existing top-four
checks are retained. Long-chain tau derivatives remain subject to FP32
subtraction/cancellation limits; the bounded tests do not certify arbitrary
lengths/tau.

The same tool also enumerates all four draws in a small independent FP64 test:
the expected output and **all five** SQ/SK/V/tau/router VJPs match complete
top-four at atol=1e-11, rtol=1e-10. This directly tests both control subtraction
and the router expectation correction.

`tools/check_topk_router.py --sampled` verifies packed vocabulary integration:
output and vector/Q-input gradients below .0024 relative L2 in the checked
case, Q-codebook and tau about 2.5e-7 and 1.2e-7. Hard K and K-codebook gradients
are exactly zero. Shared router checks also cover D32/64 and vocabularies
4/8/512.

Nine top-four/sampled regression tests pass: configuration/schema, unchanged
all-soft behavior, categorical frequency and seed separation, changing graph
counter draws, eager/graph consistency with changing documents/hard masks,
LayerNorm/sigmoid and MXFP8 optimizer integration, and restored optimizer,
counter, annealing and existing-gradient preservation. Eleven native adapter
tests pass, including original compiled checkpoint resume.

Compute Sanitizer memcheck and synccheck report zero errors; racecheck reports
zero errors, warnings, and hazards. The audited graph mutates segments/masks,
includes identical-symbol chains, and ends with an all-soft zero-output check.
The audit shape is B2,N512,H2,D64,R32,DV64,C512. No sanitizer exception applies.

Machine-code audit finds no CALL in the exercised kernels. New sampling,
histogram, prefix scan, normalizer, forward, and backward kernels have zero
spills at this shape. Reused selection has 8 spill slots and reused history
boundaries have 2, matching the previous implementation. Backward tile-64
variants with spills were tested and not selected.

## Throughput protocol

Compare scratch models with the running job's 123,473,088-parameter spec:
`ggggggdddddd`, LayerNorm/sigmoid, N2048, MXFP8 FFN/optimizer, BF16 CE, CUDA Graph,
planner off. Both modes have the same seed, initialization, generated inputs,
hard probability, optimizer and clipping. Inputs cycle full 2K, 256+1792,
4x512 and 8x256 documents. Include metadata/H2D, forward/backward, document loss
normalization, clipping and optimizer; exclude loader/evaluation/checkpoint I/O.
Repeated synthetic losses only check finiteness, not training quality.

User-authorized SIGSTOP preserves the original training's allocated memory.
The original job retains about 18 GB; B8 scratch preflight exceeded the remaining
memory even with a single resident test model. Those failed preflights produced
no valid timings, and the job was resumed after each failure. Therefore the
successful comparison uses **B4 with accumulation 16**, preserving 131072 tokens
per optimizer step. These absolute rates must not be reported as B8 rates.

To avoid multiple resident graphs, use separate processes in A–B–B–A order,
four warmup and sixteen measured optimizer steps per process. Report both
repetitions rather than treating separate-run clocks as identical. The stop
wrapper verifies PID identity and uses a separate pipe watchdog to issue CONT
even if the controller dies. Failure and controller-SIGKILL recovery were
checked on a disposable process before use on training.

At hard probability .5:

| Mode | Run 1 ms/step | Run 2 ms/step | Combined ms/step | Combined token/s |
|---|---:|---:|---:|---:|
| Full top-four | 1072.54 | 1076.44 | 1074.49 | 121986 |
| Sampled control | 1064.20 | 1049.68 | 1056.94 | 124011 |

Mean throughput improvement is **1.66%**. Both sampled run means are faster than
both full top-four run means, but the gain is modest and individual step times
vary. This is not a universal speedup claim across batches/packings/hardware.

At hard probability .95, using the same A–B–B–A protocol:

| Mode | Run 1 ms/step | Run 2 ms/step | Combined ms/step | Combined token/s |
|---|---:|---:|---:|---:|
| Full top-four | 1118.82 | 1131.37 | 1125.10 | 116498 |
| Sampled control | 1098.24 | 1095.73 | 1096.98 | 119484 |

Mean throughput improvement is **2.56%**; both sampled run means again beat
both control run means. Each probability has 32 measured optimizer steps per
backend, plus eight warmup steps per backend. `final_summary.json` aggregates
the raw step timings; `manifest.json` records source/config hashes and protocol.

The training process was resumed after every measurement window and verified
running with advancing training-log steps after the final window. Its loaded
model, optimizer, RNG counter, data iterator and configuration were not replaced.

An initial version that counted all base candidates inside history traversal
was about 1.8% slower in a preliminary A/B run. That exploratory result is kept
in the artifacts and is not the final implementation's result.

Reproduction from the workspace root (conda blkw, only this checkout on the
DISM/nanochat PYTHONPATH; an idle GPU or authorized stop wrapper is required for
meaningful throughput):

```bash
python dism_v3/tools/bench_topk_model.py \
  --model-spec nanochat/configs/sequence_true_123m_layernorm_sigmoid_top4_slimpajama_3b.json \
  --single-mode sampled --rows 4 --accum 16 --probability .5 \
  --packing changing --warmup 4 --rounds 16 \
  --output dism_v3/build/sampled_training/fresh_sampled.json
```

Repeat with `--single-mode top4` and fresh output paths in A–B–B–A order.
Numerical checks, sanitizer/SASS audits, preliminary and final throughput JSON,
logs, and runner scripts are under ignored `build/sampled_training/`. Generated
binaries/results are not source-controlled. The running top-four training is
never converted to the new mode by these tools.

## Queued full pretraining run

2026-10-26 03:54 +08:00: the user authorized a fresh sampled-control run after
the current full top-four run completes. The background queue waits for exit
code zero and the step-22888 checkpoint before launching; it does not restart
failed or already-started jobs. Only `hard_query_sampled=true` changes in the
model spec. The original B8, total batch 131072, 3B-token/22888-step budget,
linear hard schedule, old layout, LayerNorm/sigmoid, graph on, planner off,
MXFP8 FFN/optimizer and BF16 CE remain unchanged.

Config: `nanochat/configs/sequence_true_123m_layernorm_sigmoid_sampled_cv_slimpajama_3b.json`.
Ignored pointers `build/sampled_layernorm_sigmoid_training_run.txt` and
`build/sampled_training_queue.txt` locate the durable data-disk run and queue;
`build/sampled_training_queue/state.json` exposes queue state. The run contains
the exact command, source snapshot/hashes, CPU configuration/schema preflight,
and supervisor. A launch-time check requires the reviewed runtime source and
config hashes to remain unchanged. Training imports this worktree's packages.
