# Opt-in top-four hard training

2026-10-25. Implementation follow-up to [the checkpoint readout screen](TOPK_HARD_READOUT.md).
This is a training baseline, not evidence of trained quality. No long training
or checkpoint modification was performed during implementation.

## Definition and shared work

Keep the native v3 soft path, K argmax and one committed Q argmax history. At
query i, select the native Q argmax plus the three highest other Q logits. Only
that current read branches; the other candidates never enter future history.
The four logits are renormalized over the selected set, not over the full
vocabulary with missing mass discarded. SQ/SK/V/tau are shared.

For candidate c, with document/causal masking implicit:

```
A_ij = exp(tau) * (1 + H_hard[i-1,j-1])
D_ic = 1 + sum_{j: k_symbol[j]=c} A_ij
alpha_ij = pi_ic * A_ij / D_ic,  c=k_symbol[j] in selected[i]
alpha_ij = 0,                  otherwise
O_i = sum_j alpha_ij * dot(sq_i,sk_j) * v_j
```

The partition by the single K symbol lets all four candidates share one
SQ@SK.T and one weighted V read per pair tile. There are no four copies of the
history recurrence, vector forward, or vector backward. A scalar first pass
computes the four denominators before the vector readout pass.

With upstream g_i and `f_ij=alpha_ij*dot(sq_i,sk_j)*dot(g_i,v_j)`, the same
backward traversal also computes:

```
U_ic = sum_{j: k_symbol[j]=c} f_ij
(dL/d selected_logit)_ic = U_ic - pi_ic * sum_c U_ic
(dL/d tau) = sum_ij f_ij * (dlog(A_ij)/dtau - dlog(D_ic)/dtau)
```

Thus router backward needs four scalars per query, not saved four-candidate
output vectors. Selected logits feed Q and Q-codebook gradients, using BF16
operands like the native vocabulary projection. Discrete support changes,
committed Q history and K labels remain nondifferentiable. Native soft gradients
are added normally; codebook and input-projection sharing are unchanged.

## Implementation

- [topk_hard.py](../python/flash_dism/kernels/topk_hard.py): GPU selection,
  hard-history checkpoints, stable per-candidate normalizers, shared vector
  readout/backward, and router Q/codebook backward.
- [topk_training.py](../python/flash_dism/topk_training.py): autograd integration
  reusing native soft/interpolation forward and backward. The default native
  dispatch is unchanged.
- `hard_query_topk=4` in the nanochat model spec opts in. Omitted/1 preserves
  existing behavior and legacy spec identity. Only 1 and 4 are supported.
  No parameters, state_dict keys, RNG counters, schedules or optimizer groups
  are added. `hard_prob_fixed=0.5` is a separate, explicit experiment choice.
- Supported: aligned CUDA BF16 training/eval, fixed true direction, D32/64,
  R16/32, DV32/64, validated 256-aligned documents inside equal training rows,
  eager autograd and the existing `PackedTrainingGraph` with planner disabled.
- Unsupported paths reject explicitly: planner, Dynamo, Torch/cache fallback,
  SAM prefill/decode. Set `DISM_HARD_PLANNER=0` and do not pass
  `--cuda-graph-planner`. This baseline does not provide subquadratic execution.

History checkpoints occupy `B*H*floor((N-1)/32)*N` int32 elements, where N is
training-row length. No dense N-by-N weight matrix or N-by-N-by-top-k tensor is
materialized. This still has quadratic/32 saved history storage, in addition
to the native soft state. Candidate IDs/probabilities/log-denominators/derivative
moments occupy O(B*H*N*4). Router vocabulary backward uses a tiled reduction
without contended per-token codebook atomics.

The hard vector pass uses query tiles of 32 and key tiles of 64; normalization
and backward use 32-by-32 tiles. This keeps their register spills at zero on the
measured shapes. Segment handling relies on the validated 256-alignment:
32/64-wide tiles contain no interior document boundary. No shared FP32 atomicAdd
is introduced. Global FP32 gradient atomics remain, so this is not a new claim
of bitwise/batch-invariant gradients.

## Validation

`check_topk_training.py` compares forward, dSQ/dSK/dV/dtau and selected-logit
gradients against an independent FP64 sequential oracle, at N256/512/1024,
random and all-equal symbols, tau=0/.1, and document boundaries. Output relative
L2 is about .00235; vector gradients about .00165; tau/logit errors below 4e-6
in the checked N512/1024 cases. Vector cosine exceeds .999997.

`check_topk_router.py` independently checks Q and Q-codebook gradients against
FP64 differentiation with BF16 operands, including mixed hard/soft rows, D32/64
and C4/8/512. Relative errors are below 1e-5. A packed all-hard vocabulary
integration check also compares the complete autograd result against FP64:
output/vector/Q relative L2 is below .0024, Q-codebook error about 2.4e-7,
and tau error about 1.4e-6; K and K-codebook hard gradients are exactly zero.

The nanochat regression tests cover unchanged all-soft outputs/gradients,
configuration identity/parameter alias/no-decay schema, eager versus graph
replay with changing documents and hard probabilities, optimizer-state restore,
RNG/annealing buffers, pre-existing gradients and capture-state preservation.
They use scratch models only. The existing native compiled checkpoint-resume
regression also passes, checking that the default compilation path remains intact.

`audit_topk_training.py` exercises the new kernels under graph replay while
changing segments/masks and then clearing all hard flags. At B2,N512,H2,D64,
R32,DV64,C512, Compute Sanitizer memcheck and synccheck report zero errors;
racecheck reports zero hazards/errors/warnings. No tool exception is applied.
SASS audit finds no CALL instructions in all eight new kernels. Normalization,
vector forward/backward and router gradients have zero spills; selection has
8 spill slots and the history-boundary kernel has 2 for this audited shape.
This is a bounded-shape audit, not exhaustive coverage of all launch shapes.

These checks do not establish accuracy for arbitrarily long identical-symbol
chains at large tau. Log denominators prevent direct exponential overflow, but
subtracting nearby FP32 tau moments can lose relative accuracy when the true
gradient is tiny. The native BF16/relaxed-gradient limitations are unchanged.
No convergence/quality claim follows from the finite synthetic training steps.

## Throughput

RTX 5090, 123,473,088 parameters, `gggddddddggg`, 2K context, 8 rows x 8
microbatches, MXFP8 FFN/optimizer, BF16 CE, graph on. Final measurements use
`bench_topk_model.py`: same initialization and synthetic inputs, four warmup
and ten measured optimizer steps for each of two resident models. Measurement
order alternates each round. Timings include metadata/H2D, forward/backward,
document normalization, clipping, planner draining where enabled, and optimizer;
exclude data loading, training instrumentation, evaluation and checkpoint I/O.
Top-four always uses its GPU quadratic path, with planner disabled.

| Hard probability | Packing | Top-1 planner | Top-1 step | Top-4 step | Top-1 token/s | Top-4 token/s | Throughput drop |
|---|---|---|---:|---:|---:|---:|---:|
| .5 | Full 2K documents | Off | 1000.8 ms | 1058.7 ms | 130,971 | 123,805 | 5.47% |
| .5 | Changing document lengths | On | 854.1 ms | 938.6 ms | 153,464 | 139,647 | 9.00% |
| .95 | Changing document lengths | Off | 875.3 ms | 968.1 ms | 149,745 | 135,392 | 9.58% |

Changing lengths cycles through 2K, 256+1792, 4x512 and 8x256 documents within
each training row. The final kernels find each tile's containing document on
device and traverse only that interval. This removes the earlier full-row
masked work on short documents; mutable graph metadata remains supported.

The fixed-.5 baseline meets the under-10% target in the measured comparisons,
including the stronger planner-on top-one baseline for changing documents.
The margins near 9–10% are small: these are measured synthetic configurations,
not a guarantee across layouts, document distributions or hard probabilities.
In particular the .95 comparison uses planner-off top-one; no .95 planner-on
penalty is established. Mean step overhead and throughput loss are different:
9% throughput loss corresponds to about 9.9% longer steps.

Earlier separate-process measurements, before document-range skipping, gave
2.44%/7.28% loss against planner-off top-one on full/changing documents, but
15.0% loss against planner-on top-one. Separate-run clock variation made the
full-document figure optimistic; use the interleaved final results above.
The same graph geometry measured about 5.4 GiB peak allocated per scratch model;
no dense pair-weight storage was added by the interval optimization.

Ignored manifests/configs/logs live in `build/topk_training/`. Final paired
reports are `paired_p50_full_docskip.json`, `paired_p50_planner_docskip.json`, and
`paired_p95_changing_docskip.json`. Earlier `final_top*` and `limit_top*` logs
are development measurements, not final performance claims. Reproduce with
`tools/bench_topk_model.py`, the 123M model spec, `--probability .5`, and explicit
`--packing full|changing` / `--top1-planner` as applicable. Always benchmark with
no other GPU workload. The script creates scratch models and never loads a
checkpoint or dataset.

## Further work

The largest remaining structural opportunities are to fuse top-four selection
into the existing Q vocabulary projection, and
extend the event/planner route to local alternative symbols. The latter needs
weighted candidate denominators and router moments; it cannot reuse top-one
plans unchanged. It should retain one committed history rather than a beam of
four growing histories. These are not requirements to use this quadratic
training baseline, and are not implemented here.
