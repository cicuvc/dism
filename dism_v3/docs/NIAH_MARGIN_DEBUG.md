# NIAH margin probe CUDA illegal-access debug

Date: 2026-10-18

## Outcome

The illegal access was exposed by packing multiple NIAH trials into one
FLA varlen forward. The failing call was not the asynchronous
`solve_tril` / `torch.zeros_like` site in the original traceback. With CUDA
launch blocking and memcheck, the first failing launch narrowed to FLA's
`chunk_local_cumsum_scalar`; single-trial forwards, which still use the
`model.norm` capture hook and `_Stop`, complete successfully.

The standalone probe is fixed by forwarding one trial (one packed document) at
a time. This also matches the known-working in-training NIAH probe, which scores
one sample per call. `--trial-chunk` remains accepted for command compatibility,
but the value is ignored.

A second, independent probe bug was present in the current edit: sequence
assembly and the `xs` / `sl` appends were indented inside the `present` branch.
Consequently, `absent` and `counterfactual` launched on an empty tensor list.
Common assembly now runs after condition-specific insertion.

## Evidence

- The original 2048-context, 2-trial repro failed in `present`, before the
  `counterfactual` condition was reached.
- `CUDA_LAUNCH_BLOCKING=1` reported the failure at
  `fla/ops/common/chunk_scaled_dot_kkt.py:114`; a full
  `compute-sanitizer --tool memcheck --padding 4096` run reported the first
  launch failure at `fla/ops/utils/cumsum.py:268`,
  `chunk_local_cumsum_scalar`. Later sites were async fallout.
- Before the branch fix, one trial completed `present` and then failed because
  `absent` constructed an empty tensor. After the branch fix, the default
  2-trial path still failed in `present`.
- With one trial per forward, 2 trials completed all of `present`, `absent`, and
  `counterfactual` at 2048 tokens. The full 16-trial, three-context,
  three-depth, four-checkpoint run then completed without an illegal access.
- A plain full forward with the final-norm hook capturing without raising
  completed successfully, producing logits of shape `[1, 2048, 32000]`.
  This eliminates `_Stop` as the cause of the kernel fault.
- A separate inline replay of the two-document forward sometimes completed
  normally. Therefore this document does not claim a shape-only deterministic
  violation inside FLA. The fault is a memory-safety / allocation-sensitive
  issue in the installed FLA 0.4.2 + Triton 3.7.1 packed path, consistently
  exposed by this script and removed by single-document execution.

### Hypotheses eliminated

- The counterfactual edit does not trigger the original crash: `present`
  fails first.
- Counterfactual insert length is irrelevant to the illegal access.
- The `_Stop` hook is not causal: it runs after all model layers, the first
  fault is earlier, and both aborted one-trial and plain non-aborting forwards
  succeed.
- Context size is not required: the failure occurs at 2048.
- Trial count alone is not sufficient: one trial passes; the failing
  discriminator is packing multiple trials in one forward.
- The downstream evaluation is not sufficient to cause the fault. It was
  running throughout these diagnostics, while one-document forwards and all
  four full evals succeeded concurrently. No no-concurrency control was run
  because the suite had to remain undisturbed.

## Changes to `niah.py`

Backup:

`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/diagnostics/niah.py.bak`

Backup SHA-256:
`c81b1b08d3b019b25e775ff55612611e959497d473f3f58a7058789c0f5c7afe`

Changes:

1. Documented `--trial-chunk` as a deprecated compatibility option.
2. Set the effective forward step to one trial unconditionally.
3. Moved common `head`, `seq`, `xs`, and `sl` construction outside the
   condition-specific insertion branches.
4. Added an explicit error for an unknown condition instead of silently
   retaining the previous branch's behavior.

No checkpoints, kernels, training code, evaluation-suite code, or result JSONs
were modified.

## Verification and margin results

Reproduction after the fix:

```text
2048  0.50  present     1.652   0.00   0.58
2048  0.50  absent      2.870   0.00   0.08
```

The full run used 16 trials, contexts `2048 8192 16384`, depths
`0.05 0.5 0.95`, and `--trial-chunk 4`. Margin is
`counterfactual_nll - present_nll`.

| Context | Depth | qb000_nosched | qb000_tausched | qb050_nosched | qb050_tausched |
|---:|---:|---:|---:|---:|---:|
| 2048 | 0.05 | 0.616258 | 0.284456 | 0.516796 | 0.339603 |
| 2048 | 0.50 | 0.717497 | 0.920786 | 1.000095 | 0.927182 |
| 2048 | 0.95 | 1.234365 | 1.339902 | 1.274962 | 1.254334 |
| 8192 | 0.05 | 0.573774 | 0.276709 | 0.345922 | 0.186994 |
| 8192 | 0.50 | 0.379055 | 0.303156 | 0.551013 | 0.280688 |
| 8192 | 0.95 | 0.882922 | 1.199820 | 0.802075 | 0.983533 |
| 16384 | 0.05 | 0.296705 | 0.041647 | 0.344765 | 0.078317 |
| 16384 | 0.50 | 0.206204 | 0.079219 | 0.104463 | -0.096973 |
| 16384 | 0.95 | 0.716241 | 0.551745 | 0.281489 | 0.545381 |

| Checkpoint | Mean margin over 9 cells | Mean at 2048 | Mean at 8192 | Mean at 16384 |
|---|---:|---:|---:|---:|
| qb000_nosched | 0.624780 | 0.856040 | 0.611917 | 0.406383 |
| qb000_tausched | 0.555271 | 0.848381 | 0.593228 | 0.224204 |
| qb050_nosched | 0.580176 | 0.930617 | 0.566337 | 0.243572 |
| qb050_tausched | 0.499896 | 0.840373 | 0.483738 | 0.175575 |

Result files:

- `.../diagnostics/niah_results/qb000_nosched_margin.json`
- `.../diagnostics/niah_results/qb000_tausched_margin.json`
- `.../diagnostics/niah_results/qb050_nosched_margin.json`
- `.../diagnostics/niah_results/qb050_tausched_margin.json`

The component present, absent, and counterfactual NLL values are retained in
those JSON files.

## Limitation

Memcheck established the failing FLA launch but did not provide a usable
out-of-bounds address before the context became unusable. The conservative
single-document workaround is verified end to end. A future FLA/Triton upgrade
should be validated with a dedicated multi-document forward before batching
NIAH trials again.
