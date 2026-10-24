# Transformer baseline: native RoPE and inference-only YaRN

## Follow-up: independent windows must retain BOS

The original 22 jobs completed, but their long-text scores exposed a mismatch
between training inputs and the evaluation adapter. Training prepends BOS to
every independently attended document (including cropped documents). Harness
rolling windows use a predecessor token rather than BOS for subsequent windows,
while resetting attention history and positions. Generation's left truncation
also discarded BOS. The original protocol's `bos=True` did not describe these
cases accurately.

A controlled checkpoint probe held target tokens, window lengths, and weights
fixed and replaced only the oldest context token with BOS. On the first three
PG19 documents, six complete subsequent windows per context gave:

| Context / RoPE | Original-window PPL | BOS-window PPL |
| --- | ---: | ---: |
| 2K native | 246.81 | 45.42 |
| 4K native | 875.00 | 155.47 |
| 4K YaRN | 235.32 | 42.95 |
| 8K native | 2598.76 | 671.66 |
| 8K YaRN | 229.97 | 44.87 |

On four complete 2K subsequent windows from the first three GovReport documents,
PPL changed from 163.33 to 19.33. These are diagnostic subsets, not full-dataset
replacement scores. No complete subsequent 8K windows existed in those three
GovReport documents; no 8K estimate is reported for that subset.

The effect supports a BOS dependency consistent with an attention sink; headwise
attention weights were not measured, so that specific mechanism is not yet
established. Native RoPE extrapolation remains problematic even with BOS.

Other checks: the first PG19 2K prefix produced exactly equal hidden states and
losses at physical batch 1 and 4. Fused CE agreed with direct FP32-logit CE within
5.3e-6 on sampled positions. An independent PyTorch rotary/SDPA path gave mean
absolute token-loss difference 0.0424 with BF16 intermediates, far smaller than
the BOS effect; this is not an exact numerical-equivalence claim.

The corrected evaluation lives separately in
`/home/cicuvc/downstream-eval/results/bos_corrected_20261025`, driven by
`run_bos_corrected_20261025.py` and frozen `adapter_bos_*_20261025.py` files.
Every independent input starts with BOS. When a context is already full, only
its oldest context token is replaced; all target tokens and disjoint target
windows remain unchanged. Truncated generation reserves BOS within the existing
input budget. Tests checked full/tail rolling windows, unchanged target counts,
long-prompt generation, direct CE, and independent padding.

The 26-job rerun includes Transformer native/YaRN rolling evaluations at the
original contexts and SWDE/FDA, plus DISM C rolling at 2K/4K/8K and SWDE/FDA at
2K/4K under the same BOS protocol. Short, untruncated MC/LAMBADA requests need
no rerun. Original results and checkpoints remain untouched. Diagnostic raw logs
and JSON are under ignored `build/check_transformer_*`, `build/probe_rolling_bos.*`
and `build/bos_corrected_smoke.log`.

## Run and scope

Started 2026-10-24 from the completed postnorm FLA Transformer baseline,
`model_022888.pt` (123,489,024 parameters, 12 layers, width 768, FFN 1664,
training context 2048, RoPE theta 10000). The source run is recorded in
`build/fla_transformer_run.txt`. Checkpoint weights are not changed and no
long-context finetuning is performed. The concurrent LayerNorm/sigmoid DISM
pretraining run remains active.

Evaluation workspace: `/home/cicuvc/downstream-eval`.
Runner: `run_transformer_yarn_20261024.py`.
Results: `results/transformer_yarn_20261024` (state.json, summary.json,
per-job results, samples, protocol and provenance hashes).

The 22 serial jobs cover:

- Native RoPE at 2K: ARC Challenge/Easy, HellaSwag, WinoGrande, PIQA,
  LAMBADA, SWDE, FDA, PG19 and GovReport.
- Native RoPE and YaRN at 4K: all those tasks.
- Native RoPE and YaRN at 8K: PG19 and GovReport.
- Qasper and SQuAD are excluded, following the preceding evaluation protocol.

PG19 and GovReport measure rolling language-model likelihood/perplexity,
not summarization quality. Evaluations are full-dataset, zero-shot, with
physical batch 4 and 256-aligned independent padding rows. The adapter uses
BF16 fused CE for scoring, preserves MXFP8 model projections where configured,
and bounds shape-dependent MXFP8 caches. Each process caps the PyTorch CUDA
allocator at 25% of device memory to coexist with training. Datasets and tokenizer
are local; offline mode is enabled.

## YaRN protocol

`nanochat/nanochat/models/fla_transformer/yarn.py` is an opt-in evaluation-only
rotary replacement. Training defaults and checkpoint state are unchanged.
The static factor is declared evaluation context / original context: 2 at
4096 and 4 at 8192. It never depends on a request's actual length or the
combined length of packed rows. Native FLA packed rotary position resets remain
in effect.

The inverse-frequency blend uses beta_fast=32, beta_slow=1 and truncated
correction bounds. Both sine and cosine tables receive magnitude
`1 + 0.1 * log(factor)` in FP32 before conversion to model dtype. This follows
the default non-multiscale YaRN parameter computation in
[Transformers rope utilities](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py),
based on the [YaRN paper](https://arxiv.org/abs/2309.00071).

Inference-only application is an experiment, not evidence that this checkpoint
has learned long-context behavior. Even short requests use the selected static
factor, so their results may change. Paired native-RoPE runs distinguish the
effect of YaRN from simply allowing a longer input. Comparisons against earlier
DISM evaluations should retain the context and RoPE protocol labels.

## Validation before launch

- `nanochat/tests/test_yarn_eval.py`: 4 tests passed, including factors 1/2/4
  against installed Transformers parameters, exact factor-1 native parity,
  packed-vs-independent position reset parity, and 8K table magnitude checks.
- Final-checkpoint 8K YaRN smoke passed: direct vs adapter likelihood and
  independent padding checks, greedy generation, and finite full-context output.
  Peak allocated CUDA memory was 1.11 GB for this smoke (not a bound on all
  downstream requests).
- Logs are under ignored `build/transformer_yarn_tests.log` and
  `build/transformer_yarn_smoke.log`. Existing dependency/deprecation warnings
  remain visible. No CUDA kernels were changed; no sanitizer run was performed.

## Completed BOS-corrected evaluation

All 26 jobs completed successfully. The exact corrected adapter and sweep sources
are preserved in `tools/downstream_eval/`. These are workspace deployment scripts:
copy the three files together into the existing `/home/cicuvc/downstream-eval`
workspace before running them with conda `blkw` and this checkout on PYTHONPATH.
They require the existing local tasks, tokenizer, data, harness and run pointers;
they do not download assets. Completed matching jobs are skipped on restart.
Machine-specific paths are intentionally retained to reproduce this evaluation.
Raw results and logs remain outside Git.

| Task | Context | DISM C PPL | Transformer native PPL | Transformer YaRN PPL |
| --- | ---: | ---: | ---: | ---: |
| PG19 | 2048 | 39.55794 | 40.47244 | — |
| PG19 | 4096 | 38.07074 | 150.14573 | 39.31738 |
| PG19 | 8192 | 37.18578 | 705.93270 | 40.77819 |
| GovReport | 2048 | 19.01904 | 19.53797 | — |
| GovReport | 4096 | 17.90599 | 99.05997 | 18.40987 |
| GovReport | 8192 | 17.28239 | 733.72660 | 18.67110 |

| Task, 4K | DISM C contains % | Transformer native % | Transformer YaRN % |
| --- | ---: | ---: | ---: |
| SWDE | 15.3015 | 22.0522 | 31.0531 |
| FDA | 14.9728 | 3.9020 | 17.1506 |

BOS correction removes the large in-context anomaly. Transformer native 2K PPL
is only 2.31% / 2.73% above DISM C on PG19 / GovReport. YaRN restores usable
4K/8K scoring, but both Transformer PPLs worsen from 4K to 8K, while DISM improves
monotonically. Transformer plus YaRN leads on the two 4K extraction tasks.
These single-run results do not measure training-seed uncertainty or establish
mixer-only causality: Transformer trained with MXFP8 CE, DISM C with BF16 CE.
