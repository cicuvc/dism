# Local 50M autoregressive smoke test

Checkpoint: local original hybrid, 49,678,876 parameters, `latest.pt` at
step 30000 in the `dism-swa50m-softcap30-20260909-offline` run. No checkpoint,
training code, installed dependencies or running jobs were modified.

`generate_lm.py` supports original CUDA full-prefix recomputation (`prefix`,
default) and GPU projections/FFN/SWA + exact CPU hard cursor cache (`cpu-cache`).
Both are batch-one, eval/inference-mode, pure hard, bounded by context2048.
Cached decoding uses FLA shortconv state, absolute RoPE offsets and last128
SWA keys/values. DISM stores key labels/values and sparse cursors on CPU.
Sampling applies checkpoint softcap30 before temperature/top-p; no repetition
penalty, BOS insertion or instruction wrapper. EOS stops generation.

## Observations

- Existing independent pure-hard validation: 100 effective batches of64,
  13,107,200 target tokens, NLL3.44844508, perplexity31.45145, all finite.
- Four prompts: solar system, photosynthesis, village story, circle area.
  Both backends generated96 tokens per prompt at temperature0.8/top-p0.9,
  seed1234; no EOS or nonfinite-logit failures. Story continuation was locally
  coherent; science responses contained factual errors; math repeated itself.
- Original-prefix greedy generation also completed all four but showed severe
  repetition. Therefore repetition is not specific to the new CPU cache.
  This is a small qualitative smoke test, not a capability benchmark or a
  diagnosis of why the model repeats.
- After the T=1 workaround below, a24-position teacher-forced comparison of
  capped logits gave cosine>=0.9998288 and top1 agreement23/24. Max absolute
  difference0.4511; exact equivalence is NOT established. FP32 vocabulary label
  GEMM, exact CPU recurrence versus training tile approximation and BF16
  GEMM/output rounding differ. Discrete labels can amplify differences.
- Short prompts,96-token samples: prefix roughly1.96–2.12seconds including
  prefill; CPU-cache2.41–3.55seconds including prefill/first-call warmup. These
  are one-shot observations, not a controlled speed benchmark. GPU↔CPU
  synchronization per layer means this version is not a proven speedup.
- Tests: `tests/test_lm_generation.py` plus `tests/test_hard_decode_cpu.py`:
  **46 passed**. Include convolution state versus129-token full execution,
  RoPE/SWA at positions0/1/126/127/128/136, sampling and CPU core oracle tests.

### Installed FLA T=1 issue

In `fla/modules/conv/triton/kernels.py::causal_conv1d_update_kernel`, insertion
of the current input is inside `if USE_INITIAL_STATE`. Stateless T=1 forward
with no cache consequently omits x. Explicit zero state in incremental
decoding avoids this. The independent prefix helper pads T=1 with one ignored
future token, uses the regular causal convolution path, and selects position0.
No dependency or training model was patched. Initial uncorrected comparison
at position0 had cosine−0.5766; it is preserved in the first report rather
than silently overwritten.

## Run and artifacts

```bash
RUN=/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/dism-swa50m-softcap30-20260909-offline
/home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.generate_lm \
  --checkpoint "$RUN/latest.pt" --output /tmp/dism-generation-new.json \
  --backend cpu-cache --prompt 'Once upon a time, in a small village,' \
  --max-new-tokens 96 --temperature 0.8 --top-p 0.9
```

Output must be a new path. Use `--backend prefix` for the training-kernel
control, `--temperature 0` for greedy, `--compare-tokens 24` for a capped-logit
comparison. Output JSON records checkpoint step, sampling parameters, token
IDs, prompts, unedited completions, EOS and wall time.

Saved in the run directory:

- `generation-prefix.json`: original CUDA sampling; initial T=1 diagnostic.
- `generation-prefix-greedy.json`: greedy + corrected24-position comparison.
- `generation-cpu-cache.json`: incremental CPU-core sampling.
- `hard-eval-100.json`: previous independent validation result.
