# Checkpoint validation: native prefill through autoregressive decoding

Validated checkpoint: `nanochat-hybrid125m-2500m/base_checkpoints/dism125m/model_019074.pt`
under `/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs`.
Model: 8 hybrid layers, hidden768, H12, D=DV64, R32, vocab512 per head,
SWA128, untied LM head, softcap30. It is the existing v3 hybrid checkpoint,
not a trained v4 reset-gate checkpoint. Tokenizer fingerprint and BOS/EOS IDs
are checked; checkpoint SHA256 is recorded in the result JSON.

The opt-in harness `tools/checkpoint_prefill_decode_smoke.py` uses unchanged
FP32 projection/shortconv/SWA/MLP/model weights and full-hard matching. It
replaces only the DISM core:

1. Existing model projections/conv/vocabulary argmax produce labels and vectors.
2. New `HardDismPrefill`: 8-worker native planning, chunk packing, one combined
   per-layer Triton grid. Output uses the current default BF16 GEMMs/C16.
3. Existing `HardDismDecoder.prime()` builds a decoding snapshot from the same
   labels, SK and V history. No per-token replay is used to prime it.
4. Subsequent tokens use GPU-planned native decoding; explicit CPU rebuilds
   remain as before. Interval32 ensures the test crosses a rebuild boundary.
5. Compare against the original Torch recurrent full-hard model path on exactly
   the same token prefix/continuation. All layer cache positions are checked via
   the GPU status interface and recorded (320/1088 after the two forced runs).

This does not monkey-patch production model dispatch, mutate the checkpoint,
or claim that the entire model runs in BF16. This checkpoint has no reset gate;
reset semantics are covered by core batch tests, not this model-level test.

## Forced-token checks

Two configurations: TF32x3 GEMMs + FP32 cache payload as strict diagnostic,
and default BF16 GEMMs + BF16 cache payload. Prefill lengths256/1024, each
followed by64 teacher-forced autoregressive steps.

| Mode | Prefix | Prefill logit relative L2 | Prefill max absolute | Worst decode relative L2 | Worst decode cosine |
| --- | --- | --- | --- | --- | --- |
| TF32x3/FP32 | 256 | 3.38e-7 | 1.43e-5 | 4.45e-7 | >0.9999999999999 |
| TF32x3/FP32 | 1024 | 4.02e-7 | 2.48e-5 | 4.16e-7 | >0.9999999999999 |
| BF16/BF16 | 256 | 5.61e-4 | 0.290 | 4.17e-3 | 0.9999917 |
| BF16/BF16 | 1024 | 1.10e-3 | 0.438 | 1.14e-2 | 0.9999357 |

Prefill and decoding top-1 agreement are100% in all four forced runs.
TF32x3/FP32 passed the predeclared strict diagnostic checks (relative L2<1e-3,
cosine>.99999). BF16 error is measured separately, not asserted equivalent to
the strict diagnostic and not used to relax existing tolerances.

Reference/native prefix token NLL:

- N256: 0.9972448 / 0.9975284 in BF16 mode, delta +0.0002835.
- N1024: 0.3408335 / 0.3416671, delta +0.0008336.

These prefixes repeat the three test prompts to reach the requested length;
their low NLL is **not a held-out validation result**. Isolated larger BF16
logit errors are retained in JSON; model-level discrete label sensitivity has
not been separately localized. No claim of long-run error equivalence is made.

## Actual sampled generation

Three prompts, 64 generated tokens each, temperature0.8/top_p0.9,
seeds20261008/09/10. Default BF16 prefill and BF16 native decode both execute.
All outputs are finite and all three continuations complete without cache
errors. Replay reference follows exactly the native sampled token trajectory.

| Prompt topic | Same-seed next-token agreement on replay | Worst replay relative L2 |
| --- | --- | --- |
| Solar system | 64/64 | 0.001144 |
| Photosynthesis | 64/64 | 0.000375 |
| Girl finds a book | 63/64 | 0.000230 |

The last result is not a claim of independently sampled full-sequence identity:
a sampling boundary can differ even with small logits error. Full text and
token IDs are retained. For example the photosynthesis continuation starts
"a plant releases oxygen, which helps plants absorb nutrients...". The solar
system prompt's continuation incorrectly names the Sun as a planet, also
sampled by the reference; functional parity does not imply factual quality.

## Reproduce and scope

```bash
CUDA_HOME=/usr/local/cuda-13.4 OPENBLAS_NUM_THREADS=1 /home/cicuvc/miniconda3/envs/blkw/bin/python dism_v4/tools/checkpoint_prefill_decode_smoke.py
```

Results: `decoding/results/checkpoint_prefill_decode.json`. Script retains
per-step errors, prefix NLL, sampled text/tokens, cache positions, checkpoint
hash and model config. Reported prefix times include possible first-use JIT
and cache prime, so are not a steady-state model-throughput benchmark.
Functional prefill-to-cache-to-decode integration works on this checkpoint;
longer generation, v4 gated checkpoints and production model API integration
remain outside this smoke test.
