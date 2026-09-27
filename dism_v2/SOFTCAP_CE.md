# Triton softcap cross entropy

`FusedSoftcapCrossEntropyLoss(softcap=30)` in `softcap_cross_entropy.py`
computes CE of `z=c*tanh(x/c)` and gradients with respect to original logits:
`dx = (softmax(z)-one_hot(y)) * (1-tanh(x/c)^2) * upstream`.
The cap is a fixed Python scalar, not a learnable parameter.

Forward uses4096-column chunks with FP32 partial LSE, merges per-row LSE and
target logits, then optionally reduces row losses/counts. Backward uses
1024-column chunks, recomputes capped logits and multiplies the softcap
derivative in the same Triton kernel. No capped-logit/probability matrix is
materialized. The original logits remain saved for backward; a separate
input-dtype gradient matrix is allocated (unlike FLA's optional in-place CE).
At16384 rows and50257 classes, partial LSE has16384x13 FP32 entries.

NVIDIA `tanh.approx.f32` and explicit FTZ EX2 are used. User chose this
approximate path for the real large-vocabulary mean-loss workload rather
than paying for precise tanh to satisfy small-vocabulary precision probes.
The gradient uses the ideal tanh derivative evaluated at the approximate
tanh value, not the derivative of the hardware approximation's interpolation.

API: CUDA FP32/BF16/FP16 `[rows,vocab]` logits, int32/int64 `[rows]` targets,
`none/sum/mean`, configurable ignore_index. Row-padded logits, strided targets
and strided unreduced upstream gradients are supported. Last logit stride1
is required. Outputs/statistics FP32, input gradients match input dtype.
No input mutation. Empty/all-ignored mean is NaN with zero gradients, like
PyTorch; invalid non-ignored targets produce NaN, without unsafe target loads.
No class weighting,label smoothing,z-loss,distributed vocabulary or higher
derivatives in this initial implementation.

## Verification on sm120

Full approximate-path suite: **96 passed,15 strict numerical failures**.
Failures remain normal failures with unchanged tolerances, not xfail or skipped:

- Seven original small-vocabulary checks fail: six unreduced tests at V33/65
  and one FP32 sum-gradient test at V65. CE errors are around1e-4.
- Expanded cap30 GPT-2 checks have eight unreduced stress failures: FP32
  input scales4/16/64/256; BF16/FP16 scales16/64. Approximation can affect
  unreduced gradients too; large vocabulary alone is not a precision guarantee.
- **All18 GPT-2 cap30 mean-loss cases pass**, including forward/backward,
  input scales0.25/1/4/16/64/256 and FP32/BF16/FP16. This is the training path.
- Three compiled kernels (partial forward,merge,backward) show PTX TANH
  approximation and no SASS CALL. Shapes include50257 and65537 tail classes.
- Memcheck/racecheck/synccheck each7 passed, zero errors/hazards: six BF16
  GPT-2 mean cases and invalid-target memory safety.
- Updated LM wiring suite:6 passed, including actual DISM+SWA+softcap backward.

The earlier precise libdevice tanh variant passed all74 original tests, but
was replaced by the approximate variant at user request. No precise-mode
performance claim or concurrent-training timing comparison is made.

```bash
# Production-domain checks plus codegen:
OMP_NUM_THREADS=2 /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_softcap_cross_entropy.py -k 'gpt2_cap30 and mean or gpt2_codegen'
# Includes the documented strict approximation failures:
/home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_softcap_cross_entropy.py
```
