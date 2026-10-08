# v3 initial backward (R32 / D64 / DV64)

Initial R32/D64/DV64 implementation is numerically accepted by the user on
2026-09-29 after the cosine/norm/error-drift audit. This is not a claim that all
strict numerical gates pass. Default acceptance and opt-in unchanged strict
diagnostics are documented in `BACKWARD_TESTING.md`.

## Entry points

```python
from flash_dism.backward import dism_core, backward_core
from flash_dism.forward import forward_core
from flash_dism.voc import voc_dism

# Raw natural-log LSE, already-activated soft readout features.
out = dism_core(q, k, sq, sk, v, q_lse, k_lse,
                idx_q, idx_k, direction, hard, rtau)
out.backward(dout)

# Explicit first-order diagnostic interface.
out, lse2, state = forward_core(q, k, sq, sk, v, q_lse, k_lse,
                                idx_q, idx_k, direction, hard, rtau,
                                save_state=True)
grads = backward_core(state, dout, fp32_output=True)

# Existing Triton embedding forward/backward; vocab [H,V,64] or shared [V,64].
out = voc_dism(q, k, sq, sk, v, q_vocab, k_vocab, rtau,
                direction=direction, hard=hard)
```

Vectors are BF16 `[B,N,H,D]`; sq/sk are BF16 `[B,N,H,32]`. Upstream dO is BF16.
Direction is bool `[B,H]`, hard is bool `[B,H,N]`, explicit replay as in current
forward bring-up. No new production RNG, varlen or higher-order gradients.
rtau is `[H]`, raw LSE is `[B,N,H]`. SiLU remains caller-owned, including vocab
activation. Triton legacy LSE/label names are explicitly swapped to input-side
Q/K identities by the wrapper; they must not be inferred from `out_q/out_k` names.

`backward_core` returns q_vec/k_vec/sq_vec/sk_vec/v/q_lse/k_lse/rtau.
FP32 accumulation is preserved everywhere. Key-owned dV/dsk/dK/dk_lse default
to BF16 final stores, optionally FP32 (`fp32_output=True`). Cross-CTA atomic
dQ/dsq/dq_lse/dtau remain FP32. Autograd casts at upstream input dtype boundaries.
No gradient flows through the internal raw_lse-rtau preprocessing a second time.

## Kernels and state

- `src/backward_delta.cu`: FP32 `E=sum(float(O)*float(dO))`, BF16 or FP32 O input.
- `src/backward_summary.cu`: B1, dV/dsq/dsk plus reverse affine summary.
- `src/backward_chunk.cu`: B2, aligned reverse32-key passing with FP32 FMA.
- `src/backward_qk.cu`: B3, dQ/dK/dLSE/dtau after reverse prescan/postscan.
- Helpers in `include/backward/`; existing TMA maps, hardened ring protocol,
  allocator and split scan are reused. No v2 shifted-edge fixups are imported.

The three main kernels do not materialize full S/P/G. Delta and gradient-zero
launches are auxiliary and are not hidden in the count of three main stages.
`*_debug` and `*_probe` are explicit diagnostic-only dense outputs; the regular
training path passes null pointers and allocates no such buffers.

Forward training state adds `vertical[B,H,floor((N-1)/16),Np]` with
`vertical[c,i]=S2[i,16*(c+1)-1]`, directly extracted from registers. It retains
existing `horizontal[B,H,floor((N-1)/32),Np]`. Np is padded to256.
Reverse SoA summaries/G boundaries use `[B,H,ceil(N/32),Np]`. Unused query
padding is zero initialized; active partial tiles use affine identity for
padding. G boundary[c,q] is G[q,32*c], not a shifted coordinate.

Both heavy backward kernels use12 warps, inc232/dec40, independent key-prefetch
slot and two query input slots. Each consumer owns16 keys and consumes32-query
tiles in reverse order. Key ownership is interleaved0,4,1,5,2,6,3,7; reverse mail
is WG1→WG0. B3 publishes prescan state before postscan and gradient GEMMs.
Wholly noncausal query tiles below the CTA's first key are omitted symmetrically
in producer/consumers; their zero summaries remain initialized.

Shared gradient coefficients are necessary GEMM layout staging and CTA
communication. Consumer output ownership is reassigned after a256-thread
rendezvous: each active warp owns one16×16 output tile, accumulates all128 keys
in FP32 registers, then uses global atomicAdd. No shared floating-point atomic.
dLQ similarly uses warp row sums plus CTA reduction at the same rendezvous.
Held key/sk/V storage remains valid until this redistribution finishes; its
release allows next-task prefetch during final key-gradient stores/retirement.
The initial implementation uses scalar global atomic writes, not TMA reduction.
Metadata caching and B2 software pipelining are optimization follow-ups.

## Validation and precision tracking

Latest user-requested cosine/norm/systematic-error audit is in
`GRADIENT_BIAS.md`. It supplements rather than replaces the strict failures
below; statistics use paired inputs and upstream derivatives.

Commands run with blkw and `PYTHONPATH=python:..`. Build with `build.py`.

- Existing mathematical reference:33 tests pass.
- Forward vertical boundaries +16K×32Q TMA/MMA recompute + reverse primitive:
  83 tests pass; all83 memcheck passes, zero errors.
- Component gradients:21 tests pass, independently checking raw C*A/C*B/G,
  then checking all gradient GEMMs and scalar reductions with fixed verified
  coefficients. Debug-on and debug-off outputs are also compared.
- Execution covers tails, N1025, persistent CTA counts1/2/default,
  non-default stream, zero gradients for all-hard unmatched inputs, zero-sq
  gradient structure, and the soft LSE/tau chain rule (15 cases).
- Output-precision/autograd + delta smoke:3 tests pass under memcheck.
- Final causal-trimmed memcheck selection:14 pass, zero errors.
- Final default BF16 build synccheck:36 execution/component tests pass, zero
  errors (`build/backward_final_synccheck.log`).
- Final causal-trimmed B1/B3 racecheck:6 pass, zero hazards/warnings/errors.
  The filter `--kernel-name kns=dism_backward` excludes forward/setup kernels;
  no ARRIVES false-positive exception was needed for this result.
- Forward K32 regression:230 pass,36 deselected. Default K64 restored afterward.
- Full default K64 suite:681 pass,3 skip,30 ordinary precision failures,
  including both new execution structural tests. This historical run predates
  the user-requested default/strict test split. Logs are in `build/backward_full_tests.log`,
  `build/backward_forward_k32_tests.log`, `build/backward_final_memcheck.log`,
  and `build/backward_final_racecheck.log`.
- All18 oracle-delta diagnostics pass the same ideal-reference error gates.
- Actual optimizer smoke:6 cases (N65/129 × soft/mixed/hard),64 AdamW updates
  each, real V512 Triton embedding and FP32 parameter owners. All gradients
  remain finite, final fitting loss is lower, and hard-only Q/K/vocab owners
  remain bitwise unchanged. See `tests/test_voc_training.py` and
  `build/backward_training_smoke_64.log`. The earlier12-step run had one mixed
  case above its initial loss; its failure remains in
  `build/backward_training_smoke.log`. Mixed argmax labels can change during
  fitting; neither monotonic descent nor general training quality is claimed.
  These6 cases were added after the681-pass full-suite run above.

Strict independent-FP32-oracle gradient comparisons retain four failures around
BF16 conversion thresholds (initial summary hard/N65; QK soft/N33 and mixed
N17/129). The measured unquantized coefficients/G pass the independent check;
using them for GEMM verification passes tight tolerances without changing the
original strict tests. Compiler changes may change which midpoint cases fail.

Ideal reference has additional dLSE/tau cancellation-sensitive failures. In
`build/backward_error_mixed129.json`, default saved-BF16-O delta has0.2119%
relative L2 error; dLSE/tau errors are7.838%/7.013%. Replacing ONLY delta with
oracle-O delta gives0.1294%/0.0710%. The same-state, unquantized readout delta
gives0.7415%/0.3768%. All three head tau signs match in this example. This is
diagnostic evidence, not permission to replace production delta or relax tests.
Real V512 Triton end-to-end initially passes four soft/mixed direction cases;
two hard cases fail the3% tau relative-error gate (~8.7%), not embedding grads.
All original strict failures remain ordinary failures and are not xfailed.
The30 failures comprise4 same-state BF16-threshold cases,24 ideal-reference
cases, and2 real-V512 hard cases. The latter's tau values are
`[0,0.2080464]` versus oracle `[0,0.1913495]`: signs agree, but the relative
error gate does not pass. Numerical acceptance remains explicit, not inferred
from the component tests or sign agreement.

Broader sign audit (8 seeds, soft/mixed/hard, tau=2/ln64, four heads with both
directions, N129, real V512 interpolation) finds **1 sign mismatch among192
head cases**. At seed1033/hard/tau=ln64/head0, production tau is approximately
-0.0052034 while FP64 oracle is+0.00710104. Replacing only delta restores
+0.00710112. Thus prior sign agreement in a few examples must not be generalized:
saved-output error can reverse a cancellation-sensitive small gradient.
The audit leaves defaults/tolerances unchanged, checks that its all-hard
diagnostic reproduces the original output bitwise, and records all cases in
`build/backward_tau_sign_audit.json`. Reproduce with `tools/analyze_tau_sign.py`.
The user subsequently accepted this initial numerical behavior after the broader
cosine/norm/bias audit; this is still not a passing strict sign gate.

A separate FP32-O build gives delta/dLSE/tau relative errors of
0.1471%/4.913%/1.368% on the same mixed129 case. Thus final O casting is not the
only source: the forward BF16 readout coefficient also contributes. This build
is diagnostic only; default BF16 output has been restored. Report:
`build/backward_error_fp32_o_mixed129.json`.

Reproduction: `tools/analyze_backward_error.py --n 129 --mode mixed --direction query`.
This diagnostic intentionally materializes dense states; it is not production.

Initial profiler B2/H4/N1024 (five launches, before dLQ CTA reduction/causal trim):
B1~362us, B2~5.89us, B3~794us; zeroing launches listed separately in
`build/backward_timing_1024.json`. These are unoptimized profiler timings, not
CUDA-graph throughput. No splitting of B1 has been justified or implemented.
After CTA dLQ aggregation and causal trimming, ten-launch profiler timing is
B1~361.6us, B2~5.89us, B3~406.3us (`build/backward_timing_1024_final.json`).
No claim of fully tuned performance or sm90 validation is made.
At B16/H16/N2048, the same ten-launch profiler method reports
B1=11.164ms, B2=0.268ms, B3=12.172ms; six zero-fill launches per iteration are
separate (~40.6us per launch averaged across differing buffer sizes).
See `build/backward_timing_2048_final.json`. This is an initial timing baseline,
not a comparison to v2 or a continuous-stream throughput result.

Spill is retained under user authorization. Final causal build resource report:

| Kernel | Stack bytes | Spill store bytes | Spill load bytes |
| --- | ---: | ---: | ---: |
| B1 summary/readout | 432 | 480 | 504 |
| B3 QK | 384 | 416 | 528 |

These are compiler reports, not dynamic traffic totals. Both report168 static
registers and contain the intended inc232/dec40 instructions. SASS has no CALL.
Default forward checkpoint saving preserves the prior8B stack/16B spill-store/
24B spill-load report. Initial implementation committed as `76262cd`.
Subsequent opt-in atomic/TMA reduction experiments are documented in BACKWARD_TMA.md.

## Reproduce

```bash
PATH=/usr/local/cuda/bin:/home/cicuvc/miniconda3/envs/blkw/bin:$PATH \
  DISM_OUTPUT_DTYPE=bf16 DISM_LINEINFO=1 \
  /home/cicuvc/miniconda3/envs/blkw/bin/python build.py
PYTHONPATH=python:.. /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests
PYTHONPATH=python:.. /home/cicuvc/miniconda3/envs/blkw/bin/python \
  tools/analyze_backward_error.py --n 129 --mode mixed --direction query
PYTHONPATH=python:.. /home/cicuvc/miniconda3/envs/blkw/bin/python \
  tools/bench_backward.py --batch 16 --heads 16 --n 2048 --iterations 10
```

Run from this v3 directory. Default pytest now runs the separate relaxed
acceptance suite:757 passed,3 skipped,78 strict cases deselected. Use
`-m strict_gradient` to run the unchanged original gradient diagnostics, or
`-o addopts=` to include both groups. The original30 strict failures are not
converted into successful tests; see `BACKWARD_TESTING.md`.
