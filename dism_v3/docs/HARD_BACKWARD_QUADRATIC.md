# Hard quadratic CUDA backward prototype

Status (2026-10-14): explicit fixed-length prototype, not a production dispatch
change. The default hard backward still uses the asynchronous CPU event planner
and Triton event kernels. This implementation requires no CPU planning or
host/device metadata transfer. Packed documents and autograd integration are not
implemented here; do not flatten packed documents and treat them as one sequence.

## Interface

```python
from flash_dism.hard_forward import forward_hard
from flash_dism.hard_backward import backward_hard

out, norm, state = forward_hard(sq, sk, v, iq, ik, hard, tau, save_state=True)
dsq, dsk, dv, dtau = backward_hard(sq, sk, v, dout, iq, ik, hard, norm, state)
```

Vectors are contiguous BF16 BNHD, labels int32 BHN, hard flags bool `[B,H]`,
and N must be positive and 256-aligned. R16/32 and DV32/64 are supported.
Outputs are FP32 dSQ/dtau and BF16 dSK/dV. Soft sequence/head rows are skipped
and return zeros. Matching vocabulary vectors, their LSEs and direction are
absent from this kernel. `dtau` differentiates the natural-log coefficient;
the outer softplus parameterization must apply its own chain rule.

State must come from the same forward invocation. The prototype retains the
integer horizontal boundary and log-weight LUT; the summary itself is not
needed by backward. For B16/H6/N2048 the boundary alone is 49,545,216 bytes.
Recomputing summary and chunk passing in backward is an explicit benchmark
alternative, not an automatic memory policy.

## B1 structure

`src/hard_backward.cu` and `include/backward/hard.cuh` follow soft B1:

- Persistent key-major tasks, 128 keys/CTA, 384 threads, two consumer
  warpgroups and a producer warp; descending 32-query input tiles.
- SK/V are held in registers/shared storage; SQ/dO use a two-slot TMA ring.
  Query labels and a 32-key left halo are loaded by the producer into shared
  memory. Repeated matching never reloads labels from global memory.
- Directly construct a rolled integer max-scan buffer in the existing K16/Q32
  placement. The integer state is the last mismatching key coordinate.
  Reuse horizontal query32 checkpoints. Reconstruct only the missing vertical
  key16 incoming edge with at most32 shared-label comparisons per entry;
  the tile interior uses the existing parallel scan.
- Reuse B1's SQ/SK and V/dO MMAs and coefficient casts. Key-owned dSK/dV have
  private output; dSQ uses the existing FP32 global TMA add-reduction.
- No reverse affine scan, reverse boundary mail, B2 or B3. The only common
  soft helper change is inferred argument/storage template types for the
  unchanged gradient output/transpose routines.

There are global atomics for the query moments. Each warp first reduces keys
in registers, then one warp reduces eight warp partials in shared memory and
emits four additions per query per CTA. Shared storage is used for necessary
inter-warp communication, never shared FP32 atomics. The existing gradient
helper rendezvous protects moment storage from the next iteration's writers.

## Analytic tau gradient and numerical correction

Let `L` be the consecutive hard-match length, `W_L=sum(exp(r*tau), r=1..L)`,
`g_L=d ln(W_L)/d tau`, and `W_0=g_0=0`. Then

```
P_ij = W_Lij / (1 + sum_j W_Lij)
S_ij = SQ_i dot SK_j
E_ij = dO_i dot V_j

dSQ = (P*E) @ SK
dSK = (P*E).T @ SQ
dV  = (P*S).T @ dO
dtau = sum_ij P_ij*S_ij*E_ij*(g_Lij - sum_k P_ik*g_Lik)
```

The fallback has unit weight, zero value and zero log-weight derivative.
The derivative already accounts for the entire diagonal recurrence; another
reverse scan is unnecessary. `g` is a natural-tau derivative despite the
forward LUT/normalizer using log2 units; no extra LOG2E multiplier is applied.

The initial direct three-moment calculation `w-u*v` failed the N1024, tau=4
FP64-oracle test: FP32 log2-normalizer rounding was amplified by chain length.
The retained implementation instead uses `c_i=g_(i+1)`, shifted derivatives
`g_L-c_i`, and a fourth moment to explicitly renormalize the mass:

```
f = exp2(-saved_log2_Z)
z = sum(P) + f
u = sum(P*S*E)
v = sum(P*(g_L-c)) - f*c
w = sum(P*S*E*(g_L-c))
dtau_i = (w - u*v/z) / z
```

Thus common normalizer rounding cancels rather than being multiplied by L.
Vector gradients retain B1's BF16 coefficient/output behavior. Moments and
cross-CTA accumulation remain FP32 and nondeterministic; this is not a claim
of universally exact tau gradients under arbitrarily severe cancellation.
The test tolerance was not relaxed to hide the original failure.

The derivative LUT is parallel over head/length. A near-zero series handles
`abs(L*tau)<0.1`; elsewhere a stable geometric-sum derivative uses `expm1`.
Only scalar LUT/reduction work is added; no dense N-by-N pair tensor is saved.

## Reproduction

Use conda `blkw`, this worktree's PYTHONPATH and `dism_v3/build.py` with
`DISM_BUILD_JOBS=2`. The default eight configurations remain enabled.

```bash
python -m pytest dism_v3/tests/test_hard_backward.py -q
python dism_v3/tools/bench_hard_quadratic_backward.py --profile --events --soft-reference
```

The benchmark includes output allocation/zeroing, derivative LUT, main B1 and
tau reduction. A second timing also recomputes integer boundaries. Event GPU
timing excludes CPU planning and upload; CPU planning is reported separately.
Soft B1/full backward are timing-only comparisons with soft flags, not hard
numerical oracles. No model/checkpoint/optimizer state is loaded or changed.

Generated JSON, SASS and sanitizer logs belong under ignored `build/`.
Further integration needs packed-document scheduling with original training-row
hard flags, saved-state/recomputation ownership, and a full-model end-to-end
comparison against the planner overlap path. Random-label GPU-only wins must
not be inferred from eliminating CPU waits.

## Initial measurements and validation

RTX5090 sm120, B16/N2048/H6/R32/DV64, all hard, tau=0.5, BF16 vectors.
Medians of five samples, ten repeats per sample, including allocation/zeroing
and launch gaps. No clock/power settings changed. Final report:
`build/hard_quadratic_backward_20261014.json`.

| Path (ms) | Random512 labels | All labels equal |
| --- | ---: | ---: |
| Hard quadratic complete backward | 2.271 | 3.062 |
| Also recompute integer boundaries | 2.458 | 3.169 |
| Event backward, GPU only, ready plan | 0.378 | 3.652 |
| Soft B1, timing only | 1.949 | 1.941 |
| Soft complete backward, timing only | 4.106 | 4.267 |

The hard prototype includes tau, whereas soft B1 excludes B0/B2/B3. The event
comparison excludes CPU planning and upload. Neither a six-layer extrapolation
nor these standalone numbers establishes full-model training throughput.
The longer-chain improvement came from replacing per-pair backward label walks
with integer tile scans (initial all-equal backward was about15ms, then7.4ms
after moment/LUT changes, now3.1ms).

Instrumented main kernel times were about2.27/3.00ms (random/equal), tau
reduction39–42us, derivative LUT1.6us, initialization about21us. Instrumented
times are not interchangeable with uninstrumented medians. Next bottlenecks
are key16-edge reconstruction, register pressure/spills and query-gradient/
moment reductions; no further launch tuning is enabled by default.

Validation completed:

- Default eight-configuration build; 54 tests passed across
  `test_hard_backward.py`, `test_hard_training.py`, `test_production_build.py`.
  New tests cover four R/DV shapes, five label patterns, mixed sequence flags,
  all-soft zero output, tau=-4 through4 including near-zero branches, persistent
  CTA reuse and graph replay. Vector tolerances are atol=.008/rtol=.025 with
  relative-L2<.01 for pattern cases; tau atol=.003/rtol=.01. No existing
  tolerance or production launch default was changed.
- All eight hard-backward device objects are Torch-free and have no SASS CALL.
  Main kernels use168 registers. Spills remain: R16/DV32 has8-byte stack;
  R16/DV64 has104; R32/DV32 has64; R32/DV64 has192 (188 spill-store bytes,
  200 spill-load bytes reported by ptxas). This is a performance limitation,
  not a spill-free implementation claim.
- Memcheck and synccheck targeted checks returned0 errors. Isolated native-path
  racecheck (N512, random/equal/boundary, CTA1/default) returned0 hazards.
  **Limitation:** the full FP64-oracle test under racecheck reported one error
  in ATen's `reduce_kernel<double>` (6144 hazards). The report is retained in
  `build/hard_backward_racecheck.log`; it has not been dismissed as a tool
  false positive. The isolated check avoids executing that oracle reduction,
  and does not make the full test suite race-clean.

No packed hard quadratic backward, dedicated autograd adapter, training run or
checkpoint modification is included in this prototype.
