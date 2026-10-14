# Fused value-residual mix

## Current default: hardware tanh sigmoid (2026-10-15)

The final implementation follows the user's sigmoid-specific recommendation:
`s = fma(tanh.approx.f32(gate * 0.5), 0.5, 0.5)`. Both passes use the same
recomputed value. SASS confirms `MUFU.TANH`; there is no exponential, sigmoid
division, or CALL. BF16 forward/backward use 24/30 registers, FP32 26/32,
with zero spills; backward retains only 16/32 bytes of reduction scratch.
Custom-op names are `value_residual_forward_v3` / `value_residual_backward_v3`.
Launch defaults, stride handling and the FP32 mixture/reduction order are unchanged.

This is **not** an accuracy-preserving substitution for the exp2 version.
The negative tail can round to zero early because `1 + tanh` cancels; backward
uses the usual `s*(1-s)` derivative with this approximate s. Tests explicitly
bound the approximation rather than claiming bitwise equality or relative
accuracy in saturated tails. The dense [-100,100] test has 65,537 gates,
sigmoid absolute error <=4e-6, and derivative absolute error <=32*4e-6 for DV32;
it also checks gradient algebra exactly against the recomputed approximate s.

Compared with the exp2 acceptance criteria, FP32 vector relative L2 is now
<=1e-5 (dgate <=2e-5), with per-element 1e-5 of tensor scale plus output epsilon.
BF16/F16 L2 and low-precision dgate/FP64 bounds remain unchanged. Actual caller
outputs use relative L2 <=5e-4 rather than bit equality; parameter gradients
retain relative L2 <5e-3. The gate Linear's fixed absolute 1e-6 bound is replaced
by one BF16 epsilon at tensor scale: its observed error is 1.9073e-6, caused by
BF16 rounding of autocast Linear gradients. This numerical change is intentional,
not a cleanup-only tolerance adjustment.

Production BF16 `[1,32768,6,64]`, versus eager, measured relative L2:
output **7.56e-5**, dv **8.39e-5**, dgate **1.50e-4**; dvfirst **2.00e-3** retains
the documented duplicated-cast discrepancy. FP32 relative L2 is
1.97e-6 / 1.97e-6 / 1.97e-6 / 8.17e-6 for output/dv/dvfirst/dgate.
Full-autograd production BF16 median was **0.235 ms** (exp2 ~0.237 ms); this small
difference is not a demonstrated speedup. Other timing rows were noisy and are
not used to claim improvement. No full-model throughput claim for tanh is made.

Artifacts: `build/tanh_value_residual_bench.json`,
`build/tanh_value_residual_resources.json`, `build/tanh_value_residual_final_tests.log`,
and `build/tanh_value_residual_{memcheck,racecheck,synccheck}.log`.

Final validation: 22 tests plus 19 subtests pass, including actual Inductor
caller execution. Memcheck and synccheck report zero errors; racecheck reports
zero errors and zero warnings. All generated artifacts remain outside Git.

## Intermediate hardware exp2 sigmoid (superseded)

At the user's explicit request, both passes now use
`1 / (1 + exp2(-gate * log2(e)))` with hardware approximate exp2. Other FP32
arithmetic, correctly rounded division, gradient reduction order, stride
handling and launch defaults are unchanged. Custom-op identities are bumped to
`value_residual_forward_v2` / `value_residual_backward_v2` to separate compiler
caches. Eager reference remains available. The accurate-exponential results
below are historical, not current bitwise guarantees.

Tests replace applicable bit-equality assertions with explicit approximation
bounds: FP32 relative L2 <=2e-6, BF16 <=3e-4 and F16 <=1e-4, plus per-element
one output-dtype epsilon and FP32 cancellation allowance (2e-6 of tensor scale).
Existing dgate bounds, FP64 comparisons and caller-gradient tolerances remain.
A dense gate range [-100,100] checks sigmoid and derivatives including saturation.

Measured production BF16 relative L2 versus eager: output 4.97e-6, dv 3.94e-6,
dgate 1.85e-5. dvfirst remains about 2.00e-3 because of the previously documented
different BF16 rounding order, not an exp2-scale discrepancy. FP32 relative L2
is 2.38e-8 / 2.14e-8 / 2.49e-8 / 1.22e-7 for output/dv/dvfirst/dgate.
Full-autograd production BF16 median is 0.237 ms (eager 0.741 ms); it does not
establish a speedup over the previous repaired accurate version (~0.237 ms).

SASS confirms `MUFU.EX2`, 26/34 forward/backward registers and zero spills.
One CALL remains: ELF symbols identify the correctly rounded division's
`__cuda_sm3x_div_rn_ftz_f32_slowpath`, **not** the exponential. Earlier attribution
of that CALL to exp was incorrect; division precision has not been relaxed.
Artifacts: `build/exp2_value_residual_bench.json`,
`build/exp2_value_residual_resources.json`, and `build/exp2_value_residual_*tests.log`.

## Earlier accurate-exponential implementation and repair

Status: reviewed and repaired on 2026-10-15. The original implementation passed
its tests but silently assumed unit stride along DV, including dout. This broke
last-axis slices/transposes and ordinary `out.sum().backward()` (zero-stride
expanded upstream). Both kernels now consume all four vector strides; address
arithmetic is widened to int64, and device consistency is validated. No mixture
math, rounding policy, or launch defaults changed.

The original benchmark's "compiled helper" compiled the already-fused dispatch,
not the old expression. That comparison is invalid as evidence against an
Inductor-compiled reference. The script now compiles `value_residual_mix_reference`.
Caller tests now clear `v_first.grad` between A/B arms. Regressions cover last-axis
slice/transpose/expand, scalar-upstream backward and invalid dout device.
The 72M ~1.1% result below is historical and near the reported run-to-run noise;
GPU idle alone does not establish CPU-planner waiting as its cause.

### Repair validation

- 21 tests plus 19 unittest subtests passed, including actual Inductor caller
  execution. Existing tolerances were not loosened.
- Updated sanitizer smoke (including DV stride 2, transpose and scalar-expanded
  dout): memcheck/synccheck 0 errors; racecheck 0 errors and 0 warnings.
- Corrected benchmark, BF16 `[1,32768,6,64]`, 10 warmups / 30 samples, full
  autograd median: eager reference **0.740 ms**, compiled reference **0.200 ms**,
  fused **0.237 ms**. Thus fused wins over eager in this run but not over the
  compiled reference. Short/FP32 rows can also be framework-overhead dominated.
- Final BF16/FP32 DV64 kernels: forward 26 registers / 0 shared bytes, backward
  34 registers / 16 bytes BF16 or 32 bytes FP32 shared reduction scratch; zero
  spills and one accurate-exponential CALL each. Logical row division stays
  int32; only address arithmetic widens to int64. An intermediate all-int64
  index decomposition introduced an extra division CALL and was replaced
  before final validation. Launch defaults remain unchanged.
- Raw results: `build/fix_value_residual_final_tests.log`,
  `build/fix_value_residual_final_{memcheck,racecheck,synccheck}.log`,
  `build/fix_value_residual_final_resources.json`,
  `build/fix_value_residual_final_bench.json`. These supersede the incorrect compiled
  baseline interpretation in the historical tables below.

### Final repaired 72M replay

Checkpoint19074, four resident batches, linear_fla, ready policy, 16 planner
workers, chunked document-balanced CE. Four warmup steps, two phase steps and
two eight-step sustained windows per arm; no profiler. Reference ran before
the repaired fused arms, not a randomized long-run crossover.

| Metric | reference helper | final fused helper |
|---|---:|---:|
| Window step ms | 797.60 / 801.42 | 789.65 / 789.46 |
| Mean step ms | 799.51 | 789.55 |
| Mean tokens/s | 163,940 | 166,008 |
| Peak allocated bytes | 18,792,023,552 | 18,511,820,288 |
| Diagnostic routes | 16 event + 8 quadratic | 16 event + 8 quadratic |

This short replay observes -1.25% step time and about 280 MB less peak memory;
it is not a long-run convergence or statistically isolated speedup claim.
The first restored-state phase loss was 3.4894011 versus 3.4893341 and gradient
norm 0.2657191 versus 0.2659247. They are not bitwise identical: the primitive
eager contract already documents dvfirst rounding, and this whole-model run
also includes compiled reference expressions and nondeterministic reductions.
No attribution of every model-level difference to the fused helper alone is
claimed; caller gradient tests above are the controlled numerical check.

Reports: `build/fix_value_residual_model_reference.json` and
`build/fix_value_residual_model_final.json`. The intermediate repair's
`fix_value_residual_model_fused.json` predates removal of the extra int64
division CALL and is not the final kernel timing. All runs used the existing
main-worktree planner; the isolated compact planner is not integrated here.
No checkpoint writes, training restart or dataloader advancement occurred.

Implementation: `python/flash_dism/kernels/value_residual.py`
Tests: `python/flash_dism/kernels/test_value_residual.py`,
`tests/test_value_residual_integration.py`
Benchmark / report: `tools/bench_value_residual.py`

## What is fused

`module.value_residual_mix` used to run one eager expression per layer:

```python
g = torch.sigmoid(gate.float()).unsqueeze(-1)
mixed = v_first.float() + (v.float() - v_first.float()) * g
out = mixed.to(v.dtype)
```

That materialises four FP32 tensors the size of `v` (two `.float()` copies, the
difference, the mixed value) plus a `[B,N,H,1]` sigmoid, and launches about ten
elementwise kernels per layer.  On the production packed row
(`[1,32768,6,64]`, BF16, 12.6M elements per layer) that is the dominant
value-residual cost.

The fused operator keeps the whole mix in registers and writes the final dtype
directly; the backward produces all three gradients in one kernel, with the
`DV` reduction inside the CTA.  There are no atomics, no D2H copies, no
`.item()`, no device sync and no host-side data-dependent branch, and no large
tensor is made contiguous: input strides and `dout` strides are passed to the
kernel, only the freshly allocated gradients are contiguous.

Public surface (unchanged call signature for callers):

* `value_residual_mix(v, v_first, gate)` — the single common helper; dispatches
  to the fused op when eligible, otherwise to the eager reference.
* `value_residual_mix_reference(v, v_first, gate)` — the original expression,
  kept as the numerical and performance reference.
* `DISM_VALUE_RESIDUAL_KERNEL=reference` forces the eager path for an A/B or a
  rollback; the default is `fused`.  The value is read once at import time so
  the dispatch decision is a Dynamo constant, not a host call inside a graph.
* Allowed dtypes: BF16/F16/FP32 for `v`, `v_first`, `gate` (mixed combinations
  are supported; every gradient is returned in its own input's dtype).
  FP64, CPU, `numel >= 2**31`, non-4D `v`, or a `gate` that is not exactly
  `v.shape[:-1]` fall back to the eager reference rather than being narrowed.

No parameter, `state_dict`, optimizer schema, RNG counter, annealing buffer,
no-decay grouping or dataloader state is touched.  Nothing is written in place;
saved tensors are the original inputs, sigmoid is recomputed in the backward,
so a retained graph can be run twice with different upstream gradients.

## Precision contract

With `a = FP32(v)`, `b = FP32(v_first)`, `s = sigmoid(FP32(gate))`,
`delta = a - b` and an FP32 upstream gradient `dy`:

| tensor | kernel computes |
| --- | --- |
| `out` | `cast_v_dtype(b + delta * s)` |
| `dv` | `dy * s` |
| `dvfirst` | `dy - (dy * s)` |
| `dgate` | `(sum_DV(dy * delta) * (1 - s)) * s` |

Every internal multiply, add and reduction is FP32; only the final store
rounds.  The backward does not pick a mathematically equivalent but
differently rounded form: `dvfirst` is `dy - fl(dy*s)` (not `dy*(1-s)`), the
`DV` reduction happens before the sigmoid derivative, and the sigmoid
derivative is evaluated as `g*(1-y)*y`, which is the order PyTorch's CUDA
kernel uses.  Elementwise steps use the explicitly rounded
`libdevice.{add,mul,sub,div}_rn` intrinsics and `libdevice.exp` because
Triton's defaults are not bit-compatible with the eager kernels:

* `tl.exp`/`tl.sigmoid` lower to `ex2.approx`-based fast math and drift from
  `torch.exp`/`torch.sigmoid` by up to 14 FP32 ulp (measured over 1M samples).
* `a + b*c` gets contracted into an FMA when the operands allow it, which
  changes `dy - dy*s` in ~30% of samples.
* `/` is not correctly rounded in Triton; `1/(1+e)` differed from PyTorch's
  division in 15% of samples.

With `libdevice.exp` + correctly rounded add/div, `s` matches
`torch.sigmoid(gate.float())` bit for bit, and the measured elementwise
differences above drop to zero.

## Measured accuracy

`tools/bench_value_residual.py --skip-timing` on the RTX 5090 (report:
`build/value_residual_accuracy_20261014.json`).  `bit-exact` compares the fused
operator with the eager helper on identical inputs and a random, non-uniform
upstream gradient; `vs FP64` evaluates the same closed form in double
precision to separate input rounding from implementation error.

| case | `out` | `dv` | `dvfirst` | `dgate` |
| --- | --- | --- | --- | --- |
| `[1,32768,6,64]` BF16 | bit-exact | bit-exact | 1 BF16 ulp (max 0.0156) | 8.5e-6 rel-L2, max 0.0156 |
| `[1,32768,6,32]` BF16 | bit-exact | bit-exact | 1 BF16 ulp | 5.8e-6 rel-L2 |
| `[1,2048,6,64]` BF16 | bit-exact | bit-exact | 1 BF16 ulp | bit-exact |
| `[1,16384,6,64]` FP32 | bit-exact | bit-exact | bit-exact | 1e-7 rel-L2 (reduction order) |
| `[2,4096,6,64]` BF16, strided | bit-exact | bit-exact | 1 BF16 ulp | 2.1e-6 rel-L2 |

Cosine is 1.0 to double-precision round-off and norm ratios are 1.0000 for
every tensor and case; per-tensor maxima are in the JSON.

### Why BF16/F16 `dvfirst` differs by one ulp (finding, not a bug)

The eager helper writes `v_first.float()` **twice**:

```python
mixed = v_first.float() + (v.float() - v_first.float()) * g
```

Autograd therefore creates two `to(float)` nodes, and each one's backward
casts *its own* contribution back to the input dtype before the leaf
accumulates:

```text
eager dvfirst = bf16(dy) - bf16(dy * s)          # two input-dtype roundings
Fused contract = bf16(dy - dy * s)               # one FP32 arithmetic, one rounding
```

This is verified bit for bit: for BF16 and F16 the eager gradient is exactly
`(dy.float().to(dtype) - (dy.float()*s).to(dtype))`, and for FP32, where
`.float()` is a no-op, eager and fused agree exactly.  The eager form is
*lossy*: it rounds the dominant `dy` term to the input dtype before the
subtraction, so its `dvfirst` error versus FP64 is 1.4x the fused error
(rel-L2 2.3e-3 vs 1.6e-3 on the production shape).  The fused operator follows
the FP32 contract from `FUSED_VALUE_RESIDUAL_TASK.md`; reproducing the
duplicated-cast artefact would deliberately reintroduce a BF16 intermediate
product and reduction into the backward.  The deviation is bounded by one
input-dtype ulp of `max|dy|` in every measured case.

### `dgate` reduction order

`dgate` reduces `DV` in FP32.  PyTorch's `.sum(-1)` uses a stride-halving tree
(`t[:h] + t[h:]`, repeated): verified bit-exact against that closed form for
`DV = 32` and `DV = 64`.  Triton's `tl.sum` uses a different tree; for `DV = 32`
the two coincide, for `DV = 64` the sums differ by up to ~2 FP32 ulp
(measured max 5.7e-6 absolute on the FP32 case, `rel-L2 <= 1e-7`), and in BF16
the final cast absorbs it except for a few 1-ulp entries (`rel-L2 <= 8.5e-6`).
Reproducing PyTorch's tree would need a reshape/permute shuffle chain per CTA
and an unnecessary shared-memory round trip for a last-bit effect, so the
cheaper tree is kept and the difference is reported here.

## Resource use, sanitizers, CALL/spills

## Performance results (2026-10-15, RTX 5090 idle)

### Standalone operator

`tools/bench_value_residual.py`, 20 warmup + 50 measured iterations, CUDA
events, no CUDA graphs, identical fixed inputs and pre-created leaves so the
timed region contains the operator plus its own autograd/allocator cost
(`build/value_residual_bench_20261014.json`).  "compiled" is
`torch.compile(value_residual_mix, fullgraph=True, dynamic=True)`.

Production shape `[1, 32768, 6, 64]`, BF16, full autograd (median / p10 / p90):

| variant | median ms | p10 | p90 | launches | peak MB |
| --- | --- | --- | --- | --- | --- |
| eager reference helper | 0.738 | 0.735 | 0.772 | 20 | 290 |
| compiled helper | 0.386 | 0.355 | 0.636 | 2 | 145 |
| **fused** | **0.256** | 0.247 | 0.292 | 2 | 145 |

Components on the same shape: eager forward 0.354 ms (9 launches) vs compiled
forward 0.111 ms (1) vs fused forward 0.056 ms (1), raw kernel 0.041 ms;
backward through autograd 0.395 ms (eager) vs 0.143 ms (fused), raw kernel
0.093 ms.  Host wall per iteration: 0.775 ms eager vs 0.249 ms fused.

The original measurement shows 2.9x versus eager and 20 -> 2 launches. The
"compiled" column above and below is compilation of the fused dispatch, not
compilation of the reference; the claimed 1.5x advantage over a compiled
reference is withdrawn. See the corrected benchmark for that comparison.

Other configurations (median ms, eager / compiled / fused):

| case | eager | compiled | fused |
| --- | --- | --- | --- |
| `[1,32768,6,64]` BF16 (production) | 0.738 | 0.386 | **0.256** |
| `[1,32768,6,32]` BF16 | 0.283 | 0.376 | **0.224** |
| `[1,2048,6,64]` BF16 | 0.292 | 0.353 | **0.226** |
| `[1,16384,6,64]` FP32 | **0.172** | 0.364 | 0.247 |
| `[2,4096,6,64]` BF16, strided heads | 0.285 | 0.401 | **0.224** |

The FP32 row is the honest exception: with FP32 inputs the helper's `.float()`
casts are no-ops, so the eager chain is only 10 launches and already cheap,
while the fused call still pays the opaque-op autograd/dispatcher plumbing.
Training runs BF16.

That plumbing is measurable: on the production shape the raw forward+backward
launches cost 0.124 ms, the same work through the autograd API costs 0.227 ms
(forward with graph construction 0.060 ms vs 0.036 ms raw; backward through
autograd 0.144 ms vs 0.087 ms raw).  ~0.10 ms/call is dispatcher + autograd node
+ three output allocations. An alternative `torch.autograd.Function` may reduce
some overhead, but would need explicit Dynamo/AOT validation; it is not
inherently untraceable. The potential saving has not been established here.

### 72M checkpoint A/B

Same harness and arguments as the ready baseline artifact
(`build/hard_dispatch_ready_idle.json`): scratch model+optimizer replay of the
four checkpoint-prefetched resident batches, `conv_impl=linear_fla`,
`hard_backward_policy=ready`, 16 planner workers, `--review-ce
chunked-document` (document counts 34/30/32/43, total 139), 4 warmup steps,
3 phase steps, 3 windows of 8 steps.  Old = `DISM_VALUE_RESIDUAL_KERNEL=reference`,
new = default.  Legs ran old -> new -> old -> new -> old to expose drift;
nothing was written to the checkpoint directory and the dataloader never
advanced.

| leg | time | step ms (window mean) | tokens/s | peak MB |
| --- | --- | --- | --- | --- |
| old_a | 23:44 | 788.4 | 166 242 | 17 921 |
| new1 | 00:02 | 791.6 | 165 581 | 17 650 |
| old_b | 00:06 | 804.3 | 162 971 | 17 922 |
| new2 | 00:19 | 785.2 | 166 923 | 17 654 |
| old_c | 00:25 | 799.6 | 163 923 | 17 922 |

* old mean 797.4 ms/step (sd 6.6), new mean 788.4 ms/step (sd 3.2)
  -> **-9.0 ms, -1.13%** step time, **+1.14%** sustained throughput.
* A linear fit through the three old legs (slope +0.29 ms/min, i.e. the machine
  drifts ~1.6% over the session) puts the new legs 5.1 ms and 16.3 ms below the
  old trend, i.e. the effect is small but consistently in the same direction,
  inside the +-1.3% run-to-run band.
* Peak CUDA memory 17.92 GB -> 17.65 GB (**-271 MB, -1.5%**).
* Routing is unchanged: every leg's diagnostic step shows 8 quadratic fallbacks
  (`plan_unavailable`, not falling back to event) and 16 event backward routes; the
  fused path did not change planner availability or the fallback ratio.
* Losses and gradient norms match at the displayed precision
  (3.1300/2.4811/1.8154, norms 0.494/0.234/0.198). This is not proof that all
  parameter gradients are identical; the documented dvfirst rounding differs.

### Where the removed time went (profile attribution)

One warmed profiled step per arm, same harness arguments as the baseline
profile (`build/valres_old_profile.breakdown.json`,
`build/valres_new_profile.breakdown.json`, traces kept in the same directory).

| metric | old | new | delta |
| --- | --- | --- | --- |
| GPU activity union | 673.9 ms | 643.1 ms | **-30.8 ms** |
| GPU no-activity (idle) | 257.8 ms | 277.3 ms | **+19.6 ms** |
| CPU step span | 931.6 ms | 920.4 ms | -11.2 ms |
| `other_pointwise_norm_reductions_optimizer` | 153.5 ms | 128.0 ms | -25.5 ms |
| `transfers_and_memset` | 60.3 ms | 56.5 ms | -3.9 ms |
| CUDA kernel launches | 21 795 | 20 838 | -957 |

The new kernels in that step: `_value_residual_fwd_kernel` 68 calls / 2.02 ms,
`_value_residual_bwd_kernel` 68 calls / 6.76 ms, plus
`triton_poi_fused_add_value_residual_backward_v1_3` 24 calls / 0.86 ms that
Inductor fused around the compiled call site - 9.6 ms total versus ~40 ms of
old-path value-residual work (the -25.5 ms pointwise group plus the removed
casts).  The baseline profile artifact, captured two hours earlier, reports
257.08 ms idle against 257.76 ms for the old arm here, so the +19.6 ms idle
change in the fused arm is an observation, not a measured noise bound from
repeated profiles.

Interpretation:

1. The profiled step carries ~258-277 ms with no GPU activity. These traces alone
   do not attribute the idle intervals to planner waits versus CPU launch or
   dispatcher overhead, other synchronization, or profiler perturbation.
   Ready policy avoids waiting for incomplete plans. Do not extrapolate these
   intervals directly to unprofiled critical-path savings.
2. Only part of the value residual was ever eager.  Six of the twelve layers are
   hybrid `DismGdnAttention`; their DISM branch runs inside the compiled graph
   where Inductor had already fused the eager elementwise chain, and the opaque
   custom op replaces that fusion.  Per step, 24 of the 68 calls are compiled
   (6 layers x 4 microbatches) and 44 are the eager GDN-branch calls
   ((5 GDN-only + 6 hybrid GDN branches) x 4); total 17 calls per microbatch.
3. ~0.10 ms/call of framework overhead (above) costs ~7 ms/step across 68 calls,
   the same order as the net gain.

The operator is kept enabled (`fused` is the default). The historical A/B shows
271 MB less peak activation memory and unchanged diagnostic routing, but the
~1% speed gain is near timing noise and needs stronger repetition.

Design alternative: saving the FP32 sigmoid instead of recomputing it in the
backward would add one `rows * 4` byte buffer (786 KB on the production packed
row, 196 608 rows) that is allocated, written and read every microbatch, in
exchange for one `exp` per row per layer.  The recompute path is the one
implemented; the saving variant was not measured (timing deferred), so no claim
is made about which wins.  Saving FP32 `delta`/`mixed` was never considered: it
would multiply the value-residual activation footprint by 2-3x.

`compute-sanitizer` (CUDA 13.1) on BF16/F16/FP32, `DV = 32/48/64/128`, `DV = 1`,
non-contiguous inputs and a non-contiguous `dout`:

* `memcheck`: 0 errors
* `racecheck`: 0 hazards, 0 warnings
* `synccheck`: 0 errors

Original kernel resources before stride repair (final numbers above;
sm_120a, `BLOCK_ROWS=8`, one CTA covers the whole
masked `DV` axis):

| kernel | dtype | registers | spills | shared |
| --- | --- | --- | --- | --- |
| forward | BF16 | 17 | 0 | 0 B |
| forward | FP32 | 22 | 0 | 0 B |
| backward | BF16 | 25-26 | 0 | 16 B |
| backward | FP32 | 28 | 0 | 32 B |

No spills and no local memory.  The backward's small shared buffer is the
compiler's scratch for the cross-warp `tl.sum`, not an added staging area and
not an atomic.  Each kernel contains exactly one `CALL.REL.NOINC`, the
out-of-line `__nv_expf` used by the bit-exact sigmoid; it is the accurate
libdevice exponential (`torch.sigmoid` uses the same function), so the call is
the price of matching the eager rounding.  If the performance phase shows it
dominant, a fast `ex2.approx`-based sigmoid is the obvious candidate, but it
would move `out`/`dv` off bit-exactness and has not been taken here.

## Compiler contract

The operator is registered as `flash_dism::value_residual_forward_v1` /
`..._backward_v1` with `register_fake` and `register_autograd` (the same
pattern as the other opaque Triton boundaries in `compiler.py`), so Dynamo
never tries to trace a kernel launch and AOTAutograd gets an explicit
saved-state backward.

`tests/test_value_residual_integration.py` checks:

* `torch.compile(..., fullgraph=True, dynamic=True)` of a forward+backward
  closure stays in one graph (`CompileCounter.frame_count == 1`) and matches
  the eager gradients;
* `DismAttention(value_residual=True)` compiled with `fullgraph=True,
  dynamic=True` under Inductor keeps `frame_count == 1` over three packed
  boundary layouts (different document counts) and produces finite outputs;
* the real callers (`DismAttention._forward_cuda` and
  `PureGdnAttention.gdn_raw`) match the eager helper's parameter gradients with
  the fused op enabled and the reference forced, on the same weights and
  inputs.

## Limitations / not yet verified

* First-order gradients only; no second-order support is claimed or tested.
* FP64 runs on the eager fallback by design (never silently narrowed).
* `numel < 2**31` guard; no fp64 accumulation path.
* The 72M A/B is a 3-vs-2 leg interleaved comparison on one idle GPU; the
  per-leg band is +-1.3% and the measured effect is ~1.1%, so the direction is
  consistent but the size is at the edge of run-to-run noise.  It is a scratch
  replay of four fixed resident batches, not a multi-hour training run; nothing
  was downloaded, no training was restarted, and no checkpoint, RNG counter,
  annealing buffer, optimizer state or dataloader position was touched.
* The op is 2.9-3.2x faster in isolation but the eager FP32 helper is already
  cheaper than the fused call for pure-FP32 inputs; training is BF16, and no
  attempt was made to special-case FP32 dispatch beyond falling back.
* ~0.10 ms/call of the fused time is custom-op autograd/dispatcher overhead
  (see above).  Removing it (plain `autograd.Function`, fewer output
  allocations) is a possible follow-up requiring new compiler validation, so
  it was left alone.
* `out`/`dv` are bit-exact with the eager helper; `dvfirst` intentionally
  follows the FP32 contract instead of reproducing the helper's duplicated
  BF16 rounding (section above); `dgate` keeps a ~2 FP32 ulp reduction-order
  difference on DV=64.  All three are quantified above.
* Pre-existing failures unrelated to this change were observed in the same
  run (`tests/test_hybrid.py::test_hybrid_decode_chunks`, several
  `tests/test_modeling.py` decode tests): `forward_torch() got an unexpected
  keyword argument 'vocab_temperature'`.  They reproduce with this change
  stashed and are outside this task's scope.

## Commands

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/dism_v3/python/flash_dism/kernels:$PWD/nanochat"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=1

# accuracy (no timing, safe on a shared GPU)
python dism_v3/tools/bench_value_residual.py --skip-timing \
    --out dism_v3/build/value_residual_accuracy_$(date +%Y%m%d).json

# correctness
python -m pytest dism_v3/python/flash_dism/kernels/test_value_residual.py \
                 dism_v3/tests/test_value_residual_integration.py -q

# sanitizers (needs CUDA 13.1 in PATH); one tool at a time
PATH=/usr/local/cuda-13.1/bin:$PATH compute-sanitizer --tool memcheck  \
    python dism_v3/tools/value_residual_sanitizer_smoke.py
PATH=/usr/local/cuda-13.1/bin:$PATH compute-sanitizer --tool racecheck \
    python dism_v3/tools/value_residual_sanitizer_smoke.py
PATH=/usr/local/cuda-13.1/bin:$PATH compute-sanitizer --tool synccheck \
    python dism_v3/tools/value_residual_sanitizer_smoke.py

# performance, only on an idle GPU
python dism_v3/tools/bench_value_residual.py \
    --out dism_v3/build/value_residual_bench_$(date +%Y%m%d).json

# 72M A/B (old -> new -> old), same args as build/hard_dispatch_ready_idle.json
CKPT=/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/base_checkpoints/seqtrue72m-fwedu-valres
for arm in old new old; do
  [ "$arm" = old ] && export DISM_VALUE_RESIDUAL_KERNEL=reference || unset DISM_VALUE_RESIDUAL_KERNEL
  python dism_v3/build/ce_review_model.py --review-ce chunked-document \
      --checkpoint-dir "$CKPT" --step 19074 --batches 4 --warmup 4 \
      --phase-steps 3 --window-steps 8 --windows 3 --conv-impl linear_fla \
      --hard-backward-policy ready --planner-workers 16 \
      --output dism_v3/build/valres_ab_${arm}_$(date +%H%M).json
done

# profile attribution (add --profile --profile-only), then
python dism_v3/tools/analyze_model_profile.py \
    dism_v3/build/valres_new_profile.trace.json \
    --output dism_v3/build/valres_new_profile.breakdown.json

# A/B switch used by every comparison above
DISM_VALUE_RESIDUAL_KERNEL=reference ...
```

Artifacts (all ignored, `build/`): `value_residual_accuracy_20261014.json`,
`value_residual_bench_20261014.json`, `valres_ab_{old_a,new,old_b,new2,old_c}.json`,
`valres_{old,new}_profile.json`, `valres_{old,new}_profile.breakdown.json`,
`valres_{old,new}_profile.trace.json`.
