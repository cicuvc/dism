# WS backward optimization plan (2026-09-09)

Goal: optimize BOTH summary+dV (`core_dv_ws.cu`) and operand/scalar gradients
(`core_ab_ws.cu`), following the measured forward optimization workflow.
Spill is recorded, not a stop condition. Preserve all nine D/DV combinations,
RNG replay, scalar gradient routing, FP32 delta, single-BF16-P WS semantics,
FP32 scalar gradients and existing strict precision failure visibility.

## Initial inspection

Both kernels currently use nonpersistent128-key CTAs with8compute+4producer
warps, register-held16-key B/V tiles per warp, two64-query A/dO slots,
per-pair32-arrival reverse mailboxes, and CTA-wide initialization/drain.
Inputs initially use consumer scalar loads. D128 query/dO TMA emits separate
swizzle-panel transfers. Score reloads tau/labels/LSE inside scalar calls;
normalizer/delta are loaded for repeated score positions. Direction/label
width/hard selection is dynamic. Reverse sigmoid already uses tanh.approx.
Streaming visits all query tiles, including fully noncausal tiles below keys.
AB additionally has authorized per-warp Gsoft/dA shared union and dA TMA adds.

## Stages and evidence required

1. Freeze binary/codegen and actual-embedding launch baselines, profile both
   target kernels. Add an isolated benchmark with saved forward states.
2. Cache scalar metadata in registers, hoist scale/tau conversion, predicate
   score/coefficient selection, explicit FTZ EX2; evaluate direction/mode/label
   specialization and full-width TMA. Test W/G/dV/all operand/scalar gradients.
3. Derive safe reverse causal trimming, including omitted affine summaries,
   diagonal propagation and invalid warps. Do not simply omit state writes.
4. Persistent CTA and asynchronous held-B/V + first-A/dO prefetch across tasks;
   independently audit B1/B3 shared budgets, scratch lifetime, dA drain,
   register allocation, accumulated tile phases and partial tasks.
5. WG-level reverse mailbox, overlap score MMA/recompute with other WG reverse
   scan, stage-count experiments, early buffer release where safe. Prefer TK
   primitives; no new computational shared staging merely for convenience.
6. Regress nine dimensions/two directions/endpoints/mixed/tails/bitset/long
   chains; sanitizer, noCALL/native TMA/register-role codegen with spill record;
   end-to-end checks, before/after per-kernel and integrated timing. Record
   unsuccessful candidates and keep defaults evidence-based.

This goal is not complete when only scalar cleanup is done. Performance
results and per-stage progress will be appended as evidence becomes available.

## Stage1: metadata cache / scalar cleanup

Opt-in `DISM_BWD_OPT=1`; default remains0 while the larger optimization is
in progress. Both WS kernels use `backward_metadata.cuh`. Key tau/labels/LSE
are cached before the query loop; each query tile preloads two logical columns
per lane of label/LSE/normalizer/delta before the input wait. Shuffles distribute
these registers for score, P and reverse affine coefficients. No new shared
staging, no changes to RNG, input/mail protocol, dA layout, or accumulation.
Masks use explicit selp, reverse sigmoid remains tanh.approx, and weight EX2
explicitly flushes subnormal results. Invalid coordinates retain affine
identity; reachable hard breaks retain zero-map semantics.

The initial single-FFMA score candidate introduced two failures in strict
same-G BF16 GEMM probes (bounded_soft N64/65). Reverting just score arithmetic
association to `(dot*scale-lse+tau)*LOG2E`, while keeping metadata caching,
predication and FTZ, removed both. Do not describe single-FFMA recomputation
as accepted yet, nor change the independent G probe/tolerances to hide this.
Initial candidate binaries are saved in `/tmp/dism-bwd-baseline.QDt5ap/`
as `metadata_fma_full.so` and `metadata_fma_finite.so` (original PyInit names
have `_opt` suffix). The original backward full/finite binaries are also
saved in that directory for separate-process benchmarks.

### Correctness and codegen evidence so far

Full selected WS dimensions/tails/reference suite: frozen baseline192 passed,
14 failed; initial FMA190 passed,16 failed; current cached version192 passed,
the SAME14 failed (41 deselected each). Existing failures are strict reference
gradients, primarily scalar tau/production delta precision, not silently xfail.
Failure-set comparison: `benchmarks/backward_metadata_numerics_sm120a.json`.
Full and finite selected end-to-end autograd each507 passed (398 deselected).
Full targeted memcheck/racecheck/synccheck each6 passed, zero errors/hazards
(43.27s/192.29s/6.37s): B1/B3 mixed D32/DV32 and D128/DV128, plus N65
D64/DV128 tails including the formerly failing bounded-soft AB case. Both
full/finite new WS codegen suites pass. Current cached full/finite binaries
are saved as `metadata_cached_full.so` / `metadata_cached_finite.so` in the
same temporary baseline directory for the next structural A/B stage.
Remaining long-chain, broader sanitizer and final integrated coverage must be
completed after structural optimization; these checks are not whole-goal acceptance.

Current D64/DV64 static LDG instruction sites shrink B1:386→37, B3:356→37.
This is static codegen, not a claim of an equal dynamic memory-traffic ratio.
All18 WS instances retain native TMA and dec40/inc232 with no CALL.
Spill report: `benchmarks/backward_metadata_cached_codegen_sm120a.json`;
`backward_metadata_codegen_sm120a.json` records the earlier FMA candidate.
Current stack bytes (D rows,DV columns32/64/128):

| Mode/kernel | D32 | D64 | D128 |
|---|---|---|---|
| full B1 |0 /0 /136|0 /16 /184|32 /48 /344|
| full B3 |0 /0 /40|0 /16 /128|88 /192 /448|
| finite B1 |0 /40 /264|0 /48 /304|8 /48 /392|
| finite B3 |0 /0 /56|0 /16 /64|120 /264 /360|

Spills are retained per user authorization, not used to stop the optimization.
The added codegen suite reports resources without the older zero-spill gate;
those older strict gates have not been weakened or deleted.

### Timing and NCU

RTX5090, CUDA13.1/sm120a, B64/H4/N1024/D=DV64/V512, actual CUDA embedding
inputs, scale1,rtau3,hard_prob.5,finite, bitset off. Saved forward state,
normalizer and FP32 delta fixed before timing. CUPTI per-kernel durations,
10 warmups +30 launches per subprocess,3 rounds with reversed A/B order in
round2 (90 samples each); no concurrent GPU tests during measurement.
Unmodified binaries are the baseline. No clock locking for CUPTI; all samples
retained, B3 has visible round-to-round variation.

| Direction | Kernel | baseline median us | cached median us | Ratio |
|---|---|---:|---:|---:|
| q_from_k | summary+dV |2260.341|1270.778|1.779x|
| k_from_q | summary+dV |2202.053|1273.930|1.729x|
| q_from_k | dA/dB/dLSE/dtau |3131.135|2751.314|1.138x|
| k_from_q | dA/dB/dLSE/dtau |2904.930|2496.371|1.164x|

These are not full-model training speedups. Raw per-launch distributions and
auxiliary times: `benchmarks/backward_metadata_timing_sm120a.json`.
Initial one-round q/k baseline artifacts also remain, not substituted for the
controlled three-round comparison above.

NCU full-set q-direction reports (default NCU clock/replay controls, not CUPTI
timings): `/tmp/dism-backward-baseline-q-3f4e0b1.ncu-rep` and
`/tmp/dism-backward-metadata-q.ncu-rep`.40passes each baseline;40/42passes
optimized B1/B3. Source imported. B1 duration3.19→1.68ms, SM throughput
26.23→39.42%; B3 duration4.38→3.74ms, SM throughput26.46→27.05%.
Global-load sector utilization rises to30.8/30.4 bytes out of32 (B1/B3);
B1 baseline was6.9 bytes. B3 is still substantially under-utilized. Newly
measured local spill requests6.29MB/1.57MB do not erase the net improvement.
Do not infer occupancy tuning is the next priority from generic NCU advice.
Selected machine-readable NCU metrics: `benchmarks/backward_metadata_ncu_sm120a.json`.

### Next structural step

Derive and test uniform CTA query trimming below the128-key CTA base. For
q<base all valid keys are strictly noncausal; reverse zero-map summaries must
still be written, with identity for wholly padded key chunks. Invalid warps
must keep the same protocol. Then add persistent scheduling and held B/V TMA
prefetch, full-width A/dO transfers and WG mail/stage experiments. These steps
remain part of the original goal, not optional work outside its acceptance.

Reproduce current metadata candidate:

```bash
DISM_BWD_OPT=1 DISM_TILE_LSE=tanh_finite DISM_LINEINFO=1 /home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.benchmark_backward_kernels --direction q_from_k
DISM_BWD_OPT=1 DISM_TILE_LSE=full DISM_LINEINFO=1 /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_dv_ws.py tests/test_dism_v2_ab_ws.py -k 'dimensions or tails'
```

## Stages2–4: trimming, persistent inputs, WG mail

`DISM_BWD_OPT=2` adds uniform reverse query trimming. Only query tiles with
qb >=128*key_cta execute. B1 fills omitted dense summary entries with (0,0)
for chunks containing valid keys, (1,0) for wholly padded32-key chunks.
All q in the omitted interval are below every valid key of the CTA, so their
score/P/emission are zero maps; missing-key coefficients are identity. B3
omits the corresponding zero gradients. No RNG identities or boundary layouts
change. Partial valid chunks must NOT be filled with identity.

`OPT=3` adds persistent scheduling to both WS kernels, at most one CTA per
SM. A physical CTA walks logical `(batch_head,key_cta)` tasks; per-key tau
partials retain logical task indexing, independent of the persistent grid.
Initial B/V staging becomes natural128-row TK tiles; each warp selects its
interleaved16-row subtile. Each held B/V tensor uses one TMA, including tails
bounded independently in N and batch_head. A/dO full64-query tiles also each
use one TMA, including D128's swizzle panels. Their nonaligned tails retain
the cooperative warp8 scalar path and ready32/free256 protocol.

The existing initial-B/V versus A/dO-ring union is retained, without new data
staging. Producer waits for all used old input slots' read-free epochs, then
launches next held B/V while old reverse/gradient/output work can finish.
Consumers wait held_ready1, load B/V registers, and collectively release
held_free256 before the producer overwrites the union with A/dO. Query-slot
and mail phases now use accumulated tile counts across tasks; held phases
use task counts. No task-boundary CTA barrier; initialization only, then
independent128-thread WG exits. Existing B3 Gsoft/dA scratch lifetime and
per-task dA drain remain intact.

IMPORTANT: current first A/dO still loads AFTER next held B/V reaches registers.
It is NOT yet prefetched alongside next held B/V during the old workload.
That part of the forward-style prefetch objective remains to be attempted.

`OPT=4` replaces the four independent reverse mailbox ready/free epochs per
slot by one128-arrival epoch per compute WG. Four payloads per slot remain;
all WG1 writers publish before WG0 reads, and all WG0 readers release before
reuse. mbarriers fall from22 in OPT3 to10 in OPT4. Input ring remains2-stage.
Default remains OPT0 pending remaining experiments/final acceptance.

### Validation and timings

The added `test_dism_v2_backward_trim.py` compares against OPT1.108 cases
cover all nine D/DV combinations, two directions, three probabilities, N129
and257. dV/summary/passing boundary are bitwise equal; dA/dB/dLSE/dtau use
the existing3e-5 replay tolerance for atomic accumulation ordering. It also
checks omitted zero/identity maps explicitly.30 persistent epoch cases add
more sequences than SMs, N1/65/129/257/385 and D=DV32/64/128.

- OPT2 finite:108 equivalence cases + codegen pass; full selected original
  suite remains192 passed/14 same known failures.
- OPT3 full/finite:138 equivalence/epoch cases + codegen pass each. Full
  original dimensions/tails/reference remains192 passed/14 known failures.
- OPT4 full:331 passed/14 same known failures (41 deselected), including139
  new equivalence/epoch/codegen checks. Finite:139 passed. Selected full/finite
  end-to-end:507 passed each (398 deselected), no tolerance changes.
- OPT4 full persistent-epoch memcheck/racecheck/synccheck each6 passed,
  zero errors/hazards (2.99s/175.86s/2.45s), covering N1/65/129/257/385,
  D=DV32/64/128, both directions, and more sequences than SMs. Snapshots
  `wgmail_full.so` / `wgmail_finite.so` preserve this stage in the temporary
  baseline directory; their original PyInit names end in `_opt4`.
- OPT4 finite bitset suite62 + updated codegen1 passed; updated full codegen1
  also passed. Native load count and initialization/WG-exit barrier count are
  now explicitly asserted for persistent candidates, while spill is reported.

Same actual-embedding N1024/D64/DV64 mixed fixture as stage1,2 rounds x30
samples per variant/direction with order reversed in round2. No concurrent
GPU tests. Pooled CUPTI medians, us:

| Direction/kernel | OPT2 trimmed | OPT3 persistent | OPT4 WG mail |
|---|---:|---:|---:|
| q B1 |870.955|684.380|686.684|
| k B1 |870.315|684.860|685.260|
| q B3 |1658.663|1586.583|1561.575|
| k B3 |1529.847|1501.032|1442.776|

WG mail is approximately neutral for B1 and mildly helpful for B3. B3's
round-to-round timing varies; keep all samples and avoid overattribution.
Raw: `benchmarks/backward_trim_timing_sm120a.json` (OPT1/2 comparison) and
`backward_persistent_timing_sm120a.json` (OPT2/3/4). Frozen intermediate
`trim_{full,finite}.so` and `persistent_{full,finite}.so` reside alongside
earlier baselines in `/tmp/dism-bwd-baseline.QDt5ap/`; original module names
end in `_opt2` and `_opt3`, respectively.

OPT4 codegen:18 WS instances, noCALL, four native UTMALDG.5D sites each,
dec40/inc232 preserved, two BAR.SYNC sites (initialization and WG exit).
Current stack B (D rows,DV columns32/64/128), all spills retained:

| Mode/kernel | D32 | D64 | D128 |
|---|---|---|---|
| full B1 |0 /32 /160|0 /40 /256|24 /40 /392|
| full B3 |0 /0 /0|0 /8 /80|208 /288 /368|
| finite B1 |0 /40 /256|0 /40 /320|8 /72 /432|
| finite B3 |0 /0 /48|0 /40 /104|184 /200 /288|

Raw: `benchmarks/backward_wgmail_codegen_sm120a.json`. NCU q full-set,
40passes per kernel, source imported, default profiler clock/replay controls:
`/tmp/dism-backward-wgmail-q.ncu-rep`. B1 duration915.10us/SM40.70%; B3
2.12ms/SM26.91%. Main-shape dynamic shared36992B/53376B, unchanged data
allocation with fewer mail barriers. These shared numbers apply to D64/DV64,
not a blanket budget claim for the larger baseline shapes. Raw selected
metrics: `benchmarks/backward_wgmail_ncu_sm120a.json`.

### OPT5 and stage-count checkpoint

`DISM_BWD_OPT=5` additionally uses direct RHS LDSM loading for score and dP,
following the forward loader's mapping, then `mma_AB`. The default remains
OPT0. `DISM_BWD_STAGES=1/2/3` exposes input/mail ring experiments for OPT>=4;
requests exceeding the conservative candidate shared budget fall back to two
slots. B3's dA output scratch remains independently double-buffered.

Finite-mode equivalence/epoch/codegen tests passed139 cases for OPT5 with
default two stages (55.91s), and139 with requested three stages (56.14s).
The former run began before the generic stage-count edits; the latter built
the updated source. Requested three does not mean every shape uses three:
in particular B3 D64/DV64 falls back to two. These checks are not yet a
full-mode, sanitizer, or performance qualification of OPT5/stage variants;
stage1 also remains unvalidated. No default or numerical tolerance changed.

### Remaining work / first-A/dO prefetch design notes

Next verify direct RHS loading to eliminate avoidable LDSM→MMA layout work,
and test stage counts and first-query overlap. A separate first-A/dO slot fits
the B1 main shape's budget but would take B3 D64/DV64 beyond64KiB. A possible
B3 experiment is to alias the16KiB per-warp scratch region with first A/dO:
publish scratch-free only after all old dA TMA shared reads finish, prefetch
next B/V and first A/dO concurrently, and move first-tile Gsoft shared writes
AFTER dB reads A. This requires a256-reader release before any Gsoft writer
overwrites first-query input, plus independent first-query vs regular-ring
phase counters. Preserve B1/B3 tail and invalid-warp participation. This is
a proposed experiment, not implemented or validated yet. Direction/mode
specialization evaluation, broad long-chain/bitset/integrated performance and
final default selection also remain before goal completion.

Work-distribution diagnostic for the current170-SM main shape: bh-major task
order assigns96–136 query tiles per persistent CTA (mean108.42). A key-major
decode would assign104–120 at the same mean. This is a host count calculation,
not measured speedup; task-order A/B may be useful after prefetch changes.
