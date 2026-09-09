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
stage1 was subsequently validated below. No default or numerical tolerance changed.

### RHS/stage comparison follow-up

OPT5 finite requested stage1:139 passed (43.54s), including the same108
equivalence,30 persistent-epoch, and18-instance codegen checks. This does not
replace racecheck for the modified ring. OPT5 full stage2 subsequently passed
the same139 checks (168.95s including compilation); no tolerance changes.
Finite stack bytes for OPT5 stage2:

| Kernel | D32 (DV32/64/128) | D64 | D128 |
|---|---|---|---|
| B1 |0 /16 /184|0 /40 /208|16 /32 /352|
| B3 |0 /0 /40|0 /0 /32|56 /96 /176|

Direct RHS loading lowers many spill stacks, including B3 D64/DV64 from40B
to0B, but not all: B1 D128/DV32 rises8→16B. Keep spills visible; no manual
register tuning was done. Full resource strings for all four candidates are
in `benchmarks/backward_rhs_codegen_sm120a.json`.

Finite actual-embedding main shape (B64/H4/N1024/D64/DV64/V512, mixed.5),
two rounds of30 CUPTI samples each, reversed candidate order in round2:

| Direction/kernel | OPT4 stage2 | OPT5 stage2 | OPT5 stage1 | OPT5 requested3 |
|---|---:|---:|---:|---:|
| q B1 |691.740|697.260|675.372|700.796|
| k B1 |692.172|695.405|675.628|698.844|
| q B3 |1580.103|1525.143|1542.840|1535.014|
| k B3 |1697.734|1388.441|1407.976|1390.504|

Units us, pooled medians; no concurrent GPU tests, clocks not locked. B3
requested3 actually falls back to2 at this shape, so its small timing
differences are variability, not a third-stage effect. Individual B3 rounds
show substantial variability (q OPT4 medians1573.54/1791.01us; q OPT5 stage1
1792.76/1518.09us). Do not infer a stable universal speedup from pooled
medians. Stage3 does not improve B1 here; stage1 modestly improves B1 but
does not establish a whole-backward win. OPT5 stage2 remains the experimental
comparison point for first-query prefetch, not a promoted production default.
All16 runs returned finite gradients. Raw samples and per-round metadata:
`benchmarks/backward_rhs_timing_sm120a.json`.

Finite OPT5 requested stage1 and3 each passed memcheck/racecheck/synccheck
on three persistent-epoch nodes: N1/q/D32, N65/k/D64, N257/q/D128 (D=DV,
batch=SM_count+3). Each tool reported zero errors/hazards; stage1 race77.87s,
stage3 race77.49s. These nodes test single-tile/tail and fallback paths but
do not exercise a long B3 three-slot ring, so additional N385/D32 and N385/D64
checks were run separately: requested stage3 N385/q/D32 and N385/k/D64
(D=DV, batch=SM_count+3) each passed all three sanitizer tools, zero
errors/hazards. The D32 case exercises actual three-slot rings in BOTH
kernels across multiple tasks; D64 exercises B1 three-slot/B3 two-slot
fallback. This remains scoped sanitizer evidence, not whole-goal acceptance.

### OPT6: first-A/dO cross-workload prefetch

Implemented as a separate opt-in candidate, still no default promotion.
Producer drains old regular input slots, issues next held B/V, then issues
next first A/dO BEFORE waiting for next held-free. This also applies on task0,
but the intended overlap is old-task final work with next-task input loads.
The first64-query tile is excluded from the regular input ring counter;
reverse mail keeps its original all-query counter. First-ready epochs advance
once per task with32 producer arrivals, retaining safe cooperative tail loads.

B1 uses a dedicated first input slot if the conservative63-KiB budget allows
it. Old first-input-free has256 consumer arrivals after dP/dV finish reading;
next first input is disjoint from the held-B/V/input union. B3 aliases first
A/dO with its existing16-KiB Gsoft/dA scratch if the input fits. After old-task
dA shared-source reads drain, all256 consumers publish scratch-free BEFORE
their final global gradient writes. Producer waits this release before
overwriting scratch. On the new first query only, register Gsoft survives dB
MMA, then a256-reader first-consumed barrier completes before ANY Gsoft shared
write. This lets WG1 publish reverse state before waiting for WG0's input
reads, avoiding a cyclic dependency. Remaining queries use the old scratch
protocol. Full dA TMA completion remains at task end, independently of source
read completion. No extra computational shared staging or dense state added.

With requested stage2, both kernels enable first-query prefetch for D/DV
32/32,32/64,64/32,64/64; the other five shapes retain the previous protocol.
B1 stage1/3 eligibility follows its actual total budget, not this stage2 list.
B3's scratch stays at its old location when prefetch is disabled; legacy
variants are not deliberately relaid out by this experiment.

Finite first-source equivalence/epoch/codegen suite139 passed (23.35s),
including all nine dimensions and more tasks than persistent CTAs. All18 WS
instances retain noCALL, dec40/inc232, two BAR.SYNC sites, and either six native
TMA sites (prefetch enabled) or four (fallback), asserted per shape. The initial
build exposed a legacy-template task_round name-lookup error, fixed before
runtime testing; no math or tolerance was changed. Full suite139 passed
(75.76s). An equivalent counter cleanup then explicitly reused the original
tile counter for non-prefetch instances, avoiding accidental independent-ring
codegen in fallback/legacy paths. Final-source finite/full suites each139
passed again (10.19s/44.80s). Both retain the scoped18-instance codegen gates.

Finite memcheck/racecheck/synccheck each5 passed, zero errors/hazards, on
persistent-epoch N1/q/D32, N65/k/D64, N385/q/D32, N385/k/D64, N257/q/D128
(D=DV, batch=SM_count+3). Memcheck2.68s and racecheck174.76s ran before the
counter cleanup; synccheck30.63s included rebuilding the final source. These
cover first-only tasks, unpadded first-input tails, real first-input/scratch
reuse across workloads, and large-shape fallback; not blanket acceptance of
all stages or the final optimization goal.

Final stack bytes (D rows,DV columns32/64/128), spill retained:

| Mode/kernel | D32 | D64 | D128 |
|---|---|---|---|
| finite B1 |496 /56 /184|0 /56 /208|16 /32 /352|
| finite B3 |496 /0 /40|0 /0 /32|56 /96 /176|
| full B1 |648 /40 /160|24 /32 /224|24 /48 /288|
| full B3 |496 /0 /0|0 /0 /48|40 /80 /248|

Raw: `benchmarks/backward_prefetch_codegen_sm120a.json`. Initial finite
D32/DV32 B3 had608B stack before the counter cleanup. A SASS diagnostic on
that snapshot located all LDL/STL before consumer inc232, with zero such
instructions after inc232 in either B1/B3; do not attribute that large stack
to consumer gradient accumulators. This is static instruction-location
evidence, not dynamic profiler traffic. No dedicated spill fix was attempted.
Frozen pre-change OPT5 binaries for controlled timing are
`/tmp/dism-bwd-rhs-baseline.RyBviY/rhs_{full,finite}.so`, retaining `_opt5`
PyInit names. Timing results follow as collected.

Full original WS dimensions/tails suite:192 passed/14 failed/41 deselected
(16.87s), with EXACTLY the14 frozen-baseline failure IDs, no new or resolved
failures. Finite actual-embedding timing, two rounds x30 CUPTI samples,
reversed variant order in round2, B64/H4/N1024/V512/mixed.5 (us medians):

| D=DV / direction | B1 OPT5 | B1 OPT6 | B3 OPT5 | B3 OPT6 |
|---|---:|---:|---:|---:|
|64 / q|692.588|694.685|1511.735|1583.144|
|64 / k|692.764|693.484|1381.800|1453.591|
|32 / q|527.309|571.068|1027.353|1097.739|
|32 / k|526.061|571.005|926.650|1002.411|

All16 runs finite. Main D64 B1 is neutral and B3 regresses about5%; both D32
kernels regress. Keep OPT6 as an unsuccessful opt-in experiment, not a
production default or an assumed latency-hiding improvement. Raw samples:
`benchmarks/backward_prefetch_timing_sm120a.json`.

Matched q D64 NCU full sets (40passes per kernel, source imported, profiler
default clock/replay controls, NOT directly comparable to CUPTI times):
`/tmp/dism-backward-rhs-q.ncu-rep` and
`/tmp/dism-backward-prefetch-q.ncu-rep`. B1 OPT5→6 duration961.82→917.02us,
SM38.64→40.43%; B3 about2.01→2.06ms, SM27.73→27.52%. This B1 improvement
under profiler conditions does not override the ordinary timing result.
Dynamic shared main shape36,992→53,504B for B1,53,376→53,504B for B3
(plus1,024B driver allocation). B1 local spilling requests2,272,568→3,468,600;
B3 zero for both. Both B3 variants report roughly4-way shared-store bank
conflicts. Raw selected metrics: `benchmarks/backward_prefetch_ncu_sm120a.json`.

### OPT7: dA output swizzle experiment

OPT7 deliberately builds on OPT5, WITHOUT OPT6 first-query prefetch. It
replaces B3's two raw16x16 FP32 dA output slots with same-size TK
`st_fl<16,16>` (64B swizzle), and selects matching64B swizzle in the dA TMA
tensor map. Output indexing uses the TK tile operator; dimensions, arithmetic,
TMA reduction, source-read waits and slot lifecycle are unchanged. No extra
shared allocation or intermediate compute staging. This directly tests whether
dA output layout contributes to the profiled shared-store bank conflicts;
do not assume it accounts for all Gsoft/mail/output shared traffic.

Finite/full equivalence/epoch/codegen each139 passed (56.68s/75.42s), noCALL,
four native TMA load sites per instance and native dA reduction retained.
Finite B3 stack bytes D rows/DV columns32/64/128:
`(0,0,80)/(0,56,192)/(144,192,272)`. Spill retained; all full/finite resource
strings in `benchmarks/backward_da_swizzle_codegen_sm120a.json`.
Finite memcheck/racecheck/synccheck each3 passed, zero errors/hazards, on
persistent-epoch N65/k/D64, N385/q/D32, N257/q/D128 (D=DV,
batch=SM_count+3). Racecheck123.32s; synccheck29.37s included rebuilding after
adding the inactive-for-OPT7 paired-store branch.

Same two-round finite CUPTI experiment against frozen OPT5, B3 us medians:
D64 q1514.086→1581.974, k1382.888→1485.592;
D32 q1026.202→1032.170, k925.930→941.034. No default promotion.
Raw: `benchmarks/backward_da_swizzle_timing_sm120a.json`.

NCU `/tmp/dism-backward-da-swizzle-q.ncu-rep` (same main q shape/full set,
40passes per kernel/source imported) shows B3 about2.06ms/SM29.00% and
3,915,776 local spilling requests. Shared-store conflict average drops from
about4.2 to2.5-way, but requests increase14,229,504→23,666,688; total shared
store wavefronts59,474,506→59,980,902. Thus lower conflict per request did
NOT reduce total shared-store work. A plausible cause is loss of paired
FP32 store merging through swizzled address expressions; OPT8 tests explicit
pair stores rather than assuming this attribution is proven. Selected metrics
in `benchmarks/backward_da_swizzle_ncu_sm120a.json`.

### OPT8: explicit paired swizzled dA stores

Builds on OPT7, still no first-query prefetch. Calls TK `move<float2>::sts`
using TK tile address calculation for each adjacent FP32 pair. Logical j is
even and64B swizzling preserves low4 address bits, so each pair is contiguous
and8-byte aligned. Arithmetic, output storage size and all TMA lifetimes are
unchanged. Finite/full equivalence/epoch/codegen each139 passed
(42.58s/75.08s); default remains OPT0. Finite B3 stack bytes D rows/DV
columns32/64/128: `(0,0,40)/(0,0,96)/(40,80,176)`; D64/DV64 returns to0B.
SASS main B3 contains64 static STS.64 sites, versus0 in OPT7, and no local
load/store instructions in that main instance. No spill tuning was done.
Full/finite resources: `benchmarks/backward_da_pair_codegen_sm120a.json`.
Finite memcheck/racecheck/synccheck each3 passed, zero errors/hazards, using
the same N65/k/D64,N385/q/D32,N257/q/D128 persistent nodes as OPT7.
Racecheck110.94s, synccheck2.26s. This includes partial TMA reductions and
repeated reuse of both output slots across workloads.

Same two-round finite CUPTI setup, B3 OPT5→8 us medians:
D64 q1513.735→1532.487, k1383.432→1374.969;
D32 q1022.074→1028.633, k924.507→933.547. This recovers most of OPT7's
regression but is not a stable overall throughput improvement. Raw samples:
`benchmarks/backward_da_pair_timing_sm120a.json`.

NCU `/tmp/dism-backward-da-pair-q.ncu-rep`, main q D64, full set40passes:
B3 about2.04ms/SM27.19%, zero local spilling requests; shared-store requests
return to14,229,504, but conflict count35,425,732 and average about4.2-way
remain close to OPT5. Pair width fixes the request-count regression; swizzle
alone does not remove the lane-ownership conflict pattern. Selected metrics:
`benchmarks/backward_da_pair_ncu_sm120a.json`. Keep the option experimental.

A bounded next output-layout probe could gather adjacent lane pairs into
float4 stores WITHOUT swizzle. For g=lane&3,l=lane/4 and row half h=0/1,
select own accumulator pair r=h+2*(g&1) and exchange the complementary
r=h+2*((g^1)&1) pair with lane^1. Even g packs own/received, odd g packs
received/own, at row8*h+l,column4*(g/2)+8*(g&1). Four lanes then cover a
complete16-float row; each eight-lane STS.128 wavefront covers two full rows
with distinct banks. This needs four scalar shuffles per16x16 tile plus two
vector stores per lane, no extra shared. It is a mapping/design proposal,
not implemented or performance-proven; verify ownership, tail TMA reduction
and sanitizer before adopting it.

### OPT9: float4 output ownership experiment

Implements the preceding gather proposal on OPT5's raw, unswizzled output
slots and TMA map, without OPT6 prefetch. Each row-half exchanges the
complementary low/high accumulator pair with lane^1. Explicit selp chooses
the four output components; TK `move<float4>::sts` issues the aligned store.
No shared allocation, math, TMA reduction or lifetime change. A pure host
mapping test verifies all256 elements occur exactly once and each eight-lane
store wavefront covers all32 banks. GPU equivalence/epoch/codegen plus that
mapping test passed140 cases in finite mode (36.54s final source). Codegen
now additionally requires at least2*4*(D/16) STS.128 sites per B3 instance,
while retaining noCALL/native TMA/register-role checks.

Finite B3 stack bytes, D rows/DV columns32/64/128:
`(0,0,48)/(0,0,32)/(64,112,192)`. Spill recorded, not tuned.
Full mode also140 passed (75.44s). Finite memcheck/racecheck/synccheck each3
passed, zero errors/hazards, on N65/k/D64,N385/q/D32,N257/q/D128 persistent
nodes (D=DV,batch=SM_count+3);2.62s/101.69s/2.27s respectively. Full/finite
resource strings: `benchmarks/backward_da_quad_codegen_sm120a.json`.
Original full WS dimensions/tails strict-reference suite192 passed/14 failed,
41 deselected (17.28s); all14 failure IDs exactly match the frozen baseline,
no new or resolved failures and no tolerance changes.

Two-round finite CUPTI setup against frozen OPT5, B3 us medians:
D64 q1506.984→1544.966, k1376.728→1416.856;
D32 q1022.010→1040.474, k922.347→943.547. B1 remains approximately
unchanged. All16 runs finite, raw samples in
`benchmarks/backward_da_quad_timing_sm120a.json`.

NCU `/tmp/dism-backward-da-quad-q.ncu-rep`, same main q shape/full set and
40passes: B3 duration2.064192ms vs OPT5's2.013280ms, zero local spilling
requests. Shared-store requests drop14,229,504→9,510,912, conflicts
35,589,540→16,590,147, wavefronts59,474,506→40,475,855 (about32% lower).
Executed instructions rise740,490,473→765,394,762 (about3.4%). Thus this
experiment really reduces shared-store work, unlike swizzle-only, but does
not improve complete kernel time. Added gather/selection instructions are a
plausible tradeoff, not an isolated causal proof. Selected counters in
`benchmarks/backward_da_quad_ncu_sm120a.json`.

Keep OPT9 experimental, default remains OPT0. Stop this output-layout branch
for now; resume direction/label/mode specialization and whole-shape candidate
selection rather than optimize bank counters in isolation.

### OPT10: score direction/label/mode specialization (in progress)

Builds on OPT5's input/output layouts, WITHOUT OPT6 prefetch or OPT7–9 output
experiments. Host dispatch selects one of ten score policies: two column/row
LSE directions x (soft,hard32,hard64,mixed32,mixed64). Both WS kernels now
have180 total specialized instances. Key/query metadata use compile-time
label width; soft omits labels, hard omits LSE, column-LSE avoids query-LSE
shuffle. Hard score skips its QK MMA but retains all forward/reverse scan,
dV/dP, G/scalar gradient and existing dA/dB work; hard rows still contribute
rtau gradient. Input/mail synchronization, masks, delta and score arithmetic
association are unchanged. This is not a single-FFMA score reordering.

Legacy options instantiate SPEC=-1 with the old metadata behavior; their
internal CUDA symbol names gain that template parameter, not a Python ABI
change. No default promotion. Existing codegen tests now recognize both
18-instance legacy and180-instance specialized builds.

Finite equivalence/epoch/codegen139 passed (134.50s including compilation).
All180 retain noCALL, four native TMA load sites, one dec40/inc232 pair and
two BAR.SYNC sites. Additional54 int64 cases passed (1.88s), all nine D/DV,
both directions and probabilities0/.37/1 with +/-2**40 labels to detect
truncation. D64/DV64 finite B1 stack bytes by policy0..9:
24,0,0,24,24,24,0,0,32,16; B3 all ten0B. Spill elsewhere is recorded,
not tuned. Full-mode, sanitizer and per-mode performance results follow.

### OPT11: pure-hard scalar-gradient path and checkpoint validation

OPT11 builds on OPT10, without OPT6–9 experiments. For pure-hard B3 only,
retain W/dP/E recomputation, reverse G scan/mail and rtau accumulation, but
omit Gsoft/dA/dB GEMMs, dA TMA stores and dLSE reductions. Host-zeroed dA
and dLSE remain zero; the kernel explicitly writes zero dB. Hard labels are
not differentiated, but rtau and its reverse recurrence are still computed.
Soft/mixed paths are unchanged. Default remains DISM_BWD_OPT=0.

Final OPT11 finite equivalence/epoch/int64/codegen checks:193 passed,
including explicit pure-hard zero-gradient checks. Full expanded WS suite:
414 passed/26 failed; the26 numerical failure IDs match frozen OPT5's
expanded suite exactly (not the smaller suite's14 failures). No tolerance
changes or expected-failure suppression. Failure-set evidence is recorded
in `benchmarks/backward_special_numerics_sm120a.json`.
Whole-extension noCALL and specialized codegen gates pass in both full and
tanh_finite. Selected autograd wiring/training/replay checks:507 passed per
mode,398 deselected per mode; finite bitset checks:62 passed. These selected
checks do not claim that all strict end-to-end precision tests pass.

Main B64/H4/N1024/D64/DV64 finite CUPTI timing, two alternating rounds of
30 samples per configuration: OPT10 to OPT11 pure-hard B3 medians are
1419.110 to556.204 us (q_from_k) and1285.673 to555.517 us (k_from_q).
Mixed B3 is approximately unchanged:1497.463 to1502.999 us and
1367.449 to1357.624 us respectively. Do not generalize the pure-hard gain
to mixed training. D64/DV64 finite B3 has zero stack for all ten policies;
other spills remain recorded and accepted for this experiment.
Raw timings/resources are in `benchmarks/backward_hard_gradient_*_sm120a.json`;
OPT10 versus OPT5 results are in `benchmarks/backward_special_*_sm120a.json`.

### Remaining work

First-query prefetch and output-layout experiments have been measured without
a stable net gain and remain opt-in. Complete all-nine-shape performance
screening, integrated timing and final default selection before declaring
the optimization goal complete. Do not infer a default dispatch policy from
the main-shape finite measurements alone.

Work-distribution diagnostic for the current170-SM main shape: bh-major task
order assigns96–136 query tiles per persistent CTA (mean108.42). A key-major
decode would assign104–120 at the same mean. This is a host count calculation,
not measured speedup; task-order A/B may be useful after prefetch changes.
