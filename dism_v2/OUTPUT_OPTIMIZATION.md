# OUTPUT core optimization audit (2026-09-09)

Active task: apply the validated summary optimization lessons to the forward
output kernel, not just the summary. Baseline commit5645ca2. No changes to the
reference semantics, backward ownership, or external GLX/TK dependencies.

## Requirements / audit

1. Score: remove per-element runtime direction/hard/bounds branches and global
   metadata loads. Precompute scale2/tau2/bias2, cache typed int32/int64 labels,
   specialize row/column LSE and soft/hard/mixed; preserve finite/full modes,
   RNG identity, tail identity, backward checkpoints and BF16 PV conversion.
2. Input: full-width single-transfer K/V TMA including128; use validated RHS
   LDSM mapping for score. Preserve safe unpadded tails and avoid the rejected
   register-only summary store permutation (irrelevant to output writes).
3. Persistent pipeline: independent Q TMA slot, cross-workload Q/K0/V0 prefetch,
   K/V stage count budgeted with Q and communication against actual SM120
   shared limit. No metadata/intermediate shared staging for convenience.
   Don't assume summary's three K slots fit once V and output accumulators exist.
4. Synchronization: WG-level mailbox, coherent arrivals/readers, elected
   producer, deliberate K/V slot release after final read, WG-local exit.
   Do not repeat unsafe early exit or assume the failed K0/K1 peeling solved
   overlap. Separate proven protocol changes from unmeasured overlap claims.
5. Generated code/performance: CALL, spills, LDSM→HMMA mapping, TMA counts,
   register redistribution, source-correlated profiling, same-session launch
   timing against the saved baseline. Investigate output statistics/checkpoint
   global traffic, distinguishing required output from unnecessary staging.
6. Verification: all9 D/DV, both directions, hard endpoints/mixed, labels32/64,
   RNG/bitset replay, unaligned N and cross-workload reuse; full forward oracle,
   boundary/backward recomputation and end-to-end selected regressions. Preserve
   known precision failures. Run three sanitizers plus SASS regression gates.

Implementation sequence: score/metadata + K/V transfer/layout; persistent
Q/K/V and synchronization; full verification/profile and remaining hot spots.
Completion requires evidence for the whole audit, not only the first stage.

## Baseline findings

core_fwd.cu::core<...,true> uses per-element score() with conditional label/LSE
global loads, runtime direction and three-stage arithmetic. Query is copied
global→shared by compute warps, followed by two CTA barriers; shared Q aliases
two K/V stages, precluding cross-workload Q prefetch. K/V128 use multiple TMA
instructions. K uses row-layout load + mma_ABt, unlike summary's validated
load_rhs_tile. Mailboxes have individual32-thread barriers per warp pair.
Final exit is CTA-wide. Causal CTA-level key clipping, scalar-before-pair roll,
online softmax and finite-mode log-affine are already present and must remain.

## Stage1 candidate / resumed

Implemented in core_fwd.cu:90 OUTPUT specializations (D×DV×direction×5 mode/
label combinations); single-FFMA predicated scores with lane-distributed
metadata; native one-transfer K and V maps including128; summary's validated
load_rhs_tile + mma_AB; elected TMA leader. No persistent scheduling or mailbox
protocol change yet, and no claim of numerical or performance acceptance.

Baseline saved at /tmp/dism-output-baseline-5645ca2.so. Initial N1024/D64/DV64,
B64/H4/V512, mixed0.5/q_from_k, tanh_finite+lineinfo output median798.3155us
(20warmups/30CUPTI samples, unlocked clocks); no paired candidate timing yet.

Finite-mode codegen:120 allocation/deallocation sites,30summary+90output,
two TMA sites per output instance, no CALL. Zero-spill gate fails for two
OUTPUT instances: pure-soft q_from_k D64/DV32 and D128/DV128, each STACK8.
No other stack/local sizes were reported nonzero. The first test invocation
had loaded the old39-instance assertion before the test file was updated;
rerunning the current test passes the new counts and fails at LDL/STL as expected.

Initially paused before further implementation under the spill-stop convention. No
spill tuning, full-mode rebuild, numerical regression, backward regression,
sanitizer or candidate timing has been done at this point. User direction is
needed to retain these spills and continue, or authorize investigating them.
Persistent Q/K/V, WG mailbox/exit synchronization and the remainder of the
audit are still required; the overall task is not complete.

The user subsequently authorized continuing OUTPUT work with spill recorded,
without stopping or relaxing correctness. Work has resumed from this candidate.

Stage1 full-LSE core133 + label-width55 + previous exit12:200 passed before
changing the pipeline, including all9 D/DV and rtau-bound score regressions.

## Stage2 persistent candidate (validation in progress)

- Single CTA per SM, grid-stride128-row workloads; independent Q input slot.
- QROWS128 except D128/DV128, which uses64 rows in two phases. In that shape
  producer submits Q0 then K0/V0 before waiting for Q0's slot release to load
  Q1. Across workloads the first Q/K/V transfers are therefore not blocked
  behind the second Q phase. This is protocol/source-level scheduling, not
  measured instruction-level overlap yet.
- K/V slots budgeted from Q + K/V + four mailbox payloads + barriers and
  alignment under63KiB dynamic shared (reserve1KiB static). Slot counts by
  D rows / DV columns32,64,128: D32=(3,3,2), D64=(3,2,1), D128=(1,1,1).
  Host static_assert checks actual struct size; all full K/V loads are single
  TMA including128. Input and communication are the only shared writes.
- Mail ready/free128 per WG, cumulative tile phases across tasks; K/V free256
  after PV; elected producer ready1; WG-local128-thread exit, CTA init only.
- Existing scalar-before-pair scan, online softmax, required W boundaries,
  output layout and normalization semantics remain intact.

### Rejected cooperative tail producer

Initial attempt used ready32 and warp-cooperative tails for nonaligned N, to
avoid summary's serial-tail slowdown. N65 passed, but persistent N257/D32/DV32
stalled reproducibly; aligned N256 and a small nonpersistent-grid N257 passed.
Temporary timed mbarrier waits reported consumers waiting K-ready slot1 while
producer waited free slot1 (e.g.shared offsets33816/33840 for D32/DV32).
Adding a per-key cooperative syncwarp removed that particular timeout in one
run but produced1.5% output replay mismatches; racecheck+timed diagnostics then
hit a timeout, with no reported shared hazards (not sanitizer acceptance).
Exact cause is not established. It was rejected, not hidden as a known failure.

Selected candidate uses the elected thread for all input loads and scalar
safe tails, matching the validated summary protocol. The isolated N257/D32/DV32
replay now passes. All-shape dispatch has been restored, and all temporary
printf/trap/timed-wait instrumentation removed. Full/tanh_finite all-shape
tests are running; backward/sanitizer/performance/NCU audits remain required.

All-shape finite candidate:36 exit/replay cases plus codegen passed (37 total),
including the two-phase Q shape. Full suite and backward/end-to-end selected
regressions have been launched; no performance acceptance claimed yet.

## Stage2 verification / initial timing

- Full forward/core133 + labels55 + row-bitset62 + exit36 + codegen1:287 passed.
- Full transposed-boundary/G/backward suites:257 passed and1 failure in the
  existing whole-backward-extension zero-spill codegen assertion. That test
  inspects only backward._extension(), whose sources were not modified here;
  its conflict with accepted backward spills is documented in AB.md. The
  ordinary failure remains visible, no xfail/skip or tolerance change.
- Selected autograd wiring/training/replay:507 passed in full and507 passed
  in tanh_finite; existing AccumulateGrad stream warning remains visible.
- Finite labels/bitset117 and exit/codegen37 passed.
- Same-session baseline versus persistent OUTPUT, D=DV64/B64/H4/N1024/V512,
  mixed0.5, int32, tanh_finite+lineinfo, 20warmups/30CUPTI samples, three rounds,
  no concurrent GPU tests, unlocked clocks:
  q_from_k baseline797.6115us ->458.157us (1.74x),
  k_from_q baseline837.755us ->463.5495us (1.81x).
  Raw samples: benchmarks/output_persistent_sm120a.json.
  These measure output launch time in the normal forward stream, not full
  training throughput or proven instruction-level overlap.

## Stage3 final output store packing

SASS inspection found two STG.E.U16 instructions per adjacent output pair.
The output address is4-byte aligned (even j, DV multiple32), so store a
BF16x2 converted with the same RN rounding through one32-bit store. This
uses no shared scratch/shuffle and preserves FP32 normalization arithmetic.
The codegen guard rejects remaining STG.E.U16 in OUTPUT instances.
Benchmark now exposes independent --dv for all9 shape comparisons.

Final packed-store regression: full core133 + exit36 + codegen1 =170 passed;
finite labels55 + bitset62 + exit36 + codegen1 =154 passed. No tolerances changed.
Final full-mode exit/reuse suite under memcheck, racecheck and synccheck:
36 passed each, zero errors/hazards (racecheck293.09s). These cover all9 D/DV,
both directions, N65/257 and multiple workloads per CTA, including two-phase Q.

Final resource/SASS records, with binary SHA256 for each mode and baseline:
benchmarks/output_codegen_sm120a.json. All90 OUTPUT instances per mode have
native Q/K/V TMA (3 sites, or4 for split-Q D128/DV128), two BAR.SYNC sites
(initial CTA plus independent WG exit), no CALL and no STG.E.U16. Summary's
30 instances retain their existing codegen/zero-spill gates. Role budgets
remain producer40/consumer232; binary REG168 is not the consumer role budget.
Only D128/DV128 spills remain after packing:

| Mode | Direction / specialization | Stack bytes |
|---|---|---:|
| full | q_from_k mixed, int32/int64 | 8 each |
| full | q_from_k soft | 40 |
| full | k_from_q soft | 32 |
| tanh | q_from_k soft | 32 |
| tanh | k_from_q soft | 40 |
| tanh_finite | either direction mixed, int32/int64 | 8 each |
| tanh_finite | q_from_k soft | 88 |
| tanh_finite | k_from_q soft | 32 |

Other OUTPUT specializations have zero stack/local allocation. These spills
are recorded under the user's continue-with-spill authorization, not hidden
by modifying numerical tests.

Final full and finite autograd wiring/training/replay each reran507 passed;
full transposed-boundary/G/backward numerical checks reran257 passed (the
previously observed, unchanged whole-backward zero-spill test was deselected
on this rerun; its ordinary failure is documented above).

Additional tanh core/exit/codegen run:120 failed,50 passed. To distinguish
pre-existing approximation error from this patch, rebuilt baseline5645ca2's
core_fwd.cu in a separate temporary extension directory with the same tanh
flags and unchanged headers. Strict core oracle alone is120 failed/13 passed
for BOTH baseline and final, with identical failure sets. For example mixed
q_from_k D64/DV64 has exactly the same reported normalizer max absolute error
0.000591278076171875 versus tolerance2e-5. Exit36 and codegen pass on the final
tanh version. This is not a claim that tanh passes the strict full-LSE oracle;
no tests/tolerances were changed. Failure messages and comparison are retained
in benchmarks/output_tanh_oracle_regression.json.

## Final launch timing

RTX5090, driver590.48.01, CUDA13.1, BF16; B64/H4/V512, mixed0.5, int32 labels,
tanh_finite and lineinfo. Normal forward stream with saved backward boundaries,
20warmups and30CUPTI samples; baseline and final loaded in separate processes,
alternating order, no other GPU workload and no clock/cache locking. D64/DV64
N1024 uses three rounds (median of medians); other rows are single-round spot
checks. These are OUTPUT launch times, not training throughput.

| D / DV | q_from_k baseline → final (us) | k_from_q baseline → final (us) |
|---|---:|---:|
| 32 / 32 | 644.861 → 391.374 | 723.596 → 393.421 |
| 32 / 64 | 692.301 → 447.997 | 748.652 → 445.757 |
| 32 / 128 | 767.308 → 462.622 | 792.732 → 495.837 |
| 64 / 32 | 745.771 → 433.661 | 825.275 → 420.734 |
| 64 / 64 | 797.707 → 450.494 (1.77x) | 837.867 → 455.806 (1.84x) |
| 64 / 128 | 861.531 → 604.572 | 883.210 → 606.572 |
| 128 / 32 | 952.680 → 586.892 | 979.866 → 592.364 |
| 128 / 64 | 990.811 → 622.972 | 1014.395 → 623.085 |
| 128 / 128 | 1086.042 → 752.220 | 1342.569 → 761.372 |

All rows above use N1024. D64/DV64 tail spot checks:
N65 q39.840→64.976us / k39.807→65.199us (regression);
N257 q124.223→113.663us / k129.216→114.767us.
The elected producer's scalar safe tail has a substantial short-N cost, as
in accepted summary. Do not describe this as an all-shape speedup. Raw final
samples: benchmarks/output_final_sm120a.json; earlier pre-packed measurements
remain separately recorded in output_persistent_sm120a.json.

Reproduce the primary measurement (run baseline and final in separate processes):

```bash
DISM_TILE_LSE=tanh_finite DISM_LINEINFO=1 /home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.benchmark_persistent_summary --kernel output --d 64 --dv 64 --direction q_from_k
# Add --baseline-binary /tmp/dism-output-baseline-5645ca2.so for the saved baseline.
```

## Final source-correlated NCU

Report: /tmp/dism-output-persistent-lineinfo-q.ncu-rep (imported CUDA source,
40passes, full set, skip10 matching launches/collect1). Same primary shape,
q_from_k. Single-profile duration467.58us, SM37.32%, tensor37.08%, DRAM28.78%;
unlocked clocks/cache, so use CUPTI above for comparative throughput. Dynamic
shared53376B plus1024B driver allocation,170 CTAs on170 SMs. No local spilling
requests for this specialization. Do not act on NCU's generic recommendation
to increase occupancy: one CTA/SM remains the explicitly requested design.

Remaining excessive global sectors2,956,544 are source-attributed as follows:

- Horizontal W checkpoint stores:884,736 at each of core_fwd.cu lines320/321,
  total1,769,472. Four active lanes write separate8-column portions; these are
  required boundary outputs but their warp-store layout remains inefficient.
- Packed BF16 O stores:1,048,576, attributed through cuda_bf16.hpp:254; confirmed
  the associated SASS consists of the16 final32-bit STG instructions, not input
  loads. Packing removes the old32 half-width stores but does not fully
  coalesce the accumulator's per-row16-byte lane-group spans.
- Incoming checkpoint boundary loads:138,496 at core_fwd.cu:271. The shifted
  GLX HState layout accounts for the remaining load excess.

Thus about95.3% of remaining excessive sectors are required result/boundary
stores, not repeated score metadata loads. Average global-load sector usage
29.5/32B versus global-store17.4/32B. Current D64 mixed int32 SASS has no
LDG.E.64 metadata instructions (old OUTPUT50), no CALL,3 native TMA sites,
no half-width output stores. No extra shared layout buffer has been introduced;
larger store-layout changes need a separate measured tradeoff, especially with
the existing shared/stage budget and summary's rejected shuffle-store trial.

PC sampling records41,426 samples:barrier10,368, math-pipe6,852, wait6,192,
long-scoreboard2,988. Of barrier samples10,349 are at the final WG exit, which
also holds the otherwise inactive producer threads; this aggregate is NOT
evidence that the compute WGs spend25% of their work stalled on mailboxes.
Long-scoreboard source attribution includes producer slot reuse and boundary/
row-bias consumers. Exact WG-by-WG HMMA/scan overlap has not been measured;
source scheduling permits it, but these aggregate counters cannot establish
optimal overlap. Source rows and selected raw metrics are retained in
benchmarks/output_ncu_source_sm120a.json.

Profile command (requires local GPU counter permission):

```bash
DISM_TILE_LSE=tanh_finite DISM_LINEINFO=1 /usr/local/cuda-13.1/bin/ncu --set full --kernel-name-base mangled --kernel-name 'regex:.*coreILi64ELi64ELb1ELb0ELi2EiE.*' --launch-skip 10 --launch-count 1 --cache-control none --clock-control none --import-source yes --export /tmp/dism-output-persistent-lineinfo-q /home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.profile_forward_summary
```

## Completion audit / limits

The requested summary-style OUTPUT pass is implemented and verified: score
specialization/cache/FFMA, native full-width input mapping, persistent Q/K/V
prefetch and budgeted stages, WG communication/exit, output pair packing,
SASS/resource gates, numerical/replay/backward checks, sanitizers and controlled
baseline timing/source-correlated profile. Accepted summary code and external
GLX/TK files are unchanged. Only specified Q/K/V asynchronous inputs and
required WG boundary mail write shared; all score/scan/softmax intermediates
stay in registers. No dense logM/W/P is materialized; existing optional
vertical16/horizontal64 boundaries and output normalizers retain their layout.

Known precision failures (including tanh strict oracle and backward's accepted
spill-codegen conflict) remain visible. D128/DV128 spill and the serial-tail
regression are retained explicitly. Store-sector efficiency, causal workload
load balance and measured WG overlap remain future performance opportunities,
not claims of optimal throughput. No varlen/sm90 work or new math approximation
was undertaken in this pass. Changes have not been committed.
