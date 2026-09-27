# Q input storage reuse experiment (2026-09-09)

Scope:D64/DV64 only; default kv now selects the conservative Q/KV union
by the user's subsequent mixed-workload preference. Explicit none retains the
committed independent-Q persistent kernel. User requested both a conservative Q/KV union and a split
K/V producer that can prefetch the next workload after K consumption.
No math, checkpoint layout, RNG identity, external GLX/TK or summary changes.

## Variants

Set DISM_OUTPUT_Q_ALIAS before first extension use, in a separate process:

- none:independent Q, two KV stages (baseline82f9305).
- kv:three stages; Q128x64 (16KiB) overlaps KV2 (16KiB), with KV0/KV1
  independent. Producer first submits Q, KV0 and KV1; before writing KV2 it
  waits for all256 consumers' Q shared-to-register reads via qfree. A256-count
  done barrier after each consumer's final PV prevents next-workload input
  prefetch until all old KV readers have completed.
- k:three SoA K slots and three separate V slots. Q128x64 overlaps K1/K2
  (two8KiB slots), so startup can submit K0/V0 but cannot submit K1 before
  qfree. K has ready/kfree; V has vready/free. Each consumer releases K right
  after WG-local completion of shared K reads, before scan; the last K tile
  also arrives on done. The source places this synchronization after score
  MMA, but ptxas may move register-only MMA work across it.
  Next Q can therefore load while old score/scan/PV/output is still pending.
  V reuse remains protected until its PV reader is finished. Only input
  storage is overlaid; mailboxes and all barriers are outside the union.

The split candidate also uses WG-local named barriers4/5 before K release,
to complete/cohere the entire group's K reads independently of MMA scheduling.
These are separate from the existing WG exit barriers1/2/3 and CTA init0.

Both experiments reset physical slot order to0 at each workload. Per-slot
parity and ever-used bits carry across workloads in registers, rather than
using a cumulative-tile/STAGES phase formula that would mismatch this reset.
This also handles workloads with fewer than3 key tiles. Mailbox ready/free
use those same per-slot phases. No whole-CTA workload barrier or per-warp
pair barrier is introduced; WG mailbox128 and independent WG exit remain.

Only specified asynchronous inputs and existing inter-WG mail write shared.
The additional done barrier defines cross-workload storage ownership, not
an intermediate-layout convenience. Counts:baseline10, kv15, k21 mbarriers;
these include Q and mailbox barriers. No single-lane WG release experiment:
all256 consumers arrive directly, preserving the validated reader accounting.

Full Q is intentionally retained for this first SoA probe. Splitting Q into
two64-row loads could overlap only K2 and admit startup K1, but introduces
another Q phase/transfer and is not part of this comparison.

## Validation plan / progress

- Full original core oracle, both directions/hard endpoints/mixed and all9
  dimensions; D64/DV64 changes, other dimensions exercise unchanged fallback.
- New q_alias replay covers N1/65/129/257/385/1024 with more workloads than
  SMs, checking O/normalizers and both backward boundary arrays exactly.
- Full/finite codegen:CALL/native TMA/register roles, record spill, no stop.
- Finite labels/bitset and selected autograd regression.
- Three sanitizers on D64/DV64 multi-workload/tail coverage.
- Separate-process baseline/kv/k launch timing, same stream/configuration;
  keep default none until evidence supports choosing a candidate.

Initial full-mode exit/replay:D64/DV64 four cases passed for each variant.
Both full core133 + new replay12 + codegen1:146 passed each.
Both finite labels55 + bitset62 + replay12 + codegen1:130 passed each.
All10 D64/DV64 specializations in both full/finite variants have zero stack,
zero local memory and no CALL. Native3 TMA sites and inc232/dec40 codegen
gates pass. Resource/SASS data:benchmarks/output_q_alias_codegen_sm120a.json.
The conservative variant passed memcheck/racecheck/synccheck:10 new replay
cases each (N1/65/129/257/385, both directions), zero errors/hazards; racecheck
166.39s. Both full/finite selected autograd suites passed507 each for each
initial variant; none/default full core/codegen passed134.

The initial split variant released K immediately after LDSM. Despite its
ordinary regressions passing, memcheck execution found a replay mismatch:
k_from_q/N129,64 O elements in row(batch29,row128), max difference0.1640625.
Memcheck reported no access errors, but the numerical failure makes that
candidate unacceptable. No tolerance changed. Exact cause is not established.
The first revision moved the source release after score MMA. It passed146
full,130 finite,257 backward-recompute and507 selected autograd checks per
full/finite, and all three sanitizers10 cases each (racecheck169.72s). However,
SASS showed the first K-free arrive after only26 of32 score HMMAs, immediately
following a trailing LDSM. Source-level MMA position alone therefore did not
establish the intended release ordering. The final candidate adds explicit
TK WG sync before K-free/done; source statements are not used as a substitute
for the generated-code ordering check.

Final WG-synchronized split:full146 and finite130 passed; full backward
recompute257 passed (the unchanged, previously known whole-backward zero-spill
assertion excluded); selected autograd507 passed in each of full/finite.
Final memcheck/racecheck/synccheck10 cases each passed with zero errors/hazards
(racecheck179.27s), plus three additional N129 memcheck runs of2 cases each,
all passed. No debug instrumentation or relaxed numeric tolerance is present.
The last conservative finite rebuild also passed replay/codegen13 cases.
The strengthened codegen guard also passes for both full/finite:the last
128-thread WG sync must precede K release, with no intervening LDSM. Some
soft specializations put that barrier before the first HMMA; that is valid
because the ownership requirement concerns completed shared reads, not
completion of register-only MMA arithmetic. No extra shuffle/shared staging
was added to force a particular arithmetic instruction schedule.

## Final timing / decision

B64/H4/N1024/D64/DV64/V512, actual CUDA embedding inputs, tau3/scale1,
int32 labels, tanh_finite, lineinfo. Saved baseline82f9305 binary is loaded in
a separate process; each worker uses20 warmups and30 CUPTI samples in the
ordinary forward stream, with backward boundaries enabled. No concurrent GPU
test, no per-launch sync or graphs, unlocked clocks/cache. Only OUTPUT GPU
duration is counted. Mixed rows use three rounds, reversing variant order
between directions; soft and short-N rows are single-round spot checks.
No timing outliers are removed, all workers report finite output.

| N / probability / direction | Baseline us | Q/KV union us | Q/K SoA + WG sync us |
|---|---:|---:|---:|
| 1024 / 0.5 / q_from_k |451.556|417.373|462.126|
| 1024 / 0.5 / k_from_q |456.431|447.421|456.462|
| 1024 / 0 / q_from_k |468.493|482.893|466.589|
| 1024 / 0 / k_from_q |467.373|479.517|465.165|
| 65 / 0.5 / q_from_k |65.024|72.063|73.583|
| 257 / 0.5 / q_from_k |113.679|117.487|117.615|

Conservative mixed throughput improves8.19% in q direction and2.01% in k
direction. But its soft spot checks regress about2.5–3.0% in throughput, and
short-N performance regresses. The SoA variant has no consistent net gain;
its q mixed time increases2.34%, while k mixed is essentially unchanged.
The initial recommendation was to retain default none. The user subsequently
selected conservative kv as default because mixed is the main workload;
k remains opt-in. No automatic shape/probability dispatch threshold is inferred.
This tests the combined layout/stage/synchronization designs, not an isolated
causal attribution to prefetch timing alone. Other D/DV combinations still
use the original implementation; no claim of their alias-layout validation.

Raw samples:benchmarks/output_q_alias_sm120a.json. Reproduce:

```bash
DISM_TILE_LSE=tanh_finite DISM_LINEINFO=1 /home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.benchmark_q_alias --baseline-binary /tmp/dism-output-qalias-baseline-82f9305.so --output dism_v2/benchmarks/output_q_alias_sm120a.json
DISM_OUTPUT_Q_ALIAS=kv DISM_TILE_LSE=full DISM_LINEINFO=1 /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_core.py tests/test_dism_v2_q_alias.py tests/test_dism_v2_codegen.py
# Repeat with DISM_OUTPUT_Q_ALIAS=k for the SoA candidate.
```

## NCU / storage budget

Fresh full-set profiles,40passes, imported source, mixed q direction:

| Variant | Dynamic shared B | Driver shared B | Duration us | SM / tensor % |
|---|---:|---:|---:|---:|
| Q/KV union |55424|1024|441.568|42.31 / 42.31|
| Q/K SoA + WG sync |55552|1024|476.032|36.95 / 36.95|

Baseline's two stages plus independent Q use53376B dynamic. Both experiments
use slightly more shared than that baseline but fit a THIRD stage, whereas
three independent KV stages plus an independent Q would exceed64KiB.
Input storage is48KiB in either experiment; three mailbox payload sets add
6KiB, with barriers/alignment accounting for the remainder.

Reports:/tmp/dism-output-qalias-kv-lineinfo-q.ncu-rep and
/tmp/dism-output-qalias-k-lineinfo-q.ncu-rep. No clock/cache locking; profile
times are not substituted for the three-round CUPTI comparison.
Selected metrics/source attribution:benchmarks/output_q_alias_ncu_sm120a.json.
In the SoA report1,417 of42,710 PC samples are attributed to the K-release WG
barrier (source correlation at the following arrive);10,566 barrier samples
are at final WG exit, which includes inactive producer threads. The added
sync has measurable waiting, but this does not establish it as the sole cause
of the performance difference or measure exact WG-by-WG prefetch overlap.

This requested two-variant validation is complete. The unsuccessful early-LDSM
release is not selectable; only the final WG-synchronized k variant remains.
The general limitation that tanh approximations have existing strict-oracle
errors is unchanged. No new math approximation or tolerance change was made.
Changes are uncommitted.
