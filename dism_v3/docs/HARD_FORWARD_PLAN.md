# Hard forward with separate summary and readout layouts

Status: the separate-layout optimization below is deferred. A correctness-first
standalone hard forward is now implemented (2026-10-14): integer summary,
chunk passing, geometric-weight table and fused readout. It uses the existing
diagonal scan placement rather than thread-owned chains. See the contracts
and validation below. Production training/autograd dispatch remains soft-only;
the standalone hard forward is not yet wired to event-stream backward.

In the longer-term design selected on 2026-10-14, summary
assigns each complete diagonal segment inside a tile to one thread. Fused
forward/readout retains the current scan-buffer placement, replacing soft
log-domain affine arithmetic with integer state. The two stages share an
integer boundary contract, not a per-position register layout. This plan
does not require changes to production dispatch or numerical tolerances.

Subsequent CUDA work removes token-level hard handling and makes the core a
soft-only contribution with sequence-hard skips. Its zero hard outputs are
still placeholders: callers must explicitly use the standalone hard entry
below to compute the hard contribution.
See [the soft-only split and measurements](SOFT_ONLY_CUDA.md).

The target is sequence-level hard training on sm120, initially short sequences
around N2048 with R32 and DV64. Keep CUDA O(N^2) forward fused with readout and
the [asynchronous CPU planner plus event-stream backward](NATIVE_TRAINING_PLANNER.md).
Forward must not wait for CPU planning. Long-sequence training forward and
per-query tau are out of scope. Soft inputs retain their existing implementation;
the current CUDA API rejects token-level hard masks. Hard sampling remains per packed training row/head,
not per document. Document boundaries still reset attention and recurrence.

## Implemented correctness-first summary

`flash_dism.summary.summarize_hard(idx_q, idx_k, hard, layout=None, ctas=0)`
is an explicit standalone entry, not production attention dispatch. Labels are
contiguous CUDA int32 `[B,H,N]`; flags are contiguous bool `[B,H]` for fixed
length, or `[training_rows,H]` for packed batch-one input. Packed documents
cannot cross the equal-length original training rows. N and document boundaries
remain 256-aligned. Empty documents are supported inside a nonempty pack.
Neither tau nor direction is needed to compute hard match lengths.

The implementation retains the existing 384-thread layout: two consumer WGs,
one active producer warp in the third WG, interleaved consumer query blocks,
32x64 warp tiles, three K slots, persistent fixed-length scheduling, and the
same per-reader release/phase protocol. Packed documents use immutable cached
task records and the same persistent body. Soft tasks are skipped by all roles
before touching pipeline phases. Inactive final consumer warps still retire
all slots: only the 32-row block containing the final token is omitted.

Q256 and K64 labels are loaded with coalesced 16-byte `cp.async` operations.
Q stays in dedicated shared storage across key tiles; unlike soft MMA input,
it must not alias the K ring. The final slot drain protects Q as well as K
before the next task. All repeated label comparisons read this shared input,
not global memory. The first Q/K prefetch explicitly commits and waits before
publication; steady-state K loads retain asynchronous mbarrier completion.

`make_hard_summary_scan` constructs `AltLayoutSplitScanBuffer<32,64>` directly:
physical query row is `physical_row(r,c,lane>>2)`, and key column is
`c+(2*(lane&3)+half)*COL_BLOCKS`. A match contributes -1; a mismatch contributes
its document-local key coordinate. New int32 packed payloads use integer max
with the existing `reduce_forward`, left/right state exchange and bottom-edge
ownership. No QK MMA, float affine arithmetic, or input roll is performed.
The common scan implementation itself is unchanged.

Output is a separate int32 R buffer: `[B,H,(N-1)//32,N]` for fixed input;
document-major contiguous `[H,(n-1)//32,n]` regions for packed input, matching
the existing forward checkpoint offsets. R[c,j] is the last mismatch on the
diagonal segment ending at `(32*c+31,j)` within that 32-row strip, or -1 if
there is none. Only causal hard entries are computed; all other entries are
initialized to -1. Left state propagates across K tiles, while top state is
the identity because query strips are independent summaries.

The boundary-passing operation is
`prefix[c,j] = max(R[c,j], prefix[c-1,j-32])` when both predecessor indices
exist, otherwise `prefix[c,j] = R[c,j]`. Length is then `L=j-prefix[c,j]` on
causal entries. Do not pass R to the soft affine chunk kernel. GPU boundary
passing and geometric-weight readout are available through separate hard
entries below; the production training dispatcher is unchanged.

### Validation on RTX 5090

- All 8 production configurations build; device translation units remain
  Torch-free. SASS has no CALL, HMMA, TANH/EX2/LG2; integer scheduling division
  may use MUFU.RCP. Each instance reports 168 allocated registers, zero stack
  and zero spill loads/stores. Existing register-allocation policy is retained.
- 29 new tests plus selected soft/core/frontend/codegen regressions:
  **173 passed**. Integer summaries agree exactly with an independent row-major
  CPU recurrence. CPU-composed boundaries also exactly recover full-document
  match lengths. Cases cover random/extreme/equal/disjoint/periodic labels,
  mismatches around 32/64/256 boundaries, N256/512/1024/2048, packed reset boundaries,
  empty documents, mixed row/head flags, persistent CTA counts 1/2/default,
  and fixed CUDA Graph replay with changing flags. No float tolerance changed.
- Focused fixed/packed persistent-handoff tests pass memcheck and synccheck
  with zero errors, and racecheck with zero hazards. The initial asynchronous
  prefetch version reported 24 racecheck groups and had the known
  `ARRIVES.LDGSTSBAR [UR+0x8]` form. Explicit prefetch completion removed those
  reports; the final version does not rely on the suspected-tool exception.
  These are focused synchronization tests, not a proof for every shape.

Reproduce with `tests/test_hard_summary.py`; ignored build/test/sanitizer logs
are under `build/hard_summary_*20261014*`. Timing helper is
`tools/bench_hard_summary.py --profile`, using B16/H6/N2048, vocabulary512
random labels and packed ragged documents of 256/512/256/1024 per training row.
Timings include int32 output initialization but exclude boundary passing,
readout, CPU planning and transfers. Profiled device durations are attribution,
not an uninstrumented full-forward speedup.

Initial warm timings (10 warmups, 7 samples of 20 calls, current default CTA
count; eager CUDA events include allocation-related launch gaps):

| Layout / hard fraction | Summary + initialization, ms | Profiled summary kernel, ms |
| --- | ---: | ---: |
| Fixed / 100% | 0.0607 | 0.0483 |
| Fixed / 50% | 0.0421 | 0.0279 |
| Ragged / 100% | 0.0329 | 0.0272 |
| Ragged / 50% | 0.0265 | 0.0210 |

Output allocation is 49,545,216 bytes fixed and 16,515,072 bytes ragged,
independent of hard fraction. No task compaction or output-capacity reduction
is claimed. The hard integer summary is dimension-independent; the standalone
entry dispatches the first compiled backend, currently R16/D32/DV32, with the
same code as the other seven configurations.

### Integer chunk passing

`flash_dism.summary.hard_chunk_scan(summary, layout=None, heads=0)` returns a
new inclusive R boundary tensor without modifying its input. Fixed input is
`[B,H,C,N]`; packed input is flat and requires the matching layout and head
count. It uses the same document-major checkpoint offsets as hard summary.

Each thread owns a checkpoint diagonal: its key coordinate advances by 32
per checkpoint. It starts with R=-1 and executes `R=max(R,local_summary)`.
The implementation mirrors the soft chunk kernel's 128-thread CTA, four-stage
warp-local asynchronous shared input ring and retirement sync. Only one int32
input/state is needed instead of two float affine components. No intermediate
recurrence state is shared, no atomic is used, and there is no cross-CTA
communication. Noncausal outputs are explicitly written as -1 without reading
their summary inputs; the complete output is written, so no prefill is needed.
Packed documents reset state independently, including repeated empty-document
boundaries. Soft heads carry identity summaries and remain -1; this prototype
does not compact away their chunk work.

All eight configurations report 40 registers, 2 KiB shared input storage
(cubin resource usage reports 3 KiB including reserved storage), zero
stack/spills and no CALL. Tests cover arbitrary summaries with poisoned
noncausal entries, N256 through N4096, input preservation, packed documents,
full label-to-boundary agreement with scalar L recurrence, and fixed CUDA
Graph replay across changing hard flags. Run `tests/test_hard_chunk.py` and
the existing summary tests together. The benchmark helper now also reports
chunk-only and summary+chunk timings; neither includes hard readout.

The combined regression campaign passes **200 tests** (56 new summary/chunk
tests plus the selected existing regressions). Focused fixed/packed two-stage
tests pass memcheck, synccheck and racecheck with zero errors/hazards. Logs are
`build/hard_chunk_regression_20261014.log` and
`build/hard_chunk_{memcheck,synccheck,racecheck}_20261014.log`.

With the same B16/H6/N2048 benchmark and 100% hard rows, chunk alone takes
0.0199 ms fixed / 0.0138 ms ragged in eager event windows; profiled device
durations are 0.0201 / 0.0077 ms respectively. The separately measured
summary+chunk event totals are 0.0897 / 0.0406 ms, including summary output
initialization and launch gaps. Stage measurements are not additive because
allocation, cache reuse and launch gaps differ. Chunk does not skip identity
soft heads, so 50%-hard chunk times remain essentially unchanged. Results are
in ignored `build/hard_chunk_bench_20261014.json`; this is not full-forward or
training-step timing.

## Complete standalone hard forward

`flash_dism.hard_forward.forward_hard(sq, sk, v, idx_q, idx_k, hard, tau,
layout=None, ctas=0, save_state=False)` composes summary, chunk passing, a fresh
log-weight table and fused readout. It returns BF16 `[B,N,H,DV]` output and
FP32 `[B,H,N]` log2 normalizer. With `save_state=True` it additionally returns
the integer summary/boundary buffers and the invocation's log-weight table.
It is an explicit no-grad entry, not a training autograd function. No embedding
Q/K vectors or LSE inputs are needed, and direction is irrelevant. Inputs
remain contiguous BF16 vectors, int32 BHN labels and sequence-level bool flags.
Soft rows are skipped with zero output and normalizer, permitting explicit
composition with the existing soft contribution. No automatic mixed dispatch,
backward registration, vertical soft checkpoints or RNG changes are introduced.
R16/32 and DV32/64 are supported; embedding dimension D does not affect this
kernel and native dispatch selects the first compiled matching R/DV instance.

The readout retains the existing 16x64 scan placement, two consumer WGs,
128-query-row task size, two-slot SK/V TMA ring and paired-warp mailbox. Q and
K labels use shared async input buffers; Q labels remain live throughout each
task. Direct rolled construction and integer max scan omit matching QK MMA,
LSE arithmetic and input roll. The first WG loads the preceding 32-row integer
boundary; the second receives the first WG's 16-row bottom edge through an
integer mailbox. Scalar unroll reuses the original permutation before length
lookup. Noncausal coordinates are excluded before indexing the table.

The table stores `log2(sum(exp(s*tau), s=1..L))`, using the stable geometric
formula with `expm1f`, separate positive/negative tau branches, and `log2(L)`
at tau zero. Entry zero uses the existing -1e6 zero-weight sentinel. Each
invocation recomputes the table; it is not cached across tau updates. The
readout reuses the existing signed SQ/SK MMA, BF16 weight conversion, FP32
accumulation and online normalization with unit fallback. Only the argument
types of the common output-storage helper were generalized; its arithmetic,
layout and synchronization are unchanged for the soft kernels.

Packed immutable task records have explicit zero-initialized alignment fields.
An initial profile exposed repeated metadata uploads when implicit padding
varied; explicit initialization removes that cache-identity instability.
The steady-state cache test checks for no H2D copies on repeated identical
packs. This does not eliminate the first-use metadata upload or authorize
claims about asynchronous planner readiness.

Validation uses `tests/test_hard_forward.py` and the maintained mathematical
reference, with unchanged output tolerances (`atol=.008`, `rtol=.025`, relative
norm error below .01). Separate normalizer checks use a full scalar log-domain
recurrence, not the LUT formula. Tests cover negative/zero/near-zero/positive
tau, long equal-label runs, disjoint and periodic labels, signed/zero readout,
all-soft skips with NaN operands, causal independence, packed/fixed agreement,
document isolation, repeated mail/ring phases, all R/DV configurations, and
fixed CUDA Graph replay while tau changes. Packed CUDA Graph capture is not
claimed by this entry. The focused fixed/packed full-chain sanitizer campaign
passes memcheck/synccheck/racecheck with zero errors/hazards, without relying
on the ARRIVES suspected-tool exception.

All eight compiled variants have no CALL and no spills: readout reports 168
allocated registers, the table kernel 18, both with zero stack. Readout still
uses HMMA for signed similarity and weighted V, as intended. Torch is absent
from the device dependencies. FP32-output diagnostics are not exposed by this
standalone hard entry; output validation uses BF16 production precision.

Use `tools/bench_hard_forward.py --profile` for the B16/H6/N2048/R32/DV64
baseline, tau .5 and all-hard flags. Eager event totals include output
initialization, the fresh table, allocation/launch gaps and all four stages;
they exclude embedding, backward, planner, optimizer and full-model work.
Packed measurements use 256/512/256/1024 documents per original training row.
Profiled per-kernel durations are instrumented attribution, not the primary
latency numbers. Generated results/logs remain ignored under `build/`.

Final 2026-10-14 validation: **404 passed, 41 skipped** in the selected hard,
soft, fixed/varlen and codegen regression campaign. Skips are existing optional
FP32/probe instances, not hard numerical failures. Final full-chain focused
memcheck/synccheck/racecheck logs all report zero errors/hazards. Evidence is
`build/hard_forward_regression_final_20261014.log` and
`build/hard_forward_{memcheck,synccheck,racecheck}_final_20261014.log`.

Final warm event medians, milliseconds (10 warmups, 7 samples of 20 calls):

| Layout / labels | Readout including initialization | Complete hard forward |
| --- | ---: | ---: |
| Fixed / random | 0.3827 | 0.4780 |
| Fixed / identical | 0.3961 | 0.4813 |
| Ragged / random | 0.1862 | 0.2371 |
| Ragged / identical | 0.1949 | 0.2444 |

The final profile contains no memcpy events in any of these warmed cases.
First-use packed metadata uploads still exist. Stage times are measured
independently and need not sum to the full invocation. Exact event samples,
GPU status and profiled kernel attribution are in
`build/hard_forward_bench_final_20261014.json`. No training-step or
old-implementation speedup is inferred from these standalone measurements.

## Integer recurrence and weight reconstruction

Within a document, let i and j denote query and key positions. For a pure hard
chain, the consecutive match length is exactly

```text
L(i,j) = labels_q[i] == labels_k[j] ? L(i-1,j-1) + 1 : 0
```

The predecessor outside the document has length zero. A tile boundary is not
a reset: its entry length must come from the preceding segment. The recurrence
is independent of direction in the pure hard case; do not alter the model's
direction configuration or its soft-path behavior.

For the existing head-shared tau, the physical weight is
`W(L) = sum(exp(s*tau), s=1..L)`, not `exp(L*tau)`. The proposed lookup stores
`log2(W(L))` for lengths zero through the maximum document length, with entry
zero equal to negative infinity. At tau zero, use `log2(L)` for positive L.
The stable geometric-sum formula in
`python/flash_dism/inference/csrc/prefill.hpp` is the starting point for table
construction, with natural-log to log2 conversion exactly once. Handle small
positive/negative tau explicitly and validate the FP32 table numerically.
The table belongs to the invocation's tau; do not cache across tau updates.

Retain the existing stable online normalization, unit-weight zero-value
fallback, signed SQ/SK numerator, BF16 weights and FP32 accumulation. Tensor
Cores disappear from hard label matching, not from SQ*SK and weighted-V readout.
The geometric formulation need not be bit-identical to the existing approximate
softplus implementation; compare against the maintained oracle without changing
its tolerances, and use exact arithmetic only as an additional diagnostic.

## Summary with thread owned diagonal segments

Summary only needs segment summaries and exit boundaries, not all intermediate
weights. Assign each complete tile-local diagonal segment to one thread, with
one integer state per active segment and no materialized per-position outputs.
Start from the existing summary tile dimensions independently of the readout's
16x64 tile. For a Q-by-K rectangle there are Q+K-1 diagonal segments, of length
at most min(Q,K); assign multiple segments per lane as necessary.

Use a candidate ownership rule `lane = (j-i) mod 32`, with a nonnegative modulo
and coordinates local to the document. Advance each segment in increasing i
and j, using a statically scheduled row loop where practical. Neighboring lanes
can then read consecutive K labels at the same query row while Q is broadcast.
Preload/reuse query labels across key tiles. Measure actual label traffic and
edge-lane utilization rather than assuming the mapping is bandwidth optimal.

Global diagonal ownership keeps the lane unchanged across key tiles when the
same warp owns both segments. It does not guarantee the same register slot or
warp across query tiles. Define explicit slot routing for diagonals entering
or leaving the tile; distinguish continuing chains from newly entering chains
that happen to use the same lane. No recurrence-step shuffle, parallel prefix
scan, or affine tuple is required within a thread-owned segment.

This ownership is specific to summary. It is not one thread producing a full
global diagonal into a dense intermediate matrix, nor a requirement on the
fused readout's register layout. Avoid global O(N^2) materialization of lengths,
weights or probabilities.

## Entry boundaries and summary planning

Independent query tiles still need correct incoming lengths. Plan both the
top and left entry frontiers, their corner convention, and the bottom/right
exit frontiers before writing the fused kernel. Include entries that cross
key tiles, paired consumer warps, CTAs and document boundaries. Use integer
boundaries throughout; do not recover lengths from rounded log weights.

Both stages can use the last mismatch key coordinate R as integer state. For
summary, reduce a complete segment within its owner thread; for readout, use
the associative operation across the existing distributed scan placement.
If independent block summaries need composition, use the same last
mismatch key coordinate R: `L = j-R`. Before a segment starting at key j0,
initialize `R = j0-1-L_in`; a causal chain starting at key zero has R=-1.
Noncausal summary chains entering at query zero and key j0>0 instead start
with R=j0-1. Each position contributes j for a mismatch or an integer identity
for a match, and summaries combine with integer max. Readout's inclusive max
scan produces R at every position, from which L is recovered. An all-match
block must preserve and extend its incoming length.

Specify boundary coordinates and lengths independently of either stage's lane
ownership. Summary stores only the required frontiers; readout loads them into
its scan-buffer boundary layout. Prove this interface directly, including the
corner entry, rather than redistributing an entire summary tile into readout
registers. Either L or R can be stored, with explicit coordinate conversion.

Initially retain the existing checkpoint coverage and producer/consumer
lifetime contracts, including the final-token summary rule. The exact hard
boundary layout and mailbox representation remain design choices to validate;
do not reinterpret the soft float/affine buffers as integer states. Inactive
warps must still obey required pipeline phases and WG synchronization.

## Readout with the existing scan buffer layout

Retain the existing 16x64 AltLayoutSplitScanBuffer placement used by fused
forward. Directly compare labels at each rolled register's physical coordinates
to generate integer mismatch markers. Use physical_row(r,c,lane>>2) for the
query row and c+(2*(lane&3)+half)*COL_BLOCKS for the key column, with tile
offsets and bounds applied. Both packed halves share the query row. Prearrange
query labels for reuse across key tiles; the row roll does not change key
column ownership. No QK MMA tile or input roll is needed.

Perform integer max scan in this distributed placement, including incoming
boundaries and cross-key-tile state. Reuse the existing scalar unroll permutation
to reach the MMA/readout layout, then recover lengths and look up log weights.
Do not reuse floating-point affine tuple arithmetic: integer packed types,
identity values and boundary handling require their own validation. Retaining
the placement does not imply retaining every soft-scan implementation detail.

This keeps the established inexpensive conversion to the Tensor Core layout
instead of introducing a general diagonal-owned tile redistribution. Per-step
cross-lane communication remains in readout, unlike summary. Measure its cost,
register pressure and overlap with SQ/SK/V fragments in the complete kernel.
Lookup before versus after scalar unroll is a later tuning choice; no new
shared tile staging or shared FP32 atomics are part of the primary design.

Hard specialization should eventually omit embedding-space Q/K tile loads,
LSE metadata used only by soft matching, and their matching MMA. Preserve hard
labels, SQ/SK/V inputs, and the final-reader fences on shared input slots.
Remove unused producer traffic only after the recurrence and readout prototype
are correct. Merely overwriting an already computed QK tile is not the target.

## Implementation sequence after approval

1. Build CPU models for both placements and their shared boundary contract.
   Verify summary segment ownership/slot reuse, direct rolled marker generation,
   exact integer lengths, and scan-buffer unroll across multiple tiles.
2. Add separate opt-in GPU probes for thread-owned summary and scan-buffer
   readout, outside production dispatch. Validate their boundary exchange and
   length-to-weight conversion; inspect registers, CALL and spills for both.
3. Prototype hard summary/boundary preparation and fused forward, initially
   retaining each stage's current tile dimensions and readout's 16x64 placement.
   Preserve online normalization and SQ/SK/V math.
   Include any lookup preparation and boundary kernels in total timings.
4. Specialize producer inputs, then tune summary ownership and readout integer
   scan independently. Change tile sizes only after a correct measured baseline.
5. Evaluate integration with event-stream backward and asynchronous planning.
   Production dispatch changes require a separate decision; preserve RNG replay,
   hard flags, checkpoint state and invocation-owned buffers throughout.

## Validation and performance gates

Integer checks must include random, identical, disjoint and periodic labels;
long matches spanning tile/CTA boundaries; a mismatch immediately before and
after a boundary; causal diagonal and document starts; partial internal tiles
and inactive consumer warps. Training public APIs retain their 256-alignment
requirements. Test fixed and varlen paths, both direction settings, all-hard
and mixed batches of sequence-level hard/soft rows, and rejection of
token-mixed inputs. Probe-only small shapes must not weaken API validation.

Compare output and normalizer against the maintained reference and current
CUDA implementation across negative, zero, near-zero and positive tau. Include
signed/zero SQ/SK readout, long all-match chains and head-shared tau. Check that
the new forward's saved statistics remain compatible with event backward;
report tau-gradient cancellation and sign discrepancies rather than relaxing
acceptance criteria. Checkpoint replay must reproduce the original hard flags.

Run memcheck, synccheck and racecheck on new kernel paths, inspect generated
code for CALL/local-memory spills, and report any limitations explicitly.
Bounds, slot reuse, shared staging and mailbox synchronization are correctness
gates, not optional performance diagnostics.

Benchmark on the local RTX 5090 with conda blkw and worktree imports. Start
with B16/H6/N2048/R32/DV64, uniform and ragged documents, and random/repeated/
periodic labels. Record warm device timings separately for table preparation,
summary, boundary passing and fused readout, plus their end-to-end GPU total.
Compare against the current hard CUDA path at identical shapes and precision;
report metadata bytes, register/spill counts and full-forward latency rather
than only recurrence throughput. Any later overlap benchmark must additionally
include D2H/H2D contention and planner readiness at backward. No speedup or
production readiness is claimed until these measurements pass.
