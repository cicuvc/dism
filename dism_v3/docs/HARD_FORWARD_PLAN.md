# Hard forward with separate summary and readout layouts

Status: design selected on 2026-10-14; implementation is deferred. Summary
assigns each complete diagonal segment inside a tile to one thread. Fused
forward/readout retains the current scan-buffer placement, replacing soft
log-domain affine arithmetic with integer state. The two stages share an
integer boundary contract, not a per-position register layout. This plan
changes no kernels, launch defaults, production dispatch or numerical tolerances.

Subsequent CUDA work removes token-level hard handling and makes the core a
soft-only contribution with sequence-hard skips. The dedicated hard forward
described here remains unimplemented; zero hard outputs are placeholders, not
an implementation of this plan.
See [the soft-only split and measurements](SOFT_ONLY_CUDA.md).

The target is sequence-level hard training on sm120, initially short sequences
around N2048 with R32 and DV64. Keep CUDA O(N^2) forward fused with readout and
the [asynchronous CPU planner plus event-stream backward](NATIVE_TRAINING_PLANNER.md).
Forward must not wait for CPU planning. Long-sequence training forward and
per-query tau are out of scope. Soft and token-mixed inputs retain their
existing implementation; hard sampling remains per packed training row/head,
not per document. Document boundaries still reset attention and recurrence.

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
and mixed batches of sequence-level hard/soft rows, and unchanged fallback for
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
