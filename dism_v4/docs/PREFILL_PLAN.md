# Offline hard-DISM prefill: SAM + positive event streams

Status: Python numerical prototype plus an opt-in C++ planner/CPU execution
mock plus an opt-in [chunked Triton executor](PREFILL_TRITON.md); see also
[native implementation](PREFILL_NATIVE.md). No production model dispatch
change. Binary-lifting LCA is retained by user request pending profiling.

## Scope and identities

Build a single SAM on the complete key-label string. Let `p[j]` be the
non-clone state created at key endpoint j. Scan queries against that SAM,
resetting the query cursor **before** consuming a reset row. Keep both the
state `u[i]` and the actual matched length `m[i]`; the latter can lie inside
a suffix-link edge. Resets do not delete key history.

In the suffix-link tree, with node length `len`,

```
L[i,j] = min(m[i], len(LCA(u[i], p[j])))
w(l) = sum(exp(a*tau), a=1..l), w(0)=0
O[i] = sq[i]^T sum_{j<=i}(w(L[i,j]) sk[j] v[j]^T)
       / (1 + sum_{j<=i} w(L[i,j]))
```

Only full-hard labels and reset delta in {0,+inf} are covered. Signed soft
readout affects the numerator only. Future keys may affect SAM topology,
but must never enter a query's aggregate. No positional or vocabulary
parameterization changes are involved.

## Positive rectangle decomposition

Centroid-decompose the undirected suffix-link tree. At a centroid c, treat
each remaining component and the singleton c as indivisible groups.
Call the component towards the original root P, and all other groups D.
For query/key groups separated here, the weight depends on only one side:

| Query group | Key group | Match length |
| --- | --- | --- |
| D | D, different group | min(m[i], len(c)) |
| P | D | min(m[i], len(LCA(u[i],c))) |
| D | P | len(LCA(c,p[j])) |

P/P is never a cross-group pair. Pairs inside one component recurse;
singleton c/c uses m[i]. Canonical query states are essential for the D/P
formula. Each pair is assigned at its first separation, exactly once.

Do not enumerate all component pairs: high-degree stars would be quadratic.
Combine groups with a size-weighted binary grouping tree (the prototype uses
Huffman merging). Each internal grouping edge yields cross rectangles in both
directions, splitting P from D where needed. This creates positive streams,
not subtraction of nearly equal "all minus same child" vector aggregates.

Each rectangle has weight a[i]*b[j]. Merge key/query events chronologically,
keys first on equal positions. Prefix rank-one updates and readout compute

```
M += b[j] * outer(sk[j], v[j])
numerator[i] += a[i] * sq[i]^T M
```

The CPU planner handles topology, indices, scalar factors and denominator;
the eventual GPU path handles vector work. No matrix is stored at each SAM
node. Stream waves bound live matrix scratch. Later, long streams can use
chunk GEMMs plus chunk-prefix aggregation instead of sequential updates.

## Stability, complexity and implementation boundaries

Compute log(w(l)) using expm1/geometric sums, including tau=0. Maintain a
running maximum of **active** key log-factors in each stream and rescale its
matrix/count on pivot changes. First accumulate log denominators (fallback
log-weight 0), then accumulate normalized vector contributions. Never
materialize exp(tau*l). This matters when the full SAM matches future keys
much more strongly than any currently causal key.

Target per head: O(N log N) scalar/index events, O(N R DV log N) arithmetic,
O(N(R+DV)) inputs/outputs and bounded active-stream matrix scratch. Fixed
channel sizes are implicit when calling this "N log N" prefill. Weighted
group depth contributes log(size/child_size)+O(1); these ratios telescope
through centroid recursion. Keep time-ordered lists using stable partitions
and merges rather than sorting every stream independently.

The initial Python planner uses binary-lifting LCA, so its scalar planning
can incur an additional logarithmic factor. It is a numerical prototype,
not a demonstrated end-to-end N log N implementation. Dense pair audits and
the dense oracle are diagnostic-only O(N^2) memory. Production would use
constant-time LCA/RMQ preprocessing and bounded streaming GPU workspaces.
Building the final decoding snapshot once is a separate handoff step.

## Validation and next steps

`tools/prefill_reference.py` supplies the independent SAM planner and bounded
per-stream matrix evaluator. `tools/test_prefill_reference.py` compares:

1. SAM/LCA match lengths against diagonal label DP, including clones and
   implicit query lengths;
2. positive rectangles against all causal pairs, checking no omission or
   duplication;
3. FP64 outputs against exact dense logaddexp recurrence and the existing
   torch `dism_v4_ref.py` oracle (whose softplus has threshold 20);
4. resets, mismatch, repetition, periodic inputs, tau=0/tiny/ln64, negative
   tau, signed readout and unequal R/DV; long-chain overflow safety and
   future-extension invariance.

Only after these pass: measure event replication, implement CPU planner
with explicit memory budgets, then GPU vector execution and final decoding
cache handoff. No frozen-snapshot square-root scheme is required for prefill.

## Initial numerical results (2026-10-08)

Run from repository root, without using a GPU:

```bash
/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v4/tools/test_prefill_reference.py --output /tmp/dism_prefill_reference_results.json
```

60 deterministic small cases passed complete causal-pair coverage, SAM/LCA
versus diagonal DP, numerical output, denominator and prefix-extension checks.
Maximum absolute output differences:

- FP64 versus exact logaddexp recurrence: 3.20e-14.
- FP64 versus existing torch reference: 5.84e-11.
- FP32 **vector arithmetic**, retaining FP64 scalar factors/denominators:
  4.53e-6. This does not validate an entirely FP32 planner.

Full-repeat stress tests at N=256/512/1024/2048 and tau=ln64 passed, with
maximum output error 2.50e-13. At N=2048 the largest log denominator is
8517.42; direct exponentiation would overflow FP64. Stored key+query event
counts were 7120/16084/36568/80604 respectively. These are observations on
one input family, not a general complexity proof or GPU throughput result.

Known remaining work: constant-time LCA for the target planning bound,
planner/event-layout performance, large R/DV memory budgets, bounded GPU
stream scheduling, and conversion of the final prefix into decoding state.
Zero-weight and causally empty rectangles are intentionally retained in this
prototype to make pair-coverage auditing straightforward; production may
prune them. Scalar tau is fixed per head over the prefill interval.
