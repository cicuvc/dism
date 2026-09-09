# Register-only summary store packing (2026-09-09)

Not selected; the default store is unchanged. No shared staging or batching
was introduced. External GLX and the user's load_rhs_tile were not modified.

## Candidate

In summary_persistent.cuh, j ranges from -1 to30, with j=-1 in lane28.
Replace the three stores after computing h/dest with:

```cpp
float tail_a = __shfl_sync(0xffffffff, left.init[1].first.u0, 31);
float tail_b = __shfl_sync(0xffffffff, left.init[1].second.u0, 31);
bool edge = lane == 28;
int col = j & 31;
dest[col] = make_float2(
    summary_select(edge, h.first.u1, h.first.u0),
    summary_select(edge, h.second.u1, h.second.u0));
dest[col + 32] = make_float2(
    summary_select(edge, tail_a, h.first.u1),
    summary_select(edge, tail_b, h.second.u1));
```

The original three stores cover 0..30,31..62,63. Their address footprints
touch8+9+1 sectors (32B), whereas the candidate touches8+8. This is an
address-layout calculation, not a new NCU or DRAM-traffic measurement.

## Measurements

RTX5090, sm120a, CUDA13.1, BF16; B64/H4/N1024/D=DV64/V512,
scale1/tau3/hard_prob0.5, int32 labels, no bitset, tanh_finite, lineinfo enabled.
Existing benchmark_persistent_summary:20 warmups then30 CUPTI samples,
three rounds, alternating baseline/candidate order. Clocks not locked;
some samples have large scheduling outliers. Values below are per-round
medians in microseconds, not full-path or isolated-summary-only timings:
the measured launch is summary inside the normal forward stream.

| Direction | Original rounds | Candidate rounds | Median of round medians |
|---|---|---|---|
| q_from_k |220.367,221.487,222.2235|225.471,226.207,226.415|221.487 ->226.207 (+2.13%)|
| k_from_q |233.7105,233.790,233.6625|232.687,232.7825,232.367|233.7105 ->232.687 (-0.44%)|

No consistent speedup. D32/128 performance was not measured.

D64/q_from_k/mixed/int32 summary static SASS:

| Instruction | Original | Candidate |
|---|---:|---:|
| STG.E.64 (includes padding store) |4|3|
| SHFL (includes slow paths) |78|82|
| BRA.DIV |18|19|
| WARPSYNC.COLLECTIVE |24|26|

The two new source shuffles add four static SHFL instructions including
fallback copies. Predicated selection does not eliminate the compiler's
shuffle convergence fallback. These counts alone do not establish the cause
of the timing change. Forward remains no CALL, zero stack/local spill, native
TMA and register redistribution intact; reported REG168 with compute inc232.
No additional shared allocation.

## Correctness

- Full LSE: core133 + label-width55 + row-bitset62 + codegen1 =251 passed.
- Finite mode: label-width55 + row-bitset62 + codegen1 passed.
- Isolated-process old/new finite comparison:54 cases, D32/64/128,
  both directions, hard_prob0/.37/1, N65/139/257. Output, normalization,
  summaries/passing state and both saved W boundaries are bitwise identical.
  Each process loads only one extension, avoiding same-name extension cache aliasing.
- The full-oracle core suite was also initially run in finite mode and had
  122 failures/11 passes; its strict full-LSE tolerances are not an acceptance
  criterion for the tanh approximation. They were not relaxed or hidden.
- No new NCU report was collected; shared/TMA output batching remains untested.
- Compute Sanitizer memcheck: the54 isolated candidate comparisons passed,
  zero errors. Racecheck/synccheck were not repeated (no synchronization protocol change).

Baseline binary for local reproduction: /tmp/dism-summary-store-before.so.
Use DISM_TILE_LSE=tanh_finite DISM_LINEINFO=1 with
python -m dism_v2.benchmark_persistent_summary, adding
--baseline-binary /tmp/dism-summary-store-before.so for the original.
