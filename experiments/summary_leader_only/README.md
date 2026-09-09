# Leader-only producer (2026-09-09)

User-accepted current summary baseline, following the independent WG-exit change.
Producer WG executes dec40, then warp-local elect before testing
`warp == 8 && leader`. Only that thread runs the persistent task/key loops.
Other producer threads go directly to their WG-exit barrier, NOT an early
return: collective exit was necessary in the preceding tested versions.

K-ready init32 ->1; the leader's arrive.expect_tx supplies the one arrival
for full-tile TMA. No31-lane extra arrivals. K-free256, mailbox128, ring stage
counts, Q input and score-end release remain unchanged. No extra shared bytes.

Unpadded partial K tiles still cannot use the flattened TMA map safely. This
candidate has the leader copy valid logical rows and zero invalid rows, with
the inverse row permutation `(r&7)*8 + ((r>>3)&3)*2 + (r>>5)`.
Runtime row loops are not unrolled; fixed D column loops are unrolled.
This tail path is correct but substantially slower; see measurements below.

## Validation / generated code

- Full core/label-width/row-bitset/codegen/exit suites:263 passed.
- tanh_finite codegen passed. All forward instances remain zero stack/local
  spill, no CALL, native TMA, compute inc232 / producer dec40.
- Kernel-only exit/replay suite (12 persistent/tail cases) passed all three
  sanitizers: memcheck/synccheck zero errors, racecheck zero errors/warnings.
- D64/q_from_k/mixed/int32 static counts: ELECT2 ->2, TMA2 ->2,
  BRA.DIV18 ->18, collective24 ->24, SYNCS.ARRIVE8 ->7.
  No duplicate election was added at the TMA sites.

## Timing

RTX5090/sm120a/CUDA13.1, B64/H4/D=DV64/V512, BF16, int32 labels,
hard_prob0.5, scale1/tau3, tanh_finite, lineinfo, no bitset. Standard summary
CUPTI benchmark measures the summary launch in the normal forward stream:
20warmups,30samples. Alternating old/new order, unlocked clocks. No other GPU
tests were run concurrently with these timings. The baseline includes the
WG-local exit barrier, so this isolates producer changes from exit changes.

N1024, three round medians (us):

| Direction | Baseline | Leader-only |
|---|---|---|
| q_from_k |222.415,222.463,222.335|220.6545,221.055,219.951|
| k_from_q |234.0465,233.9185,234.175|233.9665,233.951,233.567|

Median of medians: q_from_k222.415 ->220.6545us (-0.79%);
k_from_q234.0465 ->233.951us (-0.04%). Small unlocked-clock changes are not
proof of a stable throughput improvement.

Tail comparison, q_from_k, two round medians (us):

| N | Baseline | Leader-only |
|---|---|---|
|65|12.992,13.0075|34.912,34.9275|
|257|36.672,36.7195|58.912,58.928|

The substantial tail regression prevents treating this candidate as a general
performance win. No D32/128 timing or new NCU report was collected.
Benchmark now accepts --n (default1024) for tail comparisons.

Local artifacts: /tmp/dism-summary-before-leader-only.so,
/tmp/dism-summary-leader-only-timings.json,
/tmp/dism-summary-leader-only-tail-timings.json.
