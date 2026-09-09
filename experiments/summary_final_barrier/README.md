# Removing the final CTA barrier (2026-09-09)

Candidate: delete only `__syncthreads(); // Final drain only, not a workload boundary.`
at the end of summary_persistent. Initialization barrier, all mbarrier counts,
mailbox protocol, stage counts and global stores remain unchanged.

Source-level motivation: Q/K consumers wait for every issued TMA; WG1 consumes
every WG0 mailbox; summary writes are ordinary global stores, not asynchronous
TMA output. Thus there is no obvious additional data transfer for this barrier
to drain. This reasoning alone does NOT establish safe early role exit with
the current generated code and protocol.

Observed on RTX5090/sm120a, CUDA13.1, lineinfo enabled:

- tanh_finite label-width55 + row-bitset62 + codegen1 =118 passed.
- D64 q_from_k mixed/int32 summary BAR.SYNC2 ->1; BRA.DIV18,
  WARPSYNC.COLLECTIVE24 and TMA emission sites2 unchanged. No spill/CALL.
- The full-LSE full regression run stopped making progress in persistent tests
  and was terminated. An isolated persistent suite passed six N65 cases, then
  stalled at N257/q_from_k/D32 and hit the45second process timeout.
- Deletion was reverted before further profiling. No new NCU report or
  sanitizer acceptance was obtained for the candidate. The exact cause of
  the stall is unresolved; this is not proof that a CTA barrier is generally
  required for all producer exits or all TMA kernels.
- After restoring the original CTA barrier, the same isolated persistent
  suite passed all12 cases (including N257, both directions and all D).

## Independent warpgroup exit barriers

Following the user's suggestion, the current candidate replaces the final
CTA barrier with `kt::warpgroup::sync(1 + warp / 4)`: WG0/WG1/producer use
IDs1/2/3 with128 arrivals each. Initialization continues to use CTA barrier0.
No extra shared allocation or mbarrier is introduced. This allows the groups
to retire independently while retaining convergence within each group.

- Full oracle/core/label-width/row-bitset/codegen suite:251 passed.
- Isolated persistent suite:12 passed, including the no-barrier stall case.
- New kernel-only tests/test_dism_v2_summary_exit.py:12 cases, N65/257,
  D32/64/128, both directions, more tasks than SMs, repeated RNG replay and
  exact output/normalizer/saved-boundary equality.
- These12 cases passed memcheck (zero errors), racecheck (zero errors/warnings),
  and synccheck (zero errors). The earlier sanitizer run containing the dense
  Torch oracle was stopped for runtime and is not counted as completed.
- SASS has one initialization BAR.SYNC0 and one dynamic-ID BAR.SYNC with0x80
  participants. No spill/CALL; native TMA and inc232/dec40 remain intact.
  The codegen regression now checks the128-thread exit barrier.

This supports using warpgroup-local exit synchronization in the tested kernel.
It does not establish which hardware/compiler interaction caused the original
no-barrier stall, or whether only the producer group's exit needs synchronization.

## NCU

Report: /tmp/dism-summary-wg-exit-lineinfo-q.ncu-rep (source imported,40passes).
B64/H4/N1024/D=DV64/V512, q_from_k, mixed0.5, int32 labels,
tanh_finite, lineinfo, no bitset, scale1/tau3. Skip10 matching launches,
capture1; cache-control none and clock-control none, matching the earlier report.

Duration229.44us, SM throughput39.7425%, tensor active38.5421%.
The earlier CTA-exit report /tmp/dism-summary-int32-lineinfo-q.ncu-rep was
230.40us. These are single profiles at unlocked clocks and different times;
the0.42% difference is not evidence of an established speedup.
