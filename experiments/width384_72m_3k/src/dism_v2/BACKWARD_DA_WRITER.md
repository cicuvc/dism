# B3 warp9 dA writer experiment

`DISM_BWD_OPT=14` inherits OPT13 arithmetic and changes only the B3 output
pipeline. User subsequently selected OPT13 as default; OPT14 stays opt-in. Performance/accuracy evaluation targets
`DISM_TILE_LSE=tanh_finite`; OPT12's known arithmetic differences remain.

Each compute warp retains its 2-KiB Gsoft transpose buffer and gets an
independent 1-KiB FP32 16x16 dA output slot. This is output-layout storage,
not an additional calculation intermediate. Gsoft may be overwritten for
the next query tile without waiting for the previous dA TMA source read.

Each compute WG has one ready128/free1 mbarrier pair. Initialization completes
free phase0. Every consumer waits for free before writing its slot, executes
an async-proxy fence and publishes its arrival to ready. All128 arrivals are
required before warp9's elected thread issues four TMA reductions, commits
one group, waits `wait_group.read 0`, then arrives on free. Output epochs
continue across persistent tasks. The writer drains global completion with
`wait_group 0` before retiring. Consumers no longer execute TMA waits.

Warp9 services all dA subtiles of WG1's query tile, then WG0's, following the
reverse-scan dependency. There is no eight-warp output rendezvous. Pure-hard
specializations skip the writer and consumer output protocol entirely.

Two input stages are retained. Actual device query reports 99 KiB opt-in
shared per CTA on this RTX5090, not the historical conservative 64 KiB budget.
Separate scratch adds 8 KiB plus synchronization/alignment. Including the
existing 1-KiB static shared, the layout estimate ranges from45.125 KiB
(D32/DV32) to93.125 KiB (D128/DV128); final compiled resources must be checked.

## Validation

RTX5090, CUDA13.1, conda blkw, tanh_finite, default two input stages:

- OPT14 vs13:192 passed. Nine D/DV combinations, both directions,
  soft/mixed/hard, N129/257 tails, high-bit int64 labels, and persistent
  multi-workload epochs at N1/65/129/257/385. dV/summary/boundary bitwise;
  dA/dB/dLSE/dtau use unchanged3e-5 tolerances (dA atomic order may differ).
- Selected autograd wiring/training/replay/contract:508 passed,397 deselected.
  Row-bitset suite:62 passed.
- Whole-extension noCALL, native TMA, setmaxnreg, and single-FFMA gates:
  2 passed. Added source-annotated SASS gate passes for both D64 mixed
  directions: all128 static UTMAREDG sites,32 commits, and all DEPBAR waits
  belong to the warp9 writer branch, not the consumer region.
- Memcheck/racecheck/synccheck: each3 passed, zero errors/hazards. Cases:
  mixed D32/DV32 N129, mixed D128/DV128 N257, D64/DV64 N65 with173 batches.

New spill is retained, not tuned: mixed B3 D64/DV64 stack rises from0 to56 B
in both directions. D32/DV32 and D32/DV64 remain zero stack. Mixed D128/DV128
has312/304 B stack (q_from_k/k_from_q). Static shared remains1024 B;
register reallocation remains producer dec40 / consumer inc232.

## Controlled timing

B64 H4 N1024 D=DV64, vocab512, hard_prob0.5, tau3, scale1, actual CUDA
embedding and fixed saved forward states. CUPTI per-launch durations,
10 warmups,40 samples per round, two rounds with reversed OPT13/14 order,
80 pooled samples each. No overlapping GPU test or benchmark workloads;
clocks not locked. Median microseconds:

| B3 direction | OPT13 | OPT14 | Change |
|---|---:|---:|---:|
| q_from_k |1435.608|1569.623|+9.34%|
| k_from_q |1282.442|1443.080|+12.53%|

The requested source-read wait relocation works but this first single-slot
writer design is slower. No isolated evidence yet assigns the regression to
spill, handshakes, or writer serialization. Do not promote it to default or
claim a training-throughput improvement. No register-budget tuning in this
experiment. Strict oracle differences inherited from OPT12 are not resolved
or hidden by this finite synchronization comparison.

Raw samples and all180 WS resources:
`benchmarks/backward_da_writer_sm120a.json`.

```bash
DISM_BWD_OPT=14 DISM_BWD_REFERENCE_OPT=13 DISM_TILE_LSE=tanh_finite /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_backward_trim.py tests/test_dism_v2_backward_optimization.py
DISM_BWD_OPT=14 DISM_TILE_LSE=tanh_finite /home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.benchmark_backward_kernels --direction q_from_k --repeats 40
```
