# Scheduler uniform-register experiments (2026-09-28)

Base: `4d8408b`. Neither experiment is enabled by default. Both patches apply
independently to that commit; do not stack them. Default source and the loaded
D128 extension were restored after the measurements.

## Findings

The original scheduler emits four static `div.s32` PTX sites (two divisions in
each role). ptxas lowers these to IABS/I2F/MUFU.RCP/F2I and integer corrections.
The loop-carried task index and Q-block count have local-memory traffic. This
is not proof that all spills are scheduler state: eight other long-lived
spilled values feed Q LDSM addresses and depend on the lane/layout.

`unsigned_scheduler.patch` uses unsigned scheduler state and simplifies KBlocks
to `(q_block + 1) * 4`. It removes signed correction work and reduces spills,
but retains RCP and does not yield a measured speedup.

`integer_scheduler.patch` additionally:

- Prepares two exact integer divisors on the host. Device division uses
  `umulhi(value, floor(2^32/divisor))` followed by one remainder correction.
  The reciprocal is saturated to UINT32_MAX for divisor=1; no device special
  case or floating-point approximation is involved.
- Gives producer and consumer separate scheduler objects.
- Separates `hasNextTask()` from value-returning `nextTask()`. TaskInfo is local
  to one loop iteration, not an output reference whose old value survives the
  failed getNextTask branch.
- Adds a task-count range check, an independent division probe and tests for
  full uint32 boundaries and different persistent CTA strides.

This changes the generated PTX and **does** allow the producer's scheduler to
use UIMAD.WIDE/UIADD3/UISETP and UR task fields. The consumer's corresponding
calculation still uses ordinary registers; its task index still spills. No
new shuffle, shared staging, synchronization or task-order change was added.
There are no MUFU.RCP or CALL instructions in the final integer experiment.
Source-level uniformity is therefore only partially reflected in allocation;
the experiment does not solve consumer uniform-register placement.

## D128 resource and throughput results

Stack/spill numbers are ptxas bytes, not dynamic traffic. All instances report
168 registers for the complete 12-warp CTA with the existing inc232/dec40.
Times are graph-replay medians, B16/H16/N2048/D128, hard_prob=.5, 20 launches per
replay, 100 warmup replays and nine samples. Clock locking was not used.

| Version | Stack | Spill stores/loads | Query us | Key us |
|---|---:|---:|---:|---:|
| Base, same-session recheck | 64 | 68 / 88 | 794.68 | 814.09 |
| Unsigned only | 48 | 56 / 68 | 797.60 | 818.28 |
| Integer divisors + local TaskInfo | 48 | 52 / 72 | 840.01 | 860.66 |

Integer divisors alone before the API cleanup had 48/56/68 bytes; separate
role-local schedulers with the old output-reference API had 56/52/68 bytes.
Neither intermediate variant was performance-qualified. Final integer version
has 2,136 static SASS instructions versus 2,264 for the base, but is about 5.7%
slower in this workload. Fewer instructions/spills are not a throughput win.
No specific scheduling/stall cause for this regression has been established.

## Validation and artifacts

Integer experiment: D32/D64/D128 each **249 passed, 3 skipped**, unchanged
tolerances. Each dimension also passed 12 allocator/persistent-summary memcheck
cases with zero errors. D32/D64 have zero spills. The divisor probe tests
10,000 random uint32 values per divisor plus quotient boundaries, including
divisors 1, powers of two, odd numbers and UINT32_MAX. Six summary tests compare
CTA strides 5/17/64 against one CTA at N769/2048, bit for bit.

Unsigned-only D128: **227 passed, 3 skipped**. It did not receive a separate
three-dimension sanitizer matrix and is not promoted to default.

Artifacts (local, ignored build directory):

- `build/scheduler_before.ptx`, `build/scheduler_final.ptx`.
- `build/allocator128_after.sass`, `build/scheduler_{unsigned,integer,roles,final}.sass`.
- `build/scheduler_{unsigned,integer,roles,ssa}_build.log`.
- `build/scheduler_final_bench.json`, `build/scheduler_baseline_recheck.json`,
  `build/scheduler_unsigned_bench.json`.
- `build/check_dims_20260928_175209_gudf8x3y/` contains the complete integer
  experiment test/memcheck matrix.
- `build/scheduler_restore_{build,tests}.log` records restoration of the default.

To reproduce either experiment, apply just its patch, then use the normal
`DISM_KC_KEY_DIM=128 DISM_LINEINFO=1 ./check_summary.py` entry point. The integer
patch also contains its standalone test probe; build.py discovers its .cu file.
