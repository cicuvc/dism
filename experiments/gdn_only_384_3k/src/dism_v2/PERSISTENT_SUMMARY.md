# Persistent forward summary, SM120a (2026-09-08)

Latest score optimization and timing supersede the unchanged-score descriptions below:
see [PREDICATED_SCORE.md](PREDICATED_SCORE.md). Single-transfer layout and restored synchronization
remain unchanged; summary now has six D/direction specializations.

## Current version: single-transfer Q/K, restored synchronization

The unsuccessful early-release/free2/ready1 experiment has been removed with user authorization.
Current K ready/free counts are again32/256, with free after score conversion; WG mailbox counts
remain128. The added named barriers and producer syncwarp experiments have been removed.
Early-release + ready32 passed two D64/N257 direction cases, but ready1 hung. Additional producer
synchronization did not fix it. CUDA debugger showed a blocked CTA's producer at final CTA barrier
and all compute warps waiting for Q ready. The cause remains unresolved; the earlier phase-race
hypothesis is not an established diagnosis. An intermediate variant had STACK8 in D64.

Current Q storage is one TK `st_bf<128,D>`, not eight separate warp tiles. Each warp loads a
`subtile<16,D>` at row-block `2*(warp&3)+warp/4`; logical ownership is unchanged.
With S=min(D,64), the Q box is `[S,128,1,D/S,1]` and coordinates `[0,base,bh,0,0]`.
Its row dimension remains sequence-local, including OOB zero fill for partial workloads.
K retains its GLX permutation with box `[S,2,4,8,D/S]`, loading all swizzle panels at once.
The default segmented map used by output/backward remains unchanged.

Consequently each workload's Q and each complete K tile require one TMA instruction each,
including D128. K tail remains the safe scalar path. Full/tanh codegen is zero-spill/no-CALL;
the regression asserts exactly two static UTMALDG.5D sites per summary specialization.
Restored baseline forward/codegen122 passed, then single-transfer forward/codegen122 passed.
Full/tanh both pass the new instruction-count codegen guard.

Same-session D64 tanh mixed-row CUPTI medians: old multi-transfer523.101us, new522.718us.
This is effectively unchanged, not an established speedup. Raw30 samples per run,20 warmup,
same B64/H4/N1024/V512 actual embedding inputs, retained outliers:
`benchmarks/single_tma_summary_sm120a.json`.
Baseline binary: `/tmp/dism-summary-wg-late-release-tanh.so` (not a repository dependency).
Single-transfer cross-workload sanitizer rerun: racecheck/memcheck/synccheck each12 passed,
zero errors/hazards; both directions, D32/64/128 and N65/257 are included.
Single-transfer full end-to-end selected regression:513 passed, with the same AccumulateGrad
stream-mismatch warning noted below. No precision tolerances or known failures were changed.

## Scope and pipeline

Default summary dispatch now uses `summary_persistent<D>`; output and passing algorithms are unchanged.
This is P1, not completion of all P0–P5 optimizations. Score arithmetic and metadata caching are unchanged.
Reference: `src/dism_fwd_nope.cu` scheduler and preprocess producer/consumer.

- Grid: min(logical tasks, SM count); 12 warps, producer-group dec40, compute groups inc232.
  Task identity is `task = blockIdx.x + iteration * gridDim.x`, decoded to batch/head/query block.
  RNG identity remains the logical query row. The simple scheduler is not yet cost-balanced.
- A separate shared input slot holds 128 Q rows; compute loads resident Q registers then releases it.
  Q's 5D TMA map includes the sequence dimension, so row OOB zero-fills without reading the next head.
  A single128-row destination and TK subtiles preserve ownership0,4,1,5,2,6,3,7.
- Producer submits all current-task K tiles, then next-task Q and K0 while compute may still be
  processing the final current-task tiles. Next K0 uses the next available ring slot, without copying.
  No task-end CTA barrier. Only initialization and final drain use CTA synchronization.
- K ring and mailbox phases advance with the cumulative tile counter across tasks; Q ready/free
  phases advance per task. A short workload may have little overlap window.
- Four separate boundary payloads share ONE warpgroup-level ready/free barrier per slot,
  each with 128 arrivals. All four writers publish before WG1 reads, and all four readers finish
  before reuse. These are not four independent per-pair epochs. No full-CTA step barrier is added.
- D32/64 use three K slots; D128 temporarily uses one to retain full Q staging within shared limits.
  D128 segmented-Q/multi-stage and stage search remain pending. Launch opts in to dynamic shared memory.
- Tail K retains safe producer scalar loading; asynchronous tail loading, score FFMA, key metadata
  prefetch, and dynamic branch reduction remain separate next steps.

TK primitives used: `warp::elect_leader`, `warpgroup::{increase,decrease}_registers`,
`tma::atoms::load_async_atom`, warp loads and warp MMA. Custom tensor maps retain the verified GLX
permutation. Existing inline mbarrier helpers retain exact arrival/expect semantics.
This vendored TK requires explicit TMA/REG_INCDEC feature enables for SM120; TMA declarations also
require enabling FP8 type declarations, without changing the BF16 computational path.
Fixed-count loops are explicitly unrolled; dynamic task/key loops use `unroll 1`.

## First timing (CUPTI)

RTX5090, B64/H4/N1024/D=DV64/V512, real CUDA embedding inputs, tau3, scale1,
q_from_k, hard_prob=.5, tanh tile LSE, full chunk passing. 20 warmup, 30 measured launches.
Only summary GPU event duration is reported; each call also executes unchanged output/passing.
No sanitizer workload was concurrent with these recorded runs. Raw outliers remain in the JSON.

| Version | Median summary us |
|---|---:|
| df561d7 nonpersistent / K stages2 | 673.277 |
| P1 persistent / K stages3 / pair mailbox (superseded) | 524.302 |
| P1 persistent / K stages3 / WG mailbox (current) | 522.717 |

Current vs baseline: about 22.4% less time (1.29x throughput).
The 0.3% difference between the two mailbox runs is not evidence of a statistically established gain.
This combines persistent, Q TMA, prefetch, ring depth and codegen changes, not a single-variable ablation.
It does not establish actual HMMA/reduce overlap or a new NCU compute-throughput figure.

Runner: `DISM_TILE_LSE=tanh python -m dism_v2.benchmark_persistent_summary`.
Baseline runner can load a pre-change binary with `--baseline-binary PATH`; the local baseline snapshot
is `/tmp/dism-summary-baseline-df561d7-tanh.so` (not a repository dependency).
Raw measurements: `benchmarks/persistent_summary_sm120a.json`.
NCU runner's new filter: `regex:.*summary_persistentILi64E.*`.

## Verification status

WG-mailbox full mode: forward plus codegen 122 passed, including 12 new cross-task reuse cases.
Cases force more tasks than SMs and cover D32/64/128, both directions, N65/257,
mixed replayable RNG, checkpoint/padding identities and saved W boundaries.
Full and tanh compilation of the initial persistent path: no CALL, zero stack/local spill,
native UTMALDG.5D and USETMAXREG, static average168 registers (consumer232/producer40).
The current WG variant full codegen also passes. Final WG variant verification:

- Racecheck, memcheck, synccheck: 12 cross-task cases each, zero hazards/errors/warnings.
- Full end-to-end wiring/replay/training-selected regression: 513 passed.
- Tanh codegen plus end-to-end replay/training-selected regression: 28 passed.
- Autograd emits a PyTorch AccumulateGrad stream-mismatch warning on one replay test in each mode;
  it is not suppressed. These selected regressions do not replace the known failing precision suite.

Commands (conda blkw Python):

```bash
python -m pytest -xq tests/test_dism_v2_codegen.py tests/test_dism_v2_core.py
compute-sanitizer --tool racecheck --kernel-name kns=_ZN7dism_v2 --error-exitcode 99 \
  python -m pytest -xq tests/test_dism_v2_core.py -k persistent_workload_reuse
# Repeat with --tool memcheck and --tool synccheck.
python -m pytest -xq tests/test_dism_v2_autograd.py \
  -k 'autograd_wiring or replay_tails or training_backward or all_cuda_no_embedding'
DISM_TILE_LSE=tanh python -m pytest -xq tests/test_dism_v2_codegen.py tests/test_dism_v2_autograd.py \
  -k 'native_tma or replay_tails or training_backward or all_cuda_no_embedding'
```
