# Chunk scan

Checkpoint coverage correction (2026-09-29): S is now floor((N-1)/32),
not8*floor((N-1)/256). Only the final warp block is omitted. Historical stage
timings/profiles below used S56 at N2048; the corrected interface uses S63,
so those numbers are not current-shape performance claims. Default stages4
is retained; this correction changes coverage, not the passing algorithm.

## Final selection: four stages (2026-09-29)

The requested2/3/4 sweep selected **stages=4** as the Python and C++ default.
`stages=2/3/4` selects compiled specializations; `pipeline=False` retains the
direct-load baseline. The output contract, full LSE and noncausal initialization
are unchanged. The earlier two-stage profile is historical, not a profile of4.

All modes were measured within each process with rotating execution order,
B16/H16/N2048, synthetic a=-.2/b=.5,20 launches per CUDA Graph replay,
100 warmup replays per variant and12 samples each:

| Path | Round1 us | Round2 us | Registers | Shared bytes |
|---|---:|---:|---:|---:|
| Direct | 171.116 | 171.027 | 19 | 0 |
| 2 stages | 164.275 | 164.380 | 34 | 2048 |
| 3 stages | 162.734 | 162.762 | 38 | 3072 |
| **4 stages** | **162.116** | **162.126** | 33 | 4096 |

Four stages reduced latency by about1.4% versus2,5.2% versus direct. All variants
have0 stack/spill and no CALL. Stage2 resource counts changed with the templated
wait helper; its old38-register measurement is not the current build.

Startup issues up to Stages checkpoints. In steady state wait<Stages-1> consumes
the oldest group; tail waits use min(Stages-1, remaining-1), ending with wait<0>.
All32 warp readers synchronize before slot reuse, and slot advance is modulo
Stages. No input/output layout or math change accompanies the sweep.

Artifacts: `build/chunk_stages_build.log`, `chunk_stages_timing1.json`,
`chunk_stages_timing2.json`. All131 stage-parameterized chunk tests passed;
filtered memcheck/racecheck/synccheck each ran those131 cases with zero errors
or hazards (`build/chunk_stages_{memcheck,racecheck,synccheck}.log`). An additional
default-interface regression brings the final chunk test count to132.

Final D32/64/128 full regressions each passed363 tests with2 skips, including
all three pipeline stages. Artifacts:
`build/check_dims_20260929_002921_pwp9fiik` and `build/chunk_stages_dims.log`.
The original summary build selection shared/D128/KStages3/tanh was restored;
chunk scan independently defaults to4 stages. This closes the chunk-scan stage,
not the as-yet unimplemented v3 output kernel.

## Interface and algorithm

`src/chunk_scan.cu` follows v2's forward passing: each thread owns a diagonal
`d = j - 32*s`, keeps one FP32 state, and visits checkpoints in increasing order.
Adjacent threads access adjacent columns. A CTA has128 threads; grid.y selects
batch/head. The default stages=4 async version uses4KiB shared input staging, with no
inter-warp synchronization. `pipeline=False` retains the original direct-load
version without shared memory for A/B testing.

```python
from flash_dism.summary import summarize, chunk_scan

summary_a, summary_b = summarize(...)  # caller already absorbed rtau into LSE
boundary = chunk_scan(summary_a, summary_b, seqlen=N)
```

Both inputs and the new output have shape `[B,H,S,padded_N]`, where
`S = floor((N-1)/32)`. Inputs are not modified. Output is FP32 log2 W at each
checkpoint's bottom row: `boundary[...,s,j] = W[32*s+31,j]`. It is not shifted
by a column and not an exclusive prefix. For a future output tile beginning at
row32*(s+1), the one-row predecessor is this boundary at column j-1; chunk-to-chunk
composition advances by32 columns. No new initial/final checkpoint is inserted.
N<=32 returns an empty checkpoint axis without launching a kernel.

For each valid checkpoint, the scalar update is

```
state = logaddexp2(state + summary_a[s,j], summary_b[s,j])
```

Missing predecessors use the existing finite LOG_ZERO=-1e6 contract. The logadd
uses `max + log1p(exp2(-abs(x-y)))*LOG2E`, not the tile's tanh/EX2 polynomial.
The exponential uses explicit PTX EX2 with FTZ, matching v2 passing. Ordinary
EX2 instructions in this file must not receive the summary's EX2 SASS patch.

Only j<=32*s+31 is meaningful in the summary buffers. Noncausal output is
initialized to -1e6 without reading either summary input. This includes padding;
all stored checkpoint rows precede the omitted final32-row block and hence N.
The test suite poisons unspecified inputs with NaNs to detect accidental use.

## Validation

`tests/test_chunk_scan.py` has45 cases: synthetic finite affine maps, zero maps,
positive long chains through N4097, empty summaries, summary+passing for both
directions and all three soft/hard modes, current-stream/CUDA Graph behavior,
input rejection, and SASS CALL/local-memory checks. The passing-only oracle uses
FP64 row-wise shifted states, independently of CUDA's diagonal traversal.
Integration preserves existing summary tolerances rather than loosening them.

Initial sm120a build:19 registers,0 stack/spill,0 shared memory/barriers, no CALL.
Build/test logs: `build/chunk_scan_build.log`, `build/chunk_scan_tests.log`.

Completed validation (2026-09-29):

- D32/64/128 full suites each276 passed,2 skipped. Matrix artifacts:
  `build/check_dims_20260929_000851_bnqn0grn`, driver log `build/chunk_scan_dims.log`.
  Original shared/D128/KStages3/tanh build restored.
- CUDA13.4 memcheck/racecheck/synccheck, filtered to chunk_scan_kernel:
  each45 passed and zero errors/hazards, no ARRIVES exception involved.
  Logs: `build/chunk_scan_{memcheck,racecheck,synccheck}.log`.
- B16/H16/N2048, synthetic finite summaries a=-.2/b=.5:171.26us median per
  launch on RTX5090 (CUDA Graph,20 launches/replay,100 warmup replays,9 samples).
  This isolates passing and excludes summary production; it is not a complete
  forward timing or a direct v2 performance comparison.

```bash
PYTHONPATH=python /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest tests/test_chunk_scan.py -q
```

This completes only the summary-to-boundary stage. The v3 output/readout kernel,
autograd, production RNG interface and varlen are not implemented by this change.

## Initial two-slot async input pipeline (historical)

The user requested a simple load_async/load_async_wait pipeline. Each warp owns
two32-column slots for each SoA input. This shared memory is an explicitly
requested async input buffer; recurrence state always remains in registers.
TK warp primitives issue16-byte cp.async loads, commit each A/B pair, then wait
with warp convergence. All coordinates are raw offsets.

Startup prefetches checkpoints0/1 relative to the warp's first valid checkpoint.
At step s, wait_group<1> completes s while s+1 may remain in flight. The warp
reads slot s into registers and synchronizes its readers before refilling that
slot with s+2; current LSE/store overlaps the prefetch. The final iteration uses
wait_group<0>, leaving no outstanding writes on exit. There are no mbarriers or
CTA barriers. Warp-aligned diagonals let the loop omit invalid prefix iterations;
noncausal warps only initialize their output and never issue input loads.

Use `chunk_scan(a,b,N,pipeline=False)` for the original baseline. Interleaved
CUDA Graph A/B timing is reproducible with:

```bash
PYTHONPATH=python /home/cicuvc/miniconda3/envs/blkw/bin/python tools/bench_chunk_scan.py
```

The async kernel uses38 registers,2048B shared memory,0 stack/spill, no CALL.
The first one-in-flight version measured171.0→170.2us, essentially no benefit.
Two-in-flight staging measured171.05→164.06us (B16/H16/N2048, synthetic finite
summaries, alternating A/B order). Logs are under `build/chunk_async2_*`;
the first experiment remains under `build/chunk_async_*`.

Repeat A/B:171.05→164.09us, about4.1% lower latency. Default is the two-in-flight
pipeline. Its45 chunk tests pass, including bitwise direct-load comparisons on
synthetic cases. Both kernels pass filtered memcheck/racecheck/synccheck with
zero errors/hazards. Full D128/tanh regression:276 passed,2 skipped. The initial
full-suite command omitted the repository parent from PYTHONPATH and failed two
v2-reference imports; rerunning with `PYTHONPATH=python:..` passed, with both logs
preserved. This iteration did not rebuild the unchanged summary for D32/64.
