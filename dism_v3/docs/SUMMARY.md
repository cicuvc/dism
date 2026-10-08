# v3 summarization bring-up

Current implementation cleanup (2026-09-28): only full32x64 summary remains.
The split-scan and half-pipeline kernels, their helper/probe and selection paths
were removed. Sections discussing selections1/2 below are historical records,
not build instructions for the current tree. See CONFIG_AUDIT.md for remaining
parameter constraints and SUMMARY_MATRIX.md for the pre-removal comparison.

## Operand dimensions

Build-time KcKeyDim32/64/128 support and the per-dimension correctness matrix
are described in KEY_DIMENSIONS.md. Default remains64. This supersedes older
D64-only statements below; it is not simultaneous runtime dimension dispatch.

## Summary causal mask removed (2026-09-28)

Score finalization now only converts to log2 and applies predicated hard-label
overwrites. The per-element causal comparisons/selects and their row/column
coordinate arithmetic are removed from all three summary variants.

Diagonal composition preserves i-j, including across key-tile boundaries. Thus
bottom-edge outputs at j<=i are unchanged by upper-triangle scores. Only these
causal entries are part of the summary contract; upper-triangle outputs are
unspecified, not guaranteed LOG_ZERO. Chunk passing must preserve diagonal
coordinates, and the eventual output softmax still needs causal masking.
CTA-level key-loop bounds and LOG_ZERO stores for omitted tiles are unchanged.

The mathematical reference remains causal. Production summary tests compare
both affine components only at valid bottom-edge columns, without changing
tolerances or ignoring hard mismatches. Separate full/split scan probes compare
masked and unmasked inputs bitwise on causal outputs, including key-tile
crossings and deliberately large upper-triangle scores. Historical descriptions
of per-element causal masking below are superseded by this section.

This is the user-selected default, not an opt-in experiment. Default variant0
passes 212 tests (3 option-specific skips), reports 168 registers and zero
stack/spill, and passes the no-CALL check. B16/N2048/H16/D64 mixed query/key
timings are 523.99/538.70 us versus fixed historical 536.51/553.94 us
(-2.33%/-2.75% latency); no baseline rebuild was timed. Full six-case results
and build/test logs: build/check_summary_20260928_151408_end88y9l.

## Current arithmetic: finite only

The optional EX2 polynomial/SASS-patch experiment is implemented and validated;
see EX2_APPROX.md for build modes, patch safeguards, accuracy and timing.
Default remains tanh: EX2 patched is about2.6–3.6% slower in the measured
B16/N2048/H16/D64 summary workload. The historical future-patch text below
predates this opt-in experiment.

The latest user decision removes the exact/-INFINITY implementation entirely.
All summary/probe paths now use one `LogAffineOp`: the existing tanh fit with
`LOG_ZERO=-1e6`. Hard mismatch, omitted output regions and the
second component of the affine identity use this same constant. There are no
infinity tests in this operation and no `approximate` Python/C++/template switch.
The new SASS-patched softplus remains future work; its formula was not changed
as part of this cleanup. Historical exact-vs-approx resource tables below are
retained as experiment records, not as currently available dispatch choices.

This deliberately changes mathematical zero/identity semantics. The supported
finite score/context regime is far from the sentinel: current tests include
N1024, rtau up to ln64, random bounded scores, hard chains and all-masked scans.
This is not a guarantee for arbitrary unbounded positive scores or infinite
caller inputs. The Python mathematical oracle retains -inf; tests classify
unreachable components separately (<-1e5), require all CUDA outputs finite,
and compare reachable first components strictly and second components using
the pre-existing .03 base2 tanh tolerance. No oracle semantics are rewritten.

## Scope and interface

### Final-warp omission (corrected2026-09-29)

S=floor((N-1)/32) warp checkpoints require ceil(S/8) CTA workloads per
sequence/head. Only the warp block containing token N-1 and later warp blocks
are omitted; previous code incorrectly omitted the whole final CTA workload.
Output is two buffers of `[B,H,S,ceil(N/256)*256]`. N<=32 launches no summary.
For example N256/500/2048 now yield7/15/63 checkpoints, respectively.
Every active warp's32 query rows are valid, so make_scan still has no tail-row
checks or padded-row identities. Inactive tail warps participate in Q-prefetch
and WG synchronization and release every K slot, but skip MMA/scan and stores.
The CTA key loop ends at the last valid checkpoint row rounded up to64 columns.
This supplies every32-row boundary needed by paired16-row output warps.
Earlier benchmark and validation records below describe the historical coverage.

All three selections passed150 tests, including aligned/unaligned scheduler
boundaries511/512/513, both directions, hard/mixed/soft, persistent reuse and
source-mapped SASS checks. All three report zero stack/spill stores/spill loads
and no CALL. Default remains0. Build/test logs: `build/skip_final_{0,1,2}*`.
Default0 also passes both racecheck smoke cases (N513/1281, two/five workloads,
both LSE directions) with zero reported hazards; log: `build/skip_final_racecheck.log`.

The implementation uses the user's persistent 32x64 warp-tile skeleton: 8
compute warps, one active producer warp inside a third warpgroup, 256 query rows
per workload, three K slots, and Q/K0 prefetch overlapping the previous
workload's reduce/writeback. TK's group tile load assigns row blocks in physical
warp order 0,4,1,5,2,6,3,7; metadata follows the same assignment explicitly.
Each32-row checkpoint produces its own affine summary; there is no dependency
between the compute warps in this stage.

`flash_dism.summary.summarize` accepts interpolated BF16 operands `[B,N,H,64]`,
natural-log FP32 preabsorbed LSE `[B,N,H]`, int labels and bool row decisions `[B,H,N]`,
direction `[B,H]`, and natural-log tau `[H]`. True direction uses **query LSE**.
Before calling, compute both `q_lse=raw_q_lse-rtau[None,None,:]` and
`k_lse=raw_k_lse-rtau[None,None,:]`. Neither the wrapper nor CUDA absorbs tau
again. The accumulator is initialized to `-preabsorbed_LSE` in natural-log
units using explicit LSE negation and TK `add_row/add_col` from zero, then MMA
accumulates the dot product.
After MMA, one multiplication by LOG2E converts the score. The producer's
elected lane converts natural-log `rtau` once per workload and publishes
`shared.tau2`; consumers use that value directly for hard matches. This reuses
the existing scalar shared slot and prefetch synchronization. Padding/packing
does not mutate caller LSE tensors. BF16 Q/K and embedding interpolation are unchanged.

Latest producer-tau2/subtraction cleanup: all three selections passed123 tests,
including source-mapped predication and no-CALL checks. However, combined-change
resource usage regressed: stack/spill stores/loads are72/76/100B for0 and
104/108/132B for1; selection2 remains8/16/24B. This is reported, not attributed
to either individual change without an A/B. The requested source changes are
retained, default remains0, and no additional spill workaround is introduced.
Logs: `build/producer_tau2_{0,1,2}.log` and corresponding `_tests.log` files.

Controlled four-way A/B on default selection0 (same compiler flags, lineinfo,
register budgets and no per-key WG sync) isolates the regression:

| Producer tau conversion | sub_row/sub_col instead of negate+add | Stack / stores / loads B |
| --- | --- | ---: |
| No | No | 8 / 16 / 24 |
| Yes | No | 8 / 16 / 24 |
| No | Yes | 72 / 76 / 100 |
| Yes | Yes | 72 / 76 / 100 |

All four passed123 tests. Thus the subtraction/broadcast rewrite alone reproduces
the stack regression; producer conversion does not. This does not yet distinguish
sub_row from sub_col or prove a defect in the TK primitive. Source, logs, cubins
and annotated SASS are saved under `build/tau_sub_ab/{baseline,tau_only,sub_only,both}`.
Per user request, current source restores negate+add while keeping producer
tau conversion. All three variants pass123 tests; stack/store/load bytes return
to8/16/24, 0/0/0, and8/16/24 respectively. Default remains0. Logs:
`build/restore_add_{0,1,2}.log` and corresponding `_tests.log` files.

The test fixtures and FP64 oracle retain raw LSE, while the caller-side test
adapter performs subtraction before invoking the summary API. Additional tests
hold absorbed LSE fixed while varying tau: pure-soft outputs must be bitwise
unchanged in both directions. Existing nonzero-tau hard/mixed tests ensure the
hard branch still receives its score contribution.
After the initial interface correction, split=0 and split=2 each passed121 tests,
including SASS checks. Current stack/spill-store/spill-load bytes are88/92/116
and8/16/24 respectively (`build/preabsorbed_lse*.log`). These predate the
accumulator-seeding update below. Default layout remains0.

### Restored accumulator seeding and predicated hard selection

The cleaned baseline was committed as `5b0ce3e` before making these changes.
All three variants now seed the FP32 accumulator with negative absorbed LSE
before MMA. LSE is NOT scaled by LOG2E at that point; the entire accumulated
score is converted afterwards. This changes floating-point addition order,
so equivalence is checked against the existing oracle tolerances, not bitwise
identity to the former zero-accumulator/FFMA path.

`finish_score_pair` uses inline PTX `@predicate selp` for hard overwrites and
`selp` for causal masking, with no hard-row C++ branch. The old skeleton's
incorrect key-register index, duplicated .x output, and non--1e6 bit pattern
are not copied back. LOG_ZERO is passed as a float operand.

Score finalization no longer takes seqlen or tests row/seqlen or col/seqlen.
For a valid query, col<=row implies col<seqlen; padded query rows are overwritten
by affine identities in make_scan. That tail operation is now behind one
warp-uniform `q_start+ROWS > seqlen` guard, so full tiles do not execute its
per-element checks. Causal masking is retained, as is the partial-checkpoint
identity contract. Input buffers remain padded; no bounds guarantees are relaxed.

With all these changes together, stack/spill-store/spill-load bytes are8/16/24
for selection0, 0/0/0 for1, and8/16/24 for2. These are combined-change results,
not a controlled attribution to accumulator seeding alone. Throughput has not
been measured and the default stays0. The codegen suite now also checks that
line-mapped hard/causal assembly contains SEL and no BRA/BRX/CALL; use
DISM_LINEINFO=1 for this check (otherwise that specific test skips).
All variants passed the numerical regression and both codegen checks (the
current full suite has122 cases). Selection2 additionally passed9 selected
tests under each of memcheck/racecheck/synccheck; the restored default0 passed
the tail/persistent smoke under racecheck. All reported zero errors/hazards.
Logs and the original-path line-mapped SASS are `build/seed_lse*`.
The restored negate+add baseline is committed as `3fff19b`.

Output is `(summary_a, summary_b)`: two independent contiguous FP32 buffers of
shape `[B,H,floor((N-1)/32),ceil(N/256)*256]` in **base2**, storing the first
and second log-affine components respectively, not only W. The kernel writes
each directly from registers, without shared-memory staging or an AoS conversion
kernel. After passing, consumers can read only the second buffer at unit stride.
Total summary element count is unchanged; this is not yet a measured end-to-end
bandwidth gain because chunk passing/output are not implemented. Only tests stack
the buffers to compare with the unchanged pair-valued oracle.
SoA validation: all three variants pass150 tests including separate allocation,
contiguity/unit-stride assertions and both affine components against the oracle.
All report168 registers, zero stack/spill, and no CALL. Default0 passes both
N513/1281 memcheck smoke cases with zero errors. Logs: `build/soa_{0,1,2}*`
and `build/soa_memcheck.log`. Default selection remains0.
The top input to each independent checkpoint
is identity; VState carries the horizontal progression across64-key tiles.
All scheduled query rows are valid. Causally masked keys are zero maps.
The key iteration ends at the CTA's causal bounding box, and
omitted output regions are explicitly initialized as zero maps.

This first interface pads and packs tensors in Python. It does not claim
unpadded TMA tail safety or production throughput. Explicit row masks are
diagnostic inputs inherited from the skeleton; production RNG is not wired.
Only D64 is currently dispatched. Soft-readout sq/sk and delta are not inputs.
Chunk passing, output and backward kernels are not implemented here yet.

## Code organization

- `src/wmma_tma_preprocess.cu`: original32x64 consumer kernel (selection0).
- `src/wmma_tma_preprocess_split_scan.cu`:32x64 MMA with two16x64 scans (selection1).
- `src/wmma_tma_preprocess_half_pipeline.cu`: sequential16x64 MMA/score/scan (selection2).
- `include/summary/kernel_common.cuh`: shared producer, score and output helpers.
- `include/summary/launch.cuh`: common host validation, descriptors and launch.
- `include/summary/primitives.cuh`: custom K tensor map, ring protocol, config,
  scheduler and FP32 log-affine scan adapters.
- `src/probe/summary_probes.cu`: independent scan, ring/cp.async, and rv-load probes (DISM_BUILD_PROBES=1).
- `python/flash_dism/summary.py`: explicit padded Python entry point.
- `tests/test_summary.py`: sequential affine oracle and component/full tests.

The compute translation units contain no diagnostic or split-mode preprocessor
branches. `DISM_SUMMARY_DIAGNOSTIC` has been removed from source/build support;
old bisection artifacts below are historical only. `DISM_SUMMARY_SPLIT_SCAN`
remains solely a build.py source selector (default0): exactly one of the three
translation units is compiled/linked, with the same Python summary API.
`cu_flash_dism.summary_variant()` reports the linked selection so codegen tests
inspect its actual cubin rather than a stale object from another variant.
After the source split, all three selections independently passed121 tests and
the loaded module's reported selection was checked against each requested build.
No CALL was introduced. Stack/spill-store/spill-load bytes are88/92/116 for0,
136/140/164 for1, and8/16/24 for2; default0 was restored afterwards. Logs:
`build/source_cleanup_{0,1,2}.log` and corresponding `_tests.log` files.

## Layout and synchronization

All coordinates are **raw element offsets**, including nonzero batch/head and
query starts. Q's tensor map sees B,H,N,D with physical B,N,H,D strides. K's
custom TMA map permutes rows so a physical MMA column
`16*block + 8*half + 2*(lane%4) + element` represents logical column
`2*block + half + 8*(2*(lane%4)+element)`. Labels and key LSE use this mapping too.

The new scan HState holds all columns0..63. Lane l stores columns
`16*(l%4)+7-l/4` and that column+8, as two FP32 affine pairs. There is no special
last-column extraction from VState. VState still owns shifted rows -1..30;
the corner convention is internal to scan and is tested across multiple tiles.
The scalar score is rolled before duplication. Summary uses reduce_forward,
which reuses prescan and does not run postscan.

There are eight mbarriers: ready/free for three K slots and the Q/K0 prefetch.
Phase bits persist across tasks; resetting a pipe resets only its slot index.
Draining peeks at the last released slot without consuming another generation.
Q/K0 storage aliases the K ring only after both WGs have read Q; next-task Q/K0
starts after both WGs have consumed the last K tile. Metadata has the same
ownership interval. Consumer input reads are complete before each reader releases.
Default selection0 no longer has a WG rendezvous between MMA and per-key-slot
release: submitToNextAndTrigger retains its warp sync and all256 consumers each
issue release-arrive1, so the producer cannot reuse a slot before its last reader.
The Q-prefetch and exit WG syncs remain. The two alternative source files retain
their existing per-key WG syncs; this change was intentionally scoped to0.
After removal, all123 regression/codegen/reference tests passed. Racecheck
passed both N257/N1025 smoke cases (each tests both LSE directions), with zero
errors/warnings/hazards, in115.69s. Stack/spill stores/loads remain8/16/24B.
Logs: `build/no_k_wg_sync_{build,tests,racecheck}.log`. Thus the sync stays
removed, and default0 uses pre-MMA LSE seeding, predicated hard overwrite and
score finalization without row/seqlen comparisons. Tail identities remain.
Named barriers1/2/3 synchronize individual WGs, not a task-boundary CTA barrier.

The initial representative-arrival protocol passed numeric tests but produced
racecheck WAR hazards in the standalone pipe probe. Current correctness baseline
uses per-thread release arrivals: init256, producer32 threads each arrive8,
consumer256 threads each arrive1. The five-task standalone probe and five mixed
N257 summary cases then passed racecheck with no hazards. This comparison is
not a proof that representative arrival is inherently invalid; it remains an
independent optimization/debugging task. Do not silently suppress these hazards.

`expect_bytes` does not also arrive: the ring submit provides that arrival.
K metadata cp.async completion attaches to the **current K slot** semaphore,
not the prefetch semaphore. Metadata loads occur after the matching ready wait.

Shared memory holds only asynchronous input staging/metadata and communication
state; no intermediate score/affine tile is written to shared memory.

## Numerical contract and tests

- The sole path uses the existing v2 tanh fit and finite -1e6 sentinel. The new
  SASS-patched approximation is **not** implemented or substituted silently.
- Oracle comparisons use FP64 sequential affine composition. Exact tolerance
  for the additive first component is atol2e-4, rtol2e-5; comparisons of the
  tanh-based second component use atol.03 in base2.
  Unreachable finite sentinels are classified separately, not compared as
  ordinary real-valued scores. These are summary tests, not long-range passing
  error guarantees or end-to-end output/gradient acceptance.
- Coverage includes both fixed directions and per-B/H mixed directions,
  soft/hard/mixed rows, N1/31/32/65/256/257/513, N1024 matching/unmatched hard
  chains, single-CTA task reuse and normal grid dispatch.
- Triton embedding integration is checked against the SAME BF16 interpolated
  operands. v2's tuple uses destination names: out_q/LSE_q/idx_q originate
  from K logits, and out_k/LSE_k/idx_k originate from Q logits; the test adapts
  them explicitly. This does not conflate FP32 interpolation error with core error.

## Build and reproduce

From this directory, with conda blkw and CUDA tools in PATH:

```bash
python build.py
PYTHONPATH=python:.. python -m pytest tests/test_summary.py tests/test_summary_codegen.py tests/test_dism_v3_ref.py -q
PYTHONPATH=python:.. compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/test_summary.py -k 'sanitizer_smoke or pipe_probe or rv_load_probe' -q
```

Repeat the last command with racecheck/synccheck. The smoke tests include N257
and N1025 tails, two/five persistent workloads and both directions. The
larger numerical suite is separate from the reduced sanitizer selection.

The build targets sm120a so setmaxnreg is supported. Device ptxas verbose output
is enabled. CUDA13's cccl include path is supplied for ATen headers. Dependency
generation keeps diagnostics separate from the makefile dependency list and
tracks compiler options; failed targets return nonzero. Future binary patching
belongs between device-object generation and fatbinary packaging.

## Finite-only cleanup validation

Both split=0 (original32-row pipeline) and split=2 (sequential16-row pipeline)
passed119 tests after removing exact-mode parameterizations and adding mixed/
all-masked finite-sentinel scan tests. Each build emits a single summary kernel,
not two arithmetic specializations. SASS checks verify no CALL, one register
redistribution pair and native TMA. Resource usage is unchanged from the prior
finite specialization: split=0 has88B stack,96B spill stores,120B spill loads;
split=2 has8B stack,16B spill stores,24B spill loads. Logs are
`build/finite_cleanup.log` and `build/finite_cleanup_split.log`.
Split=2 passed10 selected tests under each of memcheck/racecheck/synccheck,
with zero errors/hazards (`build/finite_cleanup_*check.log`). The default build
remains split=0 and was restored after testing; arithmetic is finite-only in
every split configuration.

## Historical resource status before finite-only cleanup

SASS has no CALL; setmaxnreg dec40/inc232 and native TMA are verified. Static
allocation is168 registers/thread; this is not the consumer's232-register limit.
Current full/approx summary instances each report88B stack,96B spill stores,
120B spill loads. This remains a known resource limitation, not zero-spill
acceptance. Scan probes have zero stack/spill (116/106 registers full/approx).

TK ortho rv load originally wrote a lane-dependent dst[o_dim]. The isolated
32-element load optimized to registers, but the original complete kernel showed
dynamic local-memory accesses. The local TK primitive now broadcasts a loaded
scalar into compile-time register indices. This removes that source pattern;
it does NOT remove the full kernel's genuine ptxas-reported spills. The isolated
TK/direct probes now both use14 registers and no local memory. No external TK
or v2 source was changed.

## Historical stack bisection with line information (2026-09-28)

The former `DISM_SUMMARY_DIAGNOSTIC=<mode>` build option is now removed.
The following modes/results document the investigation, not supported builds.
Line tables use Clang's `-gline-tables-only`; inspect the device object using
`nvdisasm -gi build/object/wmma_tma_preprocess.cu.dev.sm120a.o`.
Modes 1--5 intentionally changed mathematics and the host launcher rejected them;
they are code-generation experiments, not benchmarks or correctness tests.
Diagnostic mode0 was the baseline; mode6 retained mathematics but changed scheduling.

Both exact and approximate instances gave the same following resource counts:

| Mode | Change | Stack B | Spill stores B | Spill loads B |
| --- | --- | ---: | ---: | ---: |
| 0 | Full computation | 88 | 96 | 120 |
| 1 | Replace scan with checksum consuming all score elements | 24 | 32 | 44 |
| 2 | Omit MMA; real metadata and scan | 0 | 0 | 0 |
| 3 | Omit score finalization; scan raw MMA | 0 | 0 | 0 |
| 4 | Constant Q labels/bias/hard flag | 0 | 0 | 0 |
| 5 | Constant K labels/bias | 0 | 0 | 0 |
| 6 | Load K metadata after MMA, before slot release | 48 | 60 | 76 |

The full mode-0 PTX has **no `.local`, `ld.local`, or `st.local`**. Thus its
remaining stack is introduced by ptxas register allocation, not a surviving
Clang local array. This conclusion applies to the current fixed-index rv load;
it does not retroactively classify the original rv implementation's stack.
All seven cubins have no CALL instructions.

Line-mapped full SASS shows fixed stack offsets for ring arrival/control state,
scheduler values, precomputed shared-load addresses, and output addresses.
For example in the initial mode0.sass, PC0x0470 saves the arrival count, and
PC0x2f40/0x2f70/0x2fb0 save scheduler-derived values. Multiple shared-address
slots are saved during the consumer prologue then reloaded near Q input reads.
These are examples, not an assertion that all stack bytes belong to metadata.
ptxas hoists instructions; a spill's nearest source line is not by itself the
location of peak pressure, nor does attribution to setmaxnreg imply it is broken.

The controlled mode6 reduction supports overlap between K metadata and MMA
operands as a contributor. Full pressure also includes persistent Q, score,
scan state, and control/address values; no single isolated operation has been
proved responsible for all88B. Dead-code elimination changes other lifetimes
in modes1--5 (notably mode4's hard=false also removes hard-label comparisons),
so their stack reductions must not be treated as additive byte attribution.

Local artifacts are retained under `build/stack_bisect/`: mode0--6 build logs,
cubins, line-mapped SASS, and full mode0 PTX. These build artifacts are not Git
sources. The48B diagnostic variant is no longer retained in active source.

Mode6 and restored mode0 each passed all114 summary/probe/codegen/reference
tests. Restored mode0 still reports88/96/120B. The currently built extension is
mode0 with line tables; its up-to-date source mapping is `restored.sass`.
The preceding normal-path sanitizer run passed the three selected tests under
memcheck, racecheck and synccheck, with zero reported errors/hazards. Sanitizers
were not rerun for mode6, which is not selected by default.

## Sequential 16x64 halves (2026-09-28)

`DISM_SUMMARY_SPLIT_SCAN=0/1/2 python build.py` selects:

- 0 (default): original 32x64 MMA/score/reduce.
- 1: retain the 32x64 MMA and score, then reduce upper/lower 16x64 halves.
- 2: upper 16x64 MMA/score/reduce, then lower 16x64 MMA/score/reduce.

All variants still emit one32-row affine summary. Two16-row VStates persist
across key tiles. Within each key tile the upper half starts with identity
HState, and its bottom HState seeds the lower half. In particular the lower
half's shifted VState retains the upper half's previous-tile corner. No layout
conversion, extra shared intermediate, global boundary, or inter-warp mail is
introduced. Padding identities are applied with each half's actual query offset.

Mode2 retains the32-row Q operand across the key loop, but constructs only a
16-row score and scan at a time. The upper half's tile is dead before computing
the lower half; only its boundary survives. K is read via LDSM twice per tile,
and metadata is loaded separately after each half's MMA. Both halves finish
before the WG releases the K ring slot. This deliberately conservative protocol
may reduce producer overlap; less spill alone does not establish faster runtime.
The shared allocation, TMA traffic, register redistribution and CTA scheduling
are unchanged. All compile-only DIAGNOSTIC code has since been removed.

| Full kernel | Exact stack / spill stores / loads B | Approx stack / stores / loads B |
| --- | ---: | ---: |
| Original32x64 | 88 / 96 / 120 | 88 / 96 / 120 |
| Split scan only | 112 / 124 / 148 | 136 / 144 / 168 |
| Full sequential half pipeline | 8 / 16 / 24 | 8 / 16 / 24 |

The remaining two fixed stack slots in mode2 hold ring arrival count and
scheduler state (line-mapped SASS PC0x0470 and0x0680/0x0c00); this is not a
zero-spill result. All instances remain static168 registers with consumer
inc232/producer dec40, native TMA and no CALL. Independent split scan probes
use113/72 registers (exact/approx), versus116/106 for the32-row probe; all have
zero stack/spill. Full-kernel codegen, not these isolated numbers, is decisive.

The independent probe checks64/128/320/1024 columns against both FP64 sequential
composition and the original32-row scan. Both full split variants passed122
summary/probe/codegen/reference tests, including tails, mixed directions,
hard/soft rows, persistent workload reuse and Triton-interpolated inputs.
Local logs/cubins/SASS are under `build/split_scan/`; default remains0 pending
a throughput comparison. No commit was made.
Mode2 additionally passed the10 selected scan/pipe/summary smoke tests under
each of memcheck, racecheck and synccheck, with zero errors/hazards. After these
checks the extension was rebuilt as mode0, so ordinary build/import behavior
remains on the original baseline; select2 explicitly to repeat the experiment.
