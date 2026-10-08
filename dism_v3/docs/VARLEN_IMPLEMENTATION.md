# Native varlen implementation

Current registry covers R16/32 x D32/64 x DV32/64 in one extension;
see MULTI_CONFIG.md for build, dispatch and the latest validation. The initial
implementation below was brought up at R32/D64/DV64,
batch1 packed tokens with cu_seqlens. New CU files only; fixed-length bodies
remain untouched by the varlen implementation (the preceding default-path cleanup
is a separate change). Forward, backward, autograd and document isolation are
implemented and tested. This first version prioritizes correctness, not throughput.

## Direct packed layout (alignment-copy removal)

- One CPU-validated cached VarlenLayout from int32 cu_seqlens; exact-size
  allocation currently requires a host synchronization at layout construction.
- All cu_seqlens boundaries, including terminal T, must be multiples of256.
  Matrix inputs Q/K/SQ/SK/V and dO alias packed `[T,H,C]` storage: no input
  alignment allocation, padding fill or vector-copy launch. Scalar metadata
  remains global head-major `[H,T]`, with no document-major conversion. Empty documents
  consume no space/tasks. Historical `Padded*` field names now equal their
  unpadded counterparts and are not evidence of a second allocation.
- Separate SoA checkpoint families sized by sum(checkpoints_s*P_s), not T².
  Offsets are64-bit and interpreted per head; each document's full H block is
  contiguous. All six phases use one common by-value parameter object;
  matrix tensor maps cover the whole pack, scalar/checkpoint views are flat;
  only integer document/task metadata remains in global arrays.
- Native CUDA task lists address `(sequence,local_workload)` in one launch
  per phase, not invoke fixed-length entrypoints separately per document.
  Initial scheduling uses one CTA per workload; persistent cross-document
  scheduling is a later performance step, not a correctness prerequisite.
  Within each CTA, reuse the validated warp specialization, ring protocols,
  GLX scan, GEMMs and shared/TMA gradient writeback.

## API and limits

Use `flash_dism.varlen.VarlenLayout.from_cu_seqlens(cu_seqlens, T)` once per
pack, then pass `layout=layout` to `dism_core_varlen` or `voc_dism_varlen`.
The latter reuses the existing Triton embedding interpolation and its backward.
Vectors are BF16 `[1,T,H,C]`, with R16/32, D32/64, DV32/64. Core LSE is FP32 `[1,T,H]`,
labels and explicit hard flags are `[1,H,T]`, direction is `[1,H]`, tau is `[H]`.
Vocabulary and sq/sk activations remain caller-owned. This does not implement
new RNG generation or per-document directions.

The256-token contract reuses summary's256-row metadata reads and all forward/
backward tile boundaries without tail staging. Requiring terminal T as well
prevents the final workload reading past the allocation. Misaligned boundaries
fail explicitly in both Python and C++; no automatic padding/copy fallback.
Empty documents, zero total tokens and nondefault streams are supported.
Nonaligned lengths are no longer supported by this varlen entrypoint;
fixed-length kernels retain their existing tail support. CUDA Graph capture and
second derivatives are explicitly unsupported. Cached layout removes repeated
cu_seqlens synchronization. Integer document/task uploads still happen on every
call; no phase constructs/uploads per-document descriptors. Kernel parameters
contain the common tensor maps, not pointers to a global Args array.

Normalizer/delta/LSE-gradient storage is global `[H,T]`, exactly H*T elements.
Each document starts at head*T+begin. T and all starts satisfy scalar TMA alignment.
Private gradients use BF16. `fp32_output=True` requires explicitly compiling
the optional validation instances, disabled in the default multi-config build.
Cross-CTA gradients use FP32; dtau uses scalar atomic, never TMA.

Checkpoint space scales with the sum of per-document checkpoint rectangles,
not a rectangle over the full packed T. `layout.checkpoint_bytes(H)` reports
the W plus backward-summary/boundary budget, excluding operands and gradients.
For H16 and16 documents of512 tokens (T8192), that budget is47MiB, versus
767MiB if the same pack were treated as one fixed-length8192-token document.
These are allocated checkpoint capacities, not measured whole-operation peaks.
There are no private input padding regions or production pack/unpack launches.
LSE transpose/contiguous, tau absorption and metadata dtype conversions remain;
this removes document-major reordering, not all Python preprocessing operations.
The six computational phases use native task lists, not Python per-document
fixed-length calls. Delta remains an auxiliary launch. Legacy pack/unpack
utilities remain available only for isolated layout diagnostics.

## Head-major scalar metadata (2026-09-29)

The document-major scalar conversion is removed from production. Forward uses
head*T+begin for labels, hard flags and absorbed LSE; normalizer, delta and LSE
gradients use the same addressing. Recompute preserves document-local checkpoint
strides, compensating its scalar base separately. No kernel mathematical changes
or tolerance changes. This removes five pack launches and three unpack launches
per complete forward/backward call; remaining transpose/casts/tau subtraction
are not claimed eliminated. No throughput measurement is claimed here.

Validation logs in build/:
- no_pack_tests.log:868 passed,279 skipped,78 deselected.
- no_pack_{memcheck,racecheck,synccheck}.log:each10 passed across eight shapes
  plus nonuniform-LSE and no-pack tests; zero errors/hazards.
- production_only_tests.log:567 passed,581 skipped,78 deselected after probe
  exclusion. Missing optional probes/FP32 instances are explicitly skipped.
- probes_enabled_tests.log:default R32/D64/DV64 probe build329 passed,9 skipped.
- production_restore_tests.log:109 passed after restoring all eight shapes with
  probes disabled; includes oracle/autograd, codegen, public API and varlen execution.

Existing spills are retained. At R32/D64/DV64, varlen F1/F3/B1/B3 report
16/16/304/192 bytes stack per thread (previously16/24/304/192). This is a compiler
resource observation, not a performance result. Main kernel codegen tests retain
the no-CALL check.

## Historical evidence before alignment-copy removal

The following counts and seed diagnostics describe the previous arbitrary-start
staging version. Current aligned tests use explicit aligned fixtures and reject
misaligned layouts; see the direct-path validation section below. Tolerances
are unchanged.

- Layout/packing and descriptor-array probe:50 tests passed. Probe covers
  Rows32/64, H1/3, nonaligned token starts, document lengths1/17/65/129/257/513,
  empty documents, and one full TMA tile beyond each private padded extent.
  Adjacent padding has distinguishable sentinels; beyond-extent reads zero-fill.
- All six stages have separate `src/varlen_*.cu` implementations. Tests cover
  forward and all eight core gradients against independent fixed-length calls,
  real embedding autograd including vocabulary gradients, empty documents,
  document permutation/isolation, H1/2/8, and lengths through4097.
- Final default regression:936 passed,21 skipped,78 strict tests deselected,
  **one failure retained**, described below. No tolerance or xfail changes.
- All eight varlen device objects have no CALL. Packing uses typed raw elements
  and checked int32 per-document indices to avoid a 64-bit division helper.
- FP32 forward output plus compile-time backward debug build:100 selected
  varlen layout/forward/backward/autograd/execution tests passed. Final build
  is restored to BF16 output and debug disabled.
- Existing spills remain: F1 stack24B, B1 stack384B, B3 stack344B; F3 has no
  stack/spill in this default build. This is not
  a spill-free or performance-tuned implementation.
- No complete score/W/attention matrix is materialized by production kernels.
- Final memcheck:84 tests, zero errors; synccheck:28 tests, zero errors.
  B1/B3-filtered racecheck:19 tests, zero hazards. F1-filtered racecheck:9
  numerical passes and30 hazards; F3-filtered:10 numerical passes and42 hazards.
  Both F1/F3 have `ARRIVES.LDGSTSBAR [UR23+0x8]` in their own SASS, so these
  retain the user's accepted ARRIVES-immediate warning classification. They
  are **not zero-hazard results**. Raw logs:build/varlen_final_*check.log.

### Retained numerical limitation

FP64 comparison covers48 real-interpolation cases (eight seeds, soft/mixed/hard,
tau2 or ln64).47 pass. Pure-hard seed7 at ln64 has tau cosine0.94233 versus the
unchanged0.95 threshold, norm ratio0.74322. On identical inputs, fixed-length
reproduces this to8.2e-8; substituting oracle delta only for diagnosis restores
cosine approximately1 and norm ratio1.0000002.

Seed0 also exposes a weak individual tau head sign flip: reference-0.00649185,
computed+0.00129913, although aggregate tau cosine0.99219 passes. Fixed-length
reproduces it; oracle delta removes the flip. This is the inherited BF16 O/delta
precision limitation, **not evidence that each tau head's sign is reliable**.
Production delta remains unchanged. The old failing fixture has unaligned
document lengths and is now rejected by the API; its raw results remain in
build/varlen_delta_diagnostic.json and build/varlen_delta_seed0.json. The analysis
scripts now use aligned fixtures, so their seed numbers no longer reproduce
these exact historical values. This interface change is not a numerical fix.

A24-step correlated AdamW readout smoke reduced loss0.01938 to0.01252.
Vector gradient cosine was at least0.9999759 and norm ratios0.99732–1.00186;
signed projection errors were not one-sided on every step. This short test is
not a claim of long-run unbiased gradients or LM training stability.
See `tools/analyze_varlen_bias.py` and `build/varlen_correlated_bias.json`.

## Direct-path validation

- Full default regression:940 passed,21 skipped,78 strict tests deselected.
  Alignment checks cover Python and low-level C++; vector alias tests verify
  identical pointers and forbid vector packing launches in forward/backward.
  Document lengths256/512/768/1024/2048/4096, empty packs/documents, H1/2/8,
  independent-document comparisons, real interpolation/autograd, and SASS
  no-CALL checks pass. Log:build/varlen_direct_full.log.
- All48 aligned FP64 cases pass unchanged acceptance thresholds. Vector
  gradient cosine >=0.9999950, norm ratios0.999596–1.000407. Tau remains less
  accurate: minimum cosine0.95880, norm ratios0.68362–1.17719, three individual
  head sign flips across the48 cases. Aggregate acceptance does not guarantee
  individual tau signs. Full metrics:build/varlen_direct_accuracy.json.
- No gradient tolerance was changed. Historical arbitrary-start cases have
  been replaced by explicit aligned fixtures, not silently rounded by the API.
- Direct-path memcheck:84 tests, zero errors; synccheck:26 tests, zero errors;
  B1/B3-filtered racecheck:19 tests, zero hazards. Logs:build/varlen_direct_*check.log.

## Reproduction

```bash
PATH=/usr/local/cuda/bin:/home/cicuvc/miniconda3/envs/blkw/bin:$PATH DISM_LINEINFO=1 python build.py
PYTHONPATH=python:.. /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests
```

The current aligned default suite passes; the historical BF16 O/delta tau
limitation remains. FP32 output and compile-time debug support are retained;
their historical100-test run predates alignment-copy removal.

Logs: build/varlen_foundation_* (ignored). New descriptor probes use global
memory tensor-map records; do not assume that by-value grid-constant arguments
are the only valid tensor-map storage location.
