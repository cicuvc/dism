# BF16 O TMA epilogue / KV1 reuse (2026-09-09)

Opt-in `DISM_OUTPUT_TMA=1` requires conservative `DISM_OUTPUT_Q_ALIAS=kv`.
Only D64/DV64 changes; default remains direct stores. This concerns the main
OUTPUT kernel's BF16 O, NOT summary's FP32 affine checkpoints. Summary is
unchanged. Uses the existing TK TMA store atom and commit/read/full-wait
primitives; no external TK/GLX edits.

## Storage and protocol

At the workload boundary, after all current KV readers finish:

| Storage | Concurrent contents | Bytes |
|---|---|---:|
| old KV2 / Q union | next Q,128x64 BF16 |16384|
| KV0 | next K0 and V0,each64x64 BF16 |16384|
| old KV1 | current O,128x64 BF16 |16384|

Input/output data stays48KiB. Two extra uint64 mbarriers and alignment add
128B to the current struct layout:55552B dynamic shared versus55424B direct,
plus1024B driver shared, below64KiB. The output region is a single128-row
buffer, not an additional independent allocation or a double output buffer.
Its shared writes are the explicitly authorized output-layout conversion.

- Both compute WGs still arrive on done256 after their last PV. Before
  writing O, every consumer waits for the current done epoch; a faster WG
  cannot overwrite KV1 while a slower WG is still reading it.
- Each consumer normalizes its FP32 accumulator and writes adjacent BF16
  pairs directly into the swizzled O tile, preserving interleaved logical row
  ownership. No extra register tile, FP32 shared output, or BF16 re-rounding.
- Each writer performs async-proxy fence then arrives on oready256. Warp9's
  elected thread waits for this complete epoch and submits one128x64 store
  using the same row/swizzle map as Q, targeting O.
- The TK store atom commits. Warp9 waits for source-read completion then
  publishes ofree1. Full completion is drained before producer WG exit.
- Warp8 waits for old done and issues next Q/KV0 without waiting for ofree.
  Only the first KV1 load of the next workload waits for old ofree. Its later
  ring reuses use the existing KV free epochs.
- Consumers also wait for old ofree before writing their new O, covering
  N<=64 workloads that never load KV1. All new epochs are per workload, not
  per key tile. Q/KV and WG-mail phase conventions remain unchanged.
- Tail global O writes are bounded by the TMA tensor-map N dimension. Padded
  shared rows are not output; valid rows and adjacent sequences are checked
  by direct-store replay. Normalizer and scan-boundary stores remain unchanged.

This permits next Q/KV0 transfers to overlap O layout/store; it does not
claim timing traces have proved how much overlap occurs. KV1 prefetch must
wait for O read completion, and consumer done waits can add overhead.

## Validation

- Full existing core:133 passed; workload replay:12 passed.
- New direct-store vs TMA-store bitwise O/normalizer/vertical/horizontal
  comparison:60 passed in each of full and finite. Covers N1/31/64/65/127/
  128/129/257/385/1024, both directions, probability0/.37/1, more sequences
  than SMs, repeated calls and partial output tiles.
- Finite replay/bitset/label suites:129 passed.
- Selected end-to-end autograd:507 passed in each mode (398 deselected).
  These are wiring/training/replay cases, not a claim to fix the previously
  documented strict-oracle precision limitations. No tolerance changes.
- Codegen gates pass in both modes. All10 D64/DV64 output specializations
  have zero stack/local spill and no CALL, one UTMASTG.5D, three UTMALDG.5D;
  dec40/inc232 remains. New guard checks commit and DEPBAR before ofree arrive.
  Initial test incorrectly expected a SYNCS.WAIT opcode; inspection showed
  actual TK wait lowers to DEPBAR.LE, and the assertion was corrected.
- Full replay memcheck/racecheck/synccheck:10 cases each, zero errors/hazards.
  Racecheck took173.47s. No added CTA-wide barrier; WG exits remain independent.

Resources: `benchmarks/output_tma_codegen_sm120a.json`.

## Performance

RTX5090, CUDA13.1, sm120a, B64/H4/D=DV64/V512, BF16, scale1, rtau3,
actual CUDA embedding inputs, tanh_finite and lineinfo enabled. Same-stream
ordinary forward launches, no concurrent GPU tests; CUPTI OUTPUT-only times,
not embedding/core-total or CPU throughput.20 warmups,30 measured launches
per subprocess. Mixed N1024 uses three rounds with reversed A/B order in
round2,90 samples per variant. Other cases are single-round probes.
No clock locking; all samples retained. Pooled launch medians:

| Direction | hard_prob | N | direct us | TMA O us | Time change |
|---|---:|---:|---:|---:|---:|
| q_from_k |.5|1024|406.222|410.576|+1.07%|
| k_from_q |.5|1024|447.198|451.038|+0.86%|
| q_from_k |0|1024|443.213|437.486|-1.29%|
| k_from_q |0|1024|440.462|442.894|+0.55%|
| q_from_k |.5|65|72.128|65.568|-9.09%|
| q_from_k |.5|257|115.792|108.799|-6.04%|

Raw: `benchmarks/output_tma_timing_sm120a.json`. Mixed main workload is
slightly slower in all three rounds for both directions. Keep default0;
do not infer an automatic short-N dispatch threshold from one probe round.
No new NCU report was collected, so no attribution of the regression to a
specific synchronization, scheduling, or memory bottleneck is established.

```bash
DISM_OUTPUT_TMA=1 DISM_TILE_LSE=full DISM_LINEINFO=1 /home/cicuvc/miniconda3/envs/blkw/bin/python -m pytest -q tests/test_dism_v2_output_tma.py tests/test_dism_v2_codegen.py
DISM_OUTPUT_TMA=1 DISM_TILE_LSE=tanh_finite DISM_LINEINFO=1 /home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.benchmark_persistent_summary --kernel output --direction q_from_k
```

Use a separate process with `DISM_OUTPUT_TMA=0` for the performance baseline.
The equivalence suite loads both module variants explicitly in one process;
the production extension remains process-configured. Experiments are uncommitted.
