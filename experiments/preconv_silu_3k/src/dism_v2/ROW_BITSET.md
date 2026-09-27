# CUDA embedding row-bitset experiment

Enable `DISM_ROW_BITSET=1` with `embedding_backend="cuda"`. Default is0 (Philox
recompute). Mixed hard probability only: 0/1 do not allocate or load bitsets.
Triton embedding retains the original RNG path. Independent core/backward APIs
accept optional `hard_bits`; callers must associate it with the same replay state.

Storage is contiguous int32 `[B,H,ceil(N/32)]`, interpreted as unsigned words.
Bit `r%32` in word `r/32` represents logical query row r. Tail bits are zero.
At B64/H4/N1024 this costs32KiB per voc_dism, retained for backward. No full
random floats, scores, or attention matrix are stored.

The host reserves the existing Philox seed/offset before embedding. Random
direction uses its original prefix4 words; mixed rows use the following4 words.
Logical row subsequences and FP32 probability comparisons are unchanged. Replay
reserves nothing and regenerates the identical bitset. The core receives replay
metadata, so it does not reserve a second time. No extra GPU launch is introduced.

CUDA embedding warp0 generates64 logical query-row decisions after its output
stores and ballots them into two words. `elect.sync` selects the store issuer;
there is one writer per word, no atomic or shared staging. The other WG does not
generate duplicate decisions. Consumers use ordinary early scalar loads:
one word per16-row forward warp before the key loop, two words per64-query
backward tile before waiting for Q/dO readiness. The old RNG branch remains
available within the same binary; no compile-time bitset specialization yet.

## Validation (RTX5090, CUDA13.1, blkw)

- Default full LSE:133 existing core tests +62 bitset tests +1 codegen test,
  all196 pass. This also exercises the original no-bitset core path after ABI changes.
- `tanh_finite`: 62 bitset tests + forward codegen pass. Covers D32/64/128,
  N1/17/31/32/33/63/64/65/129/257/1025 for packing; all nine D/DV and both fixed
  directions plus random for end-to-end. Bits match independent CPU Philox,
  embedding/core outputs are identical, six gradients meet existing replay
  tolerances (atomic gradients need not be bit-identical), generator offset and
  selected direction agree. Endpoints allocate no mask and consume no row RNG.
- Three training-backward smoke tests pass with the experiment enabled.
- Memcheck: all62 tests, zero errors. Racecheck and synccheck:33 embedding
  packing tests each; no hazards/synchronization errors.
- Forward codegen remains zero spill/no CALL/native TMA/setmaxnreg. Embedding
  fused variants D32/64/128 (including supported BV128) have zero stack/local,
  no CALL. Backward SASS has no CALL; measured D64/DV64 WS dV and dA/dB both
  have zero stack/local. Existing other-shape backward spill work is not changed.
  Current tanh_finite WS stack bytes, DV32/64/128 respectively:
  dV D32=0/0/80, D64=0/0/144, D128=8/0/120;
  dA/dB D32=8/0/0, D64=8/0/8, D128=48/80/232.
  This includes small8B stack instances; an exact pre-bitset backward binary
  resource comparison was not retained, so this is not a claim of unchanged
  per-shape spill amounts. No spill tuning was attempted.

## Initial performance

`DISM_TILE_LSE=tanh_finite python -m dism_v2.benchmark_row_bitset` (also run
`--direction k_from_q`). B64/H4/N1024/D=DV64/V512, scale1, rtau3, hard_prob=.5,
all-CUDA embedding+core forward/backward+embedding backward, hot inputs, CUPTI,
10warmups/20samples, orders off/on then on/off. No concurrent GPU benchmarks.
Kernel medians below are averaged over the two rounds, in microseconds:

| Kernel | q_from_k off/on | k_from_q off/on |
|---|---:|---:|
| Embedding forward | 732.69 / 733.84 | 733.12 / 733.73 |
| Forward summary | 231.02 / 230.56 | 249.97 / 250.58 |
| Forward output | 837.37 / 839.59 | 963.70 / 885.68 |
| dV + summary | 2318.52 / 2286.55 | 2288.86 / 2255.16 |
| dA/dB | 3300.18 / 3314.45 | 3024.37 / 3007.62 |

Mean summed GPU durations per full iteration (including ancillary kernels),
averaged over rounds: q_from_k14.515ms off vs14.551ms on;
k_from_q14.362ms off vs14.290ms on. Individual round means vary substantially;
no established full-path speedup. These exclude host gaps, are not 3-layer model
training throughput, and are not a clock-locked statistical benchmark.
Raw per-round/per-kernel data: `benchmarks/row_bitset_sm120a.json`.

Removing repeated RNG instructions has not translated into a large backward
speedup. Whether RNG was overlapped with loads, or other bottlenecks dominate,
requires a new profile; no specific cause is claimed from timings alone.
Keep the experiment opt-in pending further profiling/specialization decisions.
