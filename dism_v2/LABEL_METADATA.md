# Label width and endpoint specialization

The high-level autograd path now keeps embedding labels as int32, removing the
two int64 casts. Low-level core/backward accept matching int32 or int64 labels;
int64 is not truncated. Args carries a width tag for output/backward; summary
uses host-dispatched label types, without per-element width branches.

Summary is specialized for soft/hard/mixed probability. Soft does not read
labels or generate row RNG; hard does not read LSE or generate row RNG. Mixed
retains the bitset/recompute choice. There are30 summary variants (D x direction
x [soft + hard/mixed x two widths]) and9 output variants. This replaces the
older six-summary codegen counts. Q/K TMA and WG mailbox protocols are unchanged,
including the user's load_rhs_tile implementation. No new shared staging.

Validation on RTX5090, CUDA13.1, blkw:

- full:251 tests pass (133 core,62 bitset,55 label-width,1 codegen).
- tanh_finite:118 tests pass; repeated with DISM_LINEINFO=1 under memcheck,
  zero errors. Includes nine D/DV combinations, both directions, endpoint/mixed
  probabilities, exact int32/int64 summary/output/edge equality and a2^32 label
  difference that must remain a hard mismatch in the int64 path.
- All forward variants have zero stack/local and no CALL, native TMA and
  register reallocation retained. int32 summaries have no LDG.E.64; int64 hard
  and mixed summaries retain four label LDG.E.64 sites. D64/DV64 WS backward
  has zero stack/local. Other backward-shape spill tuning is outside this change.

Initial same-binary CUPTI comparison, B64/H4/N1024/D=DV64/V512, actual CUDA
embedding, scale1,rtau3,hard_prob=.5,q_from_k,tanh_finite,20warmups/30samples:
int64 median236.2385us, int32 median221.151us (about6.4% shorter). One round only,
unlocked clocks, not an end-to-end speedup claim. The old RNG/bitset comparison
and old pre-load_rhs performance numbers are not controlled baselines for this.

Reproduce with `python -m dism_v2.benchmark_persistent_summary --label-dtype int32`
or `--label-dtype int64`, setting DISM_TILE_LSE before process start. NCU target
profile_forward_summary also accepts --label-dtype (default int32).

Register double-buffering of next-tile metadata and summary-store layout
repacking are deferred pending the new source-correlated profile. Endpoint
performance and complete-path performance have not yet been measured.

Source-correlated NCU (`DISM_LINEINFO=1`, --import-source yes, full40passes,
skip10/count1, cache/clock control none):
`/tmp/dism-summary-int32-lineinfo-q.ncu-rep`. Duration230.40us, SM39.52%,
Tensor38.3%, DRAM33.19%, branch efficiency99.56%, no-eligible51.59%, zero spilling
requests. SM frequency2.65GHz. Each of the two key-label load sites requests
589824 sectors, exactly half the1179648 in the earlier int64 report. Loads
remain ideally coalesced. The summary store still accounts for73728 excessive
sectors; its layout was not changed.
