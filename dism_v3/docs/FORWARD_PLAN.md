# v3 output kernel bring-up

Current dimension extension: D and DV independently support build-time32/64/128,
R remains32, default tanh/shared/BF16. D128/DV128 uses one input slot for capacity;
other pairs retain two. See FORWARD_DIMS.md. The bring-up history follows.

User-approved first shape: V channel64 and soft readout R32. KcKeyDim continues
to support32/64/128. Reuse the summary score helpers, K TMA permutation, hardened
MbarrierRingPipe and scalar-roll-before-duplicate construction.

## Mapping and computation

- Eight consumers in two WGs, plus the producer WG: logical16-row blocks are
  assigned to warps0,4,1,5,2,6,3,7. Each consumer scans16x64.
- Each WG0 warp reads the passed32-row boundary above its pair (empty for the
  first pair). Its prescan horizontal output goes to its paired WG1 warp.
  Synchronization is WG-wide, not four independent warp-pair barriers.
- Run inclusive_prescan, publish its completed boundary in the shared mailbox,
  then run inclusive_postscan with the ORIGINAL incoming HState and saved
  intermediate. Keep the first affine component dead when unrolling W.
- Online softmax includes the fixed zero-score/zero-value fallback. Soft QK
  readout multiplies only the numerator: exp(W-max)*(sq @ sk.T) @ V. Compute the
  denominator before applying readout, which can be signed. Apply causal and
  sequence masks for output probabilities even though summary omits causal mask.
- Inputs stage asynchronously; buffers cannot be released until their final
  MMA reader completes. No dense logM/W/probability allocation in production.
- Initial implementation prioritizes correctness, not tuning/benchmark claims.
  Validate mailbox handoff, padded final rows, both directions, soft/mixed/hard,
  all three KcKeyDim values, and complete summary→passing→output versus oracle.

## Checkpoint coverage correction

The old implementation incorrectly skipped the whole final256-row workload.
Correct contract is S=floor((N-1)/32). Only the warp block containing token N-1
and later blocks are skipped; earlier warps in that CTA must produce summaries.
This supplies all entry boundaries for independent output warp pairs, including
the final CTA. No special serial final-workload dependency is needed.

The corrected tail CTA still runs the producer/consumer pipe protocol. Invalid
warps acquire/release every K packet and participate in prefetch/retirement WG
sync, but skip key loads, MMA, scan and summary stores. make_scan still never
sees a partial32-row warp tile.

## Independent scan probe status

`src/probe/forward_scan_probe.cu` and `tests/test_forward_scan.py` validate16x64
prescan/postscan, nonempty incoming HState, cross-key-tile VState and unroll
layout, through320 columns.24 tests passed for tanh and an exact-LSE diagnostic.
The mathematical output and early boundary independently satisfy the existing
summary approximation tolerance. With exact LSE, early and final bottom rows
also agree within2e-5. Tanh's different composition orders gave up to~0.0016
difference between them; the initial overly strict equality check is retained
in `build/forward_probe_tests.log`, not attributed to a layout defect.
The first output implementation is now in `src/forward.cu`, with the Python
three-kernel entry point `flash_dism.forward.forward_core`. R32 and DV64 are
independent constants. The implementation uses the existing score helpers and
permuted TMA inputs, WG-wide early boundary publication, and online softmax.
K/SK/V have two input slots; Q/SQ share their storage across workloads and are
not overwritten until all readers release it. Only asynchronous inputs and
required WG mailboxes use shared memory; scan/readout intermediates remain in
registers. Cross-workload first-K co-prefetch and performance tuning are pending.
Only the combined K/SK/V pipe remains. Output defaults to direct global stores;
DISM_OUTPUT_SHARED=1 optionally adds warp-private output-layout staging.

Initial D64 validation: `build/forward_tests.log`,64 passed, covering both
directions, soft/mixed/hard, nonaligned tails, signed/zero readout, fallback,
repeated persistent tasks and SASS. `build/forward_build.log`:168 registers,
zero stack/spill and no CALL. Output is FP32; signed normalized weights are
rounded to BF16 for a single PV MMA. Output comparisons retain atol=.008,
rtol=.025 and relative L2<.01. R32 results are not evidence of R64 resource use.
The first implementation is tanh-only; ex2_patched output is rejected rather
than accidentally applying a softplus patch to normal softmax exponentials.
The explicit labels/hard-mask API is bring-up only, without autograd. D32/D128
output validation and real Triton interpolation integration remain pending.

## Tail validation

`build/warp_tail_*` records the coverage correction. D128 full regression:
397 passed,2 skipped. Ten partial-CTA cases cover1..7 active consumer warps,
including N33 where WG1 is entirely inactive and N500 with multiple workloads.
Memcheck and synccheck: zero errors. Racecheck reports48 hazards with summary
ARRIVES `[UR5+8]`; accepted under the existing user-approved immediate-address
policy, not presented as a clean racecheck. Raw SASS/logs are retained.

Final coverage regression: D32/64/128 each397 passed,2 skipped;
`build/check_dims_20260929_004852_9n76t7cd`, driver `build/warp_tail_dims.log`.
Restored shared/D128/KStages3/tanh. The independent forward probe's approximate
mode is currently tanh-only; exact mode is a diagnostic, not a production option.
