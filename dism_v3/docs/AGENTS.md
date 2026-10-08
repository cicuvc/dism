# v3 operator contracts

- Oracle: python/flash_dism/reference/dism_v3_ref.py. SQ/SK changes only the
  numerator. No delta gate in v3. Do not change reference semantics.
- Vectors BNHD, scalar labels/LSE BHN, vocabulary HVD. Fixed N and every
  cu_seqlens boundary must be256-aligned. Reject rather than pad misalignment.
- R16/32, D32/64, DV32/64; fixed and varlen. Separate configuration translation
  units. BF16 outputs, FP32 accumulation and cross-CTA atomics by default.
  FP32 diagnostics/debug output remain compile-time options.
- Embedding returns natural-log LSE-tau; apply log2 conversion exactly once.
  Finite -1e6/tanh tile softplus is intentional; chunk passing uses full LSE.
  Never pre-scale BF16 Q/K by LOG2E. Preserve RNG replay and per-head seed salt.
- Summary skips only the32-row warp block containing the final token and later
  blocks, not earlier warps in its CTA. Preserve inclusive scan/boundary layout.
- Preserve WG-level sync, pipeline phases and output buffer fences. TK coords
  are raw offsets. Roll scalar scores before affine tuple duplication.
- src/probe remains opt-in under DISM_BUILD_PROBES=1 for its unit tests.
- Default relaxed gradients and separate strict_gradient diagnostics remain.
  Compare cosine/norm/signed bias; BF16 delta/tau limitations in BACKWARD_DIMS.md
  must stay visible, not hidden through looser acceptance or oracle substitution.
- Module/cache/no-decay: MODULE.md and MODEL.md. Varlen: VARLEN_IMPLEMENTATION.md.
  Compiler contracts: DYNAMO_ASSESSMENT.md. No production behavior changes here.
- Racecheck: only documented ARRIVES [UR+nonzero immediate] reports may be
  classified as suspected tool issues, with evidence and explicit warning.
  Never extend this exception to memcheck, synccheck or numerical failures.
- Docs live here; tools contains only required test helpers. Preserve user edits.
- SAM full-hard inference lives in python/flash_dism/inference, including native
  csrc. It is shared inference infrastructure, not v4 training. Its explicit
  prefill/decode APIs accept arbitrary positive lengths; the 256-alignment rule
  above applies to training CUDA operators. Keep model/cache dispatch explicit,
  preserve complete-history prime, and test rebuild/graph boundaries. Wheel
  builds include both SAM native modules; checkout JIT must use a user cache.
