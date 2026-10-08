# Curated v3 + nanochat branch

Worktree: `/home/cicuvc/cs/project/dism-core`, branch `core/v3-nanochat`.
Original `dism-exp` workspace remains on main with its user's uncommitted edits.
The completed prefill work was committed on main as `7687cf5` before cleanup.

## Sources and exclusions

- v3 operator source: nested repository commit
  `47c8875bbf56cc7f365dad20c1154b452cde35f2`, imported as ordinary files.
  Original dirty docs were not overwritten or silently included.
- nanochat: remote `/root/autodl-tmp/dism-randomness-code`, selected
  `randomness-sequence_true-786m`, not the local training framework fork.
  Original file hashes are preserved in `nanochat/source_manifest.json`.
  All 77 downloaded Python files listed in the original manifest matched it.
  Manifest SHA256:
  `61f49b6385b8835ff865fb45e8bbc1046f875f244f38be96fd6cfa1a88e6f254`.
- Selected unchanged model JSON SHA256:
  `f3d9ba74903ebccc52049d8b64acf6649753f6c8a8eec38d6f3a0f67e3c9e26b`.
- GDN/pure-GDN module source and postnorm block semantics are imported from the
  same remote snapshot. Its per-token fused CE evaluation entrypoint is also
  retained. v3 operator math, launch defaults and gradient tolerances are unchanged.
- Remote v4/hashed/triple branches are omitted, not silently substituted in
  the selected baseline. Non-baseline readout options are explicitly rejected.
- v1/v2, experiments, profiling artifacts, one-off launchers and v4 are removed
  only in this branch. All remain recoverable on main. No checkpoint or dataset
  was deleted. v4 SAM inference is also excluded, not ported implicitly to v3.
- A frozen BHND interpolation implementation lives under v3 tests solely for
  independent BNHD migration comparisons; it has no production import path.
  Probes remain opt-in because low-level tests depend on them.
- Unused `include/ds.cuh` removed at the user's request. The actual scan path
  includes `ds_alt.cuh`; no source or generated dependency referenced ds.cuh.
- SWA control overrides the inherited vocabulary-rebinding hook with a no-op:
  it has no DISM codebooks. This fixes device transfer/checkpoint loading of
  that control without altering the selected GDN+DISM baseline.
- nanochat MIT license retained from the local fork's LICENSE (Karpathy 2025).

## Validation

- Independent sm120a source build of R32/D64/DV64 succeeded.
- Framework checkpoint/resume, varlen, registry, execution and attention tests:
  **52 passed, 11 skipped**.
- Selected baseline structure/no-decay and CUDA compiled forward/backward with
  optimizer update, checkpoint restore and stochastic replay: **2 passed**.
- Remote original Python snapshot versus curated baseline, CPU seed42:
  all 171 state_dict entries produce the identical SHA256
  `fc125d54e7a61b122e26791aeb50bc6437a5b217c6b6ef012ff339d89c0e463b`.
  Both have exactly 40,467,834 parameters. This checks initialization, not a
  claim of byte-identical optimizer trajectories across toolchains/devices.
- v3 test collection succeeds without the removed v2/v4 packages:
  1153 selected, 78 strict/diagnostic cases deselected by existing pytest config.

- Full default build succeeded for all R16/32 × D32/64 × DV32/64, including
  fixed and varlen kernels. Multi-config forward/backward, ragged replay and
  native frontend regression: **105 passed**.
- Frozen BHND versus BNHD interpolation and gradient metric helpers:
  **12 passed**. Training CLI `--help` loads successfully.
- Torch decode reference and module/cache regressions: **57 passed**.
- Final nanochat suite, including SWA/DISM and selected GDN+DISM compiled
  checkpoint replay: **58 passed, 11 skipped**. After removing ds.cuh,
  build.py also completes with the existing dependency graph unchanged.

This cleanup does not claim new sanitizer, full strict-gradient,
Hopper or trained-checkpoint evaluation coverage. No training was launched.
