# DISM v3 and nanochat development

- Scope: v3 operators, Python models, nanochat pretraining, build dependencies,
  maintained documentation and regression tests. No v4 or experiment archives.
- Read the selected version's AGENTS.md and docs/AGENTS.md before editing.
- Use only this worktree's flash_dism and cu_flash_dism on PYTHONPATH.
- nanochat originates from remote randomness-sequence_true-786m. Preserve
  postnorm, first-half GDN/rear-half GDN+DISM, fixed true direction and
  sequence-level hard sampling (packed training rows, NOT document boundaries).
- Preserve checkpoint RNG counters, annealing buffers, no-decay grouping,
  optimizer schema and consumed dataloader position. Never silently reset them.
- Use conda blkw locally and each version's build.py. CUDA device translation
  units must remain Torch-free. Keep build parallelism bounded.
- Preserve user changes. Never mutate checkpoints or restart training without
  authorization. Tests do not authorize downloads or external jobs.
- Do not change kernel math, launch defaults or tolerances during cleanup.
- Keep generated binaries, caches, profiles and result JSON out of Git.
- Shared storage is only for async input, necessary inter-warp communication,
  planned spill mitigation or output layout. No shared FP32 atomicAdd.
- Validate bounds and synchronization; inspect CALL/spills on kernel changes.
  Report numerical/sanitizer limitations; do not suppress them as cleanup.
- Historical experiments remain on main; prefill completion commit is7687cf5.
  Imported v3 source snapshot is47c8875bbf56cc7f365dad20c1154b452cde35f2.
