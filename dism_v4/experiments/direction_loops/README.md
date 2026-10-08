# Whole consumer-loop direction specialization

Base commit: `1774fb9`. Apply `split.patch` from the dism_v3 root to reproduce.
The patch duplicates the entire `task.KBlocks` consumer loop under a single
workload-uniform `if (query_lse)`. Both metadata and score initialization receive
literal true/false in their respective paths. Q loading, producer, pipeline
release protocol, scan arithmetic, output layout and scheduler are unchanged.
The loop remains explicitly non-unrolled. No runtime/build option is introduced.

Build and run with the blkw environment:

```bash
git apply experiments/direction_loops/split.patch
DISM_LINEINFO=1 DISM_KEY_METADATA=legacy python build.py
PYTHONPATH=python:.. python -m pytest tests -q
PYTHONPATH=python python bench_summary.py --output build/direction_split_timing.json
```

## Correctness and generated code

- Full suite: 204 passed, 3 option-specific skips. No tolerance changes.
- SASS contains 128 HMMA instructions, 136 SHFL instructions, no CALL, 14 LDL
  and 10 STL instruction sites. These are static code counts, not per-launch
  executed counts. The compiler retains duplicated compute paths.
- ptxas: 168 registers, 32B stack, 40B spill stores, 56B spill loads. Original
  default has zero stack/spill. Splitting does not automatically improve register
  allocation even though the directions are mutually exclusive.

## Timing

RTX5090, B16/N2048/H16/D64, tanh, legacy metadata, graph kernel-only timing.
Each run has 100 warmup replays, 20 launches/replay and 9 samples per case.
First paired comparison, microseconds/launch (median):

| Direction | hard_prob | Original | Split loops |
| --- | ---: | ---: | ---: |
| query | 0 | 536.25 | 577.22 |
| query | .5 | 536.51 | 577.07 |
| query | 1 | 521.89 | 561.36 |
| key | 0 | 554.48 | 591.84 |
| key | .5 | 553.94 | 590.99 |
| key | 1 | 538.04 | 574.65 |

A second split run gives mixed query/key 580.05/593.35 us; rebuilding the
original afterward returns to approximately 538/556 us, confirming the
regression. Spill and increased code size are candidates, not an established
single cause. No NCU attribution or sanitizer rerun is claimed for this patch;
the existing numerical suite includes mixed per-head directions, single-CTA
persistent reuse and both input tails and hard/soft cases.

Artifacts: `build/direction_{baseline,split_timing}*.json`,
`build/direction_split_{build,all_tests}.log`, `build/direction_split.sass`,
and `build/direction_split.cubin`. The original default source and binary are
restored after the experiment; this patch is retained for inspection/retesting.
