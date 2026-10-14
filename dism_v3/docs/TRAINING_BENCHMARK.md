# sm120 training performance baseline

The sections below first preserve the historical single-operator baseline.
See [full-model training replay](#72m-full-model-training-replay) for the later
complete hard/soft operator measurement and controlled core-path comparison.

Measured locally on 2026-10-14, source HEAD `cdbd33c31f9f702d6b6244a253a2e23c10ae0773`.
This is a **single DISM training operator** benchmark, including vocabulary
interpolation and its backward. It is not full-model training throughput.
Projections, short convolution, GDN, FFN, loss, optimizer, gradient accumulation,
data loading and multi-GPU communication are excluded.

## Baseline and environment

The historical run's `train.sh` and `meta_019074.json` were inspected read-only:
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m`.
They agree with `nanochat/configs/sequence_true_72m_valres.json`: 12 layers,
width384, H6, D/DV64, R32, vocabulary512, context2048, device batch16,
fixed true direction and sequence-level hard sampling. No checkpoint was loaded
or modified and no training was restarted.

- GPU: RTX5090, compute capability12.0, 170 SMs, approximately32GiB, power limit500W.
- Driver615.71.09; Torch2.13.0+cu130; Triton3.8.0; clang23 git `0cad9b7585d9`;
  compilation toolkit13.1 (`ptxas`13.1.115).
- Rebuilt with `dism_v3/build.py`, `DISM_BUILD_JOBS=4`, all eight production
  configurations, sm120a, BF16 output, no optional probes or FP32 diagnostics.
- Extension SHA256 after rebuild:
  `e9e7bd6895c5f841f82594967f5afcc761d502acc77e65a34e838e43dd098a1e`.
  It matches the preexisting extension. No kernel/default/math changes.
- Normal desktop session, unlocked clocks, no other compute job observed before
  measurement. GPU clock/temperature/power snapshots accompany every JSON case.
  This is not a measurement on RTX PRO6000; use the same script there before
  extrapolating absolute throughput.

## Measurement contract

`tools/bench_training.py` calls the `compiler.py` vocabulary forward/backward
custom operators used inside compiled training. It does not wrap the surrounding
model in `torch.compile`. Both forward paths save training checkpoints; backward
consumes those saves and computes all eight input/codebook/tau gradients.
Backward-only timing reuses immutable saves, with fresh output allocations and
no parameter `.grad` accumulation. The pair measurement runs a fresh forward
followed by backward on that forward's saves.

Inputs are seeded synthetic SiLU(Gaussian) BF16 vectors, FP32 SiLU(Gaussian)
codebooks, tau0.5, beta1 and BF16 output gradients. They are not activations or
learned codebooks from the historical run. Numerical distributions and repeated
hard chains can affect timing; this fixture does not establish trained-model
quality or long-chain numerical accuracy. Each case checks finite output and all
returned gradients.

Packed B=1 transport represents original training batches B1/B4/B16:

- `uniform`: each original row is one2048-token document.
- `ragged`: each original row contains documents256/512/256/1024.
- Hard probability0/.5/.95/1; a flag is sampled once per original row/head and
  expanded across that row, including its document boundaries. Direction is true.
- The two layouts have equal token counts, but ragged has only0.34375 times
  uniform's sum of squared document lengths. Neither is a measured dataset
  length histogram. Layout construction is outside timing, metadata caches hot.
- Realized hard fractions are recorded: B16 at p=.95 has91/96 hard row/head
  planes (94.79%). B1 at p=.95 happened to be all-hard.

Timings use CUDA-event windows around repeated calls, with synchronized wall
times also recorded. These include allocations and possible host launch gaps,
not just summed kernel execution. No CUDA Graph replay is used. Independent
stage medians need not add to a full-call median. Separate profiler captures
provide instrumented device-duration attribution and are not substituted for
unprofiled latency. Every case has an untimed settling workload plus warmup.

## Sweep and independent repeat

Primary comparison point: B16, p=.95, seven windows of100 calls after20 warmups
and two seconds of settling. Medians below, with the range of window means in
parentheses, in milliseconds:

| Layout | forward | backward | fresh forward+backward |
|---|---:|---:|---:|
| uniform |1.251 (1.214–1.279)|8.581 (8.238–8.694)|9.902 (9.855–9.917)|
| ragged |.805 (.706–.844)|5.763 (5.528–6.009)|6.430 (6.360–6.686)|

Native core forward/backward medians from the same run: uniform .945/4.885ms,
ragged .416/2.055ms. Isolated B1/B3: uniform2.435/2.373ms,
ragged1.004/1.005ms. Host and clock variation remain visible even with longer
windows; always compare repeated measurements and retain their ranges.

Milliseconds per operator invocation; each entry is the median of five windows
of20 calls, after10 warmups and one second of settling. Repeat timings are
collected before the separate profiler capture for that case.

| B16 layout | hard probability | sweep forward | sweep backward | sweep pair | repeat forward | repeat backward | repeat pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform |0|1.235|7.908|9.055|1.231|7.764|9.061|
| uniform |.5|1.228|8.252|9.326|—|—|—|
| uniform |.95|1.334|8.366|9.950|1.220|8.321|9.561|
| uniform |1|1.221|8.351|9.581|1.220|8.397|9.566|
| ragged |0|.802|5.455|6.600|.714|5.324|6.035|
| ragged |.5|.728|5.414|6.321|—|—|—|
| ragged |.95|.711|5.493|6.237|.710|5.546|6.356|
| ragged |1|.807|5.632|6.269|.808|5.532|6.952|

The short-window sweep has visible timing variation, including approximately
11% pair variation for ragged all-hard. Do not interpret small differences as
hard/soft speedups. In particular, the current all-hard training path still
launches interpolation forward/backward; it does not become a SAM inference path.

At p=.95, sweep pair latency for B1/B4/B16 is1.336/2.749/9.950ms (uniform)
and0.764/1.923/6.237ms (ragged). These are synthetic single-layer operator rates,
not whole-model tokens/second.

## Device attribution and resource evidence

Separate three-call profiler captures, B16 p=.95, microseconds per call:

| Kernel | uniform | ragged |
|---|---:|---:|
| interpolation forward `emb_fwd` |272.5|270.3|
| forward summary |223.1|106.4|
| forward readout |670.3|237.3|
| interpolation backward `_interp_bwd` |3256.5|3087.7|
| B1 backward summary |2795.0|969.2|
| B3 backward Q/K |2428.6|943.0|
| all backward device events |9069.2|5492.8|

Profiler results have instrumentation/run variability, especially on uniform;
they establish attribution, not precise unprofiled latency. The main backward
targets are interpolation and B1/B3. For ragged, interpolation backward accounts
for56.2% of device duration, B1+B3 for34.8%. Forward interpolation is only about
0.27ms here, so removing it alone cannot yield a20x training speedup.

Current sm120a cubin resources for R32/D64/DV64 packed kernels:

| Kernel | registers reported | stack bytes |
|---|---:|---:|
| forward |168|16|
| B1 |168|304|
| B3 |168|192|

No CALL found in these three disassemblies. Stack is a static compiler resource,
not dynamic spill traffic; register count under setmaxnreg is not a sufficient
optimization metric. Dynamic shared memory is not represented by the1024-byte
static shared resource reported by cuobjdump.

## Memory and validation

B16 uniform/ragged `VarlenLayout.checkpoint_bytes(6)` budgets are286.5/97.5MiB.
Measured forward+backward peak allocation increments are607.3/511.3MiB over
the live input/saved-state fixture. This increment excludes the already-live
fixture and allocator reservations; it is neither whole-model peak memory nor
the checkpoint budget. All raw allocated baselines and retained increments are
in JSON, preventing these different quantities from being conflated.

After rebuilding:121 tests passed across `test_multi_config.py`,
`test_voc_glue.py`, and `test_native_frontend.py`, covering all configurations,
fixed/packed oracle comparisons and fused-glue rounding. All benchmark outputs
and returned gradients were finite. No sanitizer or strict-gradient campaign
was run for this benchmark-only change; existing BF16 delta/tau limitations in
BACKWARD_DIMS.md remain applicable.

## Reproduction and artifacts

From repository root, with a fresh output filename:

```bash
export PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat"
/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v3/tools/bench_training.py \
  --interp-backward baseline --warmup 10 --repeats 20 --samples 5 \
  --output dism_v3/build/bench_training_sm120_20261014.json

/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v3/tools/bench_training.py \
  --interp-backward baseline --batches 16 --hard-probs 0 .95 1 --warmup 10 --repeats 20 --samples 5 \
  --profile --output dism_v3/build/bench_training_sm120_20261014_repeat_profile.json

/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v3/tools/bench_training.py \
  --interp-backward baseline --batches 16 --hard-probs .95 --warmup 20 --repeats 100 --samples 7 \
  --settle-seconds 2 --output dism_v3/build/bench_training_sm120_20261014_reference.json
```

JSON includes the exact configuration, source revision, extension and benchmark
hashes, environment, sampling fractions, document lengths, timing samples,
memory and optional kernel attribution. Generated results remain ignored under
`dism_v3/build/`; only the benchmark source and this maintained report belong in Git.

The production default subsequently changed to atomic split-attention backward
with autotune. The explicit `--interp-backward baseline` above preserves this
report's original implementation. See [INTERPOLATION_BACKWARD.md](INTERPOLATION_BACKWARD.md)
for the optimized implementation and current measurements.

## 72M full-model training replay

Measured 2026-10-14 on RTX5090/Ultra9 285K, Torch2.13.0+cu130, driver615.71.09.
GPU power limit was **450W** during all new comparisons (not the historical
500W operator baseline); no power/clocks/affinity system defaults were changed.
Source is `3b26dbf` plus the working-tree default hybrid integration. The
benchmark does not modify production kernels, launch defaults or tolerances.

`tools/bench_model_training.py` strictly loads read-only copies of the valres
checkpoint at19074, including its 72,035,808 parameters, optimizer topology and
AdamW state, RNG counter76296 and hard probability.95. It replays four real
checkpoint-owned prefetched batches, each B16/N2048, four microbatches per
131072-token optimizer step. Document counts are34/30/32/43; their quadratic
work ratios versus16 full2048-token documents are .76172/.71289/.52930/.40625.
Direction remains true, with first-half GDN/rear-half GDN+DISM, postnorm and
value residual unchanged. This is the selected fixed-true baseline, not a
claim about the random-direction checkpoint or other architectural variants.

Only disposable in-memory model/optimizer copies update. The original RNG,
weights and optimizer are restored before each measurement phase; counters
advance normally within a phase, while the final checkpoint annealing values
stay fixed. No checkpoint is written, dataloader resumed or consumed position
advanced. Inputs are resident on GPU: data loading, tokenization, W&B, disk IO
and validation are excluded. Repeated batches intentionally overfit during
the bounded throughput window; this is not continued training or a convergence
test. Loss and global gradient norm remain finite, and live plan handles return
to zero. Outputs, not training artifacts, live under ignored `build/`.

Compilation matches `base_train.py`: dynamic Inductor, duck shapes disabled,
dynamic output shape capture, GDN graph breaks retained. Three complete warmup
steps exclude JIT/autotuning. Phase events surround each forward/backward with
**no intervening synchronization**, preserving overlap; numbers below sum four
microbatches. Sustained measurements are separate three-window runs of ten
optimizer steps each, with gradient norm synchronization, AdamW, hidden-RMS and
update-RMS probes included. Phase events include host launch starvation/waits,
not just GPU-active kernel time.

### Controlled core-path comparison

The old extension is the already-built local artifact
`build/lib.linux-x86_64-cpython-312/cu_flash_dism.cpython-312-x86_64-linux-gnu.so`.
Its SHA256 equals the recorded baseline `e9e7bd68...98a1e` above; the comparison
helper rejects any other hash. It is explicitly loaded under a separate module
name, never substituted on PYTHONPATH or copied over the production extension.
`tools/bench_legacy_training.py` overrides only this benchmark process's
vocabulary custom-op implementations to use the old full-quadratic CUDA core,
expanding sequence masks into the legacy input format. Both sides retain
**current atomic interpolation**, the same surrounding model, temperature
handling, inputs, checkpoint and optimizer. This isolates core-path changes;
it is not a reconstruction of every historical training component.

All times milliseconds; forward/backward columns are sums over four microbatches.
Sustained ranges are the range of ten-step window means, not per-step extremes.

| Path | Forward | Backward | Optimizer/probes | Sustained step | tokens/s |
|---|---:|---:|---:|---:|---:|
| Old quadratic CUDA core, current interpolation |285.7|801.8|13.6|1113.5 (1109.0–1119.7)|117707|
| New hybrid,16 workers |307.4|795.3|13.6|1106.2 (1103.8–1108.7)|118483|
| New hybrid, independent repeat |303.4|790.2|13.1|1108.5 (1102.3–1112.0)|118247|
| New hybrid, no hidden/update-RMS probes |299.6|792.2|2.5|1092.7 (1092.2–1093.3)|119952|

**There is no demonstrated substantial end-to-end improvement from the new
core route at this scale:** the sustained difference is only0.5–0.7%, comparable
to run variation. The historical last100 training steps averaged1208.7ms
(108439 tokens/s), but comparing that log to the new1106ms does not isolate this
change: interpolation, data/probe costs, inputs and GPU power settings differ.
Peak allocated GPU memory in the phase measurement was16.42GiB hybrid versus
16.31GiB old core, excluding allocator reservations and pinned host memory.

### Bottleneck attribution

Separate one-step profiler captures provide attribution, **not additive latency
budgets**: copies and kernels can overlap, and CPU waits include queued GPU work.

- DISM native GPU stages fell from90.7ms (old core) to25.4ms, plus13.6ms for the
  hard event readout/summarization/passing kernels: about51.7ms less GPU work.
  The event readout alone is9.3ms, not the complete event backward.
- At16 workers, the24 plans consumed395–418ms summed CPU wall time per step,
  with35–47ms exposed `future.result()` wait. In the first measurement,329ms
  was native planning,79ms packing, and about10ms allocation/release/input work.
  Waits occur chiefly on the last DISM layer's backward: all six plans were
  queued FIFO, but backward starts at the last layer. Forward-to-backward slack
  does not guarantee that every layer's plan is ready.
- D2H event windows totaled1.9ms. Metadata H2D windows totaled52–60ms including
  enqueue gaps; the separately profiled H2D device copies totaled41ms. They
  must not be added to the step time as though all transfers were exposed.
- Metadata totals1.76GB/step. Only18.1% of padded C64 event slots are occupied,
  and96.9% of streams fit one chunk. Compact/bucket-sized short-stream storage
  is a concrete target for reducing packing/PCIe traffic without changing math.
- Short-convolution/projection kernels total276ms: `bwd_mma_gemm`176ms,
  `bwd_short_conv_silu`58ms, forward43ms. Code inspection shows the packed grid
  uses2048 for every document, or2.17 times actual rows on this fixture. The
  projection backward still iterates channels and issues dweight atomics for
  fully masked sequence tiles; its dweight atomic mask is only channel-based.
  The benefit of eliminating these padded tiles is **not yet measured**.
- Fused vocabulary CE kernels total244ms: forward61ms, dlogits65ms,
  dhidden79ms, dweight40ms. Their small fixed tiles and long vocabulary loops
  are a larger remaining GPU target than further event-readout optimization.
- `float(beta)` in compiled vocabulary backward causes24 scalar-read syncs.
  Their425.7ms profiler CPU time mostly waits for previously queued GPU work;
  **it is not425.7ms of recoverable overhead**. Packed-layout and FLA metadata
  reads create other sync boundaries. Removing/reusing these reads requires a
  controlled follow-up measurement, not adding their profiler durations.
- Hidden-RMS hooks hit Dynamo's default8-specialization limit at layer8, so
  some code falls back to eager. The no-probe control improves only about1.2%
  here; it is not the dominant source of the795ms backward duration.

The immediate planner priority is metadata footprint and tail scheduling;
the larger full-model GPU targets are projection/shortconv and CE. No changes
to those production paths were made in this measurement task. Existing tau
cancellation/long-stream numerical limitations still apply; no new sanitizer
or kernel-resource campaign was needed for benchmark-only code.

### CPU utilization and process-local affinity

The285K's PMU topology reports P cores0–7 (`cpu_core/cpus`) and E cores8–23
(`cpu_atom/cpus`), with no SMT. A12-second `pidstat -t` sample of the unbound
24-worker run averaged633% process CPU, or6.33 cores out of24. Each planner
worker averaged21–23%; main/autograd/coordinator averaged43/56/6%. This is
bursty planner work, not sustained all-core saturation. Native planning/packing
release the GIL; the worker queue mutex is released before task execution.
Low average CPU utilization does not exclude a late plan delaying backward.

The benchmark can pin each native worker to one CPU and isolate service threads
(main/autograd/coordinator) on CPUs0–1. Only this process's threads are affected;
production defaults and system affinity remain unchanged. Worker TIDs are
discovered after warming native imports; ambiguous thread discovery fails.

| Configuration | Sustained step, ms | Explicit planner wait, ms/diagnostic step | Worker migrations |
|---|---:|---:|---:|
|16 unbound, original/repeat|1106.2 /1108.5|47.2 /35.1|not captured|
|16 unbound, later control|1122.6|37.7|335|
|16 pinned2–17, service0–1|1107.8|11.7|0|
|22 pinned2–23, service0–1|1104.1|0.085|0|
|24 unbound|1100.8|1.31|not captured|

Affinity runs and the later control use two10-step windows. Migration counters
cover three phase steps plus20 throughput steps and the intervening restore,
not just the throughput windows. Pinning eliminates migration, but a large
stable end-to-end gain is **not established**. Changing worker count and service
isolation together does not isolate affinity's causal effect. An8-worker
unbound control was substantially worse:1325.2ms/step and269ms exposed wait.

### Detailed Nsight Systems and PyTorch stage captures

Saved warmed, complete optimizer-step captures (four forward/backward pairs,
gradient norm, optimizer and probes), using Nsight Systems2025.5.2.266 from
CUDA13.1. CUDA/NVTX/OSRT tracing works without root; CPU IP/backtrace and context
switch sampling were disabled because this environment's perf permissions do
not allow them. Thus these captures do not diagnose CPU cache or branch misses.

Primary artifacts, all retained locally under ignored `build/`:

- `model72m_hybrid16_stages_20261014.nsys-rep`: default16 unbound workers.
- `model72m_affinity22_stages_20261014.nsys-rep`:22 pinned workers, service0–1.
- `model72m_affinity22_torch_stages_20261014.trace.json`: PyTorch CPU/CUDA trace,
  with input shapes and explicit microbatch/stage ranges.
- Both Nsight reports have matching `.sqlite`, run `.json`/`.log` and exported
  CUDA kernel/memory/API, NVTX and OSRT CSV tables. The cross-thread analyses are
  `model72m_{hybrid16,affinity22}_nsys_analysis_20261014.json` and
  `model72m_affinity22_torch_analysis_20261014.json`.

**Attribution caveat:** built-in NVTX GPU projection is same-thread and misses
most backward launches from the autograd thread. `tools/analyze_model_profile.py`
instead maps CUDA correlations to CPU launch timestamps inside the outer
forward/backward ranges, across threads. It counts only actual kernels/copies/
memsets, not GPU user annotations. CPU scope duration is not GPU duration:
gradient-norm CPU scopes span36–45ms waiting for queued backward, while their
own GPU work is only0.56ms.

The PyTorch capture provides the following mean **GPU kernel duration sums per
microbatch**, B16/N2048. These are attribution, not additive critical-path time;
they exclude launch gaps and can overlap other streams. FFN GEMMs are identified
by the actual2576-wide intermediate tensor shapes, unavailable in Nsight's
kernel-only summary.

| Component | Forward, ms | Backward, ms |
|---|---:|---:|
|Short convolution + projection|10.62|57.23|
|Linear vocabulary cross entropy|15.19|44.65|
|FFN GEMMs|12.21|23.06|
|GDN recurrence/preparation|4.43|8.52|
|Vocabulary interpolation/glue|2.08|7.54|
|Soft CUDA core|0.82|3.60|
|Hard CUDA forward|1.90|—|
|Hard event backward|—|3.39|
|Other GEMMs|1.90|4.59|
|Other pointwise/norm/reductions|15.38|33.32|
|Total kernel duration|64.51|185.90|

Mean first-to-last-kernel spans are84.49ms forward and187.72ms backward;
the profiled whole optimizer step is1112.85ms. Use the unprofiled windows above
for throughput, not this one-step instrumented measurement. Shortconv/projection
and CE together account for510.8ms of kernel work per optimizer step; complete
hard event backward accounts for13.6ms. Other pointwise work is also material:
FP32 add37.7ms, BF16 cast19.7ms, direct copy19.0ms and fused SwiGLU backward17.3ms
per step. These categories include work outside the DISM core.

Nsight corroborates both the GPU ranking and exposed planner tail:

| Capture | Step scope, ms | GPU activity union, ms | No recorded GPU activity, ms | Longest gap, ms |
|---|---:|---:|---:|---:|
|16 unbound|1148.41|1010.18|138.23|17.43|
|22 pinned + service isolation|1140.02|1033.34|106.69|1.63|

GPU activity union merges kernels, copies and memsets across streams; it is
**not SM occupancy**. The default capture has gaps at step offsets138.44,
419.71 and703.43ms lasting17.00,17.43 and9.71ms. Each overlaps
`planner/wait_and_event_backward` and `planner/native_plan_pack`; GPU activity
resumes as the coordinator finishes packing and starts H2D. These are directly
observed planner-tail bubbles, totaling44.14ms. They disappear in the22-worker
capture, but many small launch/synchronization gaps remain. The two captures'
GPU active times differ too; do not interpret their step-time difference as an
isolated affinity speedup. Across these captures, actual D2H copies total1.91–
1.94ms/step and H2D copies43.63–51.83ms/step, mostly overlapping computation.
The `planner/bulk_H2D` NVTX scope measures host enqueue time, not copy completion.

No production optimization or default change was made as part of profiling.
Priority remains padded shortconv/projection work and CE, followed by compact
planner metadata/tail scheduling. Further event-kernel tuning has much less
full-model headroom at this fixture.

### Full-model reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
PYTHONPATH="$PWD/dism_v3/python:$PWD/nanochat" \
/home/cicuvc/miniconda3/envs/blkw/bin/python -u dism_v3/tools/bench_model_training.py \
  --checkpoint-dir /media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/base_checkpoints/seqtrue72m-fwedu-valres \
  --warmup 3 --phase-steps 3 --windows 3 --window-steps 10 --profile \
  --output dism_v3/build/model72m_hybrid_new.json
```

Use `--legacy-cuda dism_v3/build/lib.linux-x86_64-cpython-312/cu_flash_dism.cpython-312-x86_64-linux-gnu.so`
for the SHA-checked old-core comparison, `--no-training-probes` for the probe
control, or `--planner-workers N` for a process-local worker sweep. Every command
requires a fresh output filename. JSON includes the full model/config, optimizer
hyperparameters, document lengths, phase samples, window throughput, planner
diagnostics and optional profiler attribution; Chrome traces are separate files.
Primary reports: `build/model72m_{hybrid,hybrid_repeat,legacy,hybrid_noprobes}_20261014.json`.

For a bounded Nsight capture, use the same environment and checkpoint arguments
with `--warmup 3 --profile-only --nsys` (omit `--profile`), prefixed by:

```bash
/usr/local/cuda-13.1/nsight-systems-2025.5.2/bin/nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=false --output=dism_v3/build/model72m_new_capture \
  /home/cicuvc/miniconda3/envs/blkw/bin/python -u dism_v3/tools/bench_model_training.py \
  --checkpoint-dir /media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-runs/dism-fwedu-valres-72m-2500m/base_checkpoints/seqtrue72m-fwedu-valres \
  --warmup 3 --profile-only --nsys \
  --output dism_v3/build/model72m_new_capture.json
```

Add `--planner-workers 22 --planner-cpus 2-23 --service-cpus 0-1` for the pinned
case. For PyTorch, run Python directly with `--profile --profile-only` instead
of `--nsys`. Export Nsight SQLite with `nsys stats` or `nsys export`, then use:

```bash
/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v3/tools/analyze_model_profile.py \
  dism_v3/build/model72m_new_capture.sqlite \
  --output dism_v3/build/model72m_new_capture_analysis.json
```

The same analyzer accepts a PyTorch `.trace.json`. Choose fresh output names;
the original reports above are intentionally retained for inspection.
