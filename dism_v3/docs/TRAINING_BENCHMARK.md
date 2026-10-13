# sm120 training performance baseline

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
  --warmup 10 --repeats 20 --samples 5 \
  --output dism_v3/build/bench_training_sm120_20261014.json

/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v3/tools/bench_training.py \
  --batches 16 --hard-probs 0 .95 1 --warmup 10 --repeats 20 --samples 5 \
  --profile --output dism_v3/build/bench_training_sm120_20261014_repeat_profile.json

/home/cicuvc/miniconda3/envs/blkw/bin/python dism_v3/tools/bench_training.py \
  --batches 16 --hard-probs .95 --warmup 20 --repeats 100 --samples 7 \
  --settle-seconds 2 --output dism_v3/build/bench_training_sm120_20261014_reference.json
```

JSON includes the exact configuration, source revision, extension and benchmark
hashes, environment, sampling fractions, document lengths, timing samples,
memory and optional kernel attribution. Generated results remain ignored under
`dism_v3/build/`; only the benchmark source and this maintained report belong in Git.
