# Completed SWA-only versus hybrid: position NLL and needle recall

Both checkpoints step30000. Original hybrid49,678,876 parameters,FFN1024;
SWA-only49,641,856 parameters,FFN1728. Other width/depth/head/context/optimizer/
token-budget settings checked against saved configs. The FFN-capacity gap is
intentional parameter matching, not a controlled same-FFN architecture ablation.

SWA training final100-effective-batch val loss3.43468857 versus previously
recorded pure-hard hybrid3.44844508. The diagnostic sample below differs from
that validation aggregation, so compare within each matched diagnostic row.

## Execution and matching

All new model forwards ran on A100 GPU1 using the existing task-owned venv
(Torch2.6.0+cu124,FlashAttention2.7.4.post1), in the new isolated directory
`/home/chenyc/dism-swa-eval-20260910`. No training restart or checkpoint mutation;
no other users' processes/files inspected, only aggregate GPU occupancy.
The card had ample free memory; evaluation used microbatch4, no reservation.

Local work was CPU-only token preparation/plotting with CUDA_VISIBLE_DEVICES
empty. No new hybrid forwards: reused its frozen RTX5090 results. Hence this
is not a same-hardware/software numerical comparison. Both use BF16 model/head
and FP32 torch softcap30/cross_entropy; strict SWA checkpoint load passed.
The entire evaluation took~220seconds after model loading under shared load.

Input preparation reconstructs256×8192 held-out packed sequence/target pairs
and asserts the exact x/y SHA256 against the hybrid length experiment. For
NIAH, regenerates all256 main cases and asserts each prompt SHA256 against
`niah-256.json`. The128 immediate and128 no-information controls and every
matched absent/changed-color variant use the same deterministic plan.
Only a bounded token bundle was copied, not the dataset or checkpoint.

## Per-position NLL

All2,097,152 target tokens are identical for each length. Shorter lengths
reset every block within each8192-token sequence, as in the previous hybrid
study. Same RoPE frequencies and SWA128; only table capacity/length guard
extended to8192. All token losses finite.

| Independent block length | Hybrid NLL | SWA NLL | Hybrid minus SWA |
|---|---:|---:|---:|
|512|3.444778|3.409737|+0.035041|
|1024|3.400321|3.377288|+0.023034|
|2048|3.376149|3.360629|+0.015520|
|4096|3.365591|3.352135|+0.013455|
|8192|3.367980|3.347797|+0.020183|

For the first2048-token prefix alone (256sequences), mean NLL is3.382274
hybrid versus3.366637 SWA. This is the first panel of paired_positions.png,
not the four2048-block aggregate in the table. The second panel shows full8192;
the third shows the paired hybrid-minus-SWA curve.64-position bins, descriptive
95% bands from within-sequence bin averages. At8192 aggregate paired SE is
0.002237; at2048-reset it is0.001539. Packed sequences may be correlated.

SWA is generally lower across positions; the gap is visibly larger near the
start, narrows, and is positive through much of the late8192 range. There is
no evidence in this sample that hybrid wins aggregate late-position NLL.
SWA's lower loss at longer prefill does not imply access to all earlier tokens:
fewer block resets avoid cold starts, while local layers expand receptive field.

## NIAH (256 main cases, identical to expanded hybrid test)

| Case | Hybrid exact | SWA exact | Hybrid8-way | SWA8-way |
|---|---:|---:|---:|---:|
|Immediate needle control|103/128|72/128|103/128|73/128|
|2048 context|27/128|0/128|45/128|10/128|
|8192 context|33/128|0/128|51/128|15/128|
|Both lengths|60/256|0/256|96/256|25/256|

The eight-color uniform reference is12.5%; actual word priors are not uniform.
SWA main mean answer NLL5.84399 versus hybrid3.93218. Immediate-control
failures (56.25% SWA exact versus80.47% hybrid) show that task-format/copy
ability differs even before long-distance retrieval. This remains synthetic
exact repeated-prefix completion, not general instruction-following NIAH.

### Beyond SWA's structural receptive field

15layers×127 previous tokens/layer =1905 maximum predecessor distance.
96 cases (8192 length,10%/35%/65% insertion) put the entire needle outside
this range. On those identical cases:

- Hybrid exact24/96,8-way37/96; SWA exact0/96,8-way11/96.
- Changing only the needle color leaves SWA's eight candidate logits
  **bitwise unchanged in all96 cases** (maximum candidate-logit change0).
- Hybrid's original-versus-alternative margin shifts in the expected direction
  in86/96 cases. Average maximum absolute change over the eight candidate
  logits is1.18361 versus0 for SWA.

This cleanly separates natural-text average NLL from usable long-range
information access in this task. It does not claim strong general retrieval
for hybrid, whose exact recall is still modest. Different hardware, FFN
allocation and task-format baseline also prevent attributing the entire
score gap to one isolated modeling choice.

## Scripts and artifacts

- `prepare_swa_eval.py`: CPU-only token bundle, exact input checks.
- `eval_swa_bundle.py`: isolated A100 model forward/NLL/NIAH.
- `compare_swa_evaluation.py`: CPU-only paired statistics and plots.

Local root:
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs`

- `swa-eval-bundle-20260910.pt`: verified inputs, reusable without token service.
- `swa-a100-eval-20260910/report.json`, `position_losses.pt`: copied remote results.
- `swa-hybrid-comparison-20260910/summary.json`: paired statistics.
- Same directory: `paired_positions.png/svg`, `paired_niah.png`, visually inspected.

Remote model checkpoints remain in the original training task directory;
only results copied back. Source scripts compiled; runtime strict-config,
checkpoint, shape, input-hash and finite-result checks passed. No new training
or changes to the prepared vocabulary-sharing variant were performed.
