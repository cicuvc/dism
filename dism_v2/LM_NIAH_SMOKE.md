# Tiny synthetic needle smoke test

## Expanded256-main-case run

`niah-256.json`, same local checkpoint and inference semantics. Use
`--samples 256` with the command below and a new output path to reproduce.
128 distinct adjective/noun keys and128 distinct8192-token held-out packed
backgrounds, each evaluated at2048 and8192. Thus256 main cases are128 paired
key/background units, not256 independent units. Four insertion fractions
10%/35%/65%/90%,32 main cases per length/depth cell. Each cell has exactly
four of each of the eight target colors. Key and color assignment fixed by
seed9321 before scoring. Alternative colors sampled among the other seven.

| Length | Exact full-vocab top1 | Eight-color top1 | Answer NLL | NLL gain over absent |
|---|---:|---:|---:|---:|
|2048|27/128 (21.1%)|45/128 (35.2%)|4.0539|1.5459|
|8192|33/128 (25.8%)|51/128 (39.8%)|3.8105|1.4226|

Exact matches by insertion depth (each denominator32):

| Length |10%|35%|65%|90%|
|---|---:|---:|---:|---:|
|2048|0|6|9|12|
|8192|5|8|11|9|

Additional controls (not included in256-main-case denominator):

- Immediate needle:103/128 exact and eight-way (80.5%), NLL1.3507.
- No-information prefix:0/128 exact,15/128 eight-way, NLL4.6749.
- Matched long absent:0/128 exact at each length; eight-way12/128 and14/128.
- Changed-color long controls:27/128 and34/128 exact for the new color.
- Expected original-versus-alternative logit-margin shift after changing
  color:119/128 at2048,114/128 at8192 (233/256 total,91.0%).
- Restricting post hoc to keys that pass the immediate control gives20/103
  and30/103 exact. This diagnostic subset is not the main benchmark score.

Total1024 forwards:256 main +256 absent +256 changed +128 immediate +128
no-information. ~25.4seconds excluding loading. All logits finite.
New deterministic plan/layout tests:9 passed. JSON retains every result,
color logit, needle position, prompt hash and control rather than successes
only. The task format is unchanged except expanded vocabulary/background
coverage and four instead of three depth settings; the original smoke report
is preserved.

Conclusion: reproducible context sensitivity above absent/guessing controls,
but weak exact recall. Immediate-control failures show task-format/prior
limitations as well. The six-success difference between lengths is not enough
to assert better8192 capability; contexts and absolute distances differ.
This repeated-prefix single-token task is still not general NIAH QA and does
not isolate DISM from the SWA branch.

## Original12-case pilot

Local49.68M original hybrid checkpoint step30000, full-hard CUDA prefill,
BF16 model/head, FP32 softcap30. RoPE frequencies and SWA128 unchanged.
Not an official benchmark; no instruction/chat finetuning assumed.

Needle example: `The secret color of the silver telescope is yellow.`
End-of-context completion: `The secret color of the silver telescope is`
The answer is a single GPT2 token. Two keys (silver telescope, wooden compass),
held-out natural packed text backgrounds, lengths2048/8192, insertion fractions
10%/50%/90% of available background. Exact repeated prefix intentionally makes
this easier than paraphrased QA. No distractor key/value records were added.
Six prompt-layout assertions passed. No nonfinite logits.

Random seed9321 fixed before inference; both independently assigned main colors
happened to be yellow. Counterfactual colors were white/purple respectively.
This is a severe sample-diversity limitation, retained rather than changing
examples after seeing results. Colors span eight one-token candidate answers.

| Condition | Main exact matches | Mean answer NLL |
|---|---:|---:|
|Immediate needle + repeated prefix|2/2|1.2805|
|Prefix only, no information|0/2|5.0212|
|2048, six long cases|1/6|4.0660|
|8192, six long cases|2/6|2.4926|

Long cases also score eight candidate colors; candidate top1 equals the exact
match counts above. The only2048 success is silver telescope at90%;8192
successes are silver telescope at50% and90%. This tiny sample does not support
a claim that8192 is generally easier/better than2048: background suffix and
absolute distances also differ.

Every long prompt has same-length controls:

1. Absent: replace only needle span with newline tokens, keep all other tokens.
2. Counterfactual: change only the color to the alternative, same token count.

Average main answer NLL improvement over absent is1.4975nats at2048 and
3.1806 at8192. Counterfactual exact matches are0/6 and3/6 respectively.
In11/12 cases, changing the needle color moves the original-vs-alternative
logit margin in the expected direction (range−0.008 to+5.534 for the
original-minus-counterfactual margin shift). This supports context sensitivity,
not reliable recall. It does not isolate DISM from multi-layer SWA propagation.
The absent control's newline block is artificial, so its NLL effect alone is
weaker evidence than the token-matched counterfactual color change.

No full-vocabulary generation after the one-token answer was needed; exact
match is unrestricted vocabulary greedy next-token matching, not an eight-way
forced answer. No sampling, answer filtering or repetition penalty used.

## Artifacts / reproduction

```bash
RUN=/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/dism-swa50m-softcap30-20260909-offline
/home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.eval_lm_niah \
  --checkpoint "$RUN/latest.pt" --output "$RUN/niah-smoke-new.json"
```

Initial report: `$RUN/niah-smoke.json`, all36 long-condition forward passes
and4 short controls, prompts/positions/hashes, individual scores and greedy
tokens retained. Model evaluation~2.8seconds excluding load. Output must be
a new path. No weights or training jobs changed.
