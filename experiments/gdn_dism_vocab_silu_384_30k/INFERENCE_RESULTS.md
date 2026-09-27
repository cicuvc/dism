# Completed-model generation and NIAH

Checkpoint latest.pt, step30000, 72,188,142 parameters; final logged validation
NLL3.247751. Nine GDN + three DISM, no SWA/RoPE. No weight modifications.
Run directory under dism-lm-runs: gdn-dism-vocab-silu-384-72m-30k-20260911.

## Autoregressive sampling

Existing sampler applies FP32 softcap30, divides by temperature0.8, then
descending-probability top_p0.9 filtering (retains the crossing token), followed
by multinomial sampling. Seed1234, full-hard model, BF16 autocast. No repetition
penalty or postselection. Four existing default prompts, max128 new tokens each.
Saved all samples in generation-t08-p09.json: 488 tokens total, one EOS stop,
three length-limit stops, all logits finite. Complete-prefix recomputation,
not cached incremental inference; legacy CPU hybrid cache does not implement
GDN blocks. Timing includes first compilation and concurrent diagnostics and
is not an inference-throughput benchmark.

Qualitative findings: locally grammatical, topical English, but factual errors
and repetition remain. Solar-system sample invents a2.9-mile solar orbit and
repeats tilt statements; photosynthesis starts plausibly then reverses gas
conversion; story drifts into repetitive fabricated genealogy; circle-area
completion does not give the formula. Generation works mechanically, not a
claim of reliable knowledge or reasoning. All outputs retained, not best-of-N.

## Synthetic NIAH

Same deterministic128 key/background pairs as original50M evaluation, each at
2048/8192, verified all256 prompt hashes identical. Four depths10/35/65/90%,
32 cases per length/depth; eight balanced single-token colors. Repeat-prefix
completion, not instruction-following QA. Main answer scoring is deterministic
full-vocabulary greedy/CE, not temperature/top_p sampling.1024 total forwards
including immediate, absent and changed-color controls. Entire evaluation
replayed: all saved result rows exactly equal, all logits finite.

| Length | Exact full-vocab | Eight-color top1 | Answer NLL | Gain vs absent |
|---|---:|---:|---:|---:|
|2048|26/128 (20.31%)|43/128 (33.59%)|4.36541|1.23852|
|8192|3/128 (2.34%)|19/128 (14.84%)|5.29114|0.19653|

Exact successes by depth10/35/65/90%:2048=[1,1,9,15]/32;
8192=[0,0,0,3]/32. Strong recency dependence.
Immediate needle125/128 (97.66%); no-information0/128 exact,12/128 eight-color.
Long absent0/128 exact at both lengths; eight-color13/128 and16/128.
Changed-color exact28/128 and3/128. Expected color-margin shift positive in
90/128 and73/128, respectively.8192 eight-color accuracy is close to12.5%
uniform chance and the absent control; not reliable long-context retrieval.

Historical original50M DISM+SWA30k model scored27/128 and33/128 exact at2048/8192.
Current2048 performance is similar but8192 is substantially worse on these same
prompts. Architecture, DISM count, width, parameter budget and vocabulary
activation all differ, so this does not isolate GDN or SiLU as the cause.
Near-perfect immediate performance rules out gross task-format inability;
does not distinguish state forgetting from address formation/readout failure.
No RoPE exists in this model: length extension only changes context capacity.

Artifacts: niah-256-final.json (includes model metadata), niah-depth.png,
inference-summary.json. Initial niah-256.json retained, but its inherited
SWA/RoPE caveat was inapplicable; final report fixes metadata and replays all
scores. Summarize via python -m experiments.gdn_dism_vocab_silu_384_30k.summarize_inference.
Sampling and NIAH plan/layout tests:10 passed,2 unrelated CUDA tests deselected.

Example generation command (new output path required):
```bash
python -m dism_v2.generate_lm --checkpoint "$RUN/latest.pt" \
  --output "$RUN/generation-new.json" --backend prefix \
  --temperature 0.8 --top-p 0.9 --max-new-tokens 128
```
