# FineWeb-Edu decoder LM

## Queued pure-hard validation

User requested a separate pure-hard100-effective-batch validation after local
training completes. `dism_v2.eval_lm_hard` is queued as user service
`dism-hard-eval-after-training-20260910.service` (initialPID3520549).
It waits without importing torch or allocating GPU state, then requires final
checkpoint step==configured30000 before evaluating. An interrupted training
run is an error, not permission to evaluate an incomplete checkpoint.
It loads strict model state,hard_prob1,original held-out prefix,batch64/micro8,
800 microbatches (13,107,200 predicted tokens),original eval RNG seed779,
BF16 autocast/softcap30/tanh_finite,no optimizer or backward.
All microbatch losses must be finite; outputs loss/perplexity and per-batch
losses to `hard-eval-100.json`,with progress in `hard-eval-100.log` in the run
directory. This entry records a queued task, not a completed evaluation.

## Checkpoint continuation (2026-09-10)

After SIGTERM the initial softcap run saved a complete checkpoint at27957
(`status/finished=false`), not30000. User explicitly authorized continuation.
Model,394 AdamW parameter states/two decay groups,data cursor+39894 pending
tokens,and Python/CPU/CUDA/DISM RNG states were verified present. Data,CE,
autograd and backward source hashes match the original run; recent model
changes add control architectures without changing the hybrid computation.

The resume entry accepted all configuration checks and strict model loading,
and continues the original30000-update schedule from27957. No new warmup,
optimizer reset,data restart or hard-probability reset is requested.
`checkpoint-step27957.pt` and `config-before-resume27957.json` in the original
run directory preserve the pre-resume state separately from rolling latest.pt.

Current process PID3516181 is managed by the user service
`dism-hybrid-resume-20260910.service`, independent of the interactive exec
session. Same run directory,append-only console/metrics,offline W&B run ID
q65kj8pc (a new offline segment); original run artifacts remain intact.
No automatic restart loop is enabled for training failures. Both restored
token services remain separate user services and do not depend on LM lifetime.

```bash
systemctl --user status dism-hybrid-resume-20260910.service
```

Initial-run details below are historical; continuation supersedes the old
process ID and the original instruction to start the softcap experiment fresh.

## Current restart configuration

User requested a fresh speedrun with fused Triton softcap CE(cap30),peak
LR1e-3 and weight_decay1e-2. These are now the training CLI defaults;
1000-step warmup,30000 total updates,batch64/micro8,untied49.68M model and
hard-probability schedule are unchanged. `LMConfig.softcap=None` still
selects the original FLA loss for explicit model-level diagnostics.
The old offline run was gracefully stopped at **step1496** and its
`latest.pt` preserved. The new run must start from scratch, not resume it.
Approximate CE validation and limitations: [SOFTCAP_CE.md](SOFTCAP_CE.md).
Historical initial-run settings/results below are retained for comparison.

Fresh softcap run is active in
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/dism-swa50m-softcap30-20260909-offline`.
PID3399916, persistent session94817, W&B offline ID`q65kj8pc`.
Started from seed777 without a resume checkpoint. At step20: loss10.54290,
grad_norm1.45162,step0.828s,peak allocated9.53GiB. These startup observations
are not a convergence claim. Old run1496-step checkpoint remains intact.

Entry: `python -m dism_v2.train_lm`, conda `blkw`, RTX5090/sm120.
This is a new training application; existing copy-task and kernel math are
not changed. CUDA DISM uses default OPT13 and explicitly defaults this entry
to `tanh_finite`; existing core approximation/precision limitations remain.

## Model

- 49,678,876 parameters,15 blocks, residual width256, context2048.
- GPT-2 vocabulary50257. Input embedding and LM head are **untied**.
- Two LayerNorm PreNorm residual sublayers per block. Attention is
  `x + DISM(norm1(x)) + SWA(norm1(x))`, followed by a SwiGLU FFN residual.
- DISM follows copy_task: independent q/k/v projections, causal short-conv4
  with swish/silu,4 heads,D=DV64,per-head Q/K vocab512,softplus(log_sel_tau),
  low-rank output gate and FusedRMSNormGated,then output projection.
- SWA has independent q/k/v and output projections,4x64,standard RoPE
  theta10000 on q/k,FlashAttention causal `window_size=(127,0)`:128 tokens
  including self. RoPE is not added to the DISM branch.
- SwiGLU hidden1024; no dropout. FP32 master parameters, BF16 autocast;
  FLA `FusedCrossEntropyLoss(inplace_backward=True)` computes next-token CE.

## Data, optimization and evaluation

Local FineWeb-Edu parquet `train-*.parquet` and `validation-*.parquet` are
explicitly disjoint; lambada is excluded. PyArrow reads bounded batches of
64 documents, GPT-2 tokenization is online, EOS is inserted between documents,
and fixed2049-token blocks produce2048 input/label pairs. Attention may
cross document boundaries within a packed sequence. State resets between
sequences; there is no silent document truncation or full-corpus RAM cache.
Training shuffles shard order deterministically per epoch, not individual
documents. Validation starts from the same held-out prefix on every run and
refuses to repeat if insufficient data is available.

Defaults: effective batch64=micro8 x accumulation8,131072 tokens/update,
30000 optimizer updates (~3.93B predicted tokens). AdamW lr3e-4,
betas(.9,.95),eps1e-8,weight_decay0.1;1000-step linear LR warmup then cosine
to10% of peak; global grad norm clipped to1. Explicit `_no_weight_decay`
parameters plus all1D/scalar parameters receive no decay. DISM Q/K vocab,
tau,norm,bias are covered. Decay45,684,224/no_decay3,994,652 parameters.

Hard probability is linear from0 at update0 to1 at update29999. All
microbatches within an optimizer update share the probability; each DISM
call still draws one global random direction and replays row RNG backward.
Validation every1000 optimizer updates uses100 **effective batch64** batches
(800 micro8 batches), the current hard probability and a separately reset
evaluation RNG. Evaluation does not consume training RNG or advance either
schedule. Because hard_prob changes, validation losses across intervals do
not represent an identical attention mixture; hard_prob is logged with loss.

## Monitoring and recovery

W&B project `dism-finewebedu` uses the user's configured server/account.
Metrics: train/validation loss and perplexity, LR,hard_prob,pre-clipping
gradient norm,tau min/max,tokens seen,step latency/throughput,peak memory.
The W&B x-axis is optimizer_step, allowing validation and training records at
the same step. Also writes local `metrics.jsonl`,configuration and source
snapshots/hashes. No document text is logged to W&B.

Output defaults to a unique directory under the data disk's
`cicuvc/dism-lm-runs`, not the nearly-full system disk. `latest.pt` is atomically
replaced every1000 updates and on clean exit; `best.pt` is the best observed
scheduled validation loss. Checkpoints contain model,AdamW,step,best loss,
CUDA/CPU/Python/DISM RNG,and the exact parquet cursor plus remaining token
buffer. Own checkpoints are loaded as trusted pickle; never resume an
untrusted downloaded `.pt`. SIGTERM/SIGINT finish the current update then
checkpoint. Numerical/runtime failures preserve the previous valid checkpoint
instead of labeling partial/nonfinite state as resumable.

```bash
HF_ENDPOINT=https://hf-mirror.com /home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.train_lm --output /path/to/new-run
# Resume with the original model/optimizer/batch/schedule settings:
/home/cicuvc/miniconda3/envs/blkw/bin/python -m dism_v2.train_lm --output /path/to/run --resume /path/to/run/latest.pt
```

The local tokenizer asset was downloaded through the requested hf-mirror
entry (which redirected to Hugging Face). Verified GPT-2 vocab50257/EOS50256
and `Hello world!` -> `[15496,995,0]`. If future downloads fail, the user
authorized `HF_ENDPOINT=https://huggingface.co` with
`HTTPS_PROXY=http://127.0.0.1:7890`; the current run uses local tokenizer files.

## Initial checks

`DISM_TILE_LSE=tanh_finite python -m pytest -q tests/test_dism_v2_lm.py`:
6 passed. Covers parameter count/untied/no_decay,schedules,parquet resume
through shard/row-group/epoch transitions,FlashAttention against an explicit
causal128 mask,FLA CE loss+gradient,and actual DISM+SWA LM backward/RNG replay.

Full15-layer N2048 micro8/accum8 smoke: two train updates and one effective
validation batch passed. Step2 loss10.85109,grad_norm2.3320,~0.845s/update,
peak allocated8.03GiB. Resume from checkpoint2 executed update3 with the
correct LR/probability and finite gradients. First-step startup/compilation
is not steady-state timing. This does not yet demonstrate long-run learning
or long-run convergence. A subsequent full100-effective-batch validation
completed in29.41s with finite loss10.84074, without validation exhaustion.

## First long run

User approved W&B offline because the configured self-hosted server's TLS
certificate is expired. Certificate verification was not disabled.
Run directory:
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/dism-swa50m-20260909-offline`.
Launched with default30000 steps,batch64/micro8,eval100 batches every1000,
and `--wandb-mode offline`, explicitly OPT13/tanh_finite. Progress is in
`console.log` and `metrics.jsonl`; W&B binary history is under `wandb/`.
This run starts from seed777, not from the smoke-test checkpoint.
Verified training process PID3392750 (persistent execution session85860);
offline W&B run ID `z5472o1y`.

After the W&B certificate is repaired, use the blkw `wandb sync` CLI on the
specific `wandb/offline-run-*` directory. Do not treat offline logs as already
visible on the web dashboard. Completion/learning quality remains unverified
until the corresponding training and held-out validation records exist.
