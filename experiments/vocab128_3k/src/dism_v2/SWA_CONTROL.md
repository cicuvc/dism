# Parameter-matched SWA-only control

## Completed30k and paired evaluation (latest)

The resumed run finished all30000 updates. Final recorded100-effective-batch
validation loss3.43468857, perplexity31.02175. Independent SWA evaluation ran
on A100 in `/home/chenyc/dism-swa-eval-20260910`, without modifying the training
directory or stopping existing tasks. Exact paired token/hash comparisons
against frozen local hybrid results are documented in
`dism_v2/SWA_HYBRID_EVALUATION.md`. All active/recovery notes below are historical.

Latest recovery: after GPUs1–3 became idle, SWA resumed on GPU1 from3000
under remote user service `dism-swa-resume-20260910.service`,PID4048376.
Verified new update3030: loss3.79326,finite grad_norm0.20751,0.632s/update.
Full attention resumed on GPU2 from1000 under
`dism-full-resume-20260910.service`,PID4048394; update1020 loss4.80078,
finite grad_norm0.44784,0.661s/update. Both configuration/state restores
succeeded. Pre-resume checkpoints are separately preserved as
`checkpoint-step3000.pt` / `checkpoint-step1000.pt`. Logs append in their
original run directories (replayed step numbers may appear twice).
No jobs were started on GPU3. The prior no-free-GPU status below is historical.

## Recovery status (2026-09-10, supersedes active-run note below)

Both remote controls exited with `Token service unavailable; no batch was
skipped` after the local token services and SSH master stopped. The exact cause
of those service exits is not established. SWA's last logged update3060 is not
a checkpoint: its `latest.pt` contains step3000,stream cursor24000 (micro8).
Full attention's last logged update1950 has checkpoint1000,cursor8000.
Resume these checkpoints, replaying unsaved updates without advancing the
schedule/data cursor. Do not label unsaved progress as recoverable model state.

The two local services have been restarted under the user's service manager:
`dism-swa-token-20260910.service` and `dism-full-token-20260910.service`, with
Restart=on-failure,RestartSec=5. Existing source identities,cursor snapshots
and authentication files are reused; they are not tied to a foreground exec
session. A new authenticated SSH master at
`/tmp/dism-recovery-ssh.HkBVoJ/control` forwards18473/18474; that SSH connection
still needs to remain alive (it is not yet a reconnecting managed service).
At the recovery check all remote GPUs were occupied, so no training was
restarted and no existing GPU process was stopped.

The local hybrid log records SIGTERM and a clean checkpoint at27957 with
`status/finished=false`, not completion of30k. Its continuation requires user
direction; restoring the A100 data services does not restart the hybrid.

## Active run (2026-09-10)

SWA-only is now running on GPU1 (A100-SXM4-40GB), PID4019201, foreground SSH
session88460, output `/home/chenyc/dism-swa-control-20260909/run`.
There is no reservation process. Training started from scratch with seed777,
not from the preflight optimizer. Step20: loss10.54222,grad_norm1.44231,
0.627s/update (~209k predicted tokens/s),peak allocated8.297GiB.
These are startup health observations, not a convergence result.

FlashAttention2.7.4.post1 built and imported successfully in the task-local
venv (Torch2.6.0+cu124). Preflight compared output and all q/k/v gradients
against FP32 masked attention at N127/128/129/257; all passed. The full15-layer
model completed forward/backward/AdamW with finite gradients for every
parameter,peak7.728GiB for that single-microbatch check. Production mean CE
checks passed18/18 as recorded below.

Launch:
```bash
CUDA_VISIBLE_DEVICES=1 NO_PROXY=127.0.0.1,localhost OMP_NUM_THREADS=8 \
TRITON_CACHE_DIR=/home/chenyc/dism-swa-control-20260909/triton-cache \
venv/bin/python -u -m dism_v2.train_lm --architecture swa_only \
  --output /home/chenyc/dism-swa-control-20260909/run \
  --stream-url http://127.0.0.1:18473 \
  --stream-secret-file /home/chenyc/dism-swa-control-20260909/auth_token \
  --wandb-mode offline
```

The launch occurred from the task directory. Source provenance is the saved
`source_sha256` and `sources/` snapshots: the first run's `git_commit` field
picked up an ambient remote parent repository and is **not** this source's
commit. The local entry has been corrected to prefer explicit
DISM_SOURCE_COMMIT; the active run is not restarted for this metadata issue.
Local hybrid training remains untouched (observed past24k steps at launch).

## Design and preparation

The local hybrid run continues unchanged. The authorized remote run uses one
otherwise idle A100 on `chenyc@172.17.135.118`, in the new task directory
`/home/chenyc/dism-swa-control-20260909`. No other users' files, environments,
process details or jobs are inspected or modified. GPU selection uses only
aggregate utilization/memory. Source compilation uses MAX_JOBS=4 and
NVCC_THREADS=2 (at most eight compiler workers), not an unrestricted build.

## Controlled settings

Both models retain width256,15 layers,4x64 RoPE attention,window128,context2048,
untied GPT2 embedding/head,softcap30,BF16 autocast,FP32 parameters,AdamW with
explicit no_decay groups,peak LR1e-3,WD1e-2,warmup1000,total30000 updates,
effective batch64/micro8,seed777,and the same packed data order.

SWA-only removes the DISM branch and widens uniform SwiGLU hidden1024→1728.
Its49,641,856 parameters differ from the hybrid49,678,876 by−37,020 (−0.07452%).
No unused parameters are added. No DISM extension is loaded on A100; pure-SWA
construction also avoids importing DISM/FLA to calculate the parameter budget.
SWA-only has no hard schedule; the hybrid schedule remains unchanged.
The remote FlashAttention version/hardware are recorded separately: this is a
parameter/token-budget comparison, not a controlled wall-clock comparison.

## Token transport

`lm_token_stream.py` binds only local127.0.0.1:18473; the SSH reverse forward
binds remote127.0.0.1:18473. Requests additionally require a generated bearer
secret, stored in task-local mode0600 files, never in source/checkpoint/logs.
Do not expose this service on0.0.0.0 or use a general filesystem HTTP server.

Each indexed effective batch is64x2049 little-endian int32 (524,544 bytes).
Clients split it into eight microbatches and prefetch one effective batch.
Both train and validation requests are idempotent; retries never advance the
logical client cursor. SHA256 checks payloads. Train cache is16 batches,
validation cache100; parquet cursor/pending-token checkpoints every100 batches
allow bounded replay after server/client restart. No whole corpus/token cache
is copied to the remote host. Expect about15.7GB for30k training batches,
plus validation traffic (~1.57GB), excluding retries and protocol overhead.

Source files' paths/sizes/mtimes,tokenizer hash and packing settings define a
service identity. Checkpoints store client cursor/identity, not a network
secret. A resumed client can replay from the exact microbatch. Keep the local
service and SSH master alive while the remote training runs. If disconnected,
the trainer retries then errors visibly; recover from its last completed
checkpoint, reconnect the tunnel and resume with the same service identity.

Local service state:
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/swa-token-service-20260909`.

Initial transport verification: remote cached requests9.71MiB/s; first
request0.354s. The first batch matches eight independent local hybrid-style
PackedStream microbatches byte-for-byte,SHA256
`af5119c4ca9491b2b947eaaae517783b8431627ba0e6342b23d20a5be98fb94e`.
Cached throughput is not fresh-tokenization throughput or GPU training speed.
Fresh requests for16 consecutive batches took4.241s (3.77 effective batches/s,
including local tokenization and SSH transfer). CPU checks total7 pass across
the selected LM contracts and remote-stream tests; no local GPU test interrupted
the hybrid run.

## Evaluation

`eval_lm_positions.py --hybrid .../latest.pt --swa .../latest.pt --output ...`
requires matching steps (completed30k by default), settings and parameter
budget. It evaluates both on identical held-out batches, produces per-position
NLL and paired hybrid-minus-SWA differences, including an EOS-excluded view.
Loss uses chunked head projection and FP32 torch tanh/CE for diagnostics, not
the approximate unreduced CE path with known strict precision failures.

Optional `--reset-context 128 512` evaluates independent short blocks as an
additional context ablation. These are not fixed-length sliding windows:
context ramps within each block. Fifteen SWA128 layers can theoretically
propagate information across1905 preceding tokens, so a better late-position
curve alone does not establish use of arbitrarily long history. Report this
caveat and descriptive errors, not a formal independent-token significance test.

CPU checks: stream cursor replay/epoch/shard traversal,indexed retries/server
restart,microbatch resume,parameter matching and position moments pass.
Remote GPU correctness and launch status are recorded after actual execution.
On the reserved A100, `tests/test_softcap_cross_entropy.py -k
"gpt2_cap30 and mean"` passed all18 production-vocabulary/cap checks, across
FP32/BF16/FP16 and six logit scales. This does not waive the previously
documented small-vocabulary/unreduced failures.

The user additionally authorized reserving one idle GPU while dependencies
build. `reserve_gpu.py` checks aggregate idle state before allocating32GiB,
does no compute, and releases on SIGTERM/SIGINT. Initial reservation: GPU1,
PID3843296, pid file `reservation.pid` in the remote task directory. Only this
task's reservation may be terminated to hand the GPU over to its training.
