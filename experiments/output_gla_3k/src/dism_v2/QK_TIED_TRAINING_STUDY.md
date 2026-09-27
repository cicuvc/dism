# Cloud5090 tied-QK control, 3000 updates

User authorized2026-09-10 on `root@connect.weste.seetacloud.com:34026`.
Independent task root `/root/autodl-tmp/dism-qk-tied-3k-20260910`.
This is original `hybrid`, not `hybrid_shared` and not grouped-head/GQA.

## Parameter matching

Each of4 heads has its own512x64 codebook; within a head, Q and K reference the
same FP32 trainable parameter. `dism_tie_qk_vocab=True`, `dism_vocab_groups=None`.
Both shortconvs retain original SiLU; codebooks do not receive SiLU.

Baseline FFN1024:49,678,876 parameters. Tying removes1,966,080 parameters.
Increasing every layer's SwiGLU hidden dimension by one adds15*(3*256+2)=11,550
parameters, counting biases. FFN1194 restores1,963,500 parameters, leaving
**49,676,296**, just2580 fewer than baseline (0.00519%). This is the closest
integer hidden size;1195 overshoots by8970.1194 is deliberately not64-aligned;
parameter matching takes priority over choosing a convenient GEMM width here.

Remaining settings match local activation study:15 layers, residual256,H4,
D=DV64,V512,RoPE SWA128,FFN SwiGLU,PreNorm,untied GPT2 input/head,softcap30,
AdamW peak1e-3/WD.01/betas(.9,.95),effective batch64/micro8,context2048,
seed777,3000 updates,warmup100,cosine to10% peak,hard_prob0→1 over3000 updates.
393,216,000 training tokens. Every1000 updates evaluate100 effective batches;
save latest every500 updates. W&B offline, project `dism-activation-study`.
Same seed does not imply identical common tensors across architectures: tying
and FFN resizing change random-number consumption during initialization.

## Deployment and data

Reuse the prior task-owned venv read-only:
`/root/autodl-tmp/dism-shared-20260910/venv/bin/python`.
Torch2.8.0+cu128 / FlashAttention2.8.3 / CUDA12.8; local baseline uses
Torch2.13.0+cu130, so this is a cross-environment comparison, not strict same-stack.
No packages installed or shared environments changed. New independent CUDA/Triton
caches and copied TK/GLX headers. `TORCH_CUDA_ARCH_LIST=12.0a`,MAX_JOBS8,
tanh_finite,OPT13,hard_bits0,OUTPUT_Q_ALIAS=kv,OUTPUT_TMA0.
Copied GLX diagonal_scan.cuh SHA matches current local:
`e232c615b0e96bc478aa2e10d29211ed00142a2f3eb7d0c802596450b0262da8`.

Local source snapshot (outside the running three-arm queue's hashed sources):
`dism-lm-runs/qk-tied-3k-deploy-20260910/src`.
Only deployment-specific source change is adding `--steps` to this snapshot's
token server; repository/current local training sources remain unchanged.

Independent local token service unit `dism-tied-token-3k-20260910`,port18476,
state `dism-lm-runs/qk-tied-token-3k-20260910`.
SSH reverse forward18476 with master `/tmp/dism-tied-ssh.3cyVHN/control`.
Auth token is copied into new task root,0600; no SSH password is persisted.
Corpus remains local. Existing token services and training tasks are untouched.
The SSH tunnel must stay alive; there is no unattended password-based reconnect.

Source identity: `6052257b2ca98661a2e51a998b524345c5f08defdb314150fb65467a227c159c`.
First effective batch SHA256:
`af5119c4ca9491b2b947eaaae517783b8431627ba0e6342b23d20a5be98fb94e`.
Compared with locally concatenated8 microbatches from the training PackedStream:
bitwise identical. Remote checksum also matches. Throughput:8.43MiB/s cached,
3.52 fresh effective batches/s (16 batches,4.55s), sufficient for expected training.
Indexed token fetching/checkpoint cursors avoid skipping data on retry.

## Workflow and logs

Launcher source: `experiments/qk_tied_3k/run_remote.py`, copied to task root.
Detached task-owned parent process; logs:

- `job-status.json`:phase/child PID/error.
- `data-check.log`:token identity,hash,throughput.
- `gradient-check.log`:vocab-sharing suite on actual remote CUDA kernels.
- `console.log`:three-step full-size smoke,then resume to3000.
- `run/metrics.jsonl`, `run/config.json`, `run/latest.pt`, `run/best.pt`,offline W&B.

The three smoke updates count toward3000, preserving optimizer/RNG/data state.
Launcher stops on test/training failure and does not automatically retry.
Initial preflight failed before compiling because venv/bin was absent from PATH
(installed Ninja not found). Fixed only child-process PATH to include venv/bin and
CUDA/bin. Original failure log/status are retained; retry parent PID1615.
The remote gradient suite and full-size smoke must pass before formal training.
Do not interpret this deployment note as a completed training result.

Launch confirmed: retry passed all63 vocabulary-sharing tests in294.15s including
fresh extension compilation. Full-sized FFN1194 smoke completed3 updates, all losses
and gradients finite, checkpoint parameter count49,676,296 verified. Resumed trainer
PID3642, parent1615, status=training. Update4 loss10.73299, grad norm1.74055;
startup timings are not a steady-state throughput measurement. Training is ongoing,
not completed. Remote data/config identities match the intended3000-step study.
