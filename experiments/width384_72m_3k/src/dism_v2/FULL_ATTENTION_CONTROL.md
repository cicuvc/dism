# Full causal attention + RoPE control

Latest: successfully resumed from1000 on GPU2,PID4048394,user service
`dism-full-resume-20260910.service`. Update1020 loss4.80078,0.661s/update,
finite gradients. `checkpoint-step1000.pt` preserves the pre-resume checkpoint.
Logs append; the outage/no-free-GPU observations below are historical.

Recovery update2026-09-10: the initial process exited due to loss of the token
service; last logged update1950, latest completed checkpoint1000. No model
state beyond1000 is claimed recoverable. Local token service18474 has been
restored under `dism-full-token-20260910.service`; see SWA_CONTROL.md for the
new SSH tunnel and shared recovery status. No GPU job is restarted while all
cards are occupied. The launch observations below are historical.

User-authorized third run, launched2026-09-10 on idle GPU2 of
`chenyc@172.17.135.118` (A100-SXM4-40GB). PID4023852, persistent SSH
session76877; run `/home/chenyc/dism-full-control-20260910/run`.
The local hybrid and remote GPU1 SWA-only continue unchanged.

Model: same49,641,856 parameters as SWA-only,15 layers,width256,4x64 heads,
SwiGLU1728,PreNorm,untied GPT2 embedding/head,RoPE theta10000,context2048.
Only attention scope changes: `flash_attn_func(causal=True,
window_size=(-1,-1))`, recorded as architecture=`full_attention`,window=-1.
This is full **causal** attention, not bidirectional attention. No DISM branch.

Training matches both controls: seed777,30k updates,batch64/micro8,softcap30,
FP32 master weights/BF16 autocast,AdamW with no_decay,LR peak1e-3,WD1e-2,
warmup1000,validation every1000 updates with100 effective batches,W&B offline.
It starts from scratch, not the preflight model. Parameter construction is
identical to SWA-only; changing attention scope does not add RNG draws.

Independent source directory avoids replacing any running SWA source files.
The already-built task-owned FlashAttention2.7.4.post1 venv is reused read-only;
Torch2.6.0+cu124. Triton caches are separate.

Data is not copied to A100. A second local loopback service at18474, SSH reverse
forwarded to remote loopback18474, has its own cursor/cache to avoid repeated
rewinds when controls train at different speeds. Service state is
`/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/full-token-service-20260910`.
Service session45996; SSH master and both token services must remain alive.
Its source identity and first-batch SHA256 exactly match SWA's service.
Fresh16 requests measured3.48 effective batches/s, including tokenization and
SSH; do not confuse this with GPU throughput.

Preflight: N127/128/129/257 full causal attention outputs and all q/k/v
gradients passed against FP32 masked attention. Full15-layer model backward
and optimizer update had finite gradients on every parameter,peak7.728GiB.
CPU remote/dispatch/parameter tests5/5 pass, including explicit full-causal
dispatch and equal parameter counts. First actual training update completed
with loss10.84449 and finite grad_norm1.86677; initialization timing is not
steady-state throughput or evidence of convergence.

Launch (from the independent task directory):
```bash
CUDA_VISIBLE_DEVICES=2 NO_PROXY=127.0.0.1,localhost OMP_NUM_THREADS=8 \
TRITON_CACHE_DIR=/home/chenyc/dism-full-control-20260910/triton-cache \
DISM_SOURCE_COMMIT=c4cb0c4 \
/home/chenyc/dism-swa-control-20260909/venv/bin/python -u -m dism_v2.train_lm \
  --architecture full_attention --output /home/chenyc/dism-full-control-20260910/run \
  --stream-url http://127.0.0.1:18474 \
  --stream-secret-file /home/chenyc/dism-full-control-20260910/auth_token \
  --wandb-mode offline
```

Checkpoint source hashes/snapshots identify uncommitted training application
code on top of the recorded kernel repository commit. No credentials are
written in source or logs. Per-position evaluation currently targets the
hybrid/SWA pair; extending its comparison contract to this third control is a
later analysis step, not yet a completed full-attention evaluation.
