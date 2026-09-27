# Wide-attention A100 controls, 3000 steps

User selected12 heads x64, residual256, **FFN1050**, 15 layers. Both SWA-only
and full causal attention have49,675,276 parameters (hybrid49,678,876; gap3600).
The earlier estimate for FFN1152 was wrong: that configuration has50,853,376.
SWA128 includes self; full window=(-1,-1); both use unchanged RoPE theta10000.
Only attention window/architecture differs between the two controls.

Pure attention projection is256->2304 QKV and768->256 output. DISM remains
restricted to width=heads*64. This isolated application snapshot avoids changing
the source hashes used by the already-running local activation-study queue.
Changes relative to source snapshot: wide SlidingAttention projection/reshape,
relaxed width check only for pure-attention models, --attention-heads CLI, and
the existing deployment-only token-server --steps option. Production defaults unchanged.

3000 updates,batch64/micro8,context2048,seed777,AdamW peak1e-3/WD.01,
warmup100,cosine to10% peak,softcap30,offline W&B. No DISM/hard RNG in these arms.
Evaluate100 effective batches every1000 steps, save latest every500 and at finish.
Fresh training, not continuation from the old30k controls; no existing jobs stopped.

Remote account/root: `chenyc@172.17.135.118`,
`/home/chenyc/dism-wide-controls-3k-20260910`.
Uses old task venv read-only, Torch2.6+cu124 and locally built FA2.7.4.post1.
Each arm has its own triton-cache, run/, auth_token, console.log, preflight.log.

Remote user service `dism-wide-controls-3k-20260910` runs queue.py. It only reads
GPU index/memory/utilization summaries, never process details. Eligibility is
memory<256MiB and utilization0 in two consecutive30-second samples. It starts
one arm per available card, excludes its own active cards, and otherwise waits.
No GPU reservation, process termination, package changes or automatic retry.
queue-status.json shows pending/active/finished; per-arm status.json records phase.

Preflight on the chosen GPU checks identical SWA/full initial weights,12-head
attention output/input/parameter gradients against dense attention at127/129/257,
first batch SHA, and a full15-layer BF16 CE/backward/AdamW step. Preflight weights
are discarded; trainer restarts seed777. Failure prevents that arm from training.
CPU parameter counts and syntax checked locally; GPU preflight is pending until
an idle A100 is available. Do not claim GPU validation or training has begun merely
because the queue service is running.

Independent local token services:
`dism-wide-swa-token-3k-20260910` port18477,
`dism-wide-full-token-3k-20260910` port18478.
State under dism-lm-runs/wide-{swa,full}-token-3k-20260910.
SSH master `/tmp/dism-recovery-ssh.HkBVoJ/control` reverse-forwards both ports;
it must remain alive. No corpus copy. Auth files0600; credentials not in source/logs.

Launch status: service started; all cards had existing memory allocations at the
most recent inspection, so the two arms are queued rather than occupying them.
