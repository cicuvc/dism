# A100 384-wide GDN + full RoPE attention, 30k

User requested the matched control then moved execution to A100. Local service
dism-gdn-full-384-30k-20260911 was stopped during preflight before training;
its partial preflight log is retained, not a completed test result.

Model flag --gdn-full:12 layers [GDN,GDN,GDN,FullAttention] x3, width384,
H6, FA head_dim64, standard full causal FlashAttention window=(-1,-1),
RoPE theta10000. No DISM, no SWA. GDN unchanged: K64/V128, expand_v2,
shortconv4, output gate. LayerNorm PreNorm, untied GPT2 input/output.
FFN1522 gives72,182,172 parameters vs DISM control72,188,142,
difference5970 (0.0083%). Global dispatch field remains hybrid; gdn_full flag
selects no-DISM execution and the FA blocks record full_attention/window=-1.
No hard-prob RNG or DISM backward extension is used in this arm.

Fresh seed777,30000 steps, batch64/micro8, context2048, AdamW lr1e-3/WD.01,
no_decay grouping, warmup1000/cosine to.1peak, softcap30, offline W&B.
Validation100 effective batches every1000; save every1000. No previous run resumed.

Remote task directory /home/chenyc/dism-gdn-full-384-30k-20260911,
account chenyc@172.17.135.118. Read-only reuse of task venv
/home/chenyc/dism-swa-control-20260909/venv/bin/python (Torch2.6cu124,FA2.7.4.post1).
No shared environment edits. Independent source, run/, Triton cache, logs.
remote_run.py waits for two consecutive idle GPU summary readings then runs
stream preflight, dense/causal FA check and full microbatch forward/backward/
finite-gradient/AdamW test, then training. No reservation or other-user process reads.
Remote service dism-gdn-full-384-30k-20260911; status.json records actual stage.

Independent local token service dism-gdn-full-token-30k-20260911 on18482,
state under dism-lm-runs/gdn-full-token-30k-20260911. Existing SSH master
/tmp/dism-recovery-ssh.HkBVoJ/control adds reverse18482 tunnel. No corpus copied;
auth_token0600, no secret in logs. Tunnel and local service must stay available.

This control tests whether weak long-context retrieval persists with standard
full attention instead of DISM. It does not alone prove the hypothesized GDN
shortcut/weak NTP pressure mechanism. Hardware/dependency versions differ from
local5090, so throughput is not directly comparable.

Confirmed launch on A100 GPU1: stream first-batch SHA256
af5119c4ca9491b2b947eaaae517783b8431627ba0e6342b23d20a5be98fb94e,
fresh stream2.88 effective batches/s. GPU preflight2 passed in225.77s,
loss10.901881, finite gradient norm4.459622. Remote lacks causal_conv1d;
FLA explicitly falls back to its Triton convolution (validated, no install).
Training PID1024717; runner1021295. Step10 loss10.883194, finite gradient
norm1.680491,1.0435s/update (~125.6k tokens/s), peak9.688GiB.
W&B offline. First-step32.18s includes startup overhead, not steady throughput.
