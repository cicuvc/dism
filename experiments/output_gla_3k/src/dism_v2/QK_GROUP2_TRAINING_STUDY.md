# Grouped, untied Q/K codebook control

2026-09-10: cloud5090 `root@connect.weste.seetacloud.com:34026`.
Task `/root/autodl-tmp/dism-qk-group2-3k-20260910`.
`dism_vocab_groups=2`, `dism_tie_qk_vocab=False`, baseline activations:
heads0/1 share E_q[0] and independently E_k[0]; heads2/3 share E_q[1] and E_k[1].
Q/K projection, convolution, attention state and KV cache remain per head.
This shares codebooks, not actual GQA. No codeword SiLU or centering.

Total codebook parameter count equals the previous tied-within-head G4 arm,
so retain FFN1194:49,676,296 parameters,2580 below original hybrid.
All settings match:15 layers,width256,H4,D=DV64,V512,SWA128,context2048,
3000 steps,batch64/micro8,seed777,LR1e-3,WD.01,warmup100,hard_prob0→1,
softcap30,tanh_finite,OPT13,offline W&B,100-batch validation every1000 updates.

Previous tied arm confirmed completed3000, final100-batch pure-hard validation
NLL3.942519665. GPU showed2MiB/0% before new launch; nothing was stopped.
Reuse previous task's verified code, venv and compiled extensions without edits;
63 remote tests had already covered grouped/untied gradients, both directions,
soft/mixed/hard modes and full model paths. New run/triton cache/logs are separate.

Independent local service `dism-group2-token-3k-20260910`,port18479,
state dism-lm-runs/qk-group2-token-3k-20260910; SSH master
`/tmp/dism-tied-ssh.3cyVHN/control`. Data stream starts at0 with the same3000-step
identity/order; prior service and checkpoint remain untouched.

Launcher: `experiments/qk_tied_3k/launch_group2.py`.
New trainer PID5894, detached, no automatic retry. `console.log` at task root;
`run/metrics.jsonl`, `run/config.json`, checkpoints and offline W&B below run/.
Confirmed update30:loss9.34987545,finite grad norm1.39272,1.386s/update.
This is startup health only; training is not yet complete.
