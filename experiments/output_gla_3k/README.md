# DISM output GLA experiment

User design: keep the original parallel SWA128 branch. Feed the DISM retrieval
output as V into simple GLA, sharing continuous post-shortconv Q/K. Per head:
`S_t = exp(g_t) S_(t-1) + k_t r_t^T`,
`y_t = r_t + q_t^T S_t / sqrt(64)`.
`g_t = -exp(A_log) * softplus(a_proj(x_t) + dt_bias)` is FP32.
Retain the original output gated RMS norm and projection. No new Q/K
normalization, balance regularizer, or change to DISM's matching semantics.
GLA state starts at zero for each packed training sequence. Full-sequence
training/prefill only; the existing hard CPU generation cache has not been
extended for this recurrent state.

Main integration: LMConfig.dism_output_gla, CLI --dism-output-gla. Only the
original hybrid architecture accepts the flag. Fixed tuple unpacking for V and
FLA's output/state, made DISM output contiguous, removed the unused g_proj,
and prevented log(0) in A initialization. A_log/dt_bias remain no_decay.

49,694,356 parameters, +15,480 (+0.031%) over baseline; FFN remains1024.
Original baseline activation, separate Q/K vocabularies per head,15 layers,
width256,H4,D=DV64,V512,context2048,untied GPT2 input/output embeddings.
3000 steps,batch64/micro8,seed777,LR1e-3,WD.01,warmup100,softcap30,
hard_prob0→1,offline W&B. Every1000 updates evaluates100 effective batches,
checkpoint every500 updates. New parameters consume initialization RNG, so
same seed does not imply all common tensors are initialized identically.

Local tests: `tests/test_lm_output_gla.py`,5 passed in111.38s including compilation.
Covers parameter count/no_decay/config rejection; simple GLA output and all
input gradients against explicit FP32 recurrence; real CUDA DISM+GLA+SWA+CE
backward/AdamW in soft/mixed/hard; pure-hard causal-prefix comparison.
These are small-model smoke tests, not a long-context stability guarantee.

User changed deployment from a local queue to cloud5090. No local queue service
was started; balance remains running unchanged. The group2 experiment completed
3000 steps (final val NLL3.944524765); GPU summary showed2MiB/0% before launch.

Cloud: root@connect.weste.seetacloud.com:34026
Task: /root/autodl-tmp/dism-output-gla-3k-20260910
Launcher: run_cloud.py (detached parent PID9980).
Separate code, output and CUDA/Triton caches; reuse the task-owned installed venv
and read-only GLX from the preceding experiment. Remote smoke tests must pass
before training automatically starts. No automatic retry on failure.

Independent local token service dism-gla-token-3k-20260910,port18480,through SSH
reverse forwarding. Source is the existing3000-step-capable token service snapshot,
new state directory dism-lm-runs/gla-token-3k-20260910; starts at batch0.
Previous token services/checkpoints are unchanged. SSH master ControlPersist is
12h; the local token service and tunnel must remain available during training.

Remote logs: status.json,preflight.log,launcher.log,console.log;
run/config.json,run/metrics.jsonl,run/latest.pt and offline W&B.
The remote launcher validates final step/parameter count on successful completion.
No post-training vocabulary evaluation is automatically scheduled remotely yet.

Remote preflight:5 passed in354.57s including fresh CUDA/GLA compilation.
Trainer PID13366 confirmed at step40: CE8.547970772, finite gradient norm1.3387002,
1.536s/update (~85.3k tokens/s), peak allocated9.963GiB. W&B offline run
jnx4g0aw. Training is running, not complete. Token-stream identity
6052257b2ca98661a2e51a998b524345c5f08defdb314150fb65467a227c159c.
