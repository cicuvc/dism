# Pre-convolution Q/K SiLU, queued fourth local arm

User authorized2026-09-10: after no_qk_silu finishes, run a fresh3000-step arm:

```
Q: q_proj -> SiLU -> causal shortconv(activation=None)
K: k_proj -> SiLU -> causal shortconv(activation=None)
V: v_proj -> original causal shortconv(activation=SiLU), unchanged
```

Original hybrid15 layers,width256,H4,D=DV64,V512,FFN1024,parallel SWA128,
49,678,876 parameters. Independent Q/K codebooks unchanged; no codebook activation,
tying/grouping or centering. Same raw initialization seed777 and deterministic data
order as the three local arms. Effective batch64/micro8,context2048,3000 updates,
warmup100,LR1e-3/WD.01,softcap30,hard_prob0→1,offline W&B.100 validation batches
every1000 updates,save every500. About393M training tokens,not20 tokens/parameter.

Implementation in isolated src/ snapshot: dism_activation=preconv_silu. Existing
queue hashes/current training sources are not edited. Shared-hybrid support applies
preactivation only to the DISM Q/K branch, not its SWA branch. Incremental generation
in this snapshot also applies the same ordering. Use this snapshot's model loader
for its checkpoint until the option is integrated into the main application.

User service `dism-preconv-silu-3k-20260910` waits on the original study manifest's
complete status, including no_qk_silu final vocabulary evaluation. No current GPU
tests/training are interrupted. If the previous study fails/stops, this arm does not
start. After completion, run the17 activation tests (including exact hook checks of
preconv SiLU input and actual CUDA gradients at0/.5/1), then fresh training and final
256-sequence vocabulary load evaluation. Failures stop this arm; no automatic retry.
CPU initialization/parameter/chain-rule tests run before queueing; GPU checks deferred
until previous GPU work finishes. Isolated extension/Triton caches avoid changing
the previous run's compiled files.

Output: `dism-lm-runs/activation-study-3k-20260910/preconv_silu/`.
queue-status.json,preflight.log,console.log,metrics.jsonl,latest.pt,best.pt,
offline W&B and vocab-load-final/. The original manifest still describes its first
three arms; the new service/queue-status.json tracks this fourth arm separately.
Current state is queued, not a claim of completed GPU preflight or training.
