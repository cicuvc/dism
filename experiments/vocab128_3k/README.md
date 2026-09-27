# Original hybrid with Q/K vocabularies reduced to128

User authorized2026-09-10. Queue after vocab_norm completes training AND final
vocabulary evaluation. Independent baseline activation, no vocabulary normalization,
balance auxiliary, GLA, codebook sharing or tied Q/K. Each head has separate
Q/K codebooks shaped[128,64]. Token vocabulary stays GPT2 size50257.

Only codebook size changes: V512→128. FFN1024 remains unchanged to avoid adding
FFN capacity as another variable. Parameters49,678,876→46,729,756 (-2,949,120).
Same seed777, but different allocation sizes change initialization RNG positions;
common parameters are not guaranteed bitwise identical across the two sizes.
This measures sensitivity to codebook capacity, not proof of collapse: even if
NLL is unchanged, the smaller vocabulary may learn a different useful partition.

15 layers,width256,H4,D=DV64,SWA128,context2048,untied GPT2 input/head.
3000 steps,batch64/micro8,LR1e-3,WD.01,warmup100,softcap30,hard_prob0→1,
tanh_finite,OPT13,offline W&B.100 effective validation batches every1000 updates,
checkpoints every500;256-sequence final hard vocabulary-use evaluation.

Main CLI adds --qk-vocab (default512, positive). Independent src/ snapshot and
CUDA/Triton caches; existing local normalization and cloud GLA code untouched.
tests/test_lm_vocab128.py checks production parameter count and codebook shapes,
then real small-model CUDA embedding/core backward, AdamW and eval in0/.5/1 modes.
GPU tests are deferred until the previous experiment finishes and must pass before
training. Failure stops the launcher; no automatic retry or predecessor restart.

User service: dism-vocab128-3k-20260910. Launcher: run_study.py.
Output:
/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910/vocab128

queue-status.json,preflight.log,console.log,metrics.jsonl,latest.pt,best.pt,
offline W&B,vocab-load-final/. Initial phase: waiting_for_vocab_norm.
