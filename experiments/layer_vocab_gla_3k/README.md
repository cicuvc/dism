# Cross-layer codebook sharing on the GLA hybrid

User authorized2026-09-10 cloud experiment after output-GLA finished.
Previous GLA3000-step final validation NLL3.941790104 (baseline3.943258286).
Cloud GPU summary was2MiB/0% before launch; no task stopped/restarted.

Use GLA hybrid from the previous experiment, but share the Q codebook across
all15 layers and independently share K across all15 layers. Q/K remain untied;
4 heads retain distinct512x64 codebooks. Shared shape per side[4,512,64].
Q/K/V projections, convolutions, tau, GLA gates/state and all other parameters
remain layer-specific. No normalization, balance, or head grouping added.

FFN1024 unchanged:49,694,356→46,024,340 parameters (-3,670,016).
15 layers,width256,H4,D=DV64,V512,SWA128,context2048,untied GPT2 embeddings.
Same3000 steps,batch64/micro8,seed777,LR1e-3,WD.01,warmup100,softcap30,
hard_prob0→1,tanh_finite,OPT13,offline W&B.100 effective validation batches
every1000 updates; save every500. This is not an exactly parameter-matched arm.

LMConfig.dism_share_vocab_layers / CLI --dism-share-vocab-layers. Construct
layers normally, then retain layer0's codebooks under a single model-level
ParameterDict `shared_vocab`. Remove per-layer registered codebook parameters;
blocks reference the owner without registering it. Forward always reads the
current owner's parameters, so device conversion/checkpoint loading cannot
split aliases. Each layer separately casts FP32 masters to BF16; gradients
from all layer uses accumulate in FP32. Shared codebooks remain no_decay.
Non-codebook initialization RNG matches the unshared GLA construction at the
same seed, since discarded layer codebooks are still initialized first.

Tests: tests/test_lm_layer_vocab.py. CPU2 passed for tied/untied ownership,
device/dtype conversion, strict checkpoint roundtrip, optimizer uniqueness and
parameter count. Remote GPU3 compare the shared two-layer model against an
unshared model with identical codebooks: losses match and shared gradients
equal the sum over layer-local codebooks at hard probabilities0/.5/1, followed
by AdamW and finite eval. These must pass before training.

Cloud task: /root/autodl-tmp/dism-layer-vocab-gla-3k-20260910
SSH root@connect.weste.seetacloud.com:34026, detached launcher PID16881.
lm_model.py/train_lm.py are independent copies. Unchanged package files link
to the completed output-GLA snapshot; kernel loaders resolve their original
source paths and reuse that experiment's compiled extension cache. Previous
model/training source and checkpoints are not overwritten. Triton cache separate.

Token service dism-layer-vocab-token-3k-20260910 on local port18481, independent
state dism-lm-runs/layer-vocab-token-3k-20260910, SSH reverse forwarding, same
3000-step identity/order starting at0. Local token service and SSH master must
remain available. No local GPU training is started by this experiment.

Remote status.json,preflight.log,launcher.log,console.log; run/metrics.jsonl,
run/config.json,run/latest.pt and offline W&B. Launcher checks final3000 steps
and parameter count, stops on failure without automatic retry.

Remote preflight5 passed in91.48s. Training confirmed at step20: CE9.999375343,
finite gradient norm1.4091303,1.478s/update (~88.7k tokens/s), peak allocated
9.909GiB. Current state training, not a completed run.
