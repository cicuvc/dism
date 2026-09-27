# [GDN,GDN,GDN,DISM] x3, width384

User authorized2026-09-11 and explicitly removed SWA from DISM layers.
There is no SWA anywhere: GDN at layers1/2/3,5/6/7,9/10/11; pure DISM at4/8/12.
All blocks keep LayerNorm PreNorm and their SwiGLU FFN1024. Residual384,12 layers,
untied GPT2 embedding/head,context2048. DISM D=DV64,H6,V512,original shortconv
SiLU; no balance,vocab norm,GLA or vocabulary sharing.

GDN from fla.layers.gated_deltanet.GatedDeltaNet,hidden_size384,num_heads6,
head_dim64,expand_v2 (value head_dim128),mode chunk,layer_idx=physical0-based
index. Library defaults retain shortconv4 with SiLU,output gate,beta sigmoid,
allow_neg_eigval=False and Q/K L2 normalization in the recurrence kernel.
No recurrent cache carried between packed sequences. This is training/prefill;
the repository's CPU hard-generation wrapper is not extended for GDN caches.

Exact parameters66,593,550,not matched to the previous72,188,040 hybrid.
FFN remains1024 to keep that capacity unchanged. Main flag --gdn-dism / config
gdn_dism=True,architecture retains hybrid so existing CUDA setup and hard_prob
schedule apply to the3 DISM layers. GDN layers do not consume DISM row RNG.
Requires layer count divisible by4,original hybrid,no output GLA/cross-layer sharing.

Same3000 updates,batch64/micro8,seed777,LR1e-3,WD.01,warmup100,softcap30,
hard_prob0→1,tanh_finite,OPT13,offline W&B. Validation100 effective batches every
1000 steps,checkpoint every500. Module-specific GDN no_decay metadata honored.
The generic vocabulary evaluator currently assumes DISM at every layer;
automatic post-training vocabulary evaluation is therefore NOT scheduled here.

Independent src/ snapshot and CUDA/Triton caches. Local GPU summary was569MiB/0%
before launch. Service dism-gdn-dism-384-3k-20260911. Tests must pass before
training: architecture/count/no-SWA checks; full4-layer group training in
soft/mixed/hard; full12-layer microbatch8,N2048 mixed forward/backward/AdamW/eval.
No automatic retry, batch reduction or silent numerical-tolerance changes.

Output:
/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dism-lm-runs/activation-study-3k-20260910/gdn_dism_384

queue-status.json,preflight.log,console.log,metrics.jsonl,latest.pt,best.pt,
offline W&B. Initial state gpu_preflight, not successful training yet.

Preflight completed5 passed in356.98s (including fresh compilation). Full mixed
microbatch peak allocated8.695GiB,loss10.90462,finite pre-clipping grad norm3.87076.
Training confirmed step20: CE9.6909027,finite pre-clipping grad norm1.68903,
0.652s/update (~201k tokens/s),peak allocated9.458GiB. Current state training;
these startup checks are not evidence of final training quality.
