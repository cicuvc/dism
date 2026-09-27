# A100 384-wide GatedDeltaCache (GDC), 30k

Experimental arm: every token mixer is GatedDeltaCacheNet (deltastar
deltacache.py), no positional encoding, residual width 384. 12 layers, H6,
K64, V64 (dev kernel limit Dv<=64, unlike GDN's V128), FFN1664 ->
72,420,312 parameters. Chunkwise Triton kernels from deltastar/gdc_chunk.py
(chunk_size 64, fp32 state, oracle-fallback off on CUDA).

Kernel numerics note: backward tl.dot ops run in tf32 (A100/triton3.2 ieee
fp32 dot codegen is ~25x slower: 889us vs 35us per scan micro-kernel);
ieee is kept only in solve_tril_64 (the (I+X)^-1 inversion) and the
chunk_decayed_gram_fwd that feeds it. Gradient cosine similarity vs the
fp32 WY-autograd oracle is 1.000000 for every tensor; a sparse tail of
elements (<=4 per tensor) carries ~1e-3 relative magnitude noise. Self-test
tolerances relaxed accordingly (see gdc_chunk.py _self_test_bwd).

A100 baseline after the tf32 switch (idle GPU): forward 140ms, backward
199ms per layer at B8 N2048 H6; preflight steady microbatch 1.07s, DDP
2-GPU update ~8s, ~16k tokens/s, peak 11.2GiB. 30k steps ~66h.

Fresh seed777, 30000 steps, batch64/micro8, context2048, AdamW lr1e-3/WD.01,
no_decay grouping, warmup1000/cosine to .1peak, softcap30, offline W&B.
Validation 100 effective batches every 1000; save every 1000.
Same hyperparameters as the gdn_full_384_30k control for comparability.

Remote task directory /home/chenyc/gdc-384-72m-20260922,
account chenyc@172.17.135.118. Read-only reuse of task venv
/home/chenyc/dism-swa-control-20260909/venv/bin/python (Torch2.6cu124,
fla0.3.0). Training runs on GPUs 6,7 via torch.distributed.run
(nproc2, master_port 29617); remote_run.py refuses to start if either is
busy. status.json records actual stage; console.log has the metrics.

Local token service on port 18488, state under
dism-lm-runs/gdc-token-30k-20260922 (identity
2e836027b7aaf15fc8da82cd47929dcc9a0b010adab4ea6aa9f2180afc88844e).
A100 cannot reach us directly; the SSH master /tmp/gdc-ssh-ctrl carries a
reverse tunnel (-R 18488:127.0.0.1:18488). The local service and tunnel
must both stay alive for the whole run. Service restart command:
  cd ~/cs/project/dism-exp && setsid nohup \
    ~/miniconda3/envs/blkw/bin/python -u -m dism_v2.lm_token_stream \
    --data $DATA/finewebedu --tokenizer $RUNS/tokenizer-gpt2 \
    --state-dir $RUNS/gdc-token-30k-20260922 --port 18488 \
    >> $RUNS/gdc-token-30k-20260922/service.log 2>&1 &
(DATA=/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc,
RUNS=$DATA/dism-lm-runs)

Confirmed launch: stream first-batch SHA256
af5119c4ca9491b2b947eaaae517783b8431627ba0e6342b23d20a5be98fb94e (matches
control), first steps loss 10.79 -> 10.16 by step 50, grad_norm ~2.
