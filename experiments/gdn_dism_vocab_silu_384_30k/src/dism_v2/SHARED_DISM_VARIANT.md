# Shared-projection DISM + SWA (prepared, not trained)

Explicit configuration: `LMConfig(architecture='hybrid_shared')` or future
`train_lm --architecture hybrid_shared`. No run has been started. Existing
hybrid/SWA/full-attention defaults, remote source files and active processes
are unchanged. The new variant is not checkpoint-compatible with old hybrid
weights under strict loading; it must not silently resume an old hybrid run.

`SharedDismAttention` extends the existing DISM module. Its q_proj,k_proj,v_proj
each execute once. Each projected tensor feeds two separate causal depthwise
shortconvs: the existing DISM conv and new q_swa_conv,k_swa_conv,v_swa_conv.
New convs use kernel size4,bias=False,swish/silu activation,matching DISM's
convolution convention. SWA Q/K then receive the existing interleaved RoPE;
SWA uses FlashAttention causal window128. DISM receives no RoPE.

The two raw BF16 attention outputs are added directly in[B,N,H,64] before
the shared FusedRMSNormGated and o_proj. There is no separate SWA qkv/out
projection and no second SWA residual branch in Block. The outer PreNorm,
FFN1024 and residual additions remain. Sharing the nonlinear gate/norm is an
intentional architectural change, not an algebraically equivalent rewrite.
Pure-hard DISM still runs alongside SWA. DISM's global random direction,row
RNG replay,hard-probability annealing and no_decay policies are unchanged.

## Exact parameter counts (width256,15 layers,4 heads,QK vocab512)

| Model, all with FFN1024 | Total parameters | Per-layer excess vs SWA-only |
|---|---:|---:|
| SWA-only / full causal | 41,510,656 | 0 |
| Original hybrid | 49,678,876 | 544,548 |
| Shared hybrid | 45,792,796 | 285,476 |

Removed4*256^2=262,144 parameters/layer; added3*4*256=3,072.
Net reduction259,072/layer,3,886,080 total. The per-layer gap shrinks47.6%.
The two DISM vocabularies alone contribute262,144 parameters/layer,so most
remaining excess cannot be removed by projection sharing.

For an eventual approximately matched control,64-aligned FFN1408 replaces
the previous1728 suggestion. Such a control has45,945,856 parameters,+153,060
(~0.33%) relative to this variant; it is not within the previous0.1% pairing
criterion. No existing controls or comparison tolerances are changed.
The analytic helper's shortconv bias assumption was corrected to bias=False;
the original hybrid's rounded FFN match remains1728.

## Validation scope

CPU tests verify exact parameter counts,absence of duplicate SWA projections,
one execution per shared projection,the same projected tensor reaching both
convs,raw-output addition before gate/norm,and gradients through every tested
parameter using explicit CPU substitutes for the CUDA primitives. These
connectivity tests are not CUDA numerical validation.
CUDA one-layer forward/backward/RNG-replay smoke tests for soft/mixed/hard are
included but deferred while existing local training/queued evaluation runs.
No formal training or remote deployment of this variant was performed.
