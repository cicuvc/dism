# v3 soft with local top-k Q reads in hard mode

2026-10-25. This document records the initial checkpoint-only diagnostic; that
screen did not change training, checkpoints or production dispatch. The later
opt-in implementation, validation and timing are documented separately in
[Top-four hard training](TOPK_HARD_TRAINING.md). Neither establishes final trained
quality.

## Construction

Keep native v3 soft completely unchanged. In hard mode, commit only the native
argmax Q/K symbols to the mainline history. For each current query, read several
alternative Q symbols from that same prior prefix. They share SQ/SK/V/tau and
the K string. Every candidate retains its own denominator and unit fallback:

```
A[i,j] = exp(tau) * (1 + H_hard[i-1,j-1])
N[i,c] = sum_{j <= i, b[j] == c} A[i,j] * dot(sq[i],sk[j]) * v[j]
Z[i,c] = sum_{j <= i, b[j] == c} A[i,j]
pi[i,c] = pQ[i,c] / sum_{u in S[i]} pQ[i,u], for c in S[i]
O[i] = sum_{c in S[i]} pi[i,c] * N[i,c] / (1 + Z[i,c])
```

S contains the native argmax plus the highest other k-1 Q logits. Native
tie-breaking is preserved even where an FP32 projection differs slightly from
the native BF16-operand vocabulary kernel. k=1 is original hard; k=512 is full
v4-style candidate-normalized hard reading with SQ/SK, evaluated at v3 weights.
This is neither a persistent beam of Q strings nor a change to K assignments.
Later layers see the modified activations normally; their labels are recomputed.

Top-k renormalization is intentional. It preserves the k=1 control, but differs
from dropping full-vocabulary tail mass without reallocating it. This experiment
does not test the denominator-free RMSNorm variant or the new hybrid fusion.
Both checkpoints retain their own trained norm/gate and GDN configuration.

## Implementation and controls

- [topk_hard_reference.py](../tools/topk_hard_reference.py): bounded pair tiles,
  eager PyTorch diagonal match-length scans, per-symbol max scaling/scatter
  normalization, and signed SQ/SK readout. No custom CUDA/Triton code added.
- [probe_topk_hard_readout.py](../tools/probe_topk_hard_readout.py): temporary
  inference-only wrapper around `compiler.voc_forward`; restores it on exit.
  Calls the native operator for mainline labels and the numerical baseline.
  Rejects gradient-enabled use; a training backward is not installed.
- Main screen uses `native + (FP32 top-k - FP32 top-1)` before casting to BF16.
  This anchors k=1 exactly to the native model and reduces irrelevant baseline
  implementation drift. It is a numerical diagnostic correction, not the exact
  proposed operator or a proposed deployment implementation.
- Separate `--direct` controls replace the output with the FP32 candidate
  mixture itself. The top-4 benefit survives this change.
- Pure native all-hard and unchanged native all-soft are evaluated on the same
  inputs. A `sham` control directly reconstructs k=1 without the correction.
- No model parameters or registered buffers change. All four complete jobs
  verify buffers, including RNG/schedule state. No checkpoint is written.

The standalone reference check passes eight FP64 cases: k=1/2/4/8, random and
all-equal mainline symbols, tau=0/.3, packed segment boundaries, and cross-tile
history. An independent sequential candidate oracle agrees at atol=2e-10,
rtol=2e-9. Changing current read probabilities with labels fixed changes only
that output row (FP64 GEMM rounding tolerance 2e-12). Native-anchored k=1 exactly
reproduces token losses at 2K/8K for both checkpoints.

## Checkpoints and inputs

Use the same final step-22888 checkpoints as
[the soft/hard head diagnostic](SOFT_HARD_HEAD_DIAGNOSTIC.md): baseline C with
RMSNorm/SiLU (`baseline`) and the resumed LayerNorm/sigmoid scheduled run
(`ln_scheduled`). Exact paths are read from ignored
`build/soft_hard_heads/checkpoints.json` and included in every result manifest.
Only this worktree's `flash_dism` and `cu_flash_dism` are imported.

PG19 uses the first four local test documents, BOS-prefixed 2K and 8K prefixes,
all next-token targets scored, physical batch 1. All-soft/ordinary hard losses
use the checkpoint's BF16 CE path, MXFP8 FFN, TF32 disabled, no planner.

The small 8K NIAH suite has two depths (.05/.95), two trials each, seed 20261025.
It inserts the complete phrase `The special magic number is NUMBER.` into a
PG19 background; the ending query requests that number. Inputs begin with BOS.
Absent controls replace the complete needle with equal-length background text;
the answer target and sequence length are identical. Score only the six-token
answer budget through the FP32 head. This **full-phrase** suite differs from the
older bare-number-tail suite; its scores cannot be compared directly to old
NIAH reports. Input tokens and their SHA256 are saved.

## PG19 results

Mean token NLL over four equal-length documents; lower is better. Main screen:

| Checkpoint | Context | Native hard | Native soft | Top-2 | Top-4 | Top-8 |
|---|---:|---:|---:|---:|---:|---:|
| C | 2K | 3.817645 | 3.790205 | 3.812396 | 3.812958 | 3.812033 |
| C | 8K | 3.814852 | 3.784602 | 3.813279 | 3.812213 | 3.813977 |
| LN/sigmoid | 2K | 3.782918 | 3.773080 | 3.776587 | 3.776736 | 3.776244 |
| LN/sigmoid | 8K | 3.779772 | 3.765687 | 3.775423 | 3.774590 | 3.774957 |

Direct reconstruction controls, without anchoring to native output:

| Checkpoint | Context | Native hard | Direct top-4 | Direct all 512 |
|---|---:|---:|---:|---:|
| C | 2K | 3.817645 | 3.813028 | 3.815322 |
| C | 8K | 3.814852 | 3.812721 | 3.818204 |
| LN/sigmoid | 2K | 3.782918 | 3.776333 | 3.776883 |
| LN/sigmoid | 8K | 3.779772 | 3.774501 | 3.776416 |

Top-4 gives a small consistent improvement in these aggregate cells. Full
vocabulary enumeration is not better; it hurts C at 8K. The experiment supports
trying a small candidate set, not assuming more candidates are always useful.
It does not close the whole native hard/soft gap.

Average retained Q probability mass across PG19 layers/documents:

| Checkpoint | Top-2 | Top-4 | Top-8 |
|---|---:|---:|---:|
| C | .461 | .589 | .697 |
| LN/sigmoid | .487 | .594 | .683 |

Thus top-4 is not an accurate approximation to the full probability mixture.
Its favorable result can include useful truncation/reweighting, not just cheap
evaluation of almost all probability mass.

## Small 8K NIAH screen

Mean answer NLL and retrieval gain (`absent NLL - present NLL`), four paired
cases per checkpoint. Higher gain is better. Main native-anchored implementation:

| Checkpoint | Mode | Present NLL | Gain | Correct target tokens |
|---|---|---:|---:|---:|
| C | hard | .4681 | 2.8224 | 23/24 |
| C | soft | .2843 | 3.0426 | 24/24 |
| C | top-2 | .3570 | 2.9555 | 24/24 |
| C | top-4 | .3702 | 2.9331 | 24/24 |
| C | top-8 | .3386 | 2.9581 | 24/24 |
| LN/sigmoid | hard | .6424 | 2.4392 | 23/24 |
| LN/sigmoid | soft | .2570 | 2.8548 | 24/24 |
| LN/sigmoid | top-2 | .5584 | 2.5120 | 22/24 |
| LN/sigmoid | top-4 | .5503 | 2.5351 | 23/24 |
| LN/sigmoid | top-8 | .6614 | 2.4142 | 22/24 |

Top-4 improves NLL and gain for both models here; token accuracy does not
uniformly improve. LN/sigmoid top-8 regresses. Four cases are too few and token
accuracy is near ceiling: this is a signal for further work, not a general
long-context benchmark result. Direct-reconstruction control was run on PG19,
not this NIAH suite.

## Numerical and performance limits

FP32 reconstructed top-1 vs native raw DISM output has median relative L2 about
.0020, maximum .00415 across main-screen PG19 captures. Q label disagreements
between FP32 logits and native labels are below 2.2e-6 averaged over captures;
native labels are always used for the history and candidate slot zero.

Direct k=1 `sham` 2K/8K full-model mean-loss shifts on document zero range from about
-0.00147 to +0.00041 across the paired main/direct controls. Tiny numeric changes
can trigger downstream argmax differences, so individual-token losses are not
bitwise stable under direct reconstruction. Main top-4 shifts are larger and
the mean improvement survives the direct control, but fine ranking of close
variants should not be overinterpreted. No sanitizer run is claimed; no custom
kernel was changed. The eager reference is roughly 1.5 s per 8K model forward
versus about .036 s native hard on this GPU, excluding initial compilation.
This is not a candidate-kernel throughput measurement or subquadratic algorithm.

## Artifacts and next step

Ignored results: `build/topk_hard_readout/{baseline,ln_scheduled}/results.json`
and `{baseline_direct,ln_scheduled_direct}/results.json`, with inputs and completion
markers. `smoke` contains the initial single-document integration check.
`guard_smoke` additionally checks the final inference-only wrapper at N=256;
it is not included in the reported 2K/8K means.

Next useful experiment: matched short continuation with top-4 **also in training
hard rows**, while preserving native v3 soft. Current-Q mixture weights should
receive a direct gradient; discrete support/history/K selection still do not.
Keep SQ/SK, norm/gate, layer layout and hard-probability policy controlled. This
screen alone does not justify changing production defaults or launching a long run.
