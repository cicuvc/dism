# DISM pre-normalization soft/hard outputs and gradients

2026-10-25. Three final step-22888 checkpoints: original RMSNorm/SiLU gate
planner control C (`baseline`), LayerNorm/sigmoid with annealed hard probability
(`ln_scheduled`, completed continuation from step 14000), and LayerNorm/sigmoid
with fixed hard probability 0.5 (`ln_fixed05`). The RMSNorm/sigmoid training run
was left running throughout this diagnostic.

## Measurement

`tools/probe_soft_hard_heads.py` evaluates the first four local PG19 test documents
at their 2048- and 8192-token prefixes. Inputs begin with BOS; targets are identical
between paths and every token is scored. One independent document per batch.
The same model is evaluated all-soft and all-hard, in evaluation mode with
ordinary autograd enabled. No optimizer update or checkpoint mutation occurs;
the model's RNG counter is asserted unchanged. BF16 CE, MXFP8 FFN, eager execution,
planner disabled and quadratic hard backward are used. CUDA allocator capped at
30% of device memory to coexist with training; peak allocated memory was 9.25 GB.

Capture is at the input of `_combine_cuda`: the **raw DISM branch tensor before
GDN addition**, therefore before output RMSNorm/LayerNorm and gate. It is not the
post-projection mixer output or the hybrid sum. `retain_grad()` captures dL/dO
from the actual LM loss. Gradients are multiplied by document token count to
remove the explicit 1/N averaging factor; length-dependent attention effects
remain. Reduction is over token and value dimensions for each head. FP64 CPU
statistics report soft/hard RMS, cosine, ||hard-soft||/||soft|| and hard/soft norm
ratio. Additional 1024-token position bins are saved.

This is an end-to-end path comparison: later DISM layers see different upstream
inputs. It does not isolate a single head's counterfactual switch. The first
DISM layer follows identical pure GDN layers. Four document prefixes are a small
diagnostic sample, not a replacement downstream benchmark or training-seed study.

## Results

The following medians are over 4 documents × 6 DISM layers × 12 heads (288 cells),
not over tokens or over pooled model vectors. Different models' head indices are
not assumed to represent the same learned function.

| 8K statistic | Baseline C | LN + sigmoid scheduled | LN + sigmoid fixed .5 |
| --- | ---: | ---: | ---: |
| Mean soft NLL | 3.784695 | 3.765882 | 3.815972 |
| Mean hard NLL | 3.814866 | 3.779710 | 3.815945 |
| Soft output RMS, median | 2.70244e-3 | 4.02566e-3 | 6.81748e-7 |
| Hard output RMS, median | 3.22273e-3 | 2.74756e-3 | 1.31641e-6 |
| Output cosine, median | .24428 | .46810 | .11115 |
| Gradient cosine, median | .94375 | .94583 | .99789 |
| Gradient relative L2, median | .34315 | .33789 | .06546 |
| Soft gradient RMS, median | 3.99769 | 1.09560 | 2.82611 |
| Hard gradient RMS, median | 4.51483 | 1.10844 | 2.83188 |
| Cells with soft output RMS < 1e-5 | 8/288 | 6/288 | 184/288 |

The 2K output cosine medians are .22498/.42734/.07091 and gradient cosine medians
.95443/.95458/.99782, showing the same broad pattern. 2K mean soft/hard NLL:
3.790141/3.817758, 3.773115/3.782766, 3.817278/3.818346 respectively.

Fixed-.5 soft-output RMS medians by zero-based layer 6–11 are
2.36e-9, 1.19e-6, 3.60e-7, 2.34e-9, 2.51e-6, 3.36e-5. The suppression is not
limited to a single layer. Its overall median is about 5905 times smaller than
the scheduled LN/sigmoid model. However some heads retain substantial outputs:
the fixed-.5 soft-output 90th percentile is 9.21e-4. This is not complete branch
shutdown.

Raw output mismatch and gradient mismatch are distinct. The fixed-.5 model has
nearly identical losses and output-gradient directions despite poorly aligned
raw outputs, because many raw outputs have tiny amplitude. This supports the
hypothesis that this training recipe underuses many DISM heads, rather than
making soft and hard computations closely match. Output gradients remain
nonzero; that does **not** establish healthy gradients for selectors, readout
projections, or vocabulary parameters upstream. GDN-relative contributions,
gate/projection compensation, and single-head ablation are not measured here.
No causal claim about formation of long-range capability follows from this
endpoint comparison alone.

Some hard head RMS values in the baseline are very large, while corresponding
soft RMS is tiny. Arithmetic means of relative errors/norm ratios are dominated
by these cases; absolute RMS, per-head results and robust medians must accompany
any relative statistic. Zero-energy cosine is reported as zero through a
clamped denominator, not as a meaningful direction for a zero vector.

## Artifacts and checks

Ignored `build/soft_hard_heads/checkpoints.json` contains exact checkpoint paths;
per-model `results.json` includes model specs and all per-document/head/position
statistics. `tools/summarize_soft_hard_heads.py` generates `heads.csv` (432
model/context/layer/head rows averaged across four documents), `summary.json`,
and `heads_8k.png`. Original per-document statistics remain available for spread
analysis; the plot is not a significance test.

All 48 forward/backward passes had finite loss, output and gradients and unchanged
RNG counters. A repeated baseline 2K first-document probe verified exact equality
of saved forward outputs before/after backward at every DISM layer, guarding
against workspace mutation, and reproduced soft/hard loss. This integrity check
is included in the maintained probe. No CUDA kernels changed; no sanitizer or
reference-kernel backward numerical test was run. Existing relaxed BF16 training
backward is used, so these measurements do not certify upstream derivative accuracy.
