# Dynamo module assessment

## Repair and supported compilation recipe

The issues below describe the original c2a9ed9 baseline. They are now fixed
for fixed/packed DISM and hybrid training (cache disabled):

- Explicit BF16 casts outside custom autograd replace the problematic AMP
  decorators in shortconv and linear/RMSNorm/RoPE. Input/master gradients
  still pass through ordinary differentiable casts.
- compiler.py registers native vocabulary/core and stateless convolution
  forward/backward custom ops with fake implementations and saved-tensor
  autograd. Ragged checkpoints remain exact-sized, not padded to a worst-case
  allocation. Backward reuses forward checkpoints; it does not rerun forward.
  Runtime metadata validation/construction remains opaque, including alignment
  and max_seqlen validation. prepare_pack_flat returns checked boundaries and
  an owned flat CPU layout: compiled layers share this operation, and autograd
  retains the layout through backward. Flat transport avoids size-one stride
  specialization for a single document. CUDA Graph capture is still unsupported.
- Compiled gated RMSNorm uses the equivalent per-head FP32 Torch expression
  for Inductor fusion, avoiding FLA's traced device-property query.
- Packed hybrid calls FlashAttention through an allow_in_graph wrapper. This
  avoids Dynamo nested-autograd tracer failures when several layers reuse the
  same boundaries; AOTAutograd still traces FA's registered forward/backward
  operations. No boundary clone or eager graph break is needed. A two-layer
  compiled optimizer/save/resume smoke additionally covers this repeated use.

Native document/task uploads use a bounded, content-keyed, thread-local LRU
(32 entries, 8MiB payload; host keys and GPU storage each obey this bound).
Device and stream identity are part of the key. Cold misses use pinned-memory
nonblocking uploads, hot hits reuse the same stream's immutable allocation.
Never key this cache solely by cu_seqlens pointer: dataloaders overwrite it.
See UTILIZATION_OPTIMIZATION.md for measurements and validation.

Compiled voc internals now use fused Triton input preparation, operand selection
and backward gradient assembly (VOC_GLUE.md). Intermediate BF16 rounding is
preserved explicitly. Backward passes absorbed=True to native operand preparation
rather than constructing a zero tau and subtracting it. Eager comparison remains.

```python
import torch
from flash_dism.kernels.dynamo_utils import mark_cu_seqlens_dynamic

# Required for exact-sized ragged saved checkpoints, not for fixed execution.
torch._dynamo.config.capture_dynamic_output_shape_ops = True
mark_cu_seqlens_dynamic(cu_seqlens)  # before first compiled invocation
compiled = torch.compile(module, dynamic=True, fullgraph=True)
with torch.autocast('cuda', dtype=torch.bfloat16):
    output, _, _ = compiled(x, cu_seqlens=cu_seqlens, max_seqlen=1024,
                           hard=hard, direction=direction, use_cache=False)
```

Keep max_seqlen a constant upper bound across packs. If omitted, compiled mode
uses total token count as a safe upper bound rather than reading document
lengths in traced Python. Mark newly created boundary tensors before calling
the compiled module too. Do not call mark_dynamic inside compiled forward.
If local Inductor subprocess workers report `0 active drivers`, set
TORCHINDUCTOR_COMPILE_THREADS=1 before running; this is an environment workaround.

Validation: tests/test_module_dynamo.py covers actual Inductor forward,
parameter/input gradients, fixed/packed and pure/hybrid, with error_on_recompile
enabled after first use. Packs include2/4/1 documents and empty documents.
It also checks all8 R/D/DV shapes for fixed and packed opaque-op gradient
equivalence (16 cases), and runtime rejection of invalid pack metadata.
Separate probe verifies fixed-seed mixed RNG. No graph breaks in the tested
fullgraph paths. Hybrid may initially emit two backend graphs, but the count
stays unchanged across packs; tests reject retracing, not just count increases.

Limits: cached decoding, attention_mask unpadding, changing Python scalar
hard_prob/seed/max_seqlen, and a supplied torch.Generator are not covered by
the no-recompile contract. In particular annealing a Python float hard_prob
may specialize/recompile; do not infer tensor-probability support from this
test. Fixed and packed are separate compilation regimes.

### No explicit mark experiment

Using ordinary torch.compile(dynamic=True,fullgraph=True), without mark_dynamic:
the initial3-boundary example passes for both pure DISM and hybrid, but an
initial boundary count equal to the head count fails on the next pack. Tested
collisions are2 (single document/H2) and4 (three documents/H4, conv_size4).
The actual guard is:

```
kwargs['cu_seqlens'].size()[0] == kwargs['direction'].size()[1]
# duck sizing added this equality because these variables had the same size
```

Thus dynamic=True alone is not a reliable no-recompile contract, even after
fixing all module-side Python metadata specialization. The source is Dynamo
input duck-shape inference, not DISM's ragged checkpoint operator.

An alternative to per-input marking on the tested Torch version is disabling
duck-shape inference at compilation/tracing time:

```python
from torch.fx.experimental import _config as shape_config

with shape_config.patch(use_duck_shape=False), \
     torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
    compiled = torch.compile(module, dynamic=True, fullgraph=True)
    # Call/train inside this context too: torch.compile traces lazily.
    # Ordinary cu_seqlens tensors need no per-tensor dynamic marking.
```

This is a private Torch configuration, not an API stability promise. The
package does not change global compiler settings on import. Retain explicit
mark_dynamic as the existing supported alternative. Regression tests include
both successful no-mark calls and assertions that default duck inference
raises RecompileError for the two collision scenarios.
Results:12 tests passed (six collision-free no-duck cases, two default-duck
noncollision cases, four expected collision failures). Both pure DISM/hybrid
use actual Inductor forward/backward and error_on_recompile after the first call.

## Original assessment

Environment: torch2.13.0+cu130, Triton3.8.0, RTX5090, conda blkw.
Production snapshot:c2a9ed9. This assessment does not change production code.

## Reproduced blockers

- Fullgraph fixed fails at pybind prepare_embedding, which has no dispatcher/
  fake implementation visible to Dynamo.
- Fullgraph varlen fails earlier at pybind make_varlen_layout.
- Allowing graph breaks with backend=eager fails on the first forward:
  native validation reports `k: invalid device, dtype or shape`.
  Diagnostic eager boundary at voc_dism shows Q BF16 but K/V FP32 in compiled
  execution; eager has all three BF16. Sequential ShortConvSiluFunction
  custom_fwd/autocast integration needs isolation; exact compiler root cause
  is not established. Disabling shortconv tracing avoids this failure.
- Actual Inductor fixed-path run with TORCHINDUCTOR_COMPILE_THREADS=1
  reproduces the same K validation failure. Default parallel compilation first
  hit an environmental subprocess `0 active drivers` error; single-thread
  compilation bypassed that issue, not the module failure.

## Varlen shape test

Total tokens1024, B1/H2/D32/DV32/R16, explicit fixed hard/direction tensors,
max_seqlen1024 throughout. Boundary lists:
`[0,512,1024] -> [0,256,512,768,1024] -> [0,256,1024] -> [0,512,1024]`.
All cu_seqlens tensors were explicitly marked dynamic; compile dynamic=True.

To get past the independent autocast/native problems, the diagnostic process
only disables tracing of layout construction, voc_dism and three convolutions.
All four forward/parameter-gradient comparisons then pass against eager
(output atol.005/rtol.02, parameter gradients atol.001/rtol.03).
This is NOT a production compatibility pass: changing documents2->4 triggers
recompilation of the resumed frame in module.py at layout construction:

```
len(___stack0.lengths) == 2
# longest = max(layout.lengths, default=0)
```

Backend frame_count stays10 across calls despite this recompilation, since
the guarded Python-only frame does not emit another FX graph. Always inspect
recompile logs as well as backend graph counts. Marking only cu_seqlens dynamic
cannot remove guards on Python tuples derived from its contents.

Existing kernels/test_dynamo.py:2 tests pass, including primitive varlen
shape/value changes with one graph. Those tests do not cover module native
layout construction or sequential FP32-master-weight autocast convolutions.

## Proposed implementation, not yet applied

1. Isolate and correct compiled shortconv autocast; test three projections
   consecutively with FP32 inputs/master weights under BF16 autocast, including
   gradients. Do not paper over incorrect computation with an output cast.
2. Provide compiler-visible custom operator boundaries with fake shapes and
   autograd for CUDA DISM. Keep data-dependent ragged layout allocation inside
   opaque runtime code; do not return Python document-length tuples into traced
   module logic. An opaque op can keep internal ragged buffers hidden, or use
   a conservative metadata/workspace capacity bound from total tokens.
3. Validate max_seqlen and boundary values outside traced Python while preserving
   runtime rejection of invalid alignment; do not specialize on document count.
4. Test both DISM and hybrid, fixed/packed, actual Inductor forward/backward,
   dynamic document counts including empty documents, then generated RNG,
   annealed hard_prob and cached inference as separate contracts. Current probe
   uses explicit hard/direction; it does not establish RNG/cache compatibility.

## Reproduction

```bash
python tools/probe_module_dynamo.py --fullgraph
python tools/probe_module_dynamo.py --packed --fullgraph
python tools/probe_module_dynamo.py --packed
TORCH_LOGS=recompiles python tools/probe_module_dynamo.py --packed --opaque-native --opaque-conv
python python/flash_dism/kernels/test_dynamo.py -q
```

Raw logs:build/dynamo_{fixed,packed,packed_opaque,packed_isolated,primitives}_probe.log.
The probe emits caught failures as JSON; exit0 alone is not a passing verdict.
