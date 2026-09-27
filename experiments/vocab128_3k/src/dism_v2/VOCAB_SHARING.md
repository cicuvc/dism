# Prepared Q/K and grouped-head vocabulary sharing

Implemented and gradient-tested only. **No training started**, no checkpoint
conversion, remote deployment or optimizer update performed for this change.

## Configuration

```python
from dism_v2.lm_model import LMConfig, DecoderLM

# Shared Q/K within each of four independent head vocabularies:
cfg = LMConfig(dism_tie_qk_vocab=True)

# Two heads per vocabulary group, Q/K tied within each group:
cfg = LMConfig(dism_tie_qk_vocab=True, dism_vocab_groups=2)

# All four heads use one vocabulary, still independent Q/K/V projections:
cfg = LMConfig(dism_tie_qk_vocab=True, dism_vocab_groups=1)
```

`dism_vocab_groups=None` defaults to the number of heads. An explicit group
count must be a positive integer divisor of heads. Contiguous heads share a
group: for H4/G2, heads0,1 use group0; heads2,3 use group1. Setting groups
without tying Q/K is also supported as an independent ablation. Both
`hybrid` and `hybrid_shared` architectures support these fields. Pure SWA/full
attention rejects DISM vocabulary options instead of silently ignoring them.

Training CLI flags are prepared (`--dism-tie-qk-vocab`,
`--dism-vocab-groups 2`), but no training command was executed. FFN dimensions,
annealing, SWA, projections and output fusion are not automatically changed.

## Parameter ownership and gradients

- Unbound mode: `q_voc` and `k_voc` each FP32[G,V,D].
- Tied mode: `q_voc` is the sole FP32[G,V,D] parameter; `k_voc=None`.
  There is no second alias in state_dict or optimizer groups.
- `expanded_vocabularies(dtype)` produces the existing kernel ABI[H,V,D].
  It expands the FP32 group tensor before casting to BF16, separately for
  the Q and K paths. This ensures the head and branch reductions occur in
  FP32 master-parameter gradients, not prematurely in BF16.
- For group g, shared gradient is sum over heads h in g of dE_Q,h+dE_K,h.
  Without Q/K tying, each side receives only its own head sum.
- Parameters are no_decay as before and occur exactly once in optimizer
  parameter groups. GPT2 input embedding and output head remain untied.
- Standard LM construction retains FP32 parameters with BF16 autocast;
  manually converting the entire model's master weights to BF16 would also
  change accumulation precision and is not the tested training convention.

This is **not actual GQA**: K/V projections, activations, cache and per-head
DISM recurrence state remain independent. The temporary expanded tensors
still contain all head vocabularies, so no kernel bandwidth/FLOP reduction is
claimed. No CUDA source or ABI changes. CPU incremental LM labelization now
also obtains vocabularies through this common expansion method.

## Checkpoints and parameter budget

Default false/None retains old q_voc/k_voc keys/shapes and initialization
order. The real local step30000 checkpoint strictly loaded successfully,
still49,678,876 parameters. Older serialized configs gain default values when
read by LMConfig; resume compares the normalized config as before.

Shared configs serialize their explicit flags and compact group tensors.
Strict loading rejects independent→tied or incompatible group shapes; no
averaging, alias guessing or implicit migration. State round-trip tests pass.
The analytical matched-SWA budget helper accounts for the new number of
vocabulary parameters and removes DISM-only flags from its returned control.

Original hybrid, width256/H4/V512/D64/layers15/FFN1024:

| Configuration | Vocabulary parameters | Total model parameters |
|---|---:|---:|
|Original, independent Q/K per head|3,932,160|49,678,876|
|Q/K tied,4 groups|1,966,080|47,712,796|
|Q/K tied,2 groups|983,040|46,729,756|
|Q/K tied,1 group|491,520|46,238,236|

No FFN compensation is applied automatically; these are not equal-parameter
training comparisons.

## Verification

On local RTX5090, conda blkw, `DISM_TILE_LSE=tanh_finite`:

```
python -m pytest tests/test_lm_vocab_sharing.py tests/test_lm_shared.py \
  tests/test_lm_remote.py tests/test_lm_generation.py -q
```

**76 passed** (existing deprecation warnings only).

- H4 with G1/2/4, both Q/K tied and untied: parameter counts, head mapping,
  no_decay, optimizer deduplication, strict state loading and invalid options.
- Synthetic BF16 cast-path gradients exactly sum to FP32 group gradients.
- Actual CUDA core+embedding autograd:36 combinations of G1/2/4, tied/untied,
  both fixed directions and hard_prob0/.37/1, N129 tail, D=DV64, V64.
  Retained Q/K operand gradients sum to master gradients with atol1e-9,
  rtol0. Independent per-head/branch parameter clones give bitwise equal
  forward output; separate-launch gradient checks use atol2e-7/rtol.02,
  allowing kernel atomic/rounding differences without changing existing tests.
- Both whole hybrid architectures:12 one-layer V512 tests, G1/2/4,
  hard_prob.37/1, N129. All registered parameter gradients finite. Soft/mixed
  vocabulary gradients nonzero; pure-hard vocabulary gradients zero as
  required by nondifferentiable argmax (not an STE).
- Both architectures compared to explicitly expanded independent-parameter
  copies, same weights/RNG. Complete head logits bitwise equal; all parameter
  gradients match, with vocabulary gradients summed by group and Q/K branch.
- Default architecture/shared-projection regression tests, generation-state
  tests and real old-checkpoint strict-load check passed. CLI help verified.

These checks validate sharing and gradient accumulation, not improvements
to training quality, codebook balance, retrieval or performance. Existing
CUDA numerical approximations remain unchanged.
