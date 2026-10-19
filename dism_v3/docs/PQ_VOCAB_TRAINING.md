# Product-quantised q/k vocabulary training

This opt-in path preserves `qk_vocab_size=512` while representing each code as
two factorised symbols, `512 = 16 x 32`. The dense `[H,512,64]` codebook is
replaced by `[H,2,32,32]`; the first factor uses 16 of 32 allocated rows, the
second uses all 32.

The same implementation also supports `1024 = 32 x 32`. Because group storage
is padded to the maximum group size, `16x32` and `32x32` have identical codebook
parameters; only the first group's inactive tail changes and hard labels use a
1024-way mixed-radix index.

## Configuration

```json
{
  "pq_vocab_sizes": [16, 32],
  "qk_vocab_size": 512,
  "head_dim": 64
}
```

Set `pq_vocab_sizes` to an empty list for the unchanged dense vocabulary.

## Training path

`pq_voc_forward.py` calls the existing interpolation kernel once per factor,
concatenates the two embedding blocks, adds the two LSEs, and forms the hard
label with mixed-radix strides. The resulting tensors enter the unchanged DISM
core and hard planner.

The path is marked as a Dynamo graph boundary. It supports the fixed-true
training policy and packed inputs. The initial implementation rejects TransVQ,
QB prices, selector normalization, GQA variants, categorical REINFORCE and the
experimental hard-inference engine.

Metrics reconstruct the full 512-way product distribution from the two group
log-softmaxes. Hard usage counts use the mixed-radix label, so active codes,
usage entropy, max load and capacity residual remain comparable with dense
vocabularies.

## Validation

- Factorised interpolation matches the expanded dense FP32 oracle at about
  `1e-06` relative error, with exact hard labels.
- A two-layer compiled model passes forward/backward and the training q/k
  utilization probe.
- The 72M compact model has `69,823,968` parameters and uses `15.8 GiB` peak
  allocated memory in a compiled 16x2048 forward+backward smoke step.
- Warm throughput is approximately `192k tokens/s` on the local RTX 5090.
- Excluding `pq_vocab_sizes` and the explicit default recurrent-mixer fields,
  the model config is byte-for-byte equivalent to the restored valres A control.

## PQ 16x32 final diagnostics

The completed 72M run reached hard/soft BPB `1.183111/1.172432`, with
`gap_bpb=0.010678`. This is close to the restored valres baseline on in-domain
language modeling, but the gap is substantially larger.

Offline NIAH used the same 16-trial paired protocol and seeds for
2k/8k/16k contexts and depths `0.05/0.5/0.95`. Margins are
`counterfactual_nll - present_nll`:

| context | PQ16x32 mean margin | valres A mean margin | delta | PQ mean gain | A mean gain |
|---:|---:|---:|---:|---:|---:|
| 2048 | 0.545 | 0.856 | -0.311 | 0.915 | 1.303 |
| 8192 | 0.376 | 0.612 | -0.236 | 0.686 | 0.947 |
| 16384 | 0.025 | 0.406 | -0.381 | 0.312 | 0.691 |

The deficit is strongest at the hardest retrieval point. At 16k/d0.95 the PQ
margin is `-0.076` versus `0.716` for A, and present token accuracy is `0.135`
versus `0.469`. This is consistent with product matching imposing a
conjunctive (`prod_g a_g`) match probability and therefore shrinking the
effective suffix-chain reach in long contexts. NIAH, not BPB, exposes the loss.
