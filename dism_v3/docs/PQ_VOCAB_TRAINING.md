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
