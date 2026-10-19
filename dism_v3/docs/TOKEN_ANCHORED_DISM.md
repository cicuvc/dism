# Token-anchored DISM symbols

Status: design note only; no production path is enabled by this document.

## 1. Motivation

Current DISM selects q/k symbols from residual-derived features and learned
codebooks:

```text
s_q(t) = argmax_v <q_t, C_Q[v]>
s_k(t) = argmax_v <k_t, C_K[v]>
```

An alternative is to make one or both symbol streams depend directly on the
input token stream. This removes learned q/k table capacity as the address
alphabet and moves the matching alphabet to the tokenizer.

## 2. Variants

Let `x_t` be the main tokenizer id, `H` a fixed hash, and `C=512`.

### A. Learned Q, fixed token-derived K

```text
s_q(t) = argmax_v <q_t, C_Q[v]>
s_k(t) = H(x_(t-1)) mod C
```

This preserves a context-dependent learned query while replacing the K
selector with a fixed, non-collapsing token anchor. The Q codebook indices must
use the same bucket alphabet as the K hash.

### B. Token-derived Q and K

```text
s_q(t) = x_t
s_k(t) = x_(t-1)
```

or the hashed form for both sides. This removes q/k vocabulary tables and
becomes a token-level suffix automaton / induction matcher. Hard labels require
no q/k interpolation or selector relaxation.

## 3. Why the K side is shifted

For `A B ... A ?`, the previous `A` should retrieve the value at `B`, not the
value at `A`. With

```text
q_i = A
k_(j+1) = x_j = A
```

the match lands on the position after the earlier `A`, whose value can encode
the continuation `B`. This is the standard induction-head shift.

Circuit analysis of the trained model also observed `k[j] ~= q[j-1]`, so the
shifted-token construction makes an already learned behavior explicit.

The first position of every packed document must receive an invalid/sentinel K
id so this shift never crosses document boundaries.

## 4. Mechanisms

- **Learned Q + token K:** fixed-key, learned-trigger induction retrieval.
- **Token Q/K:** nonparametric variable-order token matcher; a suffix automaton
  whose alphabet is the tokenizer vocabulary.
- **Hashed K:** fixed token equivalence classes with deterministic collisions.
- **Raw token K:** exact token identity and no q/k hash limit, but no learned
  clustering across morphology, spelling or formatting.

Raw matching address space grows as `V_tok^r = 32000^r` for suffix length `r`.
Hash-to-512 matching uses `512^r` classes; it can generalize through collisions
but can also create false matches.

## 5. Training a token-to-bucket map

If the mapping is a fixed hash, there is nothing to train. A trainable mapping
is a small token-to-bucket quantizer:

```text
r_phi(c | x) = softmax_c(g_phi(e_x) / temperature)
```

A low-rank parameterization avoids a `32000 x 512` table:

```text
g_phi(e_x) = W e_x
```

The hard deployment label is `argmax_c r_phi(c|x)`. The soft training path can
reuse DISM's existing expectation lower bound:

```text
E_{c ~ r_phi(.|x_(j-1))}[ log p_q(c) ]
```

This supplies a pathwise gradient to `phi`; no REINFORCE is required. The
existing density floor `1 + m exp(tau)` and suffix recurrence remain unchanged.

Important constraints:

- use load-balancing / entropy regularization to avoid all tokens selecting a
  few buckets;
- keep the mapping token-only, not contextual, if the intended control is a
  token hash;
- detach the LM token embedding if the hash should not alter language-model
  representations;
- anneal the mapping temperature before deployment comparison.

## 6. Alternative training methods

### 6.1 Soft mapping with hard argmax deployment

Preferred first attempt. It is pathwise differentiable and follows DISM's
existing Jensen/expectation mechanism. The remaining soft/hard gap is the
token-map temperature gap, not a score-function estimator problem.

### 6.2 VQ / online k-means with EMA

Map token embeddings to the nearest of `C` centroids and update centroids with
EMA or a commitment loss. This is a learned, non-differentiable assignment
without policy gradients. It is stable but can suffer code collapse and
centroid drift; frequency-balanced initialization and dead-code restarts are
required.

### 6.3 Gumbel-softmax / Concrete

Sample or approximate `argmax` with a temperature-annealed Concrete
distribution. It is easy to add but still has train/deploy mismatch. It is a
reasonable diagnostic, not the safest production mechanism.

### 6.4 Straight-through estimator

Useful only for the narrow token-to-bucket map, not for the whole DISM path.
The previous full-path STE instability does not prove this local STE will fail,
but it should be treated as a baseline, not the first-choice method.

### 6.5 REINFORCE / REBAR / RELAX / ARM

The bucket assignment can be viewed as a discrete latent, so score-function
estimators are possible. However every token assignment changes a long suffix
chain and downstream readout, so credit assignment is global and noisy. Given
prior REINFORCE diagnostics, this is lower priority than pathwise relaxation.

### 6.6 Fixed LSH over token embeddings

Use random hyperplane signs or random projections of token embeddings to form
`log2(C)` bits. No mapping table is trained, but gradients through token
embeddings can move examples relative to hash boundaries. A soft
`sigmoid(beta * projection)` version is differentiable; deployment uses hard
bits. This is a middle ground between arbitrary token-id hashing and a learned
quantizer.

### 6.7 Balanced offline assignment

Use frequency-weighted quantiles or optimal transport to assign the 32000 token
types to 512 buckets with controlled load. This is not gradient-trained, but it
is deterministic, balanced, reproducible and avoids dead buckets. It is a
strong baseline for a learned hash.

## 7. Recommended experiment order

1. Fixed frequency-balanced hash of `x_(t-1)` for K; learned dynamic Q over the
   same 512 bucket alphabet.
2. Raw `x_(t-1)` K and raw `x_t` Q, with no q/k tables and exact equality.
3. Low-rank soft token-to-bucket mapper with temperature annealing.
4. VQ/EMA or balanced assignment mapper if the learned soft map collapses.
5. Gumbel/STE/REINFORCE only as fallback diagnostics.

The first two are parameter-free on the token-derived side and do not require a
gradient estimator. This makes them the cleanest tests of whether stable token
anchors already recover the long-range induction behavior.
