"""Exact pure-hard DISM core decoding on CPU, without a dense score/cache row.

The cache holds integer key strings, value vectors, a derived label index and
finite cursors from the previous query. No soft interpolation, Q/K-vector cache,
direction RNG, pruning or dense NxN scores. This is inference-only core code,
not yet a complete LM shortconv/RoPE/SWA generation cache.
"""
import copy
import math

import torch


def _cpu(tensor, name):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu':
        raise ValueError(f'{name} must be a CPU tensor')


def _labels(tensor, shape, name):
    _cpu(tensor, name)
    if tensor.shape != shape or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f'{name} must be int32/int64 with shape{tuple(shape)}')
    if (tensor < 0).any():
        raise ValueError(f'{name} must contain nonnegative vocabulary IDs')
    return tensor.reshape(-1).tolist()


@torch.no_grad()
def hard_labels_cpu(q, k, q_voc, k_voc, sm_scale=1.):
    """FP32 top-1 labels for a decode step [B,H,D], no softmax/interpolation.

FP32 logits match the reference convention. Near ties can differ from labels
selected by a BF16 CUDA vocabulary GEMM; callers can supply those labels
directly to step() when that exact discrete identity is required.
"""
    for name, tensor in [('q', q), ('k', k), ('q_voc', q_voc), ('k_voc', k_voc)]:
        _cpu(tensor, name)
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f'{name} must be finite floating point')
    if q.ndim != 3 or k.shape != q.shape or q_voc.ndim != 3 or k_voc.shape != q_voc.shape:
        raise ValueError('Expected q/k[B,H,D], vocabularies[H,V,D]')
    if q_voc.shape[0] != q.shape[1] or q_voc.shape[2] != q.shape[2] or q_voc.shape[1] == 0:
        raise ValueError('Vocabulary shape mismatch')
    if not math.isfinite(float(sm_scale)):
        raise ValueError('sm_scale must be finite')
    def label(x, vocab):
        logits = torch.einsum('bhd,hvd->bhv', x.float(), vocab.float()) * float(sm_scale)
        if not torch.isfinite(logits).all():
            raise ValueError('Vocabulary logits overflowed FP32')
        return logits.argmax(-1)
    return label(q, q_voc), label(k, k_voc)


class HardDismCPUCache:
    """Append one self-attention KV and decode one query per step.

rtau[H] is fixed for the cache lifetime. Each step takes labels[B,H] and
v[B,H,DV]. Cache values retain their input dtype; scores/normalization use
FP64 natural logs and output defaults to FP32. All batch entries advance in
lockstep; use separate caches for variable-length sequences.
"""
    def __init__(self, rtau, *, batch_size=1, value_dim=64, output_dtype=torch.float32):
        _cpu(rtau, 'rtau')
        if rtau.ndim != 1 or not rtau.numel() or not rtau.is_floating_point() or not torch.isfinite(rtau).all():
            raise ValueError('rtau must be finite floating point[H]')
        if not isinstance(batch_size, int) or batch_size <= 0 or not isinstance(value_dim, int) or value_dim <= 0:
            raise ValueError('batch_size and value_dim must be positive integers')
        if output_dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            raise ValueError('Unsupported output dtype')
        self.batch_size, self.heads, self.value_dim = batch_size, rtau.numel(), value_dim
        self.output_dtype = output_dtype
        self._tau = tuple(rtau.double().tolist())
        self.reset()

    def reset(self):
        self._keys, self._values = [], []
        self._index = [{} for _ in range(self.batch_size * self.heads)]
        self._active = [{} for _ in self._index]
        self._value_dtype = None

    @property
    def length(self):
        return len(self._keys)

    @property
    def active_cursors(self):
        """Independent {(key_position): W_natural_log} snapshots, flattened B,H."""
        return tuple(dict(row) for row in self._active)

    def clone(self):
        """Independent continuation from the same prefix (e.g. two hypotheses)."""
        return copy.deepcopy(self)

    @torch.no_grad()
    def step(self, q_index, k_index, value):
        shape = (self.batch_size, self.heads)
        qids = _labels(q_index, shape, 'q_index')
        kids = _labels(k_index, shape, 'k_index')
        _cpu(value, 'value')
        if value.shape != (*shape, self.value_dim) or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError('value must be finite floating point[B,H,DV]')
        if self._value_dtype is not None and value.dtype != self._value_dtype:
            raise ValueError('Value dtype changed during decoding')
        current = value.detach().reshape(-1, self.value_dim).clone()
        position = self.length
        output = torch.zeros((len(qids), self.value_dim), dtype=torch.float64)
        next_active = []
        for bh, (qid, kid) in enumerate(zip(qids, kids)):
            previous = self._active[bh]
            matches = self._index[bh].get(qid, [])
            # Include j=i (current key). Do not modify the previous-row state
            # until every new cursor has been computed.
            if qid == kid:
                matches = [*matches, position]
            tau = self._tau[bh % self.heads]
            scores = {}
            for j in matches:
                predecessor = previous.get(j - 1)
                score = tau if predecessor is None else tau + max(predecessor, 0.) + math.log1p(math.exp(-abs(predecessor)))
                if not math.isfinite(score):
                    raise OverflowError('DISM score overflowed FP64')
                scores[j] = score
            next_active.append(scores)
            if not scores:
                continue  # Only the score0/value0 fallback remains.
            logits = torch.tensor(list(scores.values()), dtype=torch.float64)
            maximum = max(0., logits.max().item())
            weights = (logits - maximum).exp()
            denominator = math.exp(-maximum) + weights.sum()
            values = torch.stack([current[bh] if j == position else self._values[j][bh]
                                  for j in scores]).double()
            output[bh] = (weights / denominator) @ values
        # Commit only after successful validation and computation. Each stored
        # value is owned by the cache, not a view of the caller's mutable buffer.
        self._keys.append(tuple(kids))
        self._values.append(current)
        for bh, kid in enumerate(kids):
            self._index[bh].setdefault(kid, []).append(position)
        self._active = next_active
        self._value_dtype = value.dtype
        return output.reshape(*shape, self.value_dim).to(self.output_dtype)

    @torch.no_grad()
    def prefill(self, q_index, k_index, value):
        """Stream a [B,H,N] prefix/chunk; can be called on a nonempty cache."""
        _cpu(q_index, 'q_index')
        if q_index.ndim != 3 or q_index.shape[:2] != (self.batch_size, self.heads):
            raise ValueError('Expected labels[B,H,N]')
        _labels(q_index, q_index.shape, 'q_index')
        _labels(k_index, q_index.shape, 'k_index')
        _cpu(value, 'value')
        if (value.shape != (*q_index.shape, self.value_dim) or not value.is_floating_point()
                or not torch.isfinite(value).all()):
            raise ValueError('Expected finite floating point value[B,H,N,DV]')
        if self._value_dtype is not None and value.dtype != self._value_dtype:
            raise ValueError('Value dtype changed during decoding')
        if q_index.shape[2] == 0:
            return torch.empty((*q_index.shape, self.value_dim), dtype=self.output_dtype)
        return torch.stack([self.step(q_index[:, :, i], k_index[:, :, i], value[:, :, i])
                            for i in range(q_index.shape[2])], dim=2)

    def step_qkv(self, q, k, value, q_voc, k_voc, sm_scale=1.):
        qid, kid = hard_labels_cpu(q, k, q_voc, k_voc, sm_scale)
        return self.step(qid, kid, value)
