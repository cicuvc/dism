"""Explicit eager, full-hard model integration for the SAM prefill core.

Retains the model's projection, convolution, vocabulary selection, auxiliary
branches and output math. Each packed document is an independent SAM stream;
no decode cache is initialized. A model must not be used concurrently while
this context is active. The caller owns and closes the reusable prefill engine.
"""
from contextlib import contextmanager
from collections import defaultdict
import torch


@contextmanager
def use_sam_prefill(model, engine):
    from ..module import DismAttention
    if model.training or torch.is_grad_enabled():
        raise ValueError('SAM model prefill requires eval() and disabled gradients')
    if engine is None or not callable(engine):
        raise TypeError('a callable HardDismPrefill engine is required')
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        raise ValueError('SAM model prefill preparation is eager-only')
    layers = [m for m in model.modules() if isinstance(m, DismAttention)
              and not getattr(m, 'is_pure_gdn', False)]
    if not layers:
        raise ValueError('model has no DISM attention layers')
    if any(getattr(m, '_sam_prefill_engine', None) is not None for m in layers):
        raise ValueError('SAM prefill context is already active')
    missing = object()
    previous = [getattr(m, '_sam_prefill_engine', missing) for m in layers]
    for m in layers:
        m._sam_prefill_engine = engine
    try:
        yield model
    finally:
        for m, value in zip(layers, previous):
            if value is missing:
                del m._sam_prefill_engine
            else:
                m._sam_prefill_engine = value


def vocabulary_prefill(engine, q, k, sq, sk, v, eq, ek, tau, beta, hard, *, layout):
    """Same fixed-true vocabulary kernel as the quadratic model forward."""
    import cu_flash_dism as cu
    from ..kernels.voc_glue import prepare_embedding
    from ..kernels.interp_true import emb_fwd_true
    if torch.is_grad_enabled() or not bool(hard.all()):
        raise ValueError('SAM prefill requires disabled gradients and all-hard flags')
    if q.shape[2] != k.shape[2]:
        raise NotImplementedError('SAM model prefill requires equal query/key heads')
    direction = torch.ones((q.shape[0], q.shape[2]), device=q.device, dtype=torch.bool)
    cu.prepare_embedding(q, k, sq, sk, v, eq, ek, tau, direction, hard, False)
    query, key, qe, ke = prepare_embedding(q, k, eq, ek)
    # Preserve quantization, tie-breaking and temperature of existing routing.
    *_, ik, iq = emb_fwd_true(query, key, qe, ke, float(beta), tau.detach())
    if layout is None:
        return engine(iq, ik, sq, sk, v, tau).to(v.dtype)
    if q.shape[0] != 1:
        raise ValueError('packed prefill requires physical batch=1')
    lengths = layout.lengths
    # Common evaluation case: fixed physical batch packed as equal-length rows.
    if lengths and min(lengths) > 0 and len(set(lengths)) == 1:
        batch, length, heads = len(lengths), lengths[0], q.shape[2]
        labels = [x.view(heads, batch, length).transpose(0, 1).contiguous() for x in (iq, ik)]
        vectors = [x.view(batch, length, heads, x.shape[-1]) for x in (sq, sk, v)]
        return engine(*labels, *vectors, tau).to(v.dtype).view_as(v)
    # Ragged documents are grouped by exact length; no pad tokens enter a SAM.
    groups = defaultdict(list)
    offset = 0
    for length in lengths:
        if length:
            groups[length].append(offset)
        offset += length
    output = torch.empty_like(v)
    for length, starts in groups.items():
        labels = [torch.stack([x[0, :, start:start+length] for start in starts]).contiguous()
                  for x in (iq, ik)]
        vectors = [torch.stack([x[0, start:start+length] for start in starts]).contiguous()
                   for x in (sq, sk, v)]
        result = engine(*labels, *vectors, tau).to(v.dtype)
        for row, start in enumerate(starts):
            output[0, start:start+length].copy_(result[row])
    return output
