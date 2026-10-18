"""Validated, cached chunk metadata for packed sliding-window DeltaNet.

cu_seqlens contains document offsets in [0, total_tokens]. Empty documents
are allowed. The first use copies offsets to CPU for validation and chunk
enumeration; a bounded identity/version cache avoids repeating that sync in
each layer. Normal in-place mutations invalidate the cache. Mutations through
.data or external CUDA code bypass PyTorch's version counter: pass a new
offsets tensor in that case. Device metadata is an owned snapshot, so later
offset mutations do not change a pending backward's document boundaries.

Numerical forward/backward is native Triton on the packed tensors. The Python
document loop here only enumerates metadata, never computes attention.
Metadata preparation is not supported inside CUDA Graph capture.
"""

from collections import OrderedDict
from dataclasses import dataclass
import threading

import torch


@dataclass(frozen=True)
class VarlenMetadata:
    cu_seqlens: torch.Tensor
    chunk_indices: torch.Tensor  # [total_chunks, 2]: document, local_chunk
    chunks: int
    max_seqlen: int


_cache = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_LIMIT = 16


def prepare_varlen_metadata(cu_seqlens, *, batch, total_tokens, chunk_size, device):
    if cu_seqlens is None:
        return None
    if batch != 1:
        raise ValueError("cu_seqlens requires batch dimension B=1")
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be an int32/int64 tensor")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a 1D tensor with at least two offsets")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must have dtype int32 or int64")
    if cu_seqlens.device.type != "cpu" and cu_seqlens.device != device:
        raise ValueError("cu_seqlens must be on CPU or the same CUDA device as Q")
    if total_tokens >= 2**31:
        raise ValueError("Packed token offsets must fit in int32")

    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Varlen metadata preparation must run outside CUDA Graph capture")
        try:
            version = cu_seqlens._version
        except RuntimeError:  # inference tensors have no mutation counter
            version = None
        key = (id(cu_seqlens), version, total_tokens, chunk_size, device)
        if version is not None:
            with _cache_lock:
                entry = _cache.get(key)
                if entry is not None:
                    source, metadata, ready = entry
                    _cache.move_to_end(key)
                    # The snapshot may have been constructed on another stream.
                    stream = torch.cuda.current_stream(device)
                    stream.wait_event(ready)
                    metadata.cu_seqlens.record_stream(stream)
                    metadata.chunk_indices.record_stream(stream)
                    return metadata

        offsets = cu_seqlens.detach().cpu().tolist()
        if offsets[0] != 0 or offsets[-1] != total_tokens:
            raise ValueError("cu_seqlens must start at 0 and end at N")
        if any(a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("cu_seqlens must be nondecreasing")
        lengths = [b - a for a, b in zip(offsets, offsets[1:])]
        indices = [(doc, local) for doc, length in enumerate(lengths)
                   for local in range((length + chunk_size - 1) // chunk_size)]
        metadata = VarlenMetadata(
            torch.tensor(offsets, dtype=torch.int32, device=device),
            torch.tensor(indices, dtype=torch.int32, device=device).reshape(-1, 2),
            len(indices), max(lengths),
        )
        if version is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
            with _cache_lock:
                # Keep a reference to the source tensor to prevent id reuse.
                _cache[key] = (cu_seqlens, metadata, ready)
                _cache.move_to_end(key)
                while len(_cache) > _CACHE_LIMIT:
                    _cache.popitem(last=False)
        return metadata
