"""
Distributed dataloaders for pretraining.

BOS-aligned varlen bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - Every document starts at a configurable token alignment (default 32)
   - Alignment gaps are padded and ignored by cross-entropy
   - Attention is isolated at document and padding boundaries
   - When no document fits remaining space, crops a document to fill exactly

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned packing ensures every trained token sees only its own document and
can attend back to that document's BOS token.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import hashlib
import os
import struct
from collections import OrderedDict

import torch
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

SEQ_ALIGN = 32
SHUFFLE_ALGORITHM = "splitmix64_fisher_yates_v1"
_UINT64_MASK = (1 << 64) - 1


def _derive_shuffle_seed(seed, *coordinates):
    """Derive a stable uint64 seed without Python's process-randomized hash()."""
    h = hashlib.sha256(SHUFFLE_ALGORITHM.encode("ascii"))
    h.update(struct.pack("<Q", int(seed) & _UINT64_MASK))
    for coordinate in coordinates:
        h.update(struct.pack("<Q", int(coordinate) & _UINT64_MASK))
    return int.from_bytes(h.digest()[:8], "little")


def _splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return state, (value ^ (value >> 31)) & _UINT64_MASK


def deterministic_permutation(length, seed, *coordinates):
    """Versioned, dependency-independent Fisher-Yates permutation."""
    if length < 0:
        raise ValueError("permutation length must be nonnegative")
    values = list(range(length))
    state = _derive_shuffle_seed(seed, *coordinates)
    for i in range(length - 1, 0, -1):
        state, random_value = _splitmix64(state)
        j = random_value % (i + 1)
        values[i], values[j] = values[j], values[i]
    return values


def max_varlen_segments(batch_size, sequence_len, seq_align=SEQ_ALIGN):
    """Static upper bound used to keep cu_seqlens shape compile-stable."""
    assert seq_align > 0
    max_docs_per_row = (sequence_len + seq_align - 1) // seq_align
    # A row alternates padding runs and documents, and always begins a new segment.
    return batch_size * (2 * max_docs_per_row + 1)


def build_varlen_metadata(document_ids, max_segments, aligned_segment_ends=False):
    """
    Turn a (B, T) document-id map into compile-stable attention metadata.

    Every contiguous run, including alignment padding, is an independent attention
    segment. cu_seqlens is padded with the final offset, i.e. zero-length segments,
    so its shape is independent of the number of documents in the batch.
    """
    assert document_ids.ndim == 2
    B, T = document_ids.shape
    boundaries = torch.ones((B, T), dtype=torch.bool, device=document_ids.device)
    if T > 1:
        boundaries[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]
        if aligned_segment_ends:
            # Causal tail padding belongs to the previous document. Targets
            # remain masked using the original document_ids, independently.
            boundaries[:, 1:] &= document_ids[:, 1:] >= 0
    flat_boundaries = boundaries.reshape(-1)
    starts = flat_boundaries.nonzero(as_tuple=False).flatten()
    num_segments = starts.numel()
    assert num_segments <= max_segments, f"Too many varlen segments: {num_segments} > {max_segments}"

    segment_ids = flat_boundaries.cumsum(0, dtype=torch.int32).sub_(1).view(B, T)
    total_tokens = B * T
    cu_seqlens = torch.full(
        (max_segments + 1,), total_tokens, dtype=torch.int32, device=document_ids.device
    )
    cu_seqlens[:num_segments] = starts.to(torch.int32)
    return cu_seqlens, segment_ids


class _DocumentBatchIterator:
    """
    Stateful iterator over deterministic, DDP-sharded Parquet document batches.

    The cursor always points at the next document that will be returned. Unlike the
    legacy state, doc_offset preserves the exact position within a row group.
    """
    STATE_VERSION = 4

    def __init__(self, split, resume_state_dict, tokenizer_batch_size, data_dir=None,
                 shuffle=None, shuffle_seed=None):
        assert split in {"train", "val"}
        _, self.rank, _, self.world_size = get_dist_info()
        warn_on_legacy = self.rank == 0 and split == "train"
        paths = list_parquet_files(data_dir=data_dir, warn_on_legacy=warn_on_legacy)
        assert paths, "No dataset parquet files found, did you run dataset.py?"
        self.parquet_paths = paths[:-1] if split == "train" else paths[-1:]
        assert self.parquet_paths, f"No parquet files available for split={split}"
        self.split = split
        self.tokenizer_batch_size = tokenizer_batch_size
        self.manifest_hash = self._manifest_hash(self.parquet_paths)
        self.pyarrow_version = pa.__version__
        saved_version = resume_state_dict.get("version") if resume_state_dict else None
        if saved_version == self.STATE_VERSION:
            saved_shuffle = bool(resume_state_dict["shuffle"])
            if shuffle is not None and bool(shuffle) != saved_shuffle:
                raise ValueError(
                    f"Dataloader resume mismatch for shuffle: {shuffle!r} != {saved_shuffle!r}"
                )
            self.shuffle = saved_shuffle
            saved_seed = int(resume_state_dict["shuffle_seed"])
            if shuffle_seed is not None and int(shuffle_seed) != saved_seed:
                raise ValueError(
                    f"Dataloader resume mismatch for shuffle_seed: {shuffle_seed!r} != {saved_seed!r}"
                )
            self.shuffle_seed = saved_seed
        elif resume_state_dict is not None:
            # v3 and older checkpoints were produced by the legacy ordered reader.
            if shuffle is True:
                raise ValueError("Legacy dataloader checkpoints must resume with shuffle disabled")
            self.shuffle = False
            self.shuffle_seed = 1337 if shuffle_seed is None else int(shuffle_seed)
        else:
            self.shuffle = split == "train" if shuffle is None else bool(shuffle)
            self.shuffle_seed = 1337 if shuffle_seed is None else int(shuffle_seed)
        if split != "train" and self.shuffle:
            raise ValueError("Only the training split may be shuffled")

        self.row_groups = None
        self.row_group_manifest_hash = None
        self._parquet_file_cache = OrderedDict()
        self._parquet_file_cache_size = 16
        self._epoch_permutation_cache = {}
        if self.shuffle:
            self.row_groups, self.row_group_manifest_hash = self._build_row_group_manifest()
            assert self.row_groups, "No row groups available for shuffled training"
        self.epoch = 1
        self.pq_idx = 0
        self.rg_idx = self.rank
        self.doc_offset = 0
        self.stream_position = self.rank if self.shuffle else None
        self._cached_key = None
        self._cached_texts = None
        if resume_state_dict is not None:
            self.load_state_dict(resume_state_dict)
        self._normalize_cursor()

    @staticmethod
    def _manifest_hash(paths):
        root = os.path.commonpath(paths)
        h = hashlib.sha256()
        for path in paths:
            relative = os.path.relpath(path, root)
            h.update(relative.encode("utf-8"))
            h.update(b"\0")
            h.update(str(os.path.getsize(path)).encode("ascii"))
            h.update(b"\0")
        return h.hexdigest()

    def __iter__(self):
        return self

    def _build_row_group_manifest(self):
        units = []
        h = hashlib.sha256()
        h.update(self.manifest_hash.encode("ascii"))
        for pq_idx, path in enumerate(self.parquet_paths):
            pf = pq.ParquetFile(path)
            h.update(struct.pack("<Q", pf.num_row_groups))
            h.update(struct.pack("<Q", pf.metadata.num_rows))
            for rg_idx in range(pf.num_row_groups):
                units.append((pq_idx, rg_idx))
                h.update(struct.pack("<QQQ", pq_idx, rg_idx, pf.metadata.row_group(rg_idx).num_rows))
            pf.close()
        return units, h.hexdigest()

    def _open_file(self, pq_idx):
        cached = self._parquet_file_cache.pop(pq_idx, None)
        if cached is not None:
            self._parquet_file_cache[pq_idx] = cached
            return cached
        pf = pq.ParquetFile(self.parquet_paths[pq_idx])
        self._parquet_file_cache[pq_idx] = pf
        if len(self._parquet_file_cache) > self._parquet_file_cache_size:
            _, evicted = self._parquet_file_cache.popitem(last=False)
            evicted.close()
        return pf

    def _open_current_file(self):
        return self._open_file(self.pq_idx)

    def _epoch_permutation(self, epoch):
        permutation = self._epoch_permutation_cache.get(epoch)
        if permutation is None:
            permutation = deterministic_permutation(
                len(self.row_groups), self.shuffle_seed, 0, epoch
            )
            # Only the current epoch is useful; avoid an epoch-proportional cache.
            self._epoch_permutation_cache = {epoch: permutation}
        return permutation

    def _set_shuffled_cursor(self):
        num_units = len(self.row_groups)
        self.epoch = self.stream_position // num_units + 1
        epoch_offset = self.stream_position % num_units
        unit_idx = self._epoch_permutation(self.epoch)[epoch_offset]
        self.pq_idx, self.rg_idx = self.row_groups[unit_idx]

    def _normalize_cursor(self):
        """Advance an exhausted file/epoch cursor to the next row group for this rank."""
        if self.shuffle:
            self._set_shuffled_cursor()
            return
        while True:
            if self.pq_idx >= len(self.parquet_paths):
                self.epoch += 1
                self.pq_idx = 0
                self.rg_idx = self.rank
                self.doc_offset = 0
            pf = self._open_current_file()
            if self.rg_idx < pf.num_row_groups:
                return
            self.pq_idx += 1
            self.rg_idx = self.rank
            self.doc_offset = 0
            self._cached_key = None
            self._cached_texts = None

    def __next__(self):
        self._normalize_cursor()
        key = (self.pq_idx, self.rg_idx)
        if self._cached_key != key:
            rg = self._open_current_file().read_row_group(self.rg_idx, columns=["text"])
            self._cached_texts = rg.column("text").to_pylist()
            if self.shuffle:
                order = deterministic_permutation(
                    len(self._cached_texts), self.shuffle_seed,
                    1, self.epoch, self.pq_idx, self.rg_idx,
                )
                self._cached_texts = [self._cached_texts[i] for i in order]
            self._cached_key = key

        start = self.doc_offset
        stop = min(start + self.tokenizer_batch_size, len(self._cached_texts))
        texts = self._cached_texts[start:stop]
        assert texts, f"Empty row group at pq={self.pq_idx}, rg={self.rg_idx}"
        self.doc_offset = stop
        source_position = (self.pq_idx, self.rg_idx, self.epoch)
        if self.doc_offset >= len(self._cached_texts):
            if self.shuffle:
                self.stream_position += self.world_size
            else:
                self.rg_idx += self.world_size
            self.doc_offset = 0
            self._cached_key = None
            self._cached_texts = None
            self._normalize_cursor()
        return texts, source_position

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "split": self.split,
            "rank": self.rank,
            "world_size": self.world_size,
            "tokenizer_batch_size": self.tokenizer_batch_size,
            "manifest_hash": self.manifest_hash,
            "pyarrow_version": self.pyarrow_version,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "shuffle_algorithm": SHUFFLE_ALGORITHM,
            "row_group_manifest_hash": self.row_group_manifest_hash,
            "epoch": self.epoch,
            "pq_idx": self.pq_idx,
            "rg_idx": self.rg_idx,
            "doc_offset": self.doc_offset,
            "stream_position": self.stream_position,
        }

    def load_state_dict(self, state):
        if state.get("version") == self.STATE_VERSION:
            expected = {
                "split": self.split,
                "rank": self.rank,
                "world_size": self.world_size,
                "tokenizer_batch_size": self.tokenizer_batch_size,
                "manifest_hash": self.manifest_hash,
                "pyarrow_version": self.pyarrow_version,
                "shuffle": self.shuffle,
                "shuffle_seed": self.shuffle_seed,
                "shuffle_algorithm": SHUFFLE_ALGORITHM,
                "row_group_manifest_hash": self.row_group_manifest_hash,
            }
            for key, value in expected.items():
                assert state.get(key) == value, \
                    f"Dataloader resume mismatch for {key}: {state.get(key)!r} != {value!r}"
            self.epoch = state["epoch"]
            self.pq_idx = state["pq_idx"]
            self.rg_idx = state["rg_idx"]
            self.doc_offset = state["doc_offset"]
            self.stream_position = state["stream_position"]
            if self.shuffle:
                assert self.stream_position % self.world_size == self.rank, \
                    "Shuffled stream position is not owned by this rank"
            return

        if state.get("version") == 3:
            # v3 already had an exact document offset, but no shuffle fields.
            expected = {
                "split": self.split,
                "rank": self.rank,
                "world_size": self.world_size,
                "tokenizer_batch_size": self.tokenizer_batch_size,
                "manifest_hash": self.manifest_hash,
                "pyarrow_version": self.pyarrow_version,
            }
            for key, value in expected.items():
                assert state.get(key) == value, \
                    f"Dataloader resume mismatch for {key}: {state.get(key)!r} != {value!r}"
            self.epoch = state["epoch"]
            self.pq_idx = state["pq_idx"]
            self.rg_idx = state["rg_idx"]
            self.doc_offset = state["doc_offset"]
            return

        # Backward-compatible approximate resume from legacy checkpoints. The old
        # cursor identified the most recently fetched row group, so skip it exactly
        # as the old generator did. Exact source resume begins at STATE_VERSION.
        self.epoch = state.get("epoch", 1)
        self.pq_idx = state.get("pq_idx", 0)
        old_rg_idx = state.get("rg_idx")
        if old_rg_idx is None:
            self.rg_idx = self.rank
        else:
            self.rg_idx = (old_rg_idx // self.world_size + 1) * self.world_size + self.rank
        self.doc_offset = 0


def _document_batches(split, resume_state_dict, tokenizer_batch_size, data_dir=None,
                      shuffle=None, shuffle_seed=None):
    return _DocumentBatchIterator(
        split, resume_state_dict, tokenizer_batch_size, data_dir,
        shuffle=shuffle, shuffle_seed=shuffle_seed,
    )


class _StatefulBOSBestfitLoader:
    """
    Varlen-aware BOS-aligned dataloader with Best-Fit Cropping.

    Each document starts at a multiple of seq_align. Alignment gaps are filled with
    BOS tokens, excluded from CE, and placed in their own attention segment.

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. Align the next document start to seq_align
    4. When nothing fits, crop a doc to fill the remaining aligned space

    Key properties:
    - Every row starts with BOS
    - Fixed-shape cu_seqlens and segment_ids (document count never recompiles)
    - No attention or CE transition across document boundaries
    """
    STATE_VERSION = 3

    def __init__(self, tokenizer, B, T, split, tokenizer_threads=4,
                 tokenizer_batch_size=128, device="cuda", resume_state_dict=None,
                 buffer_size=1000, seq_align=SEQ_ALIGN, data_dir=None, aligned_segment_ends=False,
                 shuffle=None, shuffle_seed=None):
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        assert seq_align > 0, "seq_align must be positive"
        assert T % seq_align == 0, \
            "sequence length must be divisible by seq_align so flattened row starts stay aligned"
        self.tokenizer = tokenizer
        self.B, self.T, self.split = B, T, split
        self.tokenizer_threads = tokenizer_threads
        self.tokenizer_batch_size = tokenizer_batch_size
        self.buffer_size = buffer_size
        self.seq_align = seq_align
        self.aligned_segment_ends = aligned_segment_ends
        self.device = torch.device(device)
        self.bos_token = tokenizer.get_bos_token_id()
        self.tokenizer_fingerprint = self._tokenizer_fingerprint(tokenizer)
        try:
            import tiktoken
            self.tiktoken_version = tiktoken.__version__
        except ImportError:
            self.tiktoken_version = None

        exact_state = (
            resume_state_dict
            if resume_state_dict and resume_state_dict.get("version") in {2, self.STATE_VERSION}
            and "source" in resume_state_dict and "doc_buffer" in resume_state_dict
            else None
        )
        source_resume = exact_state["source"] if exact_state is not None else resume_state_dict
        self.batches = _document_batches(
            split, source_resume, tokenizer_batch_size, data_dir=data_dir,
            shuffle=shuffle, shuffle_seed=shuffle_seed,
        )
        self.doc_buffer = []
        self._last_source_position = (0, 0, 1)
        if exact_state is not None:
            self._validate_state_config(
                exact_state["config"], legacy=exact_state.get("version") == 2
            )
            self.doc_buffer = self._unpack_doc_buffer(exact_state["doc_buffer"])

        # Pre-allocate persistent staging/device buffers.
        row_capacity = T + 1
        self.row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
        self.document_ids = torch.empty((B, row_capacity), dtype=torch.int32)
        use_cuda = self.device.type == "cuda"
        self.use_cuda = use_cuda
        self.cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
        self.gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=self.device)
        self.cpu_inputs = self.cpu_buffer[:B * T].view(B, T)
        self.cpu_targets = self.cpu_buffer[B * T:].view(B, T)
        self.inputs = self.gpu_buffer[:B * T].view(B, T)
        self.targets = self.gpu_buffer[B * T:].view(B, T)
        self.max_segments = max_varlen_segments(B, T, seq_align)
        self.cpu_cu_seqlens = torch.empty(self.max_segments + 1, dtype=torch.int32, pin_memory=use_cuda)
        self.cpu_segment_ids = torch.empty((B, T), dtype=torch.int32, pin_memory=use_cuda)
        self.cu_seqlens = torch.empty(self.max_segments + 1, dtype=torch.int32, device=self.device)
        self.segment_ids = torch.empty((B, T), dtype=torch.int32, device=self.device)

    def __iter__(self):
        return self

    @staticmethod
    def _tokenizer_fingerprint(tokenizer):
        """Fingerprint the serialized tiktoken encoding that controls packing lengths."""
        if hasattr(tokenizer, "get_fingerprint"):
            return tokenizer.get_fingerprint()
        encoding = getattr(tokenizer, "enc", None)
        ranks = getattr(encoding, "_mergeable_ranks", None)
        pattern = getattr(encoding, "_pat_str", None)
        special = getattr(encoding, "_special_tokens", None)
        if ranks is None or pattern is None or special is None:
            return None
        h = hashlib.sha256(pattern.encode("utf-8"))
        for token, rank in sorted(ranks.items(), key=lambda item: item[1]):
            h.update(len(token).to_bytes(4, "little"))
            h.update(token)
            h.update(int(rank).to_bytes(8, "little", signed=False))
        for name, token_id in sorted(special.items()):
            h.update(name.encode("utf-8"))
            h.update(b"\0")
            h.update(int(token_id).to_bytes(8, "little", signed=False))
        return h.hexdigest()

    def _config(self):
        source = self.batches.state_dict() if hasattr(self.batches, "state_dict") else {}
        result = {
            "B": self.B,
            "T": self.T,
            "split": self.split,
            "tokenizer_batch_size": self.tokenizer_batch_size,
            "buffer_size": self.buffer_size,
            "seq_align": self.seq_align,
            "bos_token": self.bos_token,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "tiktoken_version": self.tiktoken_version,
            "rank": source.get("rank"),
            "world_size": source.get("world_size"),
            "manifest_hash": source.get("manifest_hash"),
            "shuffle": source.get("shuffle"),
            "shuffle_seed": source.get("shuffle_seed"),
            "shuffle_algorithm": source.get("shuffle_algorithm"),
            "row_group_manifest_hash": source.get("row_group_manifest_hash"),
        }
        if self.aligned_segment_ends:
            result['aligned_segment_ends'] = True
        return result

    def _validate_state_config(self, saved, legacy=False):
        current = self._config()
        for key, value in current.items():
            if legacy and key not in saved:
                continue
            assert saved.get(key) == value, \
                f"Dataloader resume mismatch for {key}: {saved.get(key)!r} != {value!r}"

    @staticmethod
    def _pack_doc_buffer(doc_buffer):
        lengths = torch.tensor([len(doc) for doc in doc_buffer], dtype=torch.int64)
        offsets = torch.empty(len(doc_buffer) + 1, dtype=torch.int64)
        offsets[0] = 0
        if len(doc_buffer):
            offsets[1:] = lengths.cumsum(0)
        flat = torch.empty(int(offsets[-1]), dtype=torch.int32)
        for i, doc in enumerate(doc_buffer):
            flat[offsets[i]:offsets[i + 1]] = torch.tensor(doc, dtype=torch.int32)
        return {"tokens": flat, "offsets": offsets}

    @staticmethod
    def _unpack_doc_buffer(packed):
        tokens = packed["tokens"].cpu()
        offsets = packed["offsets"].cpu().tolist()
        return [tokens[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]

    def state_dict(self):
        assert hasattr(self.batches, "state_dict"), "Document source is not stateful"
        return {
            "version": self.STATE_VERSION,
            "config": self._config(),
            "source": self.batches.state_dict(),
            "doc_buffer": self._pack_doc_buffer(self.doc_buffer),
        }

    def fallback_state_dict(self):
        """JSON-safe source cursor used only if the binary exact state is unavailable."""
        assert hasattr(self.batches, "state_dict"), "Document source is not stateful"
        return self.batches.state_dict()

    def _refill_buffer(self):
        doc_batch, self._last_source_position = next(self.batches)
        token_lists = self.tokenizer.encode(
            doc_batch, prepend=self.bos_token, num_threads=self.tokenizer_threads
        )
        self.doc_buffer.extend(token_lists)

    def progress_state(self):
        if hasattr(self.batches, "state_dict"):
            state = self.batches.state_dict()
            return {"pq_idx": state["pq_idx"], "rg_idx": state["rg_idx"], "epoch": state["epoch"]}
        pq_idx, rg_idx, epoch = self._last_source_position
        return {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

    def __next__(self):
        B, T = self.B, self.T
        row_capacity = T + 1
        self.row_buffer.fill_(self.bos_token)
        self.document_ids.fill_(-1)
        document_serial = 0
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(self.doc_buffer) < self.buffer_size:
                    self._refill_buffer()

                start = ((pos + self.seq_align - 1) // self.seq_align) * self.seq_align
                # A document beginning at the target-only position contributes no
                # training token, so leave it as padding instead of consuming it.
                if start >= T:
                    break
                remaining = row_capacity - start

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(self.doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = self.doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    self.row_buffer[row_idx, start:start + doc_len] = torch.tensor(doc, dtype=torch.long)
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(self.doc_buffer)), key=lambda i: len(self.doc_buffer[i]))
                    doc = self.doc_buffer.pop(shortest_idx)
                    doc_len = remaining
                    self.row_buffer[row_idx, start:start + doc_len] = torch.tensor(doc[:doc_len], dtype=torch.long)

                self.document_ids[row_idx, start:start + doc_len] = document_serial
                document_serial += 1
                pos = start + doc_len

        # Copy to pinned CPU buffer, then single HtoD transfer
        self.cpu_inputs.copy_(self.row_buffer[:, :-1])
        self.cpu_targets.fill_(-1)
        valid_targets = (self.document_ids[:, :-1] >= 0) & (self.document_ids[:, :-1] == self.document_ids[:, 1:])
        self.cpu_targets[valid_targets] = self.row_buffer[:, 1:][valid_targets]
        batch_cu_seqlens, batch_segment_ids = build_varlen_metadata(
            self.document_ids[:, :-1], max_segments=self.max_segments,
            aligned_segment_ends=self.aligned_segment_ends,
        )
        self.cpu_cu_seqlens.copy_(batch_cu_seqlens)
        self.cpu_segment_ids.copy_(batch_segment_ids)

        # DISM already supports dynamic boundary shapes. Do not turn capacity
        # padding into empty attention/shortconv workloads. Compute the prefix
        # on CPU; token tensors, document boundaries and resume cursor are unchanged.
        boundary_count = self.max_segments + 1
        if self.aligned_segment_ends:
            boundary_count = int((batch_cu_seqlens < B*T).sum()) + 1

        # Single HtoD copy into persistent GPU buffer and yield
        self.gpu_buffer.copy_(self.cpu_buffer, non_blocking=self.use_cuda)
        self.cu_seqlens.copy_(self.cpu_cu_seqlens, non_blocking=self.use_cuda)
        self.segment_ids.copy_(self.cpu_segment_ids, non_blocking=self.use_cuda)
        return self.inputs, self.targets, self.cu_seqlens[:boundary_count], self.segment_ids, self.progress_state()


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000, seq_align=SEQ_ALIGN, data_dir=None, aligned_segment_ends=False,
    num_workers=0, shuffle=None, shuffle_seed=None,
):
    if num_workers < 0:
        raise ValueError('num_workers must be nonnegative')
    prefetch_state = None
    if resume_state_dict and 'prefetch_version' in resume_state_dict:
        if num_workers == 0 or resume_state_dict['prefetch_version'] != 1:
            raise ValueError('prefetched checkpoint requires workers and supported state version')
        prefetch_state = resume_state_dict
        resume_state_dict = prefetch_state['source']
    if num_workers:
        from nanochat.data_prefetch import ParallelTokenizer, StatefulPrefetchLoader
        parallel_tokenizer = ParallelTokenizer(tokenizer, num_workers)
        loader = _StatefulBOSBestfitLoader(
            parallel_tokenizer, B, T, split, tokenizer_threads=tokenizer_threads,
            tokenizer_batch_size=tokenizer_batch_size, device='cpu',
            resume_state_dict=resume_state_dict, buffer_size=buffer_size,
            seq_align=seq_align, data_dir=data_dir, aligned_segment_ends=aligned_segment_ends,
            shuffle=shuffle, shuffle_seed=shuffle_seed)
        return StatefulPrefetchLoader(loader, parallel_tokenizer, device, prefetch_state)
    return _StatefulBOSBestfitLoader(
        tokenizer, B, T, split, tokenizer_threads=tokenizer_threads,
        tokenizer_batch_size=tokenizer_batch_size, device=device,
        resume_state_dict=resume_state_dict, buffer_size=buffer_size,
        seq_align=seq_align, data_dir=data_dir, aligned_segment_ends=aligned_segment_ends,
        shuffle=shuffle, shuffle_seed=shuffle_seed,
    )

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, cu_seqlens, segment_ids, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets, cu_seqlens, segment_ids
