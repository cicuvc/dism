"""Bounded-memory parquet text streaming with exact next-batch resume state."""
from collections import deque
from pathlib import Path
import random

import pyarrow.parquet as pq
import torch


def split_files(root):
    root = Path(root)
    train = sorted(map(str, root.rglob("train-*.parquet")))
    valid = sorted(map(str, root.rglob("validation-*.parquet")))
    if not train or not valid or set(train) & set(valid):
        raise ValueError("Require disjoint train and validation parquet files")
    return train, valid


class PackedStream:
    def __init__(self, files, tokenizer, context=2048, batch_size=8, seed=1234, repeat=True):
        self.files = list(files)
        self.tokenizer, self.context, self.batch_size = tokenizer, context, batch_size
        self.seed, self.repeat = seed, repeat
        self.epoch = self.shard = self.row_group = self.row_offset = 0
        self.tokens = deque()
        self._reader = None

    def _ordered_files(self):
        order = self.files.copy()
        if self.repeat:
            random.Random(self.seed + self.epoch).shuffle(order)
        return order

    def _open(self):
        file = pq.ParquetFile(self._ordered_files()[self.shard])
        return file

    def _texts(self):
        while True:
            if self.shard >= len(self.files):
                if not self.repeat:
                    raise StopIteration("Validation split exhausted; refusing to repeat samples")
                self.epoch += 1
                self.shard = self.row_group = self.row_offset = 0
            file = self._open()
            if self.row_group >= file.num_row_groups:
                self.shard += 1
                self.row_group = self.row_offset = 0
                self._reader = None
                continue
            if self._reader is None:
                self._reader = iter(file.iter_batches(batch_size=64, row_groups=[self.row_group], columns=["text"]))
                skip = self.row_offset
                while skip:
                    batch = next(self._reader)
                    skip -= batch.num_rows
                    if skip < 0:
                        raise ValueError("Invalid parquet batch resume offset")
            try:
                batch = next(self._reader)
            except StopIteration:
                self.row_group += 1
                self.row_offset = 0
                self._reader = None
                continue
            self.row_offset += batch.num_rows
            return [x or "" for x in batch.column(0).to_pylist()]

    def next_batch(self):
        count = self.batch_size * (self.context + 1)
        while len(self.tokens) < count:
            texts = self._texts()
            encoded = self.tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            for ids in encoded:
                self.tokens.extend(ids)
                self.tokens.append(self.tokenizer.eos_token_id)
        # EOS separates documents. Attention may cross documents within a packed
        # sample; RoPE/short-conv/DISM state resets at each fixed-length sample.
        block = torch.tensor([self.tokens.popleft() for _ in range(count)], dtype=torch.long)
        block = block.reshape(self.batch_size, self.context + 1)
        return block[:, :-1].contiguous(), block[:, 1:].contiguous()

    def state_dict(self):
        return {k: getattr(self, k) for k in ("files", "context", "batch_size", "seed", "repeat", "epoch", "shard", "row_group", "row_offset")} | {"tokens": list(self.tokens)}

    def load_state_dict(self, state):
        for k in ("files", "context", "batch_size", "seed", "repeat"):
            if state[k] != getattr(self, k):
                raise ValueError(f"Streaming configuration changed: {k}")
        for k in ("epoch", "shard", "row_group", "row_offset"):
            setattr(self, k, state[k])
        self.tokens = deque(state["tokens"])
        self._reader = None
