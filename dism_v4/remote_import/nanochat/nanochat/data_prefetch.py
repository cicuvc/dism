"""Ordered CPU prefetch with an explicit snapshot of outstanding batches.

The producer never touches CUDA or training RNG. A checkpoint pauses it at a
batch boundary and saves BOTH its source state and every not-yet-consumed batch.
"""
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading

import torch


class ParallelTokenizer:
    def __init__(self, tokenizer, workers):
        self.base = tokenizer
        self.workers = workers
        self.local = threading.local()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='tokenizer')

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _encode(self, item):
        if not hasattr(self.local, 'tokenizer'):
            self.local.tokenizer = copy.deepcopy(self.base)
        docs, kwargs = item
        return self.local.tokenizer.encode(docs, **kwargs)

    def encode(self, docs, **kwargs):
        chunk = max(1, (len(docs) + self.workers - 1) // self.workers)
        items = [(docs[i:i+chunk], kwargs) for i in range(0, len(docs), chunk)]
        return [tokens for result in self.pool.map(self._encode, items) for tokens in result]


class StatefulPrefetchLoader:
    def __init__(self, loader, tokenizer, device, state=None, depth=8):
        self.loader, self.tokenizer = loader, tokenizer
        self.device = torch.device(device)
        self.depth = depth
        self.condition = threading.Condition()
        self.pending = deque(copy.deepcopy(state['pending']) if state else [])
        self.progress = copy.deepcopy(state['progress'] if state else loader.progress_state())
        self.paused = self.inflight = self.stopped = False
        self.error = None
        self.thread = threading.Thread(target=self._produce, name='batch-prefetch', daemon=True)
        self.thread.start()

    def _produce(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or (
                    not self.paused and len(self.pending) < self.depth))
                if self.stopped:
                    return
                self.inflight = True
            try:
                batch = next(self.loader)
                # The underlying loader recycles all four CPU buffers.
                owned = tuple(t.clone() for t in batch[:4]) + (copy.deepcopy(batch[4]),)
            except BaseException as error:
                with self.condition:
                    self.error = error
                    self.inflight = False
                    self.condition.notify_all()
                return
            with self.condition:
                self.pending.append(owned)
                self.inflight = False
                self.condition.notify_all()

    def __iter__(self):
        return self

    def __next__(self):
        with self.condition:
            self.condition.wait_for(lambda: self.pending or self.error or self.stopped)
            if self.error:
                raise RuntimeError('Background dataloader failed') from self.error
            if not self.pending:
                raise StopIteration
            batch = self.pending.popleft()
            self.progress = copy.deepcopy(batch[4])
            self.condition.notify_all()
        if self.device.type == 'cuda':
            values = tuple(t.pin_memory().to(self.device, non_blocking=True) for t in batch[:4])
        else:
            values = tuple(t.to(self.device) for t in batch[:4])
        return (*values, copy.deepcopy(batch[4]))

    def state_dict(self):
        with self.condition:
            self.paused = True
            try:
                self.condition.wait_for(lambda: not self.inflight)
                if self.error:
                    raise RuntimeError('Cannot snapshot failed dataloader') from self.error
                return dict(prefetch_version=1, source=copy.deepcopy(self.loader.state_dict()),
                            pending=copy.deepcopy(list(self.pending)), progress=copy.deepcopy(self.progress))
            finally:
                self.paused = False
                self.condition.notify_all()

    def progress_state(self):
        return copy.deepcopy(self.progress)

    def fallback_state_dict(self):
        # Legacy fallback is intentionally approximate, as in the serial loader.
        # Never advertise the producer's ahead-of-consumer cursor as consumed.
        return self.progress_state()

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join()
        self.tokenizer.pool.shutdown(wait=True)
