"""Authenticated loopback token service; expose remotely only through SSH.

Indexed effective batches are idempotent. Only stream cursor checkpoints, not
the corpus or all token batches, are persisted. Clients prefetch one batch.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.request import Request, urlopen

import numpy as np
import torch


class IndexedBatches:
    def __init__(self, stream, directory, identity, cache_size=16):
        self.stream, self.directory = stream, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity, self.cache_size = identity, cache_size
        self.cache, self.lock, self.index = OrderedDict(), threading.Lock(), 0
        self._save()

    def _save(self):
        path = self.directory / f'{self.index:08d}.json'
        if path.exists():
            return
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(identity=self.identity, index=self.index,
                                      stream=self.stream.state_dict())))
        os.replace(tmp, path)

    def get(self, index):
        if not 0 <= index < 30000:
            raise ValueError('Batch index outside configured run')
        with self.lock:
            if index in self.cache:
                return self.cache[index]
            # Resume near the requested index, including after server restart.
            checkpoints = [p for p in self.directory.glob('*.json') if int(p.stem) <= index]
            best = max(checkpoints, key=lambda p: int(p.stem))
            if index < self.index or int(best.stem) > self.index:
                saved = json.loads(best.read_text())
                if saved['identity'] != self.identity:
                    raise ValueError('Token source identity changed')
                self.stream.load_state_dict(saved['stream'])
                self.index = saved['index']
            while self.index <= index:
                x, y = self.stream.next_batch()
                block = torch.cat((x, y[:, -1:]), dim=1).numpy().astype('<i4')
                payload = block.tobytes()
                self.cache[self.index] = (payload, self.stream.epoch)
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
                self.index += 1
                if self.index % 100 == 0:
                    self._save()
            return self.cache[index]


def fetch(url, secret, path):
    last = None
    for attempt in range(5):
        try:
            request = Request(url.rstrip('/') + path, headers={'Authorization': 'Bearer ' + secret})
            with urlopen(request, timeout=120) as response:
                body = response.read()
                digest = response.headers.get('X-SHA256')
                if digest and hashlib.sha256(body).hexdigest() != digest:
                    raise ValueError('Token payload checksum mismatch')
                return body, int(response.headers.get('X-Epoch', 0))
        except (OSError, ValueError) as error:
            last = error
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError('Token service unavailable; no batch was skipped') from last


class RemotePackedStream:
    def __init__(self, url, secret, split, metadata, batch_size=8):
        self.url, self.secret, self.split, self.metadata = url, secret, split, metadata
        if split not in ('train', 'validation') or metadata['batch'] % batch_size:
            raise ValueError('Invalid token stream split or microbatch')
        self.batch_size, self.epoch, self.cursor = batch_size, 0, 0
        self.accum = metadata['batch'] // batch_size
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending, self.block, self.block_index = None, None, -1

    def _request(self, index):
        data, epoch = fetch(self.url, self.secret, f'/batch/{self.split}/{index}')
        expected = self.metadata['batch'] * (self.metadata['context'] + 1)
        array = np.frombuffer(data, dtype='<i4')
        if array.size != expected:
            raise ValueError('Invalid token batch size')
        return torch.from_numpy(array.copy()).long().reshape(self.metadata['batch'], -1), epoch

    def next_batch(self):
        index, micro = divmod(self.cursor, self.accum)
        if self.block_index != index:
            if self.pending is not None and self.pending[0] == index:
                self.block, self.epoch = self.pending[1].result()
            else:
                self.block, self.epoch = self._request(index)
            self.block_index = index
            limit = self.metadata['steps'] if self.split == 'train' else self.metadata['eval_batches']
            self.pending = ((index + 1, self.pool.submit(self._request, index + 1))
                            if index + 1 < limit else None)
        rows = self.block[micro * self.batch_size:(micro + 1) * self.batch_size]
        self.cursor += 1
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()

    def state_dict(self):
        return dict(identity=self.metadata['identity'], split=self.split,
                    batch_size=self.batch_size, cursor=self.cursor, epoch=self.epoch)

    def load_state_dict(self, state):
        for key in ('identity', 'split', 'batch_size'):
            if state[key] != self.state_dict()[key]:
                raise ValueError(f'Remote stream resume mismatch: {key}')
        self.cursor, self.epoch = state['cursor'], state['epoch']
        self.block_index = -1

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--port', type=int, default=18473)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--steps', type=int, default=30000)
    args = parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(2)
    from transformers import GPT2TokenizerFast
    from .lm_data import PackedStream, split_files
    tokenizer_file = Path(args.tokenizer) / 'tokenizer.json'
    tok = GPT2TokenizerFast(tokenizer_file=str(tokenizer_file), eos_token='<|endoftext|>',
                            bos_token='<|endoftext|>', unk_token='<|endoftext|>')
    tok.model_max_length = 10**12
    assert len(tok) == 50257 and tok.eos_token_id == 50256
    train, val = split_files(args.data)
    meta = dict(data=args.data, tokenizer=args.tokenizer, seed=args.seed, batch=64, context=2048,
                steps=args.steps, eval_batches=100, tokenizer_sha256=hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
                files=[(p, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in train + val])
    meta['identity'] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    directory = Path(args.state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    meta_file = directory / 'metadata.json'
    if meta_file.exists() and json.loads(meta_file.read_text()) != meta:
        # JSON normalizes tuple entries to lists.
        if json.loads(meta_file.read_text()) != json.loads(json.dumps(meta)):
            raise ValueError('Existing service state belongs to a different source')
    meta_file.write_text(json.dumps(meta, indent=2))
    secret_path = directory / 'auth_token'
    if not secret_path.exists():
        with secret_path.open('x') as file:
            os.chmod(secret_path, 0o600)
            file.write(secrets.token_hex(32))
    secret = secret_path.read_text().strip()
    sources = {split: IndexedBatches(PackedStream(files, tok, batch_size=64, seed=args.seed,
                           repeat=split == 'train'), directory / split, meta['identity'],
                           cache_size=100 if split == 'validation' else 16)
               for split, files in [('train', train), ('validation', val)]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not log tokens, credentials or document text.

        def do_GET(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + secret):
                self.send_error(403)
                return
            try:
                epoch = 0
                if self.path == '/meta':
                    payload = json.dumps(meta).encode()
                else:
                    _, kind, split, number = self.path.split('/')
                    index = int(number)
                    limit = meta['steps'] if split == 'train' else meta['eval_batches']
                    if kind != 'batch' or split not in sources or not 0 <= index < limit:
                        raise ValueError('Invalid request')
                    payload, epoch = sources[split].get(index)
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('X-SHA256', hashlib.sha256(payload).hexdigest())
                self.send_header('X-Epoch', str(epoch))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as error:
                print(type(error).__name__, str(error), flush=True)
                self.send_error(500, 'Token request failed')

    print(json.dumps(dict(event='ready', port=args.port, identity=meta['identity'])), flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
