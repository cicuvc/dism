"""Offline full-model SAM/quadratic length sweep on fixed local PG19 prefixes.

Includes CPU planning, transfers and final norm; excludes tokenization and CE.
No downloads, checkpoint mutation or automatic production backend selection.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
from statistics import median
import time

import pyarrow.parquet as pq
import torch


@torch.no_grad()
def main(observer=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, default=Path.home() / 'downstream-eval')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[8192, 16384, 24576, 32768, 49152, 65536])
    p.add_argument('--book-length', type=int, default=131072,
                   help='Select the first batch-size books with this many tokens, keeping refinements comparable')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--modes', nargs='+', choices=['quadratic', 'tf32x3', 'bf16'],
                   default=['quadratic', 'tf32x3', 'bf16'])
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if any(n <= 0 or n % 256 for n in args.lengths):
        p.error('lengths must be positive multiples of 256')
    if min(args.batch_size, args.warmup, args.repeats) < 1:
        p.error('batch-size, warmup and repeats must be positive')
    if args.book_length < max(args.lengths) - 1:
        p.error('book-length must cover all requested prefixes')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter_path = args.suite / 'adapter_layout_bounded.py'
    spec = importlib.util.spec_from_file_location('eval_adapter', adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    lm = adapter.NanochatLM(str(args.checkpoint), str(args.suite / 'tokenizer'),
                           args.batch_size, max(args.lengths), hard_backend='sam',
                           prefill_precision='tf32x3', prefill_workers=args.workers)
    result = dict(checkpoint=str(args.checkpoint), batch=args.batch_size,
                  workers=args.workers, book_length=args.book_length,
                  warmup=args.warmup, repeats=args.repeats,
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  allocator_config=os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')),
                  adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
                  scope='full model through final norm, no CE; synchronized wall time',
                  books=[], cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2))

    def select(mode):
        lm.hard_backend = 'quadratic' if mode == 'quadratic' else 'sam'
        if mode != 'quadratic':
            lm.prefill.mma_precision = mode

    try:
        books = []
        index = 0
        for batch in pq.ParquetFile(args.suite / 'data/pg19/test.parquet').iter_batches(batch_size=4, columns=['text']):
            for text in batch.column(0).to_pylist():
                tokens = lm.tok_encode(text, add_special_tokens=False)
                if len(tokens) >= args.book_length:
                    books.append(tokens)
                    result['books'].append(dict(index=index, tokens=len(tokens),
                                                text_sha256=hashlib.sha256(text.encode()).hexdigest()))
                index += 1
                if len(books) == args.batch_size:
                    break
            if len(books) == args.batch_size:
                break
        if len(books) != args.batch_size:
            raise ValueError('not enough sufficiently long local PG19 books')
        print(json.dumps(dict(books=result['books'])), flush=True)
        save()
        rng = random.Random(20261024)
        for n in args.lengths:
            gc.collect()
            torch.cuda.empty_cache()
            x, cu = lm._inputs([[lm.prefix_token_id] + book[:n-1] for book in books])
            assert x.shape == (args.batch_size, n)
            positions = torch.linspace(0, n-1, min(1024, n), device='cuda').long()
            row = dict(length=n, modes={})
            result['cases'].append(row)
            modes = list(dict.fromkeys(args.modes))
            anchors = {}
            for mode in modes:
                select(mode)
                print(f'warmup length={n} mode={mode}', flush=True)
                for _ in range(args.warmup):
                    hidden = lm._features(x, cu)
                    del hidden
                torch.cuda.synchronize()
                row['modes'][mode] = dict(wall_ms=[], peak_allocated_bytes=[], repeat_sample_relative_l2=[])
            if observer is not None:
                observer(lm, n)
            for _ in range(args.repeats):
                order = list(modes)
                rng.shuffle(order)
                for mode in order:
                    select(mode)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    hidden = lm._features(x, cu)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    data = row['modes'][mode]
                    data['wall_ms'].append(elapsed)
                    data['peak_allocated_bytes'].append(torch.cuda.max_memory_allocated())
                    sample = hidden[:, positions].float().cpu()
                    del hidden
                    if mode not in anchors:
                        anchors[mode] = sample
                    else:
                        data['repeat_sample_relative_l2'].append(float((sample-anchors[mode]).norm()/anchors[mode].norm()))
                save()
            for mode in modes:
                row['modes'][mode]['median_ms'] = median(row['modes'][mode]['wall_ms'])
            if 'quadratic' in modes:
                for mode in modes:
                    row['modes'][mode]['speedup_vs_quadratic'] = row['modes']['quadratic']['median_ms'] / row['modes'][mode]['median_ms']
                    row['modes'][mode]['sample_relative_l2_vs_quadratic'] = float((anchors[mode]-anchors['quadratic']).norm()/anchors['quadratic'].norm())
            print(json.dumps(row), flush=True)
            save()
            del x, cu, anchors, sample
    except Exception as exc:
        result['error'] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise
    finally:
        lm.close()


if __name__ == '__main__':
    main()
