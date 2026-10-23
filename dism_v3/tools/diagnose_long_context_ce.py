"""Offline long-context CE sanity check on actual long PG19 documents.

Full 128K teacher-forced scoring plus same-target, shorter-context controls.
Uses the downstream BF16-operand/FP32-accumulation CE with checkpoint softcap;
does not allocate full-vocabulary logits or change production dispatch.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.all():
        return dict(tokens=values.size, nonfinite=int((~finite).sum()))
    mean = float(values.mean())
    return dict(tokens=values.size, nonfinite=0, mean_nll=mean,
                perplexity=math.exp(mean), p99=float(np.quantile(values, .99)),
                max_nll=float(values.max()))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--suite', type=Path, default=Path.home()/'downstream-eval')
    p.add_argument('--length', type=int, default=131072)
    p.add_argument('--books', type=int, default=4)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--sam-repeats', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.length < 8192 or args.length % 8192:
        p.error('length must be an 8K multiple, at least 8K')
    if min(args.books, args.sam_repeats) < 1:
        p.error('positive books and sam-repeats required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('eval_adapter', args.suite/'adapter_layout_bounded.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    lm = adapter.NanochatLM(str(args.checkpoint), str(args.suite/'tokenizer'), 1,
                           args.length, hard_backend='sam', prefill_precision='tf32x3',
                           prefill_workers=args.workers)
    result = dict(checkpoint=str(args.checkpoint), length=args.length,
                  training_context=lm.training_context, softcap=lm.model.config.softcap,
                  workers=args.workers, ce='downstream BF16 operands, FP32 accumulation',
                  target_tokens=1024, books=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2))

    def evaluate(x, labels, mode, target_start=0):
        lm.hard_backend = 'quadratic' if mode == 'quadratic' else 'sam'
        if mode != 'quadratic':
            lm.prefill.mma_precision = mode
        cu = torch.tensor([0, x.shape[1]], dtype=torch.int32, device='cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        hidden = lm._features(x, cu)
        hidden_nonfinite = int((~torch.isfinite(hidden)).sum())
        h = hidden[:, target_start:].contiguous()
        target = labels[:, target_start:].contiguous()
        loss = adapter.fused_cross_entropy_unreduced(
            h, lm.weight, target, ignore_index=-1, softcap=lm.model.config.softcap)
        values = loss.float().cpu().numpy().reshape(-1).copy()
        return values, dict(hidden_nonfinite=hidden_nonfinite,
                            forward_and_ce_ms=(time.perf_counter()-start)*1000, **stats(values))

    try:
        selected = []
        index = 0
        for batch in pq.ParquetFile(args.suite/'data/pg19/test.parquet').iter_batches(batch_size=4, columns=['text']):
            for text in batch.column(0).to_pylist():
                tokens = lm.tok_encode(text, add_special_tokens=False)
                if len(tokens) >= args.length:
                    selected.append((index, tokens, hashlib.sha256(text.encode()).hexdigest()))
                index += 1
                if len(selected) == args.books:
                    break
            if len(selected) == args.books:
                break
        if len(selected) != args.books:
            raise ValueError('not enough sufficiently long local books')
        for index, tokens, text_hash in selected:
            print('book', index, 'tokens', len(tokens), flush=True)
            x, _ = lm._inputs([[lm.prefix_token_id] + tokens[:args.length-1]])
            labels = torch.tensor([tokens[:args.length]], device='cuda', dtype=torch.int64)
            row = dict(index=index, book_tokens=len(tokens), text_sha256=text_hash,
                       full={}, paired_windows=[])
            result['books'].append(row)
            arrays = {}
            for mode in ('quadratic', 'tf32x3', 'bf16'):
                row['full'][mode] = []
                for repeat in range(1 if mode == 'quadratic' else args.sam_repeats):
                    torch.cuda.empty_cache()
                    values, measured = evaluate(x, labels, mode)
                    arrays[f'{mode}_{repeat}'] = values
                    measured['bins_8k'] = [stats(values[i:i+8192]) for i in range(0, args.length, 8192)]
                    if mode != 'quadratic':
                        delta = values.astype(np.float64)-arrays['quadratic_0']
                        measured['mean_delta_vs_quadratic'] = float(delta.mean())
                        measured['mean_abs_delta_vs_quadratic'] = float(np.abs(delta).mean())
                    row['full'][mode].append(measured)
                    save()
                    print('full', index, mode, repeat, {k:v for k,v in measured.items() if k!='bins_8k'}, flush=True)
            endpoints = sorted(set(min(n, args.length) for n in (8192,16384,32768,65536,98304,args.length)))
            for end in endpoints:
                target_start = end-1024
                paired = dict(end=end, target_start=target_start,
                              full_prefix={mode:stats(arrays[f'{mode}_0'][target_start:end])
                                           for mode in row['full']}, shorter={})
                for context in (lm.training_context, 8192):
                    if context > end:
                        continue
                    # Same next-token targets, genuine contiguous preceding text;
                    # no synthetic BOS is inserted into the middle of a book.
                    begin = end-context
                    values, measured = evaluate(x[:, begin:end].contiguous(),
                                                labels[:, begin:end].contiguous(),
                                                'quadratic', target_start=context-1024)
                    arrays[f'context{context}_end{end}'] = values
                    paired['shorter'][str(context)] = measured
                row['paired_windows'].append(paired)
                save()
            np.savez_compressed(args.output.with_name(f'{args.output.stem}_book{index}.npz'), **arrays)
            print('paired', index, json.dumps([
                dict(end=w['end'], full=w['full_prefix']['quadratic'].get('mean_nll'),
                     shorter={k:v.get('mean_nll') for k,v in w['shorter'].items()})
                for w in row['paired_windows']]), flush=True)
            del x, labels, arrays
        save()
    except Exception as exc:
        result['error'] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise
    finally:
        lm.close()


if __name__ == '__main__':
    main()
