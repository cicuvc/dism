"""Read-only checkpoint batch replay; measures staging, not model throughput."""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from nanochat.data_prefetch import StatefulPrefetchLoader


class Replay:
    def __init__(self, batches):
        self.batches, self.cursor = batches, 0

    def __next__(self):
        b = self.batches[self.cursor % len(self.batches)]
        self.cursor += 1
        return (*b, self.state_dict())

    def state_dict(self):
        return {'cursor': self.cursor}

    progress_state = state_dict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=200)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1] / 'build'
    if args.output.exists() or not args.output.resolve().is_relative_to(root):
        p.error('output must be fresh and under dism_v3/build')
    torch.set_num_threads(1)
    # Trusted local checkpoint, never advanced or written back.
    state = torch.load(args.state, map_location='cpu', weights_only=False)
    keys = ('inputs', 'targets', 'cu_seqlens', 'segment_ids')
    batches = [tuple(state['prefetched_batch'][k] for k in keys)]
    batches += [tuple(b[:4]) for b in state['dataloader']['pending']]
    torch.cuda.init()
    results = []
    # Reverse order on the second pass to expose allocator/order effects.
    for mode in ('consumer', 'producer', 'producer', 'consumer'):
        tokenizer = SimpleNamespace(pool=SimpleNamespace(shutdown=lambda **kw: None))
        loader = StatefulPrefetchLoader(Replay(batches), tokenizer,
                                       'cuda' if mode == 'producer' else 'cpu')
        host, complete = [], []
        try:
            for i in range(args.samples + 20):
                # Isolate a ready-queue consumption from tokenizer starvation.
                with loader.condition:
                    if not loader.condition.wait_for(lambda: len(loader.pending) == loader.depth,
                                                     timeout=10):
                        raise RuntimeError('producer did not fill queue')
                start = time.perf_counter()
                b = next(loader)
                if mode == 'consumer':
                    b = (*[t.pin_memory().to('cuda', non_blocking=True) for t in b[:4]],
                         copy.deepcopy(b[4]))
                submitted = time.perf_counter()
                torch.cuda.synchronize()
                done = time.perf_counter()
                if i >= 20:
                    host.append((submitted-start)*1e3)
                    complete.append((done-start)*1e3)
                expected = batches[i % len(batches)]
                if i in (0, args.samples + 19):
                    for x, y in zip(b[:4], expected):
                        torch.testing.assert_close(x.cpu(), y, rtol=0, atol=0)
        finally:
            loader.close()
        result = dict(mode=mode, host_median_ms=statistics.median(host),
                      host_mean_ms=statistics.mean(host),
                      host_p95_ms=sorted(host)[int(.95*len(host))],
                      complete_median_ms=statistics.median(complete))
        results.append(result)
        print(result, flush=True)
    args.output.write_text(json.dumps(dict(
        method='Ready-queue staging only; excludes producer wait, model and tokenization',
        bytes_per_batch=[sum(t.numel()*t.element_size() for t in b) for b in batches],
        samples=args.samples, results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
