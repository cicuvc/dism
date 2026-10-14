"""Standalone hard summary timings, including output initialization.

Run in blkw with this worktree's dism_v3/python on PYTHONPATH. Also measures
chunk passing separately and paired with summary. No model, planner or readout
is included; this is not a training speedup.
"""
import argparse
import json

import torch

from bench_training import measure, profile_kernels
from flash_dism.summary import summarize_hard, hard_chunk_scan
from flash_dism.varlen import VarlenLayout


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(20261014)
    b, h, n = 16, 6, 2048
    cases = []
    for packed in (False, True):
        q = torch.randint(512, (b,h,n), device='cuda', dtype=torch.int32)
        k = torch.randint(512, (b,h,n), device='cuda', dtype=torch.int32)
        layout = None
        if packed:
            q = q.permute(1,0,2).reshape(1,h,b*n).contiguous()
            k = k.permute(1,0,2).reshape(1,h,b*n).contiguous()
            offsets = [0]
            for _ in range(b):
                for length in (256,512,256,1024):
                    offsets.append(offsets[-1]+length)
            layout = VarlenLayout.from_cu_seqlens(torch.tensor(offsets, dtype=torch.int32), b*n)
        for all_hard in (True, False):
            hard = torch.ones((b,h), device='cuda', dtype=torch.bool)
            if not all_hard:
                hard[:, ::2] = False
            fn = lambda: summarize_hard(q,k,hard,layout=layout)
            result = fn()
            timing = measure(fn, warmup=10, repeats=20, samples=7)
            item = dict(layout='ragged' if packed else 'fixed', hard_fraction=1. if all_hard else .5,
                        output_bytes=result.nbytes, **timing)
            chunk = lambda: hard_chunk_scan(result,layout=layout,heads=h)
            pair = lambda: hard_chunk_scan(fn(),layout=layout,heads=h)
            item['chunk'] = measure(chunk,warmup=10,repeats=20,samples=7)
            item['summary_and_chunk'] = measure(pair,warmup=10,repeats=20,samples=7)
            if args.profile:
                item['instrumented'] = profile_kernels(fn, iterations=20)
                item['chunk_instrumented'] = profile_kernels(chunk,iterations=20)
            cases.append(item)
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), batch=b, heads=h, row_tokens=n,
                         scope='hard summary + initialization and chunk passing, no readout', cases=cases), indent=2))


if __name__ == '__main__':
    main()
