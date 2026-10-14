"""Eager training handoff timings; artificial slack is not model overlap proof."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from flash_dism import dism_core, VarlenLayout
from flash_dism.hard_training import AsyncHardPlanner
from check_hard_prefill_backward_triton import inspect_kernel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--tokens', type=int, default=2048)
    p.add_argument('--heads', type=int, default=6)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--slack-ms', type=float, nargs='+', default=[0., 50.])
    p.add_argument('--patterns', nargs='+', default=['random', 'repeat'])
    p.add_argument('--layouts', nargs='+', default=['fixed', 'packed'])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a fresh output path')
    torch.set_num_threads(1)
    torch.manual_seed(519)
    reports = []
    with AsyncHardPlanner(workers=args.workers) as planner:
        for pack in args.layouts:
            b, n, h = args.batch, args.tokens, args.heads
            shape = (1, b*n, h) if pack == 'packed' else (b, n, h)
            vectors = [(torch.randn(*shape, c, device='cuda')*.2).bfloat16().requires_grad_()
                       for c in (32, 32, 32, 32, 64)]
            lq, lk = [torch.full(shape, 6., device='cuda') for _ in range(2)]
            hard = torch.ones((b, h), device='cuda', dtype=torch.bool)
            direction = torch.zeros((shape[0], h), device='cuda', dtype=torch.bool)
            tau = torch.full((h,), .5, device='cuda', requires_grad=True)
            layout = None
            if pack == 'packed':
                if n != 2048:
                    p.error('ragged benchmark currently requires N2048')
                bounds = [0]
                for _ in range(b):
                    for length in (256, 512, 256, 1024):
                        bounds.append(bounds[-1]+length)
                layout = VarlenLayout.from_cu_seqlens(torch.tensor(bounds, dtype=torch.int32), b*n)
            do = torch.randn_like(vectors[-1])*.2
            for pattern in args.patterns:
                labels = [torch.randint(0, 512, (shape[0], h, shape[1]), device='cuda', dtype=torch.int32)
                          for _ in range(2)]
                if pattern == 'repeat':
                    for x in labels:
                        x.zero_()
                for slack in args.slack_ms:
                    samples = []
                    for iteration in range(args.repeats+1):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        out = dism_core(*vectors, lq, lk, *labels, direction, hard, tau,
                                        layout=layout, hard_planner=planner)
                        ticket = out.grad_fn.ticket
                        forward_host_ms = (time.perf_counter()-start)*1000
                        time.sleep(slack/1000)  # controlled slack, not measured model computation
                        backward_start = time.perf_counter()
                        gradients = torch.autograd.grad(out, [*vectors, tau], do)
                        torch.cuda.synchronize()
                        backward_wall_ms = (time.perf_counter()-backward_start)*1000
                        if iteration:
                            samples.append(dict(ticket.metrics(), forward_host_ms=forward_host_ms,
                                                backward_wall_ms=backward_wall_ms))
                        if iteration == args.repeats:
                            resources = [inspect_kernel(k, args.output.parent/'hard_async_cubins')
                                         for k in ticket.future.result().kernel.last_kernels]
                        del out, ticket, gradients
                        gc.collect()
                    row = dict(layout=pack, pattern=pattern, artificial_slack_ms=slack,
                               samples=samples, kernels=resources)
                    reports.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != 'kernels'}), flush=True)
    args.output.write_text(json.dumps(dict(config=vars(args) | {'output': str(args.output)},
                                           results=reports), indent=2))


if __name__ == '__main__':
    main()
