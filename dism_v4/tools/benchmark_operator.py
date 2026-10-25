"""Local CUDA forward/backward and eval benchmark; no datasets or training jobs."""
import argparse
import json
import time
from pathlib import Path

import torch
from flash_dism_v4.blocked import dism_v4_blocked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--length', type=int, default=2048)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--heads', type=int, default=12)
    p.add_argument('--vocab', type=int, default=512)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--block', type=int, default=64)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--output')
    p.add_argument('--check-bf16', action='store_true', help='compare outputs and gradients to FP32 at benchmark shape')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    q = torch.randn(args.batch, args.length, args.heads, args.vocab, device='cuda', requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn(args.batch, args.length, args.heads, args.dim, device='cuda', requires_grad=True)
    tau = torch.linspace(0, 2, args.heads, device='cuda', requires_grad=True)
    cot = torch.randn_like(v)
    results = []
    for mode in ('soft', 'jensen', 'hard'):
        for precision in ('fp32', 'bf16'):
            def step(backward):
                if backward:
                    for t in (q, k, v, tau):
                        t.grad = None
                    out = dism_v4_blocked(q, k, v, tau, key_mode=mode, precision=precision, block_size=args.block)
                    (out*cot).sum().backward()
                else:
                    with torch.no_grad():
                        out = dism_v4_blocked(q, k, v, tau, key_mode=mode, precision=precision, block_size=args.block)
                return out
            for backward in (True, False):
                for _ in range(2):
                    step(backward)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for _ in range(args.steps):
                    out = step(backward)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter()-start)/args.steps
                row = dict(mode=mode, precision=precision, backward=backward, seconds=elapsed,
                           tokens_per_second=args.batch*args.length/elapsed,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(), finite=bool(torch.isfinite(out).all()))
                if backward:
                    row['finite_gradients'] = all(t.grad is None or bool(torch.isfinite(t.grad).all()) for t in (q,k,v,tau))
                print(json.dumps(row), flush=True)
                results.append(row)
    accuracy = []
    if args.check_bf16:
        for mode in ('soft', 'jensen', 'hard'):
            exact = dism_v4_blocked(q,k,v,tau,key_mode=mode,precision='fp32',block_size=args.block)
            approx = dism_v4_blocked(q,k,v,tau,key_mode=mode,precision='bf16',block_size=args.block)
            ge = torch.autograd.grad((exact*cot).sum(),(q,k,v,tau),allow_unused=True)
            ga = torch.autograd.grad((approx*cot).sum(),(q,k,v,tau),allow_unused=True)
            for name,a,e in [('output',approx,exact),*zip(('q','k','v','tau'),ga,ge)]:
                if e is None:
                    assert a is None
                    continue
                a, e = a.detach(), e.detach()
                row = dict(mode=mode,tensor=name,
                           relative_l2=float((a-e).norm()/e.norm().clamp_min(1e-12)),
                           cosine=float(torch.nn.functional.cosine_similarity(a.flatten(),e.flatten(),dim=0)),
                           max_absolute_error=float((a-e).abs().max()))
                accuracy.append(row)
                print(json.dumps(row),flush=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(config=vars(args),results=results,bf16_accuracy=accuracy), indent=2)+'\n')


if __name__ == '__main__':
    main()
