"""Bounded synthetic 2K training/eval smoke test; never starts a corpus run."""
import argparse
import json
import time
from pathlib import Path

import torch
from nanochat.models import ModelSpec, PretrainOptimizerConfig, build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='nanochat/configs/dism_v4_2k_prototype.json')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--precision', choices=['fp32', 'bf16'])
    p.add_argument('--output')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    spec_data = json.loads(Path(args.config).read_text())
    if args.precision:
        spec_data['config']['readout_precision'] = args.precision
    spec = ModelSpec.from_dict(spec_data)
    model = build_model(spec).cuda()
    model.init_weights()
    opt = model.setup_pretraining_optimizer(PretrainOptimizerConfig(1e-4, 1e-4, 1e-4, 1e-4, .1))
    n, vocab = model.config.sequence_len, model.config.vocab_size
    ids = torch.randint(vocab, (args.batch, n+1), device='cuda')
    x, y = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
    cu = torch.arange(0, (args.batch+1)*n, n, device='cuda', dtype=torch.int32)
    rows = []
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
                          context=n, batch=args.batch, spec=spec.to_dict())), flush=True)
    for probability, name in [(0., 'soft'), (1., 'hard'), (.5, 'mixed')]:
        model.train()
        model.hard_probability.fill_(probability)

        def step():
            opt.zero_grad(set_to_none=True)
            loss = model(x, y, cu_seqlens=cu)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            return loss.detach(), norm.detach()

        for _ in range(2):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.steps):
            loss, norm = step()
        torch.cuda.synchronize()
        seconds = (time.perf_counter()-start)/args.steps
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(norm))
        row = dict(mode=name, phase='train', seconds=seconds,
                   tokens_per_second=args.batch*n/seconds,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   loss=float(loss), grad_norm=float(norm), rng_counter=int(model.rng_counter))
        rows.append(row)
        print(json.dumps(row), flush=True)

    opt.zero_grad(set_to_none=True)
    model.eval()
    for hard in (False, True):
        model.eval_hard = hard
        with torch.no_grad():
            for _ in range(2):
                model(x, y, cu_seqlens=cu)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.steps):
                loss = model(x, y, cu_seqlens=cu)
            torch.cuda.synchronize()
            seconds = (time.perf_counter()-start)/args.steps
        assert bool(torch.isfinite(loss))
        row = dict(mode='hard' if hard else 'soft', phase='eval', seconds=seconds,
                   tokens_per_second=args.batch*n/seconds,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), loss=float(loss))
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(spec=spec.to_dict(), args=vars(args), results=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
