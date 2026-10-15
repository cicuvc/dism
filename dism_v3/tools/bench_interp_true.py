"""Isolated fixed-true interpolation comparison; no model/checkpoint writes."""
import argparse
import json
from pathlib import Path

import torch
import triton
from flash_dism.emb_kernel import emb_fwd_wrapper
from flash_dism.kernels.interp_true import emb_fwd_true
from flash_dism.kernels.interp_bwd_split import emb_bwd_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=32768)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build = Path(__file__).resolve().parents[1]/'build'
    if args.output.exists() or not args.output.resolve().is_relative_to(build):
        parser.error('output must be a fresh path inside dism_v3/build')
    torch.manual_seed(704)
    n, h, d, vocab = args.tokens, 6, 64, 512
    def rand(shape):
        return (torch.randn(shape,device='cuda')*.25).bfloat16()
    q,k = [rand((1,n,h,d)) for _ in range(2)]
    eq,ek = [rand((h,vocab,d)) for _ in range(2)]
    tau = torch.ones(h,device='cuda')
    doq = rand(q.shape)
    dok = torch.zeros_like(q)
    dlq = torch.zeros((1,h,n),device='cuda')
    dlk = torch.randn_like(dlq)
    old = emb_fwd_wrapper(q,k,eq,ek,1.,tau)
    new = emb_fwd_true(q,k,eq,ek,1.,tau)
    functions = {
        'dual_forward': lambda: emb_fwd_wrapper(q,k,eq,ek,1.,tau),
        'true_forward': lambda: emb_fwd_true(q,k,eq,ek,1.,tau),
        'dual_backward': lambda: emb_bwd_split(q,k,eq,ek,*old[:4],doq,dok,dlq,dlk,1.,tau),
        'true_backward': lambda: emb_bwd_split(q,k,eq,ek,*new[:4],doq,dok,dlq,dlk,1.,tau,direction_true=True),
    }
    for fn in functions.values():
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    results = []
    for repetition in range(3):
        order = list(functions) if repetition%2 == 0 else list(reversed(functions))
        result = {name: triton.testing.do_bench(functions[name],warmup=25,rep=100,
                                               quantiles=[.5,.2,.8]) for name in order}
        results.append(result)
        print(result,flush=True)
    # Inspect only the selected warmed kernels, not every autotune candidate.
    from flash_dism.kernels.interp_true import _forward, delta_true
    from flash_dism.kernels.interp_bwd_split import _dq, _dkv
    resources = []
    originals = []
    def instrument(fn):
        original = fn.run
        originals.append((fn,original))
        def run(*a,**kw):
            kernel = original(*a,**kw)
            resources.append(dict(name=kernel.name,registers=kernel.n_regs,
                spills=kernel.n_spills,shared=kernel.metadata.shared,
                has_call='CALL' in kernel.asm['sass']))
            return kernel
        fn.run = run
    try:
        for fn in (_forward,delta_true,_dq.fn,_dkv.fn):
            instrument(fn)
        functions['true_forward']()
        functions['true_backward']()
    finally:
        for fn,original in originals:
            fn.run = original
    print(resources,flush=True)
    args.output.write_text(json.dumps(dict(tokens=n,heads=h,dim=d,vocab=vocab,
        milliseconds_median_p20_p80=results,
        selected_kernel_resources=resources,
        unused_saved_bytes_removed=q.numel()*q.element_size()),indent=2)+'\n')


if __name__ == '__main__':
    main()
