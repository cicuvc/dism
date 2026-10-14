"""Read-only FLA fused-linear CE comparison on the existing real 72M fixture.

Stock FLA has neither softcap nor document weights. Its training forward already
computes dx/dw: only complete forward+backward is comparable across backends.
"""
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path

import torch
import triton
import flash_dism
import cu_flash_dism
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from flash_dism.kernels import fused_cross_entropy as ce
from bench_cross_entropy import ROOT, fixture
from bench_training import command, gpu_status, measure, memory, profile_kernels


def diagnostics(actual, reference):
    a, r = actual.float().flatten(), reference.float().flatten()
    delta = a-r
    return dict(finite=bool(torch.isfinite(a).all()),
                max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm()/r.norm().clamp_min(1e-30)),
                norm_ratio=float(a.norm()/r.norm().clamp_min(1e-30)),
                cosine=float(torch.nn.functional.cosine_similarity(a, r, dim=0)))


def benchmark(name, loss_fn, x, w, args, reference=None, prepared_backward=None):
    a, b = x.detach().requires_grad_(), w.detach().requires_grad_()
    def forward():
        return loss_fn(a, b)
    saved_loss = forward()
    # FLA modifies its saved gradients for upstream != 1; this repeated-graph
    # measurement deliberately uses only upstream == 1 (the non-mutating branch).
    def backward():
        return torch.autograd.grad(saved_loss, (a, b), retain_graph=True)
    if prepared_backward is not None:
        backward = prepared_backward
    def pair():
        return torch.autograd.grad(forward(), (a, b))
    grads = pair()
    if prepared_backward is not None:
        for direct, public in zip(prepared_backward(), grads):
            torch.testing.assert_close(direct.to(public.dtype), public, atol=0, rtol=0)
    result = dict(name=name, loss=float(saved_loss.detach()),
                  finite=all(bool(torch.isfinite(v).all()) for v in (saved_loss, *grads)))
    if not result['finite']:
        raise RuntimeError(f'non-finite outputs: {name}')
    if reference is not None:
        result['versus_local_nosoftcap'] = dict(
            loss_abs=abs(float(saved_loss.detach())-reference[0]),
            dx=diagnostics(grads[0], reference[1]),
            dw=diagnostics(grads[1], reference[2]))
    for key, fn in [('training_forward', forward), ('backward_saved', backward),
                    ('public_autograd', pair)]:
        result[key] = measure(fn, args.warmup, args.repeats, args.samples)
    # Real accumulation commonly supplies a non-unit upstream. Always create a
    # fresh graph, since FLA scales its saved gradients in place.
    quarter = torch.tensor(0.25, device='cuda')
    def scaled_pair():
        return torch.autograd.grad(forward(), (a, b), grad_outputs=quarter)
    scaled = scaled_pair()
    result['upstream_quarter_diagnostics'] = {
        key: diagnostics(value, ref*quarter)
        for key, value, ref in zip(('dx', 'dw'), scaled, grads)}
    del scaled
    result['public_autograd_upstream_quarter'] = measure(
        scaled_pair, args.warmup, args.repeats, args.samples)
    result['memory'] = memory(pair)
    if args.profile:
        result['profile'] = profile_kernels(pair, iterations=3)
    print(name, 'loss', result['loss'],
          {k: result[k]['stream_ms_median'] for k in
           ('training_forward', 'backward_saved', 'public_autograd',
            'public_autograd_upstream_quarter')}, flush=True)
    return result, (float(saved_loss.detach()), *[g.detach() for g in grads])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--step', type=int, default=19074)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--num-chunks', type=int, nargs='+', default=[8, 32])
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT/'dism_v3/build'):
        ap.error('choose fresh output under dism_v3/build')
    if min(args.warmup, args.repeats, args.samples, *args.num_chunks) < 1:
        ap.error('counts must be positive')
    for module in (flash_dism, cu_flash_dism):
        if not Path(module.__file__).resolve().is_relative_to(ROOT/'dism_v3/python'):
            raise RuntimeError(f'foreign module {module.__file__}')
    torch.set_num_threads(1)
    torch._inductor.config.compile_threads = 1
    torch._inductor.config.triton.cudagraphs = False
    gpu_before = gpu_status()
    data, info = fixture(args.checkpoint_dir, args.step)
    x, w, labels, cu = (data[k] for k in ('x', 'w', 'labels', 'cu'))
    maximum = int(cu.diff().max())
    fla_module = importlib.import_module('fla.modules.fused_linear_cross_entropy')
    report = dict(arguments={k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items()},
                  git_head=command('git', 'rev-parse', 'HEAD'),
                  git_status=command('git', 'status', '--short'),
                  torch=torch.__version__, triton=triton.__version__, gpu_before=gpu_before,
                  fixture=info, shape=dict(tokens=x.numel()//x.shape[-1],
                                          width=x.shape[-1], vocabulary=w.shape[0]),
                  ignored_labels=int((labels == -1).sum()),
                  sources={module.__name__: dict(path=module.__file__, sha256=hashlib.sha256(
                      Path(module.__file__).read_bytes()).hexdigest()) for module in (ce, fla_module)},
                  method='Resident real inputs; 5x10 calls by default; CUDA-event and wall intervals; '
                         'no CUDA graphs; local public autograd fullgraph compiled, '
                         'stock FLA class eager as required by its compiler.disable; '
                         'stock FLA computes dx/dw during forward and synchronizes host scalars; '
                         'FLA: no softcap, token mean, ignore=-1, no smoothing/scaling/l2warp; '
                         'memory is incremental allocated, not reserved; profiling separate',
                  cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    reference = None
    for name, cap, packed in [('local_nosoftcap_token', None, False),
                              ('local_softcap30_token', 30., False),
                              ('local_softcap30_document', 30., True)]:
        def local(a, b, cap=cap, packed=packed):
            return ce.fused_cross_entropy(a, b, labels, ignore_index=-1, softcap=cap,
                                         cu_seqlens=cu if packed else None,
                                         max_seqlen=maximum if packed else None)
        fn = torch.compile(local, fullgraph=True)
        def direct_fwd(a, b):
            if packed:
                return ce.torch_fwd_varlen_fused_cross_entropy(a, b, labels, cu, maximum, -1, cap)
            return ce.torch_fwd_fused_cross_entropy(a, b, labels, -1, cap)
        saved = torch.compile(direct_fwd, fullgraph=True)(x, w)
        upstream = torch.ones((), device='cuda')
        def direct_bwd():
            if packed:
                return ce.torch_bwd_varlen_fused_cross_entropy(
                    x, w, labels, saved[1], saved[2], saved[3], upstream, -1, None, cap)
            return ce.torch_bwd_fused_cross_entropy(
                x, w, labels, saved[1], saved[2], upstream, -1, None, cap)
        back = torch.compile(direct_bwd, fullgraph=True)
        row, outputs = benchmark(name, fn, x, w, args, prepared_backward=back)
        row.update(softcap=cap, reduction='document_mean' if packed else 'token_mean',
                   backend='torch.compile fullgraph',
                   backward_method='compiled direct helper on prepared LSE/count, as previous benchmark')
        if reference is None:
            reference = outputs
        del outputs
        report['cases'].append(row)
        save()
    flat_x, flat_labels = x.flatten(0, -2), labels.flatten()
    for chunks in args.num_chunks:
        loss_module = FusedLinearCrossEntropyLoss(ignore_index=-1, num_chunks=chunks)
        def stock(a, b):
            return loss_module(a, flat_labels, b)
        row, outputs = benchmark(f'fla_linear_chunks{chunks}', stock, flat_x, w, args,
                                 (reference[0], reference[1].view_as(flat_x), reference[2]))
        del outputs
        chunk_rows = triton.next_power_of_2(triton.cdiv(flat_x.shape[0], min(
            chunks, triton.cdiv(w.shape[0], w.shape[1]))))
        row.update(softcap=None, reduction='token_mean', backend='stock eager class',
                   requested_num_chunks=chunks, actual_num_chunks=triton.cdiv(flat_x.shape[0], chunk_rows),
                   chunk_rows=chunk_rows, logits_chunk_bytes=chunk_rows*w.shape[0]*x.element_size())
        report['cases'].append(row)
        save()
        gc.collect()
    report['gpu_after'] = gpu_status()
    save()
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()
