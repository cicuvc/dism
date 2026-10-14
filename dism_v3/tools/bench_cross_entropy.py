"""Standalone current CE baseline on a read-only 72M checkpoint's real head input.

No training updates, data loading continuation, production tile changes or downloads.
Flat token mean and packed document mean are timed separately, not substituted.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import triton
import flash_dism
import cu_flash_dism
from flash_dism.kernels import fused_cross_entropy as ce
from nanochat.models import build_model, model_spec_from_metadata
from nanochat.models.spec import ModelSpec
from bench_conv_frontend import mapped_name, to_fla
from bench_training import measure, memory, command, gpu_status, profile_kernels

ROOT = Path(__file__).resolve().parents[2]


def fixture(path, step):
    """Capture only one eager/no-grad forward; the original checkpoint is untouched."""
    from nanochat.models.dism import model as model_module
    meta = json.loads((path/f'meta_{step:06d}.json').read_text())
    spec = model_spec_from_metadata(meta)
    state = torch.load(path/f'train_state_{step:06d}_rank0.pt', map_location='cpu', weights_only=False)
    if state['model_fingerprint'] != spec.fingerprint():
        raise ValueError('source checkpoint fingerprint mismatch')
    weights = torch.load(path/f'model_{step:06d}.pt', map_location='cpu', weights_only=True)
    weights = {n.removeprefix('_orig_mod.'):v for n,v in weights.items()}
    source_spec = spec.to_dict()
    if spec.config.get('conv_impl','fused') == 'fused':
        spec = ModelSpec(spec.architecture,spec.architecture_version,
                         dict(spec.config,conv_impl='linear_fla',conv_backend='cuda'))
        weights = {mapped_name(n):to_fla(v) if n.endswith('.conv_weight') else v for n,v in weights.items()}
    with torch.device('meta'):
        model = build_model(spec)
    model.to_empty(device='cuda')
    model.init_weights()
    model.load_state_dict(weights,strict=True)
    model.train()
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])
    batch = {k:v.cuda() for k,v in state['prefetched_batch'].items()
             if k in ('inputs','targets','cu_seqlens','segment_ids')}
    captured = {}
    original = model_module.fused_cross_entropy
    def capture(hidden, head, labels, **kwargs):
        captured.update(x=hidden.detach(),w=head.detach(),labels=labels.detach(),options=kwargs)
        return hidden.new_zeros((),dtype=torch.float32)
    model_module.fused_cross_entropy = capture
    try:
        with torch.no_grad():
            model(batch['inputs'],batch['targets'],cu_seqlens=batch['cu_seqlens'],segment_ids=batch['segment_ids'])
    finally:
        model_module.fused_cross_entropy = original
    info = dict(source=source_spec, capture=spec.to_dict(), checkpoint_step=step,
                checkpoint_rng_counter=int(weights['rng_counter']),
                method='One eager/no-grad forward on the saved first prefetched batch with linear_fla; '
                       'capture before CE; no optimizer constructed or state written',
                document_lengths=batch['cu_seqlens'].diff().tolist(),
                ce_options=captured.pop('options'))
    captured['cu'] = batch['cu_seqlens'].contiguous()
    del model, weights, state, batch
    gc.collect()
    torch.cuda.synchronize()
    return captured, info


def run_mode(data, packed, args):
    x,w,labels,cu = (data[k] for k in ('x','w','labels','cu'))
    maximum = int(cu.diff().max())
    upstream = torch.ones((),device='cuda',dtype=torch.float32)
    def forward(a,b,c):
        if packed:
            return ce.torch_fwd_varlen_fused_cross_entropy(a,b,c,cu,maximum,-1,30.)
        return ce.torch_fwd_fused_cross_entropy(a,b,c,-1,30.)
    def backward(a,b,c,lse,count,inv,dy):
        if packed:
            return ce.torch_bwd_varlen_fused_cross_entropy(a,b,c,lse,count,inv,dy,-1,None,30.)
        return ce.torch_bwd_fused_cross_entropy(a,b,c,lse,count,dy,-1,None,30.)
    # Compiled validation mirrors training: no eager label/boundary D2H checks.
    fwd = torch.compile(forward,fullgraph=True)
    bwd = torch.compile(backward,fullgraph=True)
    saved = fwd(x,w,labels)
    loss,lse,count = saved[:3]
    inv = saved[3] if packed else lse
    def back():
        return bwd(x,w,labels,lse,count,inv,upstream)
    def pair():
        out = fwd(x,w,labels)
        return bwd(x,w,labels,out[1],out[2],out[3] if packed else out[1],upstream)
    a,b = x.detach().requires_grad_(), w.detach().requires_grad_()
    def public(u,v):
        return ce.fused_cross_entropy(u,v,labels,ignore_index=-1,softcap=30.,
                                     cu_seqlens=cu if packed else None,
                                     max_seqlen=maximum if packed else None)
    compiled = torch.compile(public,fullgraph=True)
    def public_pair():
        out = compiled(a,b)
        return torch.autograd.grad(out,(a,b))
    result = dict(loss=float(loss),normalization_count=int(count),
                  mode='document_mean' if packed else 'token_mean')
    for name,fn in [('forward',lambda:fwd(x,w,labels)),('backward',back),
                    ('combined_lowlevel',pair),('public_autograd',public_pair)]:
        result[name] = measure(fn,args.warmup,args.repeats,args.samples)
    dx,dw = back()
    ax,aw = public_pair()
    result['finite'] = all(bool(torch.isfinite(v).all()) for v in (loss,dx,dw,ax,aw))
    torch.testing.assert_close(ax,dx,atol=0,rtol=0)
    torch.testing.assert_close(aw,dw.to(aw.dtype),atol=0,rtol=0)
    result['lowlevel_memory'] = memory(pair)
    result['public_memory'] = memory(public_pair)
    if args.profile:
        result['profile'] = profile_kernels(pair,iterations=3)
    print(result['mode'], 'loss',result['loss'],
          {k:result[k]['stream_ms_median'] for k in ('forward','backward','combined_lowlevel','public_autograd')},flush=True)
    return result


def forward_probe(data, args):
    """Only benchmark launch arguments vary; source/defaults remain unchanged."""
    x,w,labels = (data[k] for k in ('x','w','labels'))
    m,c = x.shape[1:]
    v = w.shape[0]
    reference_lse,reference_nll = ce._forward_rows(x,w,labels,-1,30.)
    results=[]
    for bm in args.probe_bm:
        lse,nll = torch.empty_like(reference_lse),torch.empty_like(reference_nll)
        def call():
            return ce._linear_cross_entropy_fwd[(triton.cdiv(m,bm),)](
                x,w,labels,lse,nll,m,c,v,30.,IGNORE_INDEX=-1,HAS_SOFTCAP=True,
                BLOCK_M=bm,BLOCK_V=ce._FWD_BV,BLOCK_K=ce._FWD_BK,num_warps=4,num_stages=3)
        kernel=call()
        torch.cuda.synchronize()
        errors=dict(lse_max_abs=float((lse-reference_lse).abs().max()),
                    nll_max_abs=float((nll-reference_nll).abs().max()))
        torch.testing.assert_close(lse,reference_lse,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(nll,reference_nll,atol=2e-5,rtol=2e-5)
        timing=measure(call,args.warmup,args.repeats,args.samples)
        ptx=kernel.asm['ptx']
        # Persist generated cubin/PTX as ignored diagnostic artifacts, never sources.
        prefix=args.output.with_suffix('').with_name(args.output.stem+f'_bm{bm}')
        prefix.with_suffix('.cubin').write_bytes(kernel.asm['cubin'])
        prefix.with_suffix('.ptx').write_text(ptx)
        row=dict(bm=bm,bv=ce._FWD_BV,bk=ce._FWD_BK,num_warps=4,num_stages=3,
                 timing=timing,errors=errors,registers=kernel.n_regs,spills=kernel.n_spills,
                 shared_bytes=kernel.metadata.shared,cubin=str(prefix.with_suffix('.cubin')))
        results.append(row)
        print('BM probe',bm,timing['stream_ms_median'],'registers/spills',kernel.n_regs,kernel.n_spills,flush=True)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint-dir',type=Path,required=True)
    ap.add_argument('--step',type=int,default=19074)
    ap.add_argument('--warmup',type=int,default=5)
    ap.add_argument('--repeats',type=int,default=10)
    ap.add_argument('--samples',type=int,default=5)
    ap.add_argument('--profile',action='store_true')
    ap.add_argument('--probe-bm',type=int,nargs='*',default=[])
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT/'dism_v3/build'):
        ap.error('choose fresh output under dism_v3/build')
    if min(args.warmup,args.repeats,args.samples)<1:
        ap.error('counts must be positive')
    if any(bm not in (16,32,64,128) for bm in args.probe_bm):
        ap.error('BM probes must be16/32/64/128')
    for module in (flash_dism,cu_flash_dism):
        if not Path(module.__file__).resolve().is_relative_to(ROOT/'dism_v3/python'):
            raise RuntimeError(f'foreign module {module.__file__}')
    torch.set_num_threads(1)
    torch._inductor.config.compile_threads=1
    torch._inductor.config.triton.cudagraphs=False
    data,info=fixture(args.checkpoint_dir,args.step)
    x,w,labels=(data[k] for k in ('x','w','labels'))
    if info['ce_options'] != dict(ignore_index=-1,softcap=30.):
        raise ValueError(f"unexpected CE options: {info['ce_options']}")
    report=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                git_head=command('git','rev-parse','HEAD'),git_status=command('git','status','--short'),
                gpu_before=gpu_status(),torch=torch.__version__,triton=triton.__version__,fixture=info,
                source_sha256=hashlib.sha256(Path(ce.__file__).read_bytes()).hexdigest(),
                shape=dict(tokens=x.shape[1],width=x.shape[2],vocabulary=w.shape[0]),
                ignored_labels=int((labels==-1).sum()),chunk_rows=ce._chunk_rows(x.shape[1],w.shape[0],None),
                method='Warm resident inputs; CUDA-event interval and host wall timing; '
                       'compiled direct forward/backward plus actual public autograd; '
                       'no CUDA graphs, no model/optimizer time; profiling separate')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    report['modes']=[]
    for packed in (False,True):
        report['modes'].append(run_mode(data,packed,args))
        save()
    if args.probe_bm:
        report['forward_bm_probe']=forward_probe(data,args)
    report['gpu_after']=gpu_status()
    save()
    print('saved',args.output,flush=True)


if __name__=='__main__':
    main()
