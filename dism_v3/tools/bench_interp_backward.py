"""Interleaved old / split-attention interpolation backward comparison on sm120."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
import triton
import cu_flash_dism as cu
from flash_dism.emb_kernel import emb_fwd_wrapper, emb_bwd_wrapper
from flash_dism.kernels.interp_bwd_split import emb_bwd_split, _dq, _dkv, _reduce, _tables
from bench_training import measure, memory, gpu_status, ROOT


def metrics(new, old):
    a,b = new.double().flatten(), old.double().flatten()
    bn = b.norm().item()
    return {'relative_l2': (a-b).norm().item()/max(bn,1e-30),
            'cosine': torch.nn.functional.cosine_similarity(a,b,dim=0).item(),
            'norm_ratio': a.norm().item()/max(bn,1e-30),
            'signed_bias_over_rms': (a-b).mean().item()/max(b.square().mean().sqrt().item(),1e-30)}


def resources(directory):
    directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    for jit in (_dq,_dkv,_reduce,_tables):
        for i,kernel in enumerate(jit.device_caches[torch.cuda.current_device()][0].values()):
            path=directory/f'{jit.__name__}_{i}.cubin'
            path.write_bytes(kernel.asm['cubin'])
            sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','-sass',str(path)],text=True)
            path.with_suffix('.sass').write_text(sass)
            rows.append({'kernel':jit.__name__,'index':i,'registers':getattr(kernel,'n_regs',None),
                         'spills':getattr(kernel,'n_spills',None),'shared':kernel.metadata.shared,
                         'calls':sum(' CALL' in line for line in sass.splitlines()),
                         'metadata':str(kernel.metadata)})
    return rows


def full_operator(q,k,eq,ek,tau,args):
    import flash_dism.compiler as compiler
    from flash_dism.varlen import VarlenLayout
    n,h,d=q.shape[1:]
    if n%2048:
        raise ValueError('full operator fixtures require multiples of2048')
    result=[]
    original=compiler.emb_bwd_wrapper
    try:
        for pattern,row_lengths in [('uniform',[2048]),('ragged',[256,512,256,1024])]:
            lengths=row_lengths*(n//2048)
            boundaries=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],device='cuda',dtype=torch.int32)
            table=VarlenLayout.from_cu_seqlens(boundaries,n).table.flatten()
            sq,sk=[torch.nn.functional.silu(torch.randn(1,n,h,32,device='cuda')).bfloat16() for _ in range(2)]
            v=torch.randn_like(q)
            flags=torch.rand(1,h,n//2048,1,device='cuda')<.95
            hard=flags.expand(1,h,n//2048,2048).reshape(1,h,n).contiguous()
            inputs=[q,k,sq,sk,v,eq.float(),ek.float(),tau,torch.tensor(1.),
                    torch.ones(1,h,device='cuda',dtype=torch.bool),hard]
            saved=compiler.voc_forward(*inputs,boundaries,table)
            dout=torch.randn_like(v)*.1
            def backward():
                return compiler.voc_backward(inputs,saved,boundaries,dout,table)
            def pair():
                current=compiler.voc_forward(*inputs,boundaries,table)
                return compiler.voc_backward(inputs,current,boundaries,dout,table)
            compiler.emb_bwd_wrapper=original
            expected=backward()
            compiler.emb_bwd_wrapper=emb_bwd_split
            actual=backward()
            comparison=[metrics(a,b) for a,b in zip(actual,expected)]
            assert all(torch.isfinite(x).all() for x in actual)
            del actual,expected
            row={'pattern':pattern,'tokens':n,'hard_fraction':flags.float().mean().item(),
                 'gradient_comparison':comparison,'memory':{},'variants':{}}
            backends={'baseline':original,'fa2':emb_bwd_split}
            if args.compare_reduction:
                backends['atomic']=partial(emb_bwd_split,reduction='atomic')
                backends['register']=partial(emb_bwd_split,reduction='register',splits=1)
            samples={name:{'backward':[],'pair':[]} for name in backends}
            for name,backend in backends.items():
                compiler.emb_bwd_wrapper=backend
                for _ in range(10):
                    pair()
                row['memory'][name]=memory(pair)
            for round_index in range(args.samples):
                for name in (list(backends) if round_index%2==0 else list(reversed(backends))):
                    compiler.emb_bwd_wrapper=backends[name]
                    for label,fn in [('backward',backward),('pair',pair)]:
                        samples[name][label].append(measure(fn,5,args.repeats,1)['stream_ms_median'])
            for name,measurements in samples.items():
                row['variants'][name]={key:{'median_ms':statistics.median(values),'samples_ms':values}
                                       for key,values in measurements.items()}
                print(f'FULL N{n} {pattern} {name}: '+str({k:round(v["median_ms"],4)
                      for k,v in row['variants'][name].items()}),flush=True)
            result.append(row)
    finally:
        compiler.emb_bwd_wrapper=original
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',nargs='+',type=int,default=[2048,8192,32768])
    parser.add_argument('--splits',nargs='+',type=int,default=[1,4,8,16,32])
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--samples',type=int,default=5)
    parser.add_argument('--tune-tiles',action='store_true')
    parser.add_argument('--tune-register',action='store_true')
    parser.add_argument('--full-operator',action='store_true')
    parser.add_argument('--compare-reduction',action='store_true',help='same tiles, paired partial vs global FP32 atomics')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('choose a fresh output path')
    if any(n<=0 or n%256 for n in args.tokens) or min(args.splits)<1:
        parser.error('positive 256-aligned token counts and positive splits required')
    if not Path(cu.__file__).resolve().is_relative_to(ROOT/'dism_v3/python'):
        raise RuntimeError('foreign extension')
    torch.set_num_threads(1)
    report={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'torch':torch.__version__,'triton':triton.__version__,'gpu_initial':gpu_status(),
            'extension_sha256':hashlib.sha256(Path(cu.__file__).read_bytes()).hexdigest(),
            'candidate_sha256':hashlib.sha256((ROOT/'dism_v3/python/flash_dism/kernels/interp_bwd_split.py').read_bytes()).hexdigest(),
            'method':'Synthetic BF16 SiLU inputs H6,D64,V512,beta1,tau0.5; unchanged forward. '
                     'Interleaved reverse-order alternating CUDA-event windows; includes allocation and dispatch. '
                     'All gradients active; not a full model or zero-gradient-skipping benchmark.',
            'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'cases':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for n in args.tokens:
        torch.manual_seed(1729)
        h,d,vocab=6,64,512
        def rand(shape):
            return torch.nn.functional.silu(torch.randn(shape,device='cuda')).bfloat16()
        q,k=[rand((1,n,h,d)) for _ in range(2)]
        eq,ek=[rand((h,vocab,d)) for _ in range(2)]
        tau=torch.full((h,),.5,device='cuda')
        saved=emb_fwd_wrapper(q,k,eq,ek,1.,tau)
        doq,dok=[torch.randn_like(q)*.1 for _ in range(2)]
        dlq,dlk=[torch.randn((1,h,n),device='cuda')*.1 for _ in range(2)]
        inputs=(q,k,eq,ek,*saved[:4],doq,dok,dlq,dlk,1.,tau)
        variants={'baseline':partial(emb_bwd_wrapper,*inputs)}
        variants.update({f'split{s}':partial(emb_bwd_split,*inputs,splits=s,dq_v=32,dkv_m=32,dkv_v=64)
                         for s in args.splits})
        variants['fa2']=partial(emb_bwd_split,*inputs)
        if args.compare_reduction:
            variants['register']=partial(emb_bwd_split,*inputs,reduction='register',splits=1)
            # Both modes use the selected tile schedule and same split counts.
            for splits in args.splits:
                for reduction in ('partial','atomic'):
                    variants[f'{reduction}{splits}']=partial(emb_bwd_split,*inputs,
                                                           splits=splits,reduction=reduction)
        if args.tune_tiles:
            for dq_m,dq_v,kv_m,kv_v in [(64,64,32,64),(64,64,64,64),
                                       (64,64,64,32),(128,32,32,64),
                                       (32,64,32,64),(64,32,32,32)]:
                name=f's16_dq{dq_m}x{dq_v}_kv{kv_m}x{kv_v}'
                variants[name]=partial(emb_bwd_split,*inputs,splits=16,
                                       dq_m=dq_m,dq_v=dq_v,dkv_m=kv_m,dkv_v=kv_v)
        if args.tune_register:
            for kv_m,kv_v in [(32,16),(64,16),(128,16),(32,32),(128,32),(32,64),(64,64)]:
                variants[f'register{kv_m}x{kv_v}']=partial(emb_bwd_split,*inputs,splits=1,
                    reduction='register',dkv_m=kv_m,dkv_v=kv_v)
            for kv_v in (16,32):
                for stages in (1,2):
                    variants[f'register64x{kv_v}_stages{stages}']=partial(emb_bwd_split,*inputs,splits=1,
                        reduction='register',dkv_m=64,dkv_v=kv_v,num_stages=stages)
        oracle=variants['baseline']()
        accuracy={}
        unsupported={}
        for name,fn in list(variants.items()):
            try:
                actual=fn()
            except triton.runtime.errors.OutOfResources as error:
                unsupported[name]=str(error)
                del variants[name]
                print(f'{name}: unsupported resources: {error}',flush=True)
                continue
            accuracy[name]=[metrics(a,b) for a,b in zip(actual,oracle)]
            assert all(torch.isfinite(x).all() for x in actual),name
            # Tight regression gate, separate from comparison to FP64 math.
            assert max(m['relative_l2'] for m in accuracy[name])<1e-4,(name,accuracy[name])
            del actual
        until=time.perf_counter()+1.
        while time.perf_counter()<until:
            variants['baseline']()
        torch.cuda.synchronize()
        samples={name:[] for name in variants}
        for round_index in range(args.samples):
            order=list(variants) if round_index%2==0 else list(reversed(variants))
            for name in order:
                samples[name].append(measure(variants[name],5,args.repeats,1)['stream_ms_median'])
        row={'tokens':n,'accuracy':accuracy,'unsupported':unsupported,'gpu':gpu_status(),'variants':{}}
        if args.compare_reduction:
            row['memory']={name:memory(fn) for name,fn in variants.items()}
        baseline=statistics.median(samples['baseline'])
        for name,values in samples.items():
            median=statistics.median(values)
            row['variants'][name]={'median_ms':median,'min_ms':min(values),'max_ms':max(values),
                                   'samples_ms':values,'speedup':baseline/median}
            print(f'N{n} {name}: {median:.4f} ms ({baseline/median:.2f}x)',flush=True)
        report['cases'].append(row)
        if args.full_operator:
            row['full_operator']=full_operator(q,k,eq,ek,tau,args)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    report['resources']=resources(args.output.with_suffix(''))
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
