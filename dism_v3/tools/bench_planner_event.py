"""Paired event-backward diagnostics; artifacts stay in ignored build/.

Uses the precision-fix commit as baseline, never edits installed kernels.
R64 is a diagnostic wrapper extension (native CUDA supports R16/32 only).
CUDA graphs measure execute time; eager enqueue and synchronized wall time
are measured separately. Timing never includes compilation or CPU planning.
"""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import triton
import flash_dism
import cu_flash_dism
from flash_dism.inference.training_planner import NativeTrainingPlanner
from flash_dism.kernels.hard_backward_scan import HardPrefillBackward
from check_hard_prefill_backward_triton import inspect_kernel

ROOT = Path(__file__).resolve().parents[2]
BASE = '2eb846ae92cc92fdc2572e5f1b36e098823fcdac'
SOURCE = 'dism_v3/python/flash_dism/kernels/hard_backward_fused.py'


def variant_source(source, variant):
    from planner_algorithm_variants import ALGORITHM_VARIANTS, algorithm_source
    if variant in ALGORITHM_VARIANTS:
        return algorithm_source(source, variant)
    if variant == 'split3_combined':
        return variant_source(variant_source(source, 'combined'), 'split3')
    if variant in ('reuse', 'split2', 'split3'):
        start = source.index('        if BF16:\n            hq =')
        end = source.index('        hq *=', start)
        source = source[:start] + '''        hq = tl.dot(do, tl.trans(h), input_precision="tf32x3")
        tq = hq
''' + source[end:]
        start = source.index('        if BF16:\n            hk =')
        end = source.index('        hk *=', start)
        source = source[:start] + '''        hk = tl.dot(v, tl.trans(g), input_precision="tf32x3")
        tk = hk
''' + source[end:]
    if variant in ('split2', 'split3'):
        # Exact only for BF16 source vectors; FP32 keeps TF32x3 fallback.
        helper = '''
@tr.jit
def _split_dot(a, b):
    hi = b.to(tl.bfloat16)
    residual = b - hi.to(tl.float32)
    lo = residual.to(tl.bfloat16)
    low = tl.dot(a.to(tl.bfloat16), lo)
REMAINDER
    return tl.dot(a.to(tl.bfloat16), hi, low)

'''.replace('REMAINDER', '''    tiny = (residual - lo.to(tl.float32)).to(tl.bfloat16)
    low = tl.dot(a.to(tl.bfloat16), tiny) + low''' if variant == 'split3' else '')
        source = source.replace('@tr.jit\ndef _chunk_backward', helper + '@tr.jit\ndef _chunk_backward', 1)
        for expr in ('tl.dot(do, tl.trans(h), input_precision="tf32x3")',
                     'tl.dot(v, tl.trans(g), input_precision="tf32x3")'):
            a,b = ('do', 'tl.trans(h)') if 'do,' in expr else ('v', 'tl.trans(g)')
            source = source.replace('        hq = '+expr, '        hq = _split_dot('+a+', '+b+')') if a=='do' else source.replace('        hk = '+expr, '        hk = _split_dot('+a+', '+b+')')
        source = source.replace('h = tl.dot(tl.trans(sk*ks[:, None]), v, input_precision="tf32x3")', 'h = tl.trans(_split_dot(tl.trans(v), sk*ks[:, None]))')
        source = source.replace('g = tl.dot(tl.trans(sq), do*qs[:, None], input_precision="tf32x3")', 'g = _split_dot(tl.trans(sq), do*qs[:, None])')
    if variant in ('split2', 'split3'):
        lines=[]
        for line in source.splitlines(keepends=True):
            if '= _split_dot(' in line or '= tl.trans(_split_dot(' in line:
                indent=line[:len(line)-len(line.lstrip())]
                lhs=line.strip().split(' = ')[0]
                fallback={'hq':'tl.dot(do, tl.trans(h), input_precision="tf32x3")',
                          'hk':'tl.dot(v, tl.trans(g), input_precision="tf32x3")',
                          'h':'tl.dot(tl.trans(sk*ks[:, None]), v, input_precision="tf32x3")',
                          'g':'tl.dot(tl.trans(sq), do*qs[:, None], input_precision="tf32x3")'}[lhs]
                lines.append(indent+'if BF16 and SQ.dtype.element_ty == tl.bfloat16 and SK.dtype.element_ty == tl.bfloat16 and V.dtype.element_ty == tl.bfloat16 and DO.dtype.element_ty == tl.bfloat16:\n')
                lines.append('    '+line)
                lines.append(indent+'else:\n'+indent+'    '+lhs+' = '+fallback+'\n')
            else: lines.append(line)
        source=''.join(lines)
    if variant == 'reorder':
        for x, a, state, scalar, vec in [('q','do','h','qs','sq'), ('k','v','g','ks','sk')]:
            line = f'            h{x} = tl.dot({a}.to(tl.bfloat16), tl.trans({state}).to(tl.bfloat16))\n'
            source = source.replace(line, '')
            # Move vector contraction after tau consumption, same arithmetic.
            source = source.replace(f'        h{x} *= {scalar}[:, None]\n', '')
            target = f'        dtau += tl.sum(dw*tl.sum({vec}*t{x}, 1)*{scalar}, 0)\n'
            source = source.replace(target, target + f'''        if BF16:
            h{x} = tl.dot({a}.to(tl.bfloat16), tl.trans({state}).to(tl.bfloat16))
        h{x} *= {scalar}[:, None]
''')
    if variant == 'combined':
        start=source.index('    if BF16:\n        dv =')
        end=source.index('    if CHUNKS is None:',start)
        old=source[start:end]
        local=old[:old.index('    if CHUNKS is not None:')]
        cross=old[old.index('    if CHUNKS is not None:'):]
        common=cross[cross.index('        pos ='):cross.index('        h =')]
        h=cross[cross.index('        h ='):cross.index('        g =')]
        g=cross[cross.index('        g ='):cross.index('        if BF16:\n            hv =')]
        hv=cross[cross.index('        if BF16:\n            hv ='):]
        output='    if CHUNKS is not None:\n'+common
        for grad,ptr,extra in [('dv','DV_OUT',hv),('dsq','DSQ',h),('dsk','DSK',g)]:
            token=f'    tl.atomic_add({ptr}'
            a=local.index(token); b=local.index('sem="relaxed")',a)+len('sem="relaxed")')
            before=local[:a]; atomic=local[a:b]; local=local[b:].lstrip('\n')
            output+=before+'    if CHUNKS is not None:\n'
            if grad=='dv':
                output+=cross[cross.index('        g ='):cross.index('        if BF16:\n            hk =')]
            cut=extra.index(f'        tl.atomic_add({ptr}')
            output+=extra[:cut]
            output+=f'        {grad} += '+{'dv':'hv*ks[:, None]','dsq':'hq','dsk':'hk'}[grad]+'\n'
            output+=atomic+'\n'
        source=source[:start]+output+source[end:]
    if variant == 'separate':
        start = source.index('    if CHUNKS is not None:')
        end = source.index('    if CHUNKS is None:', start)
        branch = source[start:end]
        # Auxiliary kernel contains only accurate boundary tau, no local pairs.
        pre = source[source.index('    local ='):source.index('    valid =')]
        tau = branch[branch.index('        pos ='):]
        tau = '\n'.join(line[4:] for line in tau.split('\n'))
        for x, a, state, scalar, vec, out, idx in [('q','do','h','qs','sq','DSQ','qi'),('k','v','g','ks','sk','DSK','ki')]:
            astart = tau.index('    if BF16:')
            aend = tau.index(f'    dtau += tl.sum(dw*tl.sum({vec}', astart)
            tau = tau[:astart] + f'    t{x} = tl.dot({a}, tl.trans({state}), input_precision="tf32x3")\n' + tau[aend:]
            astart = tau.index(f'    tl.atomic_add({out}')
            aend = tau.index('sem="relaxed")', astart)+len('sem="relaxed")')
            tau = tau[:astart]+tau[aend:]
        tau = tau[:tau.index('    if BF16:')]
        signature = source[source.index('@tr.jit\ndef _chunk_backward'):source.index('    # LOCAL')].replace('_chunk_backward', '_boundary_tau')
        extra = signature + pre + '    dtau = tl.full((), 0., tl.float32)\n' + tau + '    tl.atomic_add(TAU + stream, dtau, sem="relaxed")\n\n'
        for x, a, state, vec, scalar in [('q','do','h','sq','qs'),('k','v','g','sk','ks')]:
            source = source.replace(f'            t{x} = tl.dot({a}, tl.trans({state}), input_precision="tf32x3")\n', '')
            source = source.replace(f'            t{x} = h{x}\n', '')
            source = source.replace(f'        dtau += tl.sum(dw*tl.sum({vec}*t{x}, 1)*{scalar}, 0)\n','')
        source = source.replace('@tr.jit\ndef _chunk_summaries', extra + '@tr.jit\ndef _chunk_summaries')
        # Duplicate only multi readout launches, using the new kernel.
        lines=source.splitlines(keepends=True); result=[]; i=0
        while i<len(lines):
            if 'self.last_kernels.append(_chunk_backward[(self.multi_chunks,)]' in lines[i]:
                block=[]
                while True:
                    block.append(lines[i]); i+=1
                    if 'num_stages=1))' in block[-1]: break
                result.extend(block)
                result.extend([line.replace('_chunk_backward[','_boundary_tau[') for line in block])
            else: result.append(lines[i]); i+=1
        source=''.join(result)
    # Diagnostics only: R64 isn't exposed by the production API.
    source = source.replace('sq.shape[1] not in (16, 32)', 'sq.shape[1] not in (16, 32, 64)')
    return source


def load_variant(name, directory):
    source = subprocess.check_output(['git','show',f'{BASE}:{SOURCE}'], cwd=ROOT, text=True)
    if name == 'current': source=(ROOT/SOURCE).read_text()
    source = variant_source(source, name)
    path=directory/f'{name}.py'; path.write_text(source)
    module_name=f'flash_dism.kernels._perf_{name}'
    spec=importlib.util.spec_from_file_location(module_name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[module_name]=module
    spec.loader.exec_module(module)
    return module


class Recorder:
    def __init__(self,kernel,records): self.kernel,self.records=kernel,records
    def __getitem__(self,grid):
        def call(*args,**kw):
            kernel=self.kernel
            fn=lambda: kernel[grid](*args,**kw)
            compiled=fn()
            self.records.append((compiled.name,grid,fn,compiled))
            return compiled
        return call


def stats(xs):
    return dict(median_us=float(np.median(xs)),p95_us=float(np.percentile(xs,95)))


def measure(fn,samples=11,inner=8,setup=None):
    if setup is not None:
        inner=1
        setup()
    for _ in range(4): fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner): retained=fn()
    for _ in range(5):
        if setup is not None: setup()
        graph.replay()
    times=[]
    for _ in range(samples):
        a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        if setup is not None: setup()
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b)*1000/inner)
    return stats(times)


def eager(fn,samples):
    enqueue=[]; wall=[]
    for _ in range(samples):
        torch.cuda.synchronize(); t=time.perf_counter_ns(); fn(); u=time.perf_counter_ns()
        torch.cuda.synchronize(); v=time.perf_counter_ns()
        enqueue.append((u-t)/1000); wall.append((v-t)/1000)
    return dict(enqueue=stats(enqueue),wall=stats(wall))


def build_plan(module,device,host,compact,warps):
    if compact:
        return module.CompactFusedHardPrefillBackward.from_compact(device,num_warps=warps)
    source=HardPrefillBackward.from_native(device,mma_precision='bf16')
    return module.FusedHardPrefillBackward.from_plan(source,host.data['offsets'].numpy(),
                host_lengths=host.data['lengths'].numpy(),num_warps=warps)


def diagnostic_c8(planner, labels, tau, compact):
    """Rechunk native C16 into logical C8, padded to the minimum MMA width16.

    CPU repacking is diagnostic only and is included in CPU-plan timing.
    The native production planner continues to reject C8.
    """
    from flash_dism.inference.training_planner import (
        HostTrainingPlan, HostCompactPlan, _specification, _views,
        _compact_specification, _compact_view)
    original=planner.prepare(labels,tau,chunk_size=16,compact=False)
    data={k:v.numpy() for k,v in original.data.items()}
    fields={k:[] for k in ('rows','prefixes','weights','dlog','lengths','resets')}
    offsets=[0]
    for begin,end in zip(data['offsets'][:-1],data['offsets'][1:]):
        for chunk in range(begin,end):
            length=int(data['lengths'][chunk])
            for start in range(0,length,8):
                stop=min(start+8,length)
                for key in ('rows','prefixes','weights','dlog'):
                    values=data[key][chunk*16+start:chunk*16+stop].copy()
                    if key=='prefixes' and start:
                        values-=data[key][chunk*16+start-1]
                    fields[key].extend(np.pad(values,(0,16-len(values))))
                fields['lengths'].append(stop-start)
                fields['resets'].append(int(data['resets'][chunk]) if start==0 else 0)
        offsets.append(len(fields['lengths']))
    fields['offsets']=offsets; fields['stream_heads']=data['stream_heads']
    stats=dict(original.statistics,logical_chunk_size=8,chunks=len(fields['lengths']))
    spec,size=_specification(stats)
    storage=torch.empty(size,dtype=torch.uint8,pin_memory=True); views=_views(storage,spec)
    for key,value in fields.items(): views[key].copy_(torch.as_tensor(np.asarray(value)))
    host=HostTrainingPlan(storage,views,spec,stats,{})
    if not compact: return host
    spans=np.diff(offsets); singles=np.flatnonzero(spans==1); multis=np.flatnonzero(spans>1)
    stats=dict(stats,compact=True,multi_streams=len(multis),multi_chunks=int(spans[multis].sum()),
               bucket_blocks=[len(singles),0,0],single_blocks=len(singles))
    spec,size=_compact_specification(stats)
    storage=torch.empty(size,dtype=torch.uint8,pin_memory=True); packed=_views(storage,spec)
    buckets,multi=_compact_view(storage,packed)
    single_chunks=np.asarray(offsets)[singles]
    chunks=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in multis]) if len(multis) else np.array([],np.int64)
    for key in ('rows','prefixes','weights','dlog'):
        buckets[0][key].copy_(views[key].view(-1,16)[single_chunks].reshape(-1))
        multi[key].copy_(views[key].view(-1,16)[chunks].reshape(-1))
    buckets[0]['lengths'].copy_(views['lengths'][single_chunks])
    buckets[0]['stream_heads'].copy_(views['stream_heads'][singles])
    for key in ('lengths','resets'): multi[key].copy_(views[key][chunks])
    multi['stream_heads'].copy_(views['stream_heads'][multis])
    multi['offsets'].copy_(torch.as_tensor(np.r_[0,np.cumsum(spans[multis])]))
    return HostCompactPlan(storage,buckets,multi,spec,stats,{})


def main():
    global BASE
    from planner_algorithm_variants import ALGORITHM_VARIANTS
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--baseline',default=BASE,help='Pinned source revision for paired diagnostics')
    ap.add_argument('--variants',nargs='+',choices=['baseline','current','reuse','reorder','split2','split3','separate','combined','split3_combined',*ALGORITHM_VARIANTS],default=['baseline','reuse','reorder','split2','split3','separate'])
    ap.add_argument('--chunks',nargs='+',type=int,choices=[8,16,32,64],default=[16,32,64])
    ap.add_argument('--warps',nargs='+',type=int,choices=[2,4,8],default=[2,4,8])
    ap.add_argument('--shapes',nargs='+',default=['32x64','64x64','64x32'])
    ap.add_argument('--cases',nargs='+',default=['1x1x65:repeat','1x1x512:repeat','2x6x2048:repeat','2x6x2048:random'])
    ap.add_argument('--compact',nargs='+',type=int,default=[0,1])
    ap.add_argument('--samples',type=int,default=15)
    ap.add_argument('--references',action='store_true')
    args=ap.parse_args()
    BASE = subprocess.check_output(['git','rev-parse',args.baseline],cwd=ROOT,text=True).strip()
    if not any(args.output.resolve().is_relative_to(p) for p in (ROOT/'build',ROOT/'dism_v3/build')):
        ap.error('output must be inside an ignored worktree build directory')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists(): ap.error('output must be fresh')
    for module in (flash_dism,cu_flash_dism):
        if not Path(module.__file__).resolve().is_relative_to(ROOT/'dism_v3/python'): raise RuntimeError('foreign module')
    torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.09)
    (args.output.parent/'variants').mkdir(exist_ok=True)
    modules={v:load_variant(v,args.output.parent/'variants') for v in args.variants}
    planner=NativeTrainingPlanner(2)
    from bench_training import gpu_status
    result=dict(baseline=BASE,torch=torch.__version__,triton=triton.__version__,device=torch.cuda.get_device_name(),gpu_before=gpu_status(),
                arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                extension_sha256=hashlib.sha256(Path(cu_flash_dism.__file__).read_bytes()).hexdigest(),
                source_sha256={name:hashlib.sha256((args.output.parent/'variants'/f'{name}.py').read_bytes()).hexdigest() for name in args.variants},
                method='Warmed CUDA graphs: 8 calls/replay (1 when state snapshots exceed 128 MiB), 5 warm replays; scan uses 1 call and rebuilds summaries before each sample. Eager enqueue and synchronized wall include completed upload waits. CPU plan excludes initial JIT. Stages independent, not additive. C8 is diagnostic rechunking padded to physical width16; R64 is diagnostic only.',cases=[])
    reference_cache={}
    for case in args.cases:
        shape,pattern=case.split(':'); batch,heads,n=map(int,shape.split('x'))
        labels=np.random.default_rng(120).integers(0,512,(2,batch,heads,n),dtype=np.int32)
        if pattern=='repeat': labels.fill(0)
        for c in args.chunks:
            for compact in args.compact:
                (diagnostic_c8(planner,labels,np.full(heads,.5),bool(compact)) if c==8 else planner.prepare(labels,np.full(heads,.5),chunk_size=c,compact=bool(compact)))
                times=[]
                for _ in range(9):
                    t=time.perf_counter_ns(); host=(diagnostic_c8(planner,labels,np.full(heads,.5),bool(compact)) if c==8 else planner.prepare(labels,np.full(heads,.5),chunk_size=c,compact=bool(compact))); times.append((time.perf_counter_ns()-t)/1000)
                device=host.upload(); device.wait(); torch.cuda.synchronize()
                for dims in args.shapes:
                    r,dv=map(int,dims.split('x')); torch.manual_seed(318)
                    xs=[(torch.randn((batch*n*heads,d),device='cuda')*.2).bfloat16() for d in (r,r,dv,dv)]
                    references={}
                    if args.references:
                        key=(case,r,dv)
                        if key not in reference_cache:
                            references={}
                            if r in (16,32) and n%256==0:
                                from flash_dism.hard_backward import backward_hard_recompute
                                vectors=[x.reshape(batch,n,heads,-1) for x in xs]
                                iq,ik=[torch.as_tensor(x,device='cuda') for x in labels]
                                flags=torch.ones((batch,heads),device='cuda',dtype=torch.bool)
                                tau=torch.full((heads,),.5,device='cuda')
                                fn=lambda:backward_hard_recompute(*vectors,iq,ik,flags,tau)
                                q=fn()
                                references['quadratic']={k:v.reshape(-1).cpu() for k,v in q.items()}
                                references['quadratic_time']=dict(full=measure(fn,args.samples),eager=eager(fn,args.samples))
                            if batch*heads==1 and n<=512:
                                from check_hard_prefill_training import dense_oracle
                                cpu=[x.float().cpu().double().numpy() for x in xs]
                                _,expected=dense_oracle(labels[0,0,0],labels[1,0,0],*cpu[:3],.5,cpu[3],direction=True)
                                references['fp64']={k:torch.as_tensor(v).reshape(-1) for k,v in zip(('sq_vec','sk_vec','v','rtau'),expected)}
                            if pattern=='repeat':
                                from bench_hard_backward_kernels import repeated_tau_oracle
                                references['dense_fp64_tau']=repeated_tau_oracle(xs,batch,n,heads,[n],.5).cpu()
                            reference_cache[key]=references
                        references=reference_cache[key]
                    baseline=None
                    for warps in args.warps:
                        for name,module in modules.items():
                            row=dict(case=case,r=r,dv=dv,c=c,compact=bool(compact),warps=warps,variant=name,cpu_plan=stats(times),statistics=host.statistics)
                            try:
                                t=time.perf_counter_ns(); plan=build_plan(module,device,host,compact,warps); torch.cuda.synchronize()
                                row['consumer_prepare_us']=(time.perf_counter_ns()-t)/1000
                                actual=plan.backward(*xs)
                                torch.cuda.synchronize()
                                if baseline is None: baseline={k:v.clone() for k,v in actual.items()}
                                row['error_vs_first']={k:float((actual[k].double()-baseline[k].double()).norm()/baseline[k].double().norm().clamp_min(1e-30)) for k in ('sq_vec','sk_vec','v','rtau')}
                                row['tau']=actual['rtau'].cpu().tolist()
                                if 'dense_fp64_tau' in references:
                                    ref=references['dense_fp64_tau']
                                    row['tau_fp64']=ref.tolist()
                                    row['tau_error_fp64']=float((actual['rtau'].cpu().double()-ref).norm()/ref.norm().clamp_min(1e-30))
                                    try:
                                        torch.testing.assert_close(actual['rtau'].cpu().double(), ref.double(), rtol=2e-4, atol=2e-6)
                                        row['tau_fp64_gate_failure']=None
                                    except AssertionError as exc:
                                        row['tau_fp64_gate_failure']=str(exc)
                                for kind in ('quadratic','fp64'):
                                    if kind in references:
                                        row['error_vs_'+kind]={k:float((actual[k].detach().cpu().double().reshape(-1)-v.double()).norm()/v.double().norm().clamp_min(1e-30)) for k,v in references[kind].items()}
                                if 'quadratic_time' in references: row['quadratic_time']=references['quadratic_time']
                                if 'fp64' in references:
                                    from check_hard_prefill_training import assert_gradient
                                    failures={}
                                    for k,ref in references['fp64'].items():
                                        try:
                                            assert_gradient(actual[k].cpu().reshape(-1),ref,k)
                                            if k=='rtau':
                                                torch.testing.assert_close(actual[k].cpu().double().reshape(-1),ref.double(),rtol=2e-4,atol=2e-6)
                                        except AssertionError as exc: failures[k]=str(exc)
                                    row['fp64_gate_failures']=failures
                                row['eager']=eager(lambda:plan.backward(*xs),args.samples)
                                # Completed external upload waits cannot enter a CUDA graph.
                                # Keep device storage alive, omit waits only after synchronize.
                                torch.cuda.synchronize()
                                if hasattr(plan,'device_plan'): del plan.device_plan
                                plan.bucket_ready=None
                                if compact: plan.ready=torch.cuda.Event()
                                row['full']=measure(lambda:plan.backward(*xs),args.samples,inner=1 if plan.matrix_snapshot_bytes>128*2**20 else 8)
                                # Record actual stage calls with persistent argument buffers.
                                records=[]; originals={}
                                for attr in ('_chunk_backward','_chunk_summaries','_summary_scan','_boundary_tau','_query_backward','_key_backward'):
                                    if hasattr(module,attr):
                                        originals[attr]=getattr(module,attr); setattr(module,attr,Recorder(originals[attr],records))
                                try: plan.backward(*xs)
                                finally:
                                    for attr,kernel in originals.items(): setattr(module,attr,kernel)

                                row['stages']=[]
                                for stage,grid,fn,kernel in records:
                                    setup=next((f for sn,sg,f,sk in records if sn=='_chunk_summaries'),None) if stage=='_summary_scan' else None
                                    row['stages'].append(dict(name=stage,grid=grid,time=measure(fn,args.samples,setup=setup),resources=inspect_kernel(kernel,args.output.parent/'cubins')))
                                row['snapshot_bytes']=plan.matrix_snapshot_bytes
                                row['peak_allocated']=torch.cuda.max_memory_allocated()
                                if records: del fn,kernel,setup
                                del records,plan,actual
                                print(case,dims,c,compact,warps,name,row['full'],row['error_vs_first']['rtau'],flush=True)
                            except torch.OutOfMemoryError as exc:
                                # Retained capture arguments may keep the failed arena alive.
                                # Save evidence and stop rather than contaminate later timings.
                                row['error']=repr(exc)
                                result['cases'].append(row)
                                args.output.write_text(json.dumps(result,indent=2)+'\n')
                                raise
                            except Exception as exc:
                                row['error']=repr(exc); print('ERROR',row,flush=True)
                            result['cases'].append(row)
                            args.output.write_text(json.dumps(result,indent=2)+'\n')
                    del xs,baseline
                del device,host
    result['gpu_after']=gpu_status()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('peak MiB',torch.cuda.max_memory_allocated()/2**20)

if __name__=='__main__': main()
