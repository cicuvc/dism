"""Validate and benchmark isolated sparse-readout plans on captured real labels."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch

from flash_dism.inference.training_planner import NativeTrainingPlanner
from flash_dism.inference.prefill_native import NativePrefillPlanner
from flash_dism.kernels.hard_backward_fused import CompactFusedHardPrefillBackward
from hard_prefill_grad_reference import Program, dense_oracle
from check_hard_prefill_training import assert_gradient, metrics


class Adapter:
    def __init__(self, native, limit):
        self.native, self.limit = native, limit

    def plan(self, *args):
        return self.native.plan(*args, self.limit)

    def plan_prefill(self, *args):
        return self.native.plan_prefill(*args, self.limit)


def planner(native, limit, prefill=False):
    cls = NativePrefillPlanner if prefill else NativeTrainingPlanner
    obj = cls.__new__(cls)
    obj.native, obj.workers = Adapter(native, limit), 8
    return obj


def load_input(path):
    with path.open('rb') as f:
        b,h,n,nd = np.fromfile(f,np.int32,4)
        labels = np.fromfile(f,np.int32,2*b*h*n).reshape(2,b,h,n)
        hard = np.fromfile(f,np.uint8,b*h).reshape(b,h)
        tau = np.fromfile(f,np.float64,h)
        docs = np.fromfile(f,np.int64,nd*3).reshape(nd,3)
    return labels,tau,hard,docs


def scalar_checks(module):
    rng = np.random.default_rng(123)
    count = changed = 0
    for n in (17,65,129):
        for pattern in ('single','four','random','repeat'):
            q = rng.integers(0,8,n,dtype=np.int32)
            k = rng.integers(0,8,n,dtype=np.int32)
            if pattern in ('single','four'):
                q.fill(0);k=np.arange(n,dtype=np.int32)+1
                k[np.linspace(0,n//2,1 if pattern=='single' else 4,dtype=int)]=0
            elif pattern=='repeat':
                q.fill(0);k.fill(0)
            xs = [rng.normal(0,.2,(n,d)) for d in (4,4,8,8)]
            for tau in (-.2,0.,.5):
                p=module.plan_training(q.tolist(),k.tolist(),tau)
                baseline=Program(**p.arrays()).evaluate(*xs)
                for width in (16,32,64):
                    for limit in (1,4):
                        after=module.split_sparse(p,width,limit)
                        arrays=after.arrays()
                        result=Program(**arrays).evaluate(*xs)
                        np.testing.assert_array_equal(arrays['logden'],p.arrays()['logden'])
                        np.testing.assert_array_equal(arrays['dlogden'],p.arrays()['dlogden'])
                        for a,b in zip((result[0],*result[1],result[2]),(baseline[0],*baseline[1],baseline[2])):
                            np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-11)
                        count+=1;changed+=after.statistics()['streams']!=p.statistics()['streams']
    assert changed>0
    print('scalar PASS',count,'changed',changed,flush=True)
    return dict(cases=count,changed=changed)


def gpu_checks(native, consumer_class=CompactFusedHardPrefillBackward, resource_dir=None):
    results=[]
    for count in (1,4):
        n=257;labels=np.zeros((2,1,1,n),np.int32)
        labels[1,0,0]=np.arange(n)+1
        labels[1,0,0,np.linspace(0,n//2,count,dtype=int)]=0
        for precision in ('bf16','tf32x3'):
            dtype=torch.bfloat16 if precision=='bf16' else torch.float32
            torch.manual_seed(921)
            xs=[(torch.randn((n,d),device='cuda')*.2).to(dtype) for d in (16,16,32,32)]
            expected,grads=dense_oracle(*labels[:,0,0],*[x.cpu().double().numpy() for x in xs[:3]],.5,xs[3].cpu().double().numpy(),direction=True)
            for limit in (0,1,4,24):
                host=planner(native,limit).prepare(labels,np.array([.5]),chunk_size=64,compact=True)
                consumer=consumer_class.from_compact(host.upload(),mma_precision=precision)
                actual=consumer.backward(*xs)
                error={}
                resources=[]
                for k in consumer.last_kernels:
                    ptx=k.asm['ptx'];key=hashlib.sha256(ptx.encode()).hexdigest()[:12]
                    resources.append(dict(name=k.name,registers=k.n_regs,spills=k.n_spills,
                                          ptx_call_instructions=len(re.findall(r'^\s*(?:@\S+\s+)?call(?:\.\w+)*\s',ptx,re.M)),key=key))
                    if resource_dir is not None:
                        resource_dir.mkdir(exist_ok=True)
                        (resource_dir/f'{key}.ptx').write_text(ptx)
                        (resource_dir/f'{key}.cubin').write_bytes(k.asm['cubin'])
                for name,ref in zip(('sq_vec','sk_vec','v','rtau'),grads):
                    ref=torch.as_tensor(ref).reshape(actual[name].shape)
                    assert_gradient(actual[name].cpu(),ref,name)
                    error[name]=metrics(actual[name].cpu(),ref)
                torch.testing.assert_close(actual['rtau'].cpu().double(),torch.as_tensor(grads[-1]).reshape(1),rtol=2e-4,atol=2e-6)
                hostf=planner(native,limit,True).plan(labels,np.array([.5]),chunk_size=16)
                forward=hostf.upload(device='cuda',mma_precision=precision)
                out=forward.execute_dispatch(*xs[:3])
                assert_gradient(out.cpu(),torch.as_tensor(expected),'out')
                error['out']=metrics(out.cpu(),expected)
                results.append(dict(keys=count,precision=precision,limit=limit,error=error,resources=resources))
    print('GPU oracle PASS',len(results),flush=True)
    return results


def measure(functions):
    graphs=[];outputs=[]
    for fn in functions:
        for _ in range(3):fn()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):result=fn()
        graphs.append(graph);outputs.append(result)
    for g in graphs:
        for _ in range(5):g.replay()
    torch.cuda.synchronize()
    times=[[] for _ in graphs];rng=random.Random(123)
    for _ in range(21):
        order=list(range(len(graphs)));rng.shuffle(order)
        for i in order:
            a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record();graphs[i].replay();b.record();b.synchronize()
            times[i].append(a.elapsed_time(b))
    return [dict(median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95))) for t in times]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--contexts',type=int,nargs='+',default=[2048,8192])
    ap.add_argument('--layers',type=int,nargs='+',default=list(range(6)))
    ap.add_argument('--checks-only',action='store_true')
    ap.add_argument('--kernel',type=Path)
    ap.add_argument('--limits',type=int,nargs='+')
    ap.add_argument('--resources',type=Path)
    args=ap.parse_args();torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('_dism_sparse_prefill',args.experiment/'_dism_sparse_prefill.so')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native=module.TrainingPlanner(8)
    optimized=CompactFusedHardPrefillBackward
    if args.kernel:
        name='flash_dism.kernels._sparse_readout_experiment'
        spec=importlib.util.spec_from_file_location(name,args.kernel)
        kernel=importlib.util.module_from_spec(spec);sys.modules[name]=kernel;spec.loader.exec_module(kernel)
        optimized=kernel.CompactFusedHardPrefillBackward
    result=dict(scalar=scalar_checks(module),gpu_checks=gpu_checks(native,optimized,args.resources),cases=[])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if args.checks_only:return
    for n in args.contexts:
        for layer in args.layers:
            labels,tau,hard,docs=load_input(args.inputs/f'n{n}_layer{layer}.bin')
            total=int(np.prod(labels.shape[1:]));torch.manual_seed(2026)
            xs=[(torch.randn((total,d),device='cuda')*.2).bfloat16() for d in (32,32,64,64)]
            for stage in ('backward','prefill'):
                consumers=[];functions=[];records=[];hosts=[]
                variants=[(0,CompactFusedHardPrefillBackward),(1,optimized),(4,optimized)]
                if args.kernel and stage=='backward':variants.insert(1,(0,optimized))
                if args.limits is not None:
                    variants=[(limit,optimized) for limit in args.limits]
                for limit,consumer_class in variants:
                    obj=planner(native,limit,stage=='prefill')
                    prepare=(lambda:obj.prepare(labels,tau,hard=hard,documents=docs,chunk_size=64,compact=True)) if stage=='backward' else (lambda:obj.plan(labels,tau,chunk_size=16))
                    for _ in range(2):host=prepare()
                    samples=[]
                    for _ in range(7):
                        start=time.perf_counter();host=prepare();samples.append((time.perf_counter()-start)*1000)
                    hosts.append(host)
                    if stage=='backward':
                        consumer=consumer_class.from_compact(host.upload(),mma_precision='bf16')
                        fn=lambda consumer=consumer:consumer.backward(*xs)
                    else:
                        consumer=host.upload(device='cuda',mma_precision='bf16')
                        fn=lambda consumer=consumer:consumer.execute_dispatch(*xs[:3])
                    consumers.append(consumer);functions.append(fn)
                    records.append(dict(limit=limit,optimized_kernel=consumer_class is not CompactFusedHardPrefillBackward,cpu_ms=float(np.median(samples)),stats=host.statistics))
                if args.limits and 0 in args.limits:
                    a=hosts[args.limits.index(0)]
                    for limit,b in zip(args.limits,hosts):
                        if limit & 7 == 0:
                            assert torch.equal(a.storage,b.storage),('coefficient cache changed packed bytes',stage,limit)
                # Same real symbols, independent random vector payloads, same dtype.
                baseline=functions[0]();errors=[]
                for fn in functions[1:]:
                    actual=fn();err={}
                    for name in (('sq_vec','sk_vec','v','rtau') if stage=='backward' else ('out',)):
                        a=actual[name] if stage=='backward' else actual;b=baseline[name] if stage=='backward' else baseline
                        assert_gradient(a,b,name);err[name]=metrics(a.cpu(),b.cpu())
                    errors.append(err)
                del baseline,actual
                # Prepared-plan GPU microbenchmark, after all uploads finished.
                # Match bench_planner_event: eager upload events are not part of
                # the dedicated planner_graph protocol and cannot be imported
                # into a standalone capture on its private stream.
                torch.cuda.synchronize()
                if stage=='backward':
                    for consumer in consumers:
                        consumer.bucket_ready=None
                        consumer.ready=torch.cuda.Event()
                times=measure(functions)
                for record,timing in zip(records,times):record.update(gpu=timing)
                result['cases'].append(dict(context=n,layer=layer,stage=stage,variants=records,errors=errors))
                args.output.write_text(json.dumps(result,indent=2)+'\n')
                print(n,layer,stage,[(x['limit'],round(x['cpu_ms'],2),round(x['gpu']['median_ms'],3)) for x in records],flush=True)
                del functions,consumers,hosts,consumer,fn,host,obj
                gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
