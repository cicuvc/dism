"""Record/compare decode snapshots and outputs using an explicit old extension.

Artifacts belong under build/. The optional old extension must be a saved
binary from this worktree; it runs in a separate process before the new binary.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch


def legacy_extension(path):
    spec=importlib.util.spec_from_file_location('dism_decode_cuda',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    original=module.NativeCache
    class LegacyCache:
        def __init__(self,*args):self.inner=original(*args[:10])
        def __getattr__(self,name):return getattr(self.inner,name)
        def refresh_gpu(self,history,position,band=None):self.inner.refresh_gpu(history,position)
    module.NativeCache=LegacyCache


def fixture(index):
    configs=[(79,11,'random',True),(79,11,'repeat',False),(79,11,'mismatch',True),
             (257,128,'periodic',True),(257,128,'random',False),
             (1025,512,'repeat',False),(1025,512,'random',True),
             (2049,1024,'repeat',True),(2049,1024,'periodic',False)]
    n,interval,pattern,reset=configs[index];h=3;count=2*interval+3
    gen=torch.Generator().manual_seed(551+index)
    labels=torch.randint(0,8,(n+count,3,h),generator=gen,dtype=torch.int32)
    labels[:,2]=0
    if pattern=='repeat':labels[:,:2]=0
    if pattern=='mismatch':labels[:,0]=0;labels[:,1]=1
    if pattern=='periodic':
        labels[:,0]=torch.arange(n+count)[:,None]%7
        labels[:,1]=(torch.arange(n+count)[:,None]+torch.arange(h)[None,:])%7
    if reset:labels[17::23,2]=1
    keys=torch.randn(n+count,h,32,generator=gen).bfloat16()*.2
    queries=torch.randn(count,h,32,generator=gen).bfloat16()*.2
    values=torch.randn(n+count,h,64,generator=gen).bfloat16()*.2
    return n,interval,labels,keys,queries,values


@torch.no_grad()
def run(index,workers):
    from flash_dism.inference import GpuPlannerCache
    n,interval,labels,keys,queries,values=fixture(index)
    count=queries.shape[0];h=keys.shape[1]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    results=[];snapshots=[]
    def snapshot(cache,pos):
        saved={i:cache.snapshot[i].cpu().clone() for i in (2,3,4,5,6,8,9)}
        saved[7]=cache.snapshot[7][:pos].cpu().clone()
        snapshots.append(saved)
    with torch.cuda.stream(stream):
        cache=GpuPlannerCache(h,32,64,n+count,[0.,.35,1.5],device='cuda',
                              rebuild_interval=interval,rebuild_workers=workers)
        labels,keys,queries,values=[x.cuda() for x in (labels,keys,queries,values)]
        cache.prime(labels[:n].contiguous(),keys[:n].transpose(0,1).contiguous(),
                    values[:n].transpose(0,1).contiguous())
        snapshot(cache,n)
        static=[torch.empty_like(x) for x in (labels[0],keys[0],queries[0],values[0])]
        graph=None
        for i in range(count):
            if i and i%interval==0:
                cache.rebuild();graph=None;snapshot(cache,n+i)
            for dest,source in zip(static,(labels[n+i],keys[n+i],queries[i],values[n+i])):dest.copy_(source)
            if graph is None:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):out=cache.step(*static)
            graph.replay();results.append(out.clone())
        assert cache.check_status()==n+count
        output=torch.stack(results).cpu()
        assert output.isfinite().all()
        cache.rebuild();snapshot(cache,n+count)
    stream.synchronize()
    return dict(output=output,snapshots=snapshots)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-extension',type=Path)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--workers',type=int,nargs='+',default=[1,4,8])
    p.add_argument('--cases',type=int,nargs='+',default=list(range(9)))
    args=p.parse_args();torch.set_num_threads(1)
    if args.legacy_extension:
        if args.reference.exists():p.error('fresh reference path required')
        legacy_extension(args.legacy_extension)
        data={i:run(i,1) for i in args.cases}
        args.reference.parent.mkdir(parents=True,exist_ok=True);torch.save(data,args.reference)
        print('saved',len(data),'legacy cases',flush=True)
    else:
        reference=torch.load(args.reference,weights_only=True)
        for workers in args.workers:
            for i in args.cases:
                actual=run(i,workers);expected=reference[i]
                torch.testing.assert_close(actual['output'],expected['output'],rtol=0,atol=0)
                for a,b in zip(actual['snapshots'],expected['snapshots']):
                    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                print(json.dumps(dict(case=i,workers=workers,bitwise_equal=True)),flush=True)


if __name__=='__main__':main()
