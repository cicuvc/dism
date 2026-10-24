"""CPU-only native math/packing regression without importing the CUDA package.

Pass this worktree's independently built _dism_prefill extension explicitly.
Uses ordinary numpy buffers, so it never initializes CUDA or pinned allocation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import numpy as np
import torch  # Load extension dependencies; CUDA remains disabled.
from hard_prefill_grad_reference import SymbolicPlanner, Program, assert_equivalent_programs

FIELDS = ('rows', 'prefixes', 'weights', 'dlog', 'lengths', 'resets', 'stream_heads')

def allocate(streams, chunks, width):
    sizes = dict(offsets=streams+1, rows=chunks*width, prefixes=chunks*width,
                 weights=chunks*width, dlog=chunks*width, lengths=chunks,
                 resets=chunks, stream_heads=streams)
    return {k: np.empty(n, np.int64 if k=='offsets' else
                       np.float32 if k in ('prefixes','weights','dlog') else np.int32)
            for k,n in sizes.items()}


def check_batch(module, pool, seed, width, compact):
    rng=np.random.default_rng(seed);b,h,n=2,3,65
    labels=rng.integers(-3,4,(2,b,h,n),dtype=np.int32)
    labels[:,0,0]=0;labels[0,1,1]=0;labels[1,1,1]=1
    docs=np.array([[0,0,1],[0,1,17],[0,17,n],[1,0,33],[1,33,n]],np.int64)
    hard=(rng.random((b,h))<.7).astype(np.uint8)
    if seed%5==0:hard.fill(0)
    tau=np.array([-.5,0.,2.]) if seed%2 else np.array([[-2.,1e-8,.5],[16.,0.,-1e-8]])
    native=pool.plan(labels,tau,docs,hard,width,compact);stats=dict(native.statistics())
    expected={k:[] for k in ('offsets',)+FIELDS};expected['offsets']=[0]
    buckets=[{k:[] for k in ('rows','prefixes','weights','dlog','lengths','stream_heads')} for _ in range(3)]
    for row,start,end in docs:
        for head in range(h):
            if not hard[row,head]:continue
            t=tau[head] if tau.ndim==1 else tau[row,head]
            p=module.plan_training(*[a.tolist() for a in labels[:,row,head,start:end]],t).chunks(width)
            for stream in range(len(p['offsets'])-1):
                first,last=p['offsets'][stream:stream+2];count=last-first
                bucket=compact and count==1
                length=int(p['lengths'][first]);bi=0 if length<=16 else 1 if length<=32 else 2
                dst=buckets[bi] if bucket else expected;w=(16,32,64)[bi] if bucket else width
                for chunk in range(first,last):
                    le=int(p['lengths'][chunk]);slots=slice(chunk*width,chunk*width+le)
                    local=p['rows'][slots].astype(np.int64);absolute=(np.where(local>=0,local,-local-1)+row*n+start)*h+head
                    dst['rows'].extend(np.where(local>=0,absolute,-absolute-1));dst['rows'].extend([(row*n+start)*h+head]*(w-le))
                    for field in ('prefixes','weights','dlog'):
                        dst[field].extend(p[field][slots] if field in p else np.zeros(le));dst[field].extend([0.]*(w-le))
                    dst['lengths'].append(le)
                    if not bucket:dst['resets'].append(p['resets'][chunk])
                dst['stream_heads'].append(head if tau.ndim==1 else row*h+head)
                if not bucket:expected['offsets'].append(expected['offsets'][-1]+count)
    actual=allocate(stats['streams'],stats['chunks'],width)
    if compact:
        bs=[]
        for w,blocks in zip((16,32,64),stats['bucket_blocks']):
            a=allocate(blocks,blocks,w);del a['offsets'],a['resets'];bs.append(a)
        native.pack_into_compact(bs,actual)
        for got,want in zip(bs,buckets):
            for k in got:assert got[k].tobytes()==np.asarray(want[k],dtype=got[k].dtype).tobytes(),k
    else:native.pack_into(actual)
    for k in actual:assert actual[k].tobytes()==np.asarray(expected[k],dtype=actual[k].dtype).tobytes(),k
    return 1


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--native',type=Path,required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    if not args.native.resolve().is_relative_to(root/'build'):ap.error('native must be in this worktree build directory')
    spec=importlib.util.spec_from_file_location('_dism_prefill',args.native);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cases=0
    for n in (1,65,513):
        for pattern in ('random','repeat','mismatch','periodic'):
            q,k=np.random.default_rng(318).integers(0,7,(2,n),dtype=np.int32)
            if pattern=='repeat':q.fill(0);k.fill(0)
            if pattern=='mismatch':q.fill(0);k.fill(1)
            if pattern=='periodic':q=np.arange(n)%3;k=(np.arange(n)+1)%3
            for tau in (-2.,-1e-8,0.,1e-8,.5,2.,16.):
                ref=SymbolicPlanner(q,k).compile(tau);got=module.plan_training(q.tolist(),k.tolist(),tau).arrays()
                assert_equivalent_programs(Program(**got),ref)
                cases+=1
    print(f'{cases} symbolic numerical oracle cases passed',flush=True)
    pool=module.TrainingPlanner(4)
    jobs=[(seed,width,compact) for seed in range(16) for width in (16,32,64) for compact in (False,True)]
    with ThreadPoolExecutor(6) as executor:
        counts=list(executor.map(lambda args:check_batch(module,pool,*args),jobs))
    assert not torch.cuda.is_initialized()
    print(f'{sum(counts)} concurrent batch packing cases byte-identical to independent document plans; CUDA uninitialized')

if __name__=='__main__':main()
