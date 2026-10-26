"""Exact final-metadata and parallel packing checks for the direct planner."""
import argparse
from pathlib import Path

import numpy as np

from check_decoupled_readout import load_native


def identical(old,new):
    for a,b in zip([*old['buckets'],old['multi'],old['split']],
                   [*new['buckets'],new['multi'],new['split']]):
        assert a.keys()<=b.keys()
        for name in a:
            if isinstance(a[name],np.ndarray):
                assert a[name].dtype==b[name].dtype and a[name].shape==b[name].shape,name
                assert a[name].tobytes()==b[name].tobytes(),name
            else:assert a[name]==b[name],name
    for name,x in old['statistics'].items():
        if not name.endswith('_ms'):assert x==new['statistics'][name],name
    multi=new['multi']
    if 'stream_ids' in multi:
        offsets=multi['offsets'];spans=np.diff(offsets);longs=np.flatnonzero(spans>64)
        chunks=[np.arange(offsets[i],offsets[i+1],dtype=np.int64) for i in longs]
        expected=dict(stream_ids=np.arange(len(spans)),chunk_ids=np.arange(offsets[-1]),local_offsets=offsets[:-1],
            short_stream_ids=np.flatnonzero(spans<=64),long_stream_ids=longs,
            long_chunk_ids=np.concatenate(chunks) if chunks else np.zeros(0,np.int64),
            long_local_offsets=np.r_[0,np.cumsum([len(x) for x in chunks])[:-1]] if chunks else np.zeros(0,np.int64))
        for name,value in expected.items():np.testing.assert_array_equal(multi[name],value)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native',type=Path,required=True);args=ap.parse_args()
    module=load_native(args.native);rng=np.random.default_rng(719);cases=0
    for workers in (1,8):
        planner=module.Planner(workers)
        for n in (1,65,129,257):
            for pattern in ('random','repeat','empty'):
                labels=rng.integers(0,7,(2,2,3,n),dtype=np.int32)
                if pattern=='repeat':labels.fill(0)
                if pattern=='empty':labels[1]+=100
                tau=np.array([-.2,0.,.5])
                for k in (1,2,4):
                    candidates=np.stack([(labels[0]+i)%7 for i in range(k)],axis=-1)
                    for gradients in (False,True):
                        for threshold in (-1,0,1,64):
                            params=(labels,tau,candidates,k,gradients,64,64,threshold)
                            old=planner.prepare(*params);new=planner.prepare_direct(*params);identical(old,new);cases+=1
        print('PASS workers',workers,'cumulative cases',cases,flush=True)
    print('PASS',cases,'parallel metadata cases',flush=True)
    params=(np.zeros((2,1,1,1),np.int32),np.zeros(1,np.float64),np.zeros(0,np.int32),1,False,64,64,64)
    def readonly(size):
        array=np.empty(size,np.uint8);array.flags.writeable=False;return array
    for allocator in (lambda size:np.empty(size+1,np.uint8),lambda size:np.empty(size,np.float32),readonly):
        try:planner.prepare_direct(*params,allocator=allocator)
        except ValueError:pass
        else:raise AssertionError('invalid arena allocator accepted')
    packed=planner.prepare_direct(*params,allocator=lambda size:np.empty(size,np.uint8))
    for group in [*packed['buckets'],packed['multi'],packed['split']]:
        for array in group.values():
            if isinstance(array,np.ndarray) and array.size:assert np.shares_memory(array,packed['_storage'])
    print('PASS allocator validation and shared arena ownership',flush=True)


if __name__=='__main__':main()
