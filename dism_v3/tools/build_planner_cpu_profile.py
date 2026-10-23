"""Build isolated current-source CPU profiler targets, never install extensions.

The timed binary includes unchanged production headers. The phase binary copies
headers into ignored build output and inserts coarse timers/allocation counters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SUPPORT = r'''#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
namespace prof {
constexpr int K=13;
inline std::atomic<uint64_t> total_ns[K]{}, total_alloc[K]{}, total_bytes[K]{}, total_free[K]{};
inline thread_local uint64_t ns[K]{},alloc[K]{},bytes[K]{},frees[K]{};
inline thread_local bool active=false;
inline thread_local int phase=10;
inline thread_local std::chrono::steady_clock::time_point last;
inline void begin(int p) {
    for(int i=0;i<K;++i)ns[i]=alloc[i]=bytes[i]=frees[i]=0;
    phase=p;active=true;last=std::chrono::steady_clock::now();
}
inline void tick(int p) {
    auto now=std::chrono::steady_clock::now();
    if(active) ns[phase]+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-last).count();
    last=now;phase=p;
}
inline void end() {
    tick(10);active=false;
    for(int i=0;i<K;++i) {
        total_ns[i].fetch_add(ns[i],std::memory_order_relaxed);
        total_alloc[i].fetch_add(alloc[i],std::memory_order_relaxed);
        total_bytes[i].fetch_add(bytes[i],std::memory_order_relaxed);
        total_free[i].fetch_add(frees[i],std::memory_order_relaxed);
    }
}
inline void allocation(size_t n) {if(active){++alloc[phase];bytes[phase]+=n;}}
inline void release() {if(active)++frees[phase];}
inline void report(size_t iterations) {
    const char* names[]={"tau_precompute","SAM","adjacency","query_match","decompose_emit","scalar_denominator",
                        "coefficients","builder_free","boundary_count","compact_pack","other","program_output_free","output_allocate"};
    std::cout<<"phase,worker_ms_per_batch,allocations_per_batch,allocated_bytes_per_batch,frees_per_batch\n";
    for(int i=0;i<K;++i)std::cout<<names[i]<<","<<double(total_ns[i])/iterations/1e6<<","<<double(total_alloc[i])/iterations
      <<","<<double(total_bytes[i])/iterations<<","<<double(total_free[i])/iterations<<"\n";
}
}
'''

def once(source,old,new):
    if source.count(old)!=1:raise ValueError(f'Expected one source anchor: {old[:90]}')
    return source.replace(old,new,1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--geometry-json',type=Path,help='CPU diagnosis JSON supplying geometry/documents for iid inputs')
    args=ap.parse_args();root=Path(__file__).resolve().parents[2]
    if not args.output.resolve().is_relative_to(root/'dism_v3/build'):
        ap.error('output must be under dism_v3/build')
    args.output.mkdir(exist_ok=True,parents=True)
    src=root/'dism_v3/python/flash_dism/inference/csrc'
    profile=args.output/'instrumented';profile.mkdir(exist_ok=True)
    (profile/'profile_support.hpp').write_text(SUPPORT)
    for name in ('planner.hpp','prefill_matcher.hpp','planning_workers.hpp'):
        (profile/name).write_text((src/name).read_text())
    s=(src/'prefill.hpp').read_text()
    s='#include "profile_support.hpp"\n'+s
    for old,new in (
        ('        if (gradients) {\n            if (cached_tau!=tau)', '        prof::tick(0);\n        if (gradients) {\n            if (cached_tau!=tau)'),
        ('        // Build the offline SAM','        prof::tick(1);\n        // Build the offline SAM'),
        ('        int count=int(sam.nodes.size());','        prof::tick(2);\n        int count=int(sam.nodes.size());'),
        ('        int state=0, length=0;','        prof::tick(3);\n        int state=0, length=0;'),
        ('        ids.resize(n);','        prof::tick(4);\n        ids.resize(n);'),
        ('        decompose(0,ids,{0,n},ids,{0,n},0);','        decompose(0,ids,{0,n},ids,{0,n},0);\n        prof::tick(10);'),
        ('    Program finish() const {','    Program finish() const {\n        prof::tick(5);'),
        ('        // Compile all floating point transcendental work','        prof::tick(6);\n        // Compile all floating point transcendental work'),
        ('        return out;\n    }\n};','        prof::tick(7);\n        return out;\n    }\n};'),
    ):s=once(s,old,new)
    (profile/'prefill.hpp').write_text(s)
    s=(src/'prefill_training.hpp').read_text()
    s=once(s,'            const auto &job=jobs[i];','            prof::begin(10);\n            const auto &job=jobs[i];')
    s=once(s,'            event_counts[i]=p.rows.size();','            prof::tick(8);\n            event_counts[i]=p.rows.size();')
    s=once(s,'            programs[i]=std::move(p);','            programs[i]=std::move(p);\n            prof::end();')
    s=once(s,'            const Program &p=programs[i]; const auto &job=jobs[i];','            prof::begin(9);\n            const Program &p=programs[i]; const auto &job=jobs[i];')
    s=once(s,'            if (msl!=job_ms) throw std::logic_error("multi-chunk stream count mismatch");','            if (msl!=job_ms) throw std::logic_error("multi-chunk stream count mismatch");\n            prof::end();')
    (profile/'prefill_training.hpp').write_text(s)
    common=['/usr/bin/g++','-O3','-g','-fno-omit-frame-pointer','-std=c++17','-pthread']
    target=Path(__file__).with_name('profile_planner_native.cpp')
    commands=[]
    for include,name,extra in ((src,'native',[]),(profile,'phases',['-DPHASE_PROFILE'])):
        cmd=common+extra+['-I',str(include),str(target),'-o',str(args.output/name)]
        subprocess.run(cmd,check=True);commands.append(cmd)
    counter=Path(__file__).with_name('count_planner_events.cpp')
    cmd=['/usr/bin/g++','-O2','-g','-std=c++17',str(counter),'-o',str(args.output/'count_events')]
    subprocess.run(cmd,check=True);commands.append(cmd)
    if args.geometry_json:
        import numpy as np
        geometry=json.loads(args.geometry_json.read_text())
        b,h,n=geometry['geometry'];docs=np.asarray(geometry['documents'],dtype=np.int64)
        for symbols in (512,32,8):
            labels=np.random.default_rng(730).integers(0,symbols,(2,b,h,n),dtype=np.int32)
            hard=(np.random.default_rng(19231).random((b,h))<.95).astype(np.uint8)
            blob=b''.join(a.tobytes() for a in (np.array([b,h,n,len(docs)],dtype=np.int32),labels,hard,np.full(h,.5),docs))
            path=args.output/f'iid{symbols}.bin'
            if path.exists() and path.read_bytes()!=blob:
                raise ValueError(f'Refusing to replace different workload: {path}')
            path.write_bytes(blob)
    sources=[target,counter,Path(__file__),*(src/n for n in ('planner.hpp','prefill_matcher.hpp','planning_workers.hpp','prefill.hpp','prefill_training.hpp'))]
    (args.output/'build_manifest.json').write_text(json.dumps(dict(commands=commands,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')

if __name__=='__main__':main()
