// -pthread; optionally -fsanitize=address,undefined or -fsanitize=thread.
// An old worktree snapshot can be supplied as DISM_BASELINE_DECODE_HEADER.
#ifdef DISM_BASELINE_DECODE_HEADER
#define dism_decode baseline_decode
#include DISM_BASELINE_DECODE_HEADER
#undef dism_decode
#endif
#include "decode_rebuild.hpp"
#ifndef DISM_BASELINE_DECODE_HEADER
namespace baseline_decode=dism_decode;
#endif
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>

template<class T> void equal(const std::vector<T> &a,const std::vector<T> &b) {
    if(a.size()!=b.size() || (!a.empty() && std::memcmp(a.data(),b.data(),a.size()*sizeof(T))))
        throw std::runtime_error("decode rebuild byte mismatch");
}
template<class A,class B> void compare(const A &a,const B &b) {
    assert(a.nodes.size()==b.nodes.size());
    for(size_t i=0;i<a.nodes.size();++i) {
        assert(a.nodes[i].link==b.nodes[i].link && a.nodes[i].length==b.nodes[i].length &&
               a.nodes[i].position==b.nodes[i].position && a.nodes[i].next==b.nodes[i].next);
    }
    equal(a.order,b.order);equal(a.materialized,b.materialized);equal(a.samples,b.samples);
    equal(a.mat_index,b.mat_index);equal(a.sample_index,b.sample_index);equal(a.nearest,b.nearest);
    equal(a.band,b.band);equal(a.subtree_count,b.subtree_count);equal(a.prefix_weight,b.prefix_weight);
    equal(a.edge_weight,b.edge_weight);equal(a.edge_decay,b.edge_decay);
    assert(a.state==b.state && a.match==b.match);
}

void trial(int seed) {
    std::mt19937 rng(seed);
    int interval=std::vector<int>{11,64,128}[seed%3];
    double tau=std::vector<double>{0.,.35,1.5}[seed%3];
    dism_decode::Planner reuse(16,interval,5,32,tau);
    dism_decode::RebuildWorkspace workspace;
    std::vector<int> k(1+rng()%257),q(k.size()),reset(k.size());
    for(size_t i=0;i<k.size();++i) {
        k[i]=seed%4==0?0:int(rng()%8);q[i]=seed%4==0?0:int(rng()%8);
        reset[i]=seed%4==1 && rng()%11==0;
    }
    reuse.initialize(k,q,reset);
    for(int round=0;round<4;++round) {
        reuse.rebuild();
        baseline_decode::Planner fresh(16,interval,5,32,tau);
        fresh.initialize(reuse.keys,reuse.queries,reuse.resets);fresh.rebuild();
        compare(reuse,fresh);
        workspace.prepare(reuse);
        int n=reuse.nodes.size(),pitch=(n+1)&~1,ep=std::max(1,workspace.edges);
        int matrices=reuse.materialized.size()+reuse.samples.size();
        std::vector<int> topo((8+workspace.levels)*n),snapshot(13*pitch),edges(2*ep),state(8),band(2*interval);
        std::vector<float> coefficients((workspace.levels+1)*n),counts(7+matrices,-666.f);
        float scalar_tau=0;
        workspace.pack_summary(reuse,topo.data(),coefficients.data());
        workspace.pack_snapshot(reuse,pitch,ep,4096,7,reuse.materialized.size(),snapshot.data(),
            edges.data(),state.data(),band.data(),counts.data(),&scalar_tau);
        for(int i=0;i<7;++i) assert(counts[i]==-666.f);
        for(int i=0;i<n;++i) assert(topo[6*n+i]==workspace.inverse[i] && topo[7*n+i]==workspace.end[i]);
        assert(state[0]==int(reuse.keys.size()) && scalar_tau==float(tau));
        for(int i=0;i<interval;++i) {
            int key=seed%4==0?0:int(rng()%8),query=seed%4==0?0:int(rng()%8);
            bool rst=seed%4==1 && rng()%13==0;
            auto a=reuse.append(key,query,rst);
            auto b=fresh.append(key,query,rst);
            assert(a.size()==b.size());
            for(size_t j=0;j<a.size();++j)
                assert(a[j].kind==b[j].kind && a[j].index==b[j].index && a[j].coefficient==b[j].coefficient);
            equal(reuse.band,fresh.band);
        }
    }
    // Explicit complete-history replacement resets incremental state.
    reuse.keys.clear();reuse.initialize(k,q,reset);reuse.rebuild();
    baseline_decode::Planner fresh(16,interval,5,32,tau);fresh.initialize(k,q,reset);fresh.rebuild();
    compare(reuse,fresh);
}

int main() {
    dism_prefill::PlanningWorkers workers(4);
    std::thread a([&]{workers.run(16,[](size_t i){trial(i);});});
    std::thread b([&]{workers.run(16,[](size_t i){trial(i+16);});});
    a.join();b.join();
    std::cout<<"32 concurrent incremental planners: snapshots/tasks identical across 4 rebuilds and replacement\n";
}
