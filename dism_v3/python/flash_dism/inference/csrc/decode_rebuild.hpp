#pragma once
// CPU-only reusable per-head metadata workspace; CUDA launch stays on owner thread.
#include "planner.hpp"
#include "planning_workers.hpp"

namespace dism_decode {
struct RebuildWorkspace {
    std::vector<int> inverse, end, next_valid;
    std::vector<double> rho, next_rho;
    std::vector<std::pair<int,int>> sorted;
    int levels=0, edges=0;

    void prepare(const Planner &p) {
        int n=int(p.nodes.size());
        inverse.resize(n); end.resize(n); next_valid.resize(n+1);
        for(int i=0;i<n;++i) {inverse[p.order[i]]=i;end[p.order[i]]=i+1;}
        for(int i=n-1;i>0;--i) {
            int u=p.order[i];end[p.nodes[u].link]=std::max(end[p.nodes[u].link],end[u]);
        }
        levels=0;
        if(!p.samples.empty()) while((int64_t(1)<<levels)<n) ++levels;
        edges=0;for(const auto &node:p.nodes) edges+=node.next.size();
    }

    void pack_summary(const Planner &p,int *data,float *coefficient_data) {
        int n=p.nodes.size();
        for(int i=0;i<n;++i) {
            data[i]=p.nodes[i].link;data[n+i]=p.nodes[i].length;
            data[2*n+i]=p.nodes[i].position;data[3*n+i]=p.order[i];
            data[4*n+i]=p.mat_index[i];
            data[5*n+i]=p.sample_index[i]<0?-1:int(p.materialized.size())+p.sample_index[i];
            data[6*n+i]=inverse[i];data[7*n+i]=end[i];
            coefficient_data[i]=p.edge_weight[i];
        }
        for(int level=0;level<levels;++level) for(int i=0;i<n;++i) {
            int a=level?data[(7+level)*n+i]:p.nodes[i].link;
            data[(8+level)*n+i]=level && a>=0?data[(7+level)*n+a]:a;
        }
        rho=p.edge_decay;next_rho.resize(n);
        for(int level=0;level<levels;++level) {
            for(int i=0;i<n;++i) coefficient_data[(level+1)*n+i]=rho[i];
            for(int i=0;i<n;++i) {
                int a=data[(8+level)*n+i];next_rho[i]=a<0?0.:rho[i]*rho[a];
            }
            rho.swap(next_rho);
        }
    }

    void pack_snapshot(const Planner &p,int pitch,int edge_pitch,int capacity,int offset,int matrices,
                       int *t,int *e,int *s,int *band,float *counts,float *tau) {
        int n=p.nodes.size(),position=p.keys.size();
        std::fill(t,t+13*int64_t(pitch),-1);std::fill(e,e+2*int64_t(edge_pitch),0);
        std::fill(s,s+8,0);std::fill(band,band+2*int64_t(p.interval),0);
        s[0]=position;s[1]=p.state;s[2]=p.match;s[3]=s[4]=-1;
        int cursor=0;
        for(int u=0;u<n;++u) {
            const auto &node=p.nodes[u];
            t[u]=node.link;t[pitch+u]=node.length;t[2*pitch+u]=node.position;
            t[3*pitch+u]=cursor;
            sorted.assign(node.next.begin(),node.next.end());std::sort(sorted.begin(),sorted.end());
            for(auto pair:sorted) {e[cursor]=pair.first;e[edge_pitch+cursor]=pair.second;++cursor;}
            t[4*pitch+u]=cursor;t[5*pitch+u]=inverse[u];t[6*pitch+u]=end[u];
            t[7*pitch+u]=p.mat_index[u]<0?-1:offset+p.mat_index[u];
            t[8*pitch+u]=p.nearest[u];
            t[9*pitch+u]=p.sample_index[u]<0?-1:offset+matrices+p.sample_index[u];
            t[10*pitch+u]=p.order[u];
        }
        next_valid[n]=n;
        for(int i=n-1;i>=0;--i) {
            int u=p.order[i],matrix=t[7*pitch+u];
            int id=matrix>=0?capacity+matrix:p.nodes[u].position;
            next_valid[i]=id>=0?i:next_valid[i+1];
            t[11*pitch+2*i]=id;t[11*pitch+2*i+1]=next_valid[matrix>=0?end[u]:i+1];
        }
        for(int i=0;i<int(p.band.size());++i) {
            int absolute=position-int(p.band.size())+i;
            band[(position&1)*p.interval+absolute%p.interval]=p.band[i];
        }
        for(int i=0;i<int(p.materialized.size());++i) counts[offset+i]=p.subtree_count[p.materialized[i]];
        for(int i=0;i<int(p.samples.size());++i) counts[offset+matrices+i]=p.prefix_weight[p.samples[i]];
        *tau=p.tau;
    }
};
} // namespace dism_decode
