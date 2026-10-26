#include "prefill.hpp"
#include "../../tools/planner_sparse_reads.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <random>

// Scalar payloads expose the exact event transform independently of packing.
using Result=std::array<std::vector<double>,5>;
Result evaluate(const dism_prefill::Program&p,const std::array<std::vector<double>,4>&x) {
    Result out;for(auto &v:out)v.assign(p.n,0.);
    for(size_t s=1;s<p.offsets.size();++s) {
        double state=0.,tangent=0.,adjoint=0.;
        for(auto e=p.offsets[s-1];e<p.offsets[s];++e) {
            int row=p.rows[e];double w=p.weight[e],dw=p.dlogweight[e];
            if(row>=0){double z=w*x[1][row]*x[2][row];state=p.decay[e]*state+z;tangent=p.decay[e]*tangent+dw*z;}
            else {int i=-row-1;out[0][i]+=w*x[0][i]*state;out[1][i]+=w*x[3][i]*state;out[4][i]+=w*x[0][i]*(dw*state+tangent)*x[3][i];}
        }
        for(auto e=p.offsets[s];e-->p.offsets[s-1];) {
            int row=p.rows[e];double w=p.weight[e];
            if(row<0){int i=-row-1;adjoint+=w*x[0][i]*x[3][i];}
            else {out[2][row]+=w*adjoint*x[2][row];out[3][row]+=w*adjoint*x[1][row];adjoint*=p.decay[e];}
        }
    }return out;
}
int main() {
    std::mt19937 rng(921);int cases=0,changed=0;
    for(int n:{1,15,16,17,63,64,65,129,257})for(int pattern=0;pattern<4;++pattern)for(double tau:{-.2,0.,.5}) {
        std::vector<int>q(n),k(n),reset(n,0);
        std::array<std::vector<double>,4>x;for(auto &v:x){v.resize(n);for(auto &a:v)a=(int(rng()%200)-100)/100.;}
        for(int i=0;i<n;++i){q[i]=pattern<2?0:rng()%7;k[i]=pattern<2?i+1:rng()%7;if(pattern==3)q[i]=k[i]=0;}
        if(pattern<2){k[0]=0;if(pattern==1)for(int i=n/4;i<n;i+=std::max(1,n/4))k[i]=0;}
        auto p=dism_prefill::Builder(q,k,reset,tau,false,true).finish();auto ref=evaluate(p,x);
        for(int width:{16,32,64})for(int limit:{0,1,4}) {
            auto got=split_sparse_reads(p,width,limit);auto values=evaluate(got,x);
            assert(got.logden==p.logden&&got.dlogden==p.dlogden);
            int queries=0,original=0;for(int row:p.rows)original+=row<0;for(int row:got.rows)queries+=row<0;assert(queries==original);
            for(int a=0;a<5;++a)for(int i=0;i<n;++i)assert(std::abs(values[a][i]-ref[a][i])<=2e-10*(1.+std::abs(ref[a][i])));
            ++cases;changed+=got.offsets!=p.offsets;
        }
    }
    assert(changed>0);std::cout<<"PASS sparse replay forward/backward/tau: "<<cases<<" cases, "<<changed<<" rewritten\n";
}
