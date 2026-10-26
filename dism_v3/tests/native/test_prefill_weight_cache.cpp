#include "prefill.hpp"
#ifdef DISM_PREFILL_BASELINE_HEADER
#include DISM_PREFILL_BASELINE_HEADER
#endif
#include <cassert>
#include <iostream>
#include <random>

template<class A,class B>void equal_program(const A&a,const B&b) {
    assert(a.n==b.n&&a.states==b.states&&a.offsets==b.offsets&&a.rows==b.rows);
    assert(a.decay==b.decay&&a.weight==b.weight&&a.logden==b.logden);
    assert(a.dlogweight==b.dlogweight&&a.dlogden==b.dlogden);
}
int main() {
    std::mt19937 rng(813);dism_prefill::Builder reused;
#ifdef DISM_PREFILL_BASELINE_HEADER
    dism_prefill_original::Builder original;
#endif
    int cases=0;
    // Grow, shrink, reuse tau across modes, change it, then return to old tau.
    for(int n:{1,257,17,1025,65,257,3})for(double tau:{0.,.5,.5,-.2,0.,1.5,.5})
    for(bool gradients:{true,false,true,false})for(int pattern=0;pattern<3;++pattern) {
        std::vector<int>q(n),k(n),reset(n,0);
        for(int i=0;i<n;++i){q[i]=rng()%9;k[i]=rng()%9;}
        if(pattern==0){std::fill(q.begin(),q.end(),0);std::fill(k.begin(),k.end(),0);}
        if(pattern==1){std::fill(q.begin(),q.end(),0);for(int i=0;i<n;++i)k[i]=i+1;k[0]=0;}
        if(n>64&&pattern==2)reset[n/2]=1;
        reused.build(q.data(),k.data(),reset.data(),n,tau,false,gradients);
        auto got=reused.finish();
        auto fresh=dism_prefill::Builder(q,k,reset,tau,false,gradients).finish();
        equal_program(got,fresh);
#ifdef DISM_PREFILL_BASELINE_HEADER
        original.build(q.data(),k.data(),reset.data(),n,tau,false,gradients);
        equal_program(got,original.finish());
#endif
        ++cases;
    }
    std::cout<<"PASS weight cache: "<<cases<<" mode/tau/length/reset cases\n";
}
