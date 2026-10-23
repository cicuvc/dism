// Independent long-double oracle for repeated labels and query reset boundaries.
// Compile with -I pointing at the isolated candidate (or baseline) headers.
#include "prefill.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace dism_prefill;
int main() {
    static_assert(sizeof(long double)>sizeof(double), "extended precision required");
    Builder builder; int cases=0, failures=0; double max_scaled=0;
    auto check=[&](double actual,long double expected,const char* field,int n,double tau) {
        long double error=std::abs(static_cast<long double>(actual)-expected);
        long double tolerance=2e-10L+2e-9L*std::abs(expected);
        max_scaled=std::max(max_scaled,double(error/tolerance));
        if (!std::isfinite(actual) || error>tolerance) {
            if(failures++<8)std::cerr<<field<<" N="<<n<<" tau="<<tau<<" error="<<double(error)<<" tolerance="<<double(tolerance)<<"\n";
        }
    };
    for(int n:{1,65,513,1542,4096})for(double tau:{-16.,-2.,-1e-8,0.,1e-8,.5,2.,16.})for(bool resets:{false,true}) {
        Ids q(n,0),k(n,0),reset(n,0);
        if(resets)for(int i=0;i<n;++i)reset[i]=(i%37==0 || i%113==0);
        builder.build(q.data(),k.data(),reset.data(),n,tau,false,true);
        auto p=builder.finish();
        std::vector<long double> expected_dlogden(n),local_derivative(n);
        int length=0;
        for(int i=0;i<n;++i) {
            length=reset[i]?1:length+1;
            // For repeated labels, exactly i-r+2 causal keys match >=r tokens.
            // Summing over match length r is independent of the event planner.
            long double pivot=std::max(0.L,length*static_cast<long double>(tau));
            long double z=std::exp(-pivot),dz=0,w=0,dw=0;
            for(int r=1;r<=length;++r) {
                long double e=std::exp(r*static_cast<long double>(tau)-pivot);
                z+=(i-r+2)*e;dz+=(i-r+2)*r*e;w+=e;dw+=r*e;
            }
            expected_dlogden[i]=dz/z;local_derivative[i]=dw/w;
            check(p.logden[i],pivot+std::log(z),"logden",n,tau);
            check(p.dlogden[i],expected_dlogden[i],"dlogden",n,tau);
        }
        // At tau=16, different positive integer match lengths have derivative
        // gaps near one. A near-zero query dlog must therefore use the full
        // matched length, exposing cancellation that relative checks miss.
        if(tau==16.)for(size_t e=0;e<p.rows.size();++e)
            if(p.rows[e]<0 && std::abs(p.dlogweight[e])<1e-5) {
                int i=-p.rows[e]-1;
                check(p.dlogweight[e],local_derivative[i]-expected_dlogden[i],"near_zero_dlogweight",n,tau);
            }
        ++cases;
    }
    std::cout<<"cases="<<cases<<" failures="<<failures<<" max_tolerance_fraction="<<max_scaled<<"\n";
    return failures?1:0;
}
