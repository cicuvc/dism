// Independent dense long-double match-weight oracle for isolated CPU planners.
// No SAM/LCA/decomposition code is used by the oracle. Also checks the derivative
// of each normalized pair weight, covering all vector forward/backward operands.
#include "prefill.hpp"
#include <iostream>
#include <random>
#include <climits>
using namespace dism_prefill;
int main() {
    static_assert(sizeof(long double)>sizeof(double),"extended precision required");
    std::mt19937 rng(4761); Builder builder;
    double maximum=0.; int cases=0;
    auto check=[&](double actual,long double expected,const char *field,int trial) {
        long double error=std::abs(static_cast<long double>(actual)-expected);
        long double limit=2e-9L+2e-9L*std::abs(expected);
        maximum=std::max(maximum,double(error));
        if (!std::isfinite(actual) || error>limit) {
            std::cerr<<field<<" trial="<<trial<<" actual="<<actual<<" expected="<<double(expected)
                     <<" error="<<double(error)<<" limit="<<double(limit)<<"\n";
            throw std::runtime_error("dense oracle mismatch");
        }
    };
    for (int trial=0;trial<500;++trial) {
        int n=1+rng()%128; Ids q(n),k(n),reset(n);
        for (int i=0;i<n;++i) {
            int symbols=trial%7==0 ? 512 : 7;
            q[i]=int(rng()%symbols)-symbols/2;k[i]=int(rng()%symbols)-symbols/2;
            if (trial%5==0) q[i]=k[i]=INT_MIN;
            if (trial%5==1) {q[i]=i%3;k[i]=(i+1)%3;}
            if (trial%5==2) {q[i]=0;k[i]=1;}
            reset[i]=trial%2 && rng()%19==0;
        }
        double tau=std::vector<double>{-16.,-2.,-1e-8,0.,1e-8,.5,2.,16.}[(trial/3)%8];
        bool gradients=trial%3!=0;
        builder.build(q.data(),k.data(),reset.data(),n,tau,trial%4==0,gradients);
        auto p=builder.finish();
        std::vector<double> actual(n*n,0.),tangent(n*n,0.),state(n),ds(n);
        for (size_t s=1;s<p.offsets.size();++s) {
            std::fill(state.begin(),state.end(),0.);std::fill(ds.begin(),ds.end(),0.);
            for (auto e=p.offsets[s-1];e<p.offsets[s];++e) {
                int row=p.rows[e];double w=p.weight[e],dw=gradients?p.dlogweight[e]:0.;
                if (row>=0) {
                    for (int j=0;j<n;++j) {state[j]*=p.decay[e];ds[j]*=p.decay[e];}
                    state[row]+=w;ds[row]+=w*dw;
                } else {
                    int i=-row-1;
                    for (int j=0;j<n;++j) {actual[i*n+j]+=w*state[j];tangent[i*n+j]+=w*(ds[j]+dw*state[j]);}
                }
            }
        }
        std::vector<int> lengths(n*n,0);
        long double pivot=n*std::max(0.,tau);
        std::vector<long double> f(n+1,0),df(n+1,0);
        for (int l=1;l<=n;++l) {
            long double term=std::exp(l*static_cast<long double>(tau)-pivot);
            f[l]=f[l-1]+term;df[l]=df[l-1]+l*term;
        }
        for (int i=0;i<n;++i) {
            long double z=std::exp(-pivot),dz=0;
            for (int j=0;j<=i;++j) {
                int l=q[i]==k[j] ? 1+(i && j && !reset[i] ? lengths[(i-1)*n+j-1] : 0) : 0;
                lengths[i*n+j]=l;z+=f[l];dz+=df[l];
            }
            check(p.logden[i],pivot+std::log(z),"logden",trial);
            if (gradients) check(p.dlogden[i],dz/z,"dlogden",trial);
            for (int j=0;j<n;++j) {
                int l=lengths[i*n+j];
                check(actual[i*n+j],f[l]/z,"pair_weight",trial);
                if (gradients) check(tangent[i*n+j],(df[l]-f[l]*(dz/z))/z,"pair_derivative",trial);
            }
        }
        ++cases;
    }
    std::cout<<cases<<" dense long-double cases passed; max_abs="<<maximum<<"\n";
}
