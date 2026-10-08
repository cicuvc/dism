// Standalone host sanitizer smoke; does not link Python/Torch/CUDA.
#include "prefill.hpp"
#include <cassert>
#include <random>

int main() {
    std::mt19937 rng(612);
    for (int trial=0;trial<32;++trial) {
        int n=1+trial*33;
        dism_prefill::Ids q(n),k(n),reset(n);
        for (int i=0;i<n;++i) {
            q[i]=rng()%7; k[i]=rng()%7; reset[i]=(rng()%13)==0;
            if (trial%3==0) q[i]=k[i]=0;
            if (trial%3==1) q[i]=k[i]=i;
        }
        auto p=dism_prefill::Builder(q,k,reset,4.2).finish();
        std::vector<double> input(n,1.),out(n);
        dism_prefill::execute(p,input.data(),input.data(),input.data(),1,1,out.data());
        for (int i=0;i<n;++i) {
            assert(std::isfinite(out[i]));
            assert(std::abs(out[i] - (-std::expm1(-p.logden[i]))) < 1e-10);
        }
    }
}
