// Standalone CPU sanitizer harness; no Torch/Python/CUDA linkage.
#include "flash_dism/inference/csrc/prefill_training.hpp"
#include <cassert>
#include <iostream>

using namespace dism_prefill;

int main() {
    constexpr int b=2,h=3,n=65;
    std::vector<int> labels(2*b*h*n);
    for (size_t i=0;i<labels.size();++i) labels[i]=int(i%3);
    double tau[h]={0.,.5,2.};
    int64_t docs[]={0,0,17,0,17,65,1,0,33,1,33,65};
    uint8_t hard[b*h]={1,0,1,0,1,1};
    auto pool=std::make_shared<PlanningWorkers>(4);
    for (int iteration=0;iteration<12;++iteration) {
        TrainingBatchProgram p(pool,labels.data(),b,h,n,tau,1,docs,4,hard,16);
        int64_t s=p.stream_base.back(),c=p.chunk_base.back(),e=c*16;
        std::vector<int64_t> offsets(s+1);
        std::vector<int> rows(e),lengths(c),resets(c),heads(s);
        std::vector<float> prefixes(e),weights(e),derivs(e);
        TrainingBuffers out{offsets.data(),rows.data(),lengths.data(),resets.data(),heads.data(),
                            prefixes.data(),weights.data(),derivs.data()};
        p.pack_into(out);
        assert(offsets.back()==c);
        for (int i:rows) assert((i>=0 ? int64_t(i) : -int64_t(i)-1)<b*h*n);
        for (float x:derivs) assert(std::isfinite(x));
    }
    double bad[h]={1e308,.5,2.};
    bool failed=false;
    try { TrainingBatchProgram p(pool,labels.data(),b,h,n,bad,1,docs,4,hard,16); }
    catch (const std::invalid_argument &) { failed=true; }
    assert(failed);
    TrainingBatchProgram recovered(pool,labels.data(),b,h,n,tau,1,docs,4,hard,16);
    assert(recovered.jobs.size()==8);
    std::cout << "native planner sanitizer harness passed\n";
}
