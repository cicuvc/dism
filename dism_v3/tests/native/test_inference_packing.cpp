// Standalone ASan/UBSan/TSan probe for concurrent native inference packing.
#include "prefill_inference.hpp"
#include <climits>
#include <cstring>
#include <iostream>
#include <random>

template<class T> void equal(const std::vector<T> &a,const std::vector<T> &b) {
    if (a.size()!=b.size() || (!a.empty() && std::memcmp(a.data(),b.data(),a.size()*sizeof(T))))
        throw std::runtime_error("packed byte mismatch");
}

void trial(std::shared_ptr<dism_prefill::PlanningWorkers> pool,int seed) {
    using namespace dism_prefill;
    std::mt19937 rng(seed);
    const int b=2,h=3,n=1+rng()%513,c=std::vector<int>{16,32,64}[seed%3];
    const int threshold=std::vector<int>{0,1,64}[seed%3],total=b*h*n;
    std::vector<int> labels(3*total);
    for (int j=0;j<total;++j) {
        labels[j]=rng()%8; labels[total+j]=rng()%8;
        if (seed%7==0) labels[j]=labels[total+j]=INT_MIN;
        if (seed%11==0) { labels[j]=0; labels[total+j]=1; }
        labels[2*total+j]=rng()%19==0;
    }
    std::vector<double> tau{0.,-.5,.5,.1,0.,-.3};
    PrefillBatchProgram plan(pool,labels.data(),b,h,n,tau.data(),b,labels.data()+2*total,c,threshold);
    const auto &p=*plan.program;
    size_t s=p.stream_base.back(),chunks=p.chunk_base.back(),slots=chunks*c;
    std::vector<int64_t> offsets(s+1),sid(s),cid(chunks),local(s),shorts(plan.short_base.back()),
        longs(plan.long_base.back()),lc(plan.long_chunk_base.back()),lo(longs.size());
    std::vector<int> rows(slots),lengths(chunks),resets(chunks);
    std::vector<float> prefixes(slots),weights(slots);
    TrainingBuffers out{offsets.data(),rows.data(),lengths.data(),resets.data(),nullptr,
                        prefixes.data(),weights.data(),nullptr};
    PrefillRoutingBuffers ids{sid.data(),cid.data(),local.data(),shorts.data(),longs.data(),lc.data(),lo.data()};
    plan.pack_into(out,ids);
    std::vector<int64_t> expected_offsets{0},expected_sid,expected_cid,expected_local,
        expected_short,expected_long,expected_lc,expected_lo;
    std::vector<int> expected_rows,expected_lengths,expected_resets;
    std::vector<float> expected_prefixes,expected_weights;
    int64_t cb=0,sb=0;
    for (int i=0;i<b*h;++i) {
        Ids q(labels.begin()+i*n,labels.begin()+(i+1)*n);
        Ids k(labels.begin()+total+i*n,labels.begin()+total+(i+1)*n);
        Ids reset(labels.begin()+2*total+i*n,labels.begin()+2*total+(i+1)*n);
        auto program=Builder(q,k,reset,tau[i]).finish();
        auto packed=chunk_program(program,c);
        if (!p.programs[i].dlogweight.empty()) throw std::runtime_error("inference derivatives allocated");
        for (int row:packed.rows) {
            int idx=row>=0?row:-row-1;
            int abs=((i/h)*n+idx)*h+i%h;
            expected_rows.push_back(row>=0?abs:-abs-1);
        }
        expected_lengths.insert(expected_lengths.end(),packed.lengths.begin(),packed.lengths.end());
        expected_resets.insert(expected_resets.end(),packed.resets.begin(),packed.resets.end());
        expected_prefixes.insert(expected_prefixes.end(),packed.prefixes.begin(),packed.prefixes.end());
        expected_weights.insert(expected_weights.end(),packed.weights.begin(),packed.weights.end());
        for (size_t j=1;j<packed.offsets.size();++j,++sb) {
            int64_t begin=cb+packed.offsets[j-1],end=cb+packed.offsets[j];
            expected_offsets.push_back(end); expected_sid.push_back(sb); expected_local.push_back(begin);
            if (end-begin>threshold) {
                expected_long.push_back(sb); expected_lo.push_back(expected_lc.size());
                for (auto k=begin;k<end;++k) expected_lc.push_back(k);
            } else expected_short.push_back(sb);
        }
        for (size_t j=0;j<packed.lengths.size();++j) expected_cid.push_back(cb+j);
        cb+=packed.lengths.size();
    }
    equal(offsets,expected_offsets); equal(rows,expected_rows); equal(prefixes,expected_prefixes);
    equal(weights,expected_weights); equal(lengths,expected_lengths); equal(resets,expected_resets);
    equal(sid,expected_sid); equal(cid,expected_cid); equal(local,expected_local);
    equal(shorts,expected_short); equal(longs,expected_long); equal(lc,expected_lc); equal(lo,expected_lo);
}

int main() {
    auto pool=std::make_shared<dism_prefill::PlanningWorkers>(4);
    std::vector<std::thread> callers;
    for (int caller=0;caller<4;++caller) callers.emplace_back([&,caller] {
        for (int i=0;i<25;++i) trial(pool,100*caller+i);
    });
    for (auto &thread:callers) thread.join();
    std::cout<<"100 concurrent inference plans/packed/routing buffers byte identical\n";
}
