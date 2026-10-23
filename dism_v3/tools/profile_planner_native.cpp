// Standalone CPU profiler target; production headers and compact C64 math.
// Build this diagnostic with -O3 -g -std=c++17 -pthread and the inference include path.
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#ifdef PHASE_PROFILE
#include "profile_support.hpp"
void* operator new(std::size_t n) {
    void* p=std::malloc(n?n:1);
    if(!p) throw std::bad_alloc();
    prof::allocation(n); return p;
}
void operator delete(void* p) noexcept { if(p) prof::release(); std::free(p); }
void operator delete(void* p,std::size_t) noexcept {::operator delete(p);}
void* operator new[](std::size_t n) {return ::operator new(n);}
void operator delete[](void* p) noexcept {::operator delete(p);}
void operator delete[](void* p,std::size_t) noexcept {::operator delete(p);}
#endif
#include "prefill_training.hpp"
using namespace dism_prefill;
using Clock=std::chrono::steady_clock;
struct Output {
    std::vector<std::unique_ptr<int[]>> ints;
    std::vector<std::unique_ptr<float[]>> floats;
    std::unique_ptr<int64_t[]> offsets;
    std::vector<CompactBucketBuffers> buckets;
    TrainingBuffers multi{};
    size_t bytes=0;
    int* integer(size_t n) {bytes+=4*n;ints.emplace_back(new int[n]);return ints.back().get();}
    float* real(size_t n) {bytes+=4*n;floats.emplace_back(new float[n]);return floats.back().get();}
    explicit Output(const TrainingBatchProgram& p) {
        for(int b=0;b<3;++b) {
            size_t count=p.bucket_blocks[b], slots=count*compact_widths[b];
            CompactBucketBuffers v;
            v.rows=integer(slots);v.prefixes=real(slots);v.weights=real(slots);v.dlog=real(slots);
            v.lengths=integer(count);v.heads=integer(count);buckets.push_back(v);
        }
        size_t streams=p.stream_base.back(),chunks=p.chunk_base.back(),slots=chunks*p.width;
        offsets.reset(new int64_t[streams+1]);bytes+=8*(streams+1);multi.offsets=offsets.get();
        multi.rows=integer(slots);multi.prefixes=real(slots);multi.weights=real(slots);multi.dlog=real(slots);
        multi.lengths=integer(chunks);multi.resets=integer(chunks);multi.stream_heads=integer(streams);
    }
};
template<class T> void read(std::ifstream& f,T* out,size_t n) {
    f.read(reinterpret_cast<char*>(out),n*sizeof(T));if(!f)throw std::runtime_error("input truncated");
}
int main(int argc,char** argv) {
    if(argc<4) {std::cerr<<"input.bin seconds workers [output.bin]\n";return 2;}
    std::ifstream f(argv[1],std::ios::binary);int shape[4];read(f,shape,4);
    int b=shape[0],h=shape[1],n=shape[2],nd=shape[3];
    if(b<1||h<1||n<1||nd<1)throw std::runtime_error("invalid input shape");
    std::vector<int> labels(size_t(2)*b*h*n);std::vector<uint8_t> hard(b*h);
    std::vector<double> tau(h);std::vector<int64_t> docs(nd*3);
    read(f,labels.data(),labels.size());read(f,hard.data(),hard.size());
    read(f,tau.data(),tau.size());read(f,docs.data(),docs.size());
    auto pool=std::make_shared<PlanningWorkers>(std::stoi(argv[3]));
    size_t iterations=0,events=0,metadata=0;double seconds=std::stod(argv[2]);
    auto start=Clock::now();
    do {
        auto p=std::make_unique<TrainingBatchProgram>(pool,labels.data(),b,h,n,tau.data(),1,docs.data(),nd,hard.data(),64,true);
#ifdef PHASE_PROFILE
        prof::begin(12);
#endif
        auto output=std::make_unique<Output>(*p);
#ifdef PHASE_PROFILE
        prof::end();
#endif
        p->pack_into_compact(output->buckets,output->multi);
        events=p->events;metadata=output->bytes;
        if(iterations==0 && argc>4) {
            std::ofstream out(argv[4],std::ios::binary);
            auto write=[&](const void* data,size_t bytes){out.write(static_cast<const char*>(data),bytes);};
            for(int i=0;i<3;++i) {
                size_t blocks=p->bucket_blocks[i],slots=blocks*compact_widths[i];auto& v=output->buckets[i];
                write(v.rows,4*slots);write(v.prefixes,4*slots);write(v.weights,4*slots);write(v.dlog,4*slots);
                write(v.lengths,4*blocks);write(v.heads,4*blocks);
            }
            size_t streams=p->stream_base.back(),chunks=p->chunk_base.back(),slots=chunks*64;auto& v=output->multi;
            write(v.offsets,8*(streams+1));write(v.rows,4*slots);write(v.prefixes,4*slots);write(v.weights,4*slots);write(v.dlog,4*slots);
            write(v.lengths,4*chunks);write(v.resets,4*chunks);write(v.stream_heads,4*streams);
        }
#ifdef PHASE_PROFILE
        prof::begin(11);
#endif
        output.reset();p.reset();
#ifdef PHASE_PROFILE
        prof::end();
#endif
        ++iterations;
    } while(std::chrono::duration<double>(Clock::now()-start).count()<seconds);
    double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
    std::cout<<"{\"iterations\":"<<iterations<<",\"seconds\":"<<elapsed<<",\"ms_per_iteration\":"<<elapsed*1000/iterations
             <<",\"events\":"<<events<<",\"metadata_bytes\":"<<metadata<<"}\n";
#ifdef PHASE_PROFILE
    prof::report(iterations);
#endif
    return 0;
}
