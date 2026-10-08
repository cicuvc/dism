#include "summary/primitives.cuh"
#include "backward/config.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace {
constexpr int Step = dism_backward::SummaryK;
using Global = kt::gl<float,-1,-1,-1,-1>;
// Each thread owns one diagonal. Coordinates are aligned, with no v2 corner
// fixup: summary[c,q] maps G[q+Step,Step*(c+1)] to G[q,Step*c].
__global__ void reverse_chunk(const float* a,const float* b,float* output,int chunks,int padded) {
    int diagonal=int(blockIdx.x*blockDim.x+threadIdx.x)-(chunks-1)*Step;
    if (diagonal>=padded) return;
    int64_t base=int64_t(blockIdx.y)*chunks*padded;
    float state=0.f;
#pragma unroll 1
    for (int c=chunks-1;c>=0;--c) {
        int query=diagonal+c*Step;
        if (query>=0 && query<padded) {
            int64_t offset=base+int64_t(c)*padded+query;
            state=fmaf(a[offset],state,b[offset]);
            output[offset]=state;
        }
    }
}

template<int Pending>
__device__ __forceinline__ void wait_input(int remaining) {
    if constexpr (Pending==0) {
        kt::warp::load_async_wait<0>();
    } else {
        if (remaining>Pending) kt::warp::load_async_wait<Pending>();
        else wait_input<Pending-1>(remaining);
    }
}

// Each warp owns its input ring.16-element halves keep every async transfer
// aligned and in bounds for diagnostic inputs with a16-element tail.
template<int Stages, class Configuration = ActiveConfig>
__global__ void reverse_chunk_async(Global a,Global b,float* output,int chunks,int padded) {
    __shared__ kt::sv_fl<16> staged_a[4][Stages][2];
    __shared__ kt::sv_fl<16> staged_b[4][Stages][2];
    int warp=threadIdx.x/32,lane=threadIdx.x%32;
    int negative_span=((chunks-1)*Step+31)/32*32;
    int diagonal=int(blockIdx.x*blockDim.x)+warp*32-negative_span;
    if (diagonal>=padded) return;
    int first=min(chunks-1,(padded-1-diagonal)/Step);
    int last=max(0,(-diagonal-32+Step)/Step);
    if (first<last) return;
    int64_t base=int64_t(blockIdx.y)*chunks*padded;
    auto prefetch=[&](int slot,int chunk) {
#pragma unroll
        for (int half=0;half<2;++half) {
            int column=diagonal+chunk*Step+half*16;
            if (column>=0 && column+16<=padded) {
                kt::warp::load_async(staged_a[warp][slot][half],a,
                    {int(blockIdx.y),0,chunk,column});
                kt::warp::load_async(staged_b[warp][slot][half],b,
                    {int(blockIdx.y),0,chunk,column});
            } else if (lane<16) {
                staged_a[warp][slot][half][lane]=0.f;
                staged_b[warp][slot][half][lane]=0.f;
            }
        }
        kt::warp::load_async_commit_group();
    };
#pragma unroll
    for (int stage=0;stage<Stages;++stage) {
        if (first-stage>=last) prefetch(stage,first-stage);
    }
    float state=0.f;
    int slot=0;
#pragma unroll 1
    for (int chunk=first;chunk>=last;--chunk) {
        wait_input<Stages-1>(chunk-last+1);
        float aa=staged_a[warp][slot][lane/16][lane%16];
        float bb=staged_b[warp][slot][lane/16][lane%16];
        kt::warp::sync(); // All readers finish before this slot is refilled.
        if (chunk-Stages>=last) prefetch(slot,chunk-Stages);
        slot=(slot+1)%Stages;
        state=fmaf(aa,state,bb);
        int query=diagonal+chunk*Step+lane;
        if (query>=0 && query<padded)
            output[base+int64_t(chunk)*padded+query]=state;
    }
}
}

const void* backward_chunk_address_0() { return reinterpret_cast<const void*>(reverse_chunk_async<4>); }
const void* backward_chunk_address_1() { return reinterpret_cast<const void*>(reverse_chunk); }
} // namespace DISM_VARIANT
