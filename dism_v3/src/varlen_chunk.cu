#include "varlen/chunk.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_varlen {
template<int Pending>
__device__ __forceinline__ void chunk_wait(int remaining) {
    if constexpr (Pending==0) kt::warp::load_async_wait<0>();
    else {
        if (remaining>Pending) kt::warp::load_async_wait<Pending>();
        else chunk_wait<Pending-1>(remaining);
    }
}

template<bool Reverse, class Configuration = ActiveConfig>
__global__ void varlen_chunk_kernel(const __grid_constant__ PackedArgs<ChunkArgs> packed) {
    constexpr int Stages=4;
    constexpr int Direction=Reverse ? -1 : 1;
    __shared__ kt::sv_fl<32> staged_a[4][Stages],staged_b[4][Stages];
    const auto& args=packed.common;
    int2 work=packed.tasks[blockIdx.x];
    const auto* document=packed.documents+int64_t(work.x)*Fields;
    int padded=document[Length];
    int chunks=Reverse?padded/32:(padded-1)/32;
    int diagonal_blocks=(padded+(chunks-1)*32+127)/128;
    int head=work.y/diagonal_blocks,block=work.y%diagonal_blocks;
    int warp=threadIdx.x/32,lane=threadIdx.x%32;
    int diagonal=block*128+warp*32-(chunks-1)*32;
    if (diagonal>=padded) return;
    int64_t base=document[Reverse?BackwardOffset:ForwardOffset]*args.heads+
                 int64_t(head)*chunks*padded;
    if constexpr (!Reverse) {
        if (diagonal>0) {
            int end=min(chunks,(padded-diagonal)/32);
            for (int c=0;c<end;++c)
                args.output[base+int64_t(c)*padded+diagonal+c*32+lane]=LOG_ZERO;
            return;
        }
    }
    int first,last;
    if constexpr (Reverse) {
        first=min(chunks-1,(padded-1-diagonal)/32);
        last=max(0,-diagonal/32);
    } else {
        first=-diagonal/32;
        last=chunks-1;
    }
    int count=Direction*(last-first)+1;
    if (count<=0) return;
    auto prefetch=[&](int slot,int c) {
        int column=diagonal+c*32;
        int row=int((base+int64_t(c)*padded+column)/32);
        kt::warp::load_async(staged_a[warp][slot],args.a,{0,0,row,0});
        kt::warp::load_async(staged_b[warp][slot],args.b,{0,0,row,0});
        kt::warp::load_async_commit_group();
    };
#pragma unroll
    for (int stage=0;stage<Stages;++stage)
        if (stage<count) prefetch(stage,first+Direction*stage);
    float state=Reverse ? 0.f : LOG_ZERO;
    int slot=0;
#pragma unroll 1
    for (int step=0;step<count;++step) {
        int c=first+Direction*step;
        chunk_wait<Stages-1>(count-step);
        float aa=staged_a[warp][slot][lane],bb=staged_b[warp][slot][lane];
        kt::warp::sync();
        if (step+Stages<count) prefetch(slot,c+Direction*Stages);
        slot=(slot+1)%Stages;
        if constexpr (Reverse) state=fmaf(aa,state,bb);
        else {
            float sum=state+aa,exponential;
            asm("ex2.approx.ftz.f32 %0,%1;":"=f"(exponential):"f"(-fabsf(sum-bb)));
            state=fmaxf(sum,bb)+log1pf(exponential)*1.4426950408889634f;
        }
        args.output[base+int64_t(c)*padded+diagonal+c*32+lane]=state;
    }
}

template<bool Reverse> const void* varlen_chunk_address() { return reinterpret_cast<const void*>(varlen_chunk_kernel<Reverse>); }
template const void* varlen_chunk_address<false>();
template const void* varlen_chunk_address<true>();
} // namespace dism_varlen
} // namespace DISM_VARIANT
