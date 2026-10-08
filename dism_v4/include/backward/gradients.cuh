#pragma once
#include "backward/types.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_backward {
__device__ __forceinline__ float exp2_ftz(float x) {
    float result;
    asm("ex2.approx.ftz.f32 %0,%1;":"=f"(result):"f"(x));
    return result;
}
__device__ __forceinline__ float sigmoid2(float x) {
    float t;
    asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(x*0.34657359027997265f));
    return fmaf(.5f,t,.5f);
}

template<int Channels, int Output, bool FP32Output>
__device__ __forceinline__ void store_key_gradient(
        const kt::rt_fl<K,Channels>& accum,void* dest,const RecomputeArgs& a,int bh,int k0,
        const Args& args, Shared& shared) {
    if constexpr (!FP32Output) {
        // query_gradient's final consumer rendezvous finished every transpose
        // reader. This warp may now reuse its private slot for output layout.
        constexpr int Panel = Channels<32 ? Channels : 32;
        auto scratch=shared.transpose[kt::warpid()].template subtile<K,Panel>({0,0});
#pragma unroll
        for (int panel=0;panel<Channels/Panel;++panel) {
            kt::rt_fl<K,Panel> part;
#pragma unroll
            for (int c=0;c<Panel/16;++c) part.tiles[0][c]=accum.tiles[0][Panel/16*panel+c];
            kt::warp::store(scratch,part);
            __syncwarp();
            kt::warp::store(args.key_output[Output],scratch,
                kt::coord<>{bh/a.heads,bh%a.heads,k0,Panel*panel});
            __syncwarp(); // Finish shared reads before the next panel/task.
        }
        return;
    }
    int lane = kt::warp::laneid();
#pragma unroll
    for (int c=0;c<Channels/16;++c) {
#pragma unroll
        for (int r=0;r<4;++r) {
            int key = k0+8*(r%2)+lane/4;
            int channel = 16*c+8*(r/2)+2*(lane&3);
            auto value = accum.tiles[0][c].data[r];
            if (key<a.n) {
                int64_t offset = ((int64_t(bh/a.heads)*a.n+key)*a.heads+bh%a.heads)*Channels+channel;
                static_cast<float*>(dest)[offset] = value.x;
                static_cast<float*>(dest)[offset+1] = value.y;
            }
        }
    }
}

template<int Channels, bool PackedLSE=false>
__device__ __forceinline__ void query_gradient(
        const kt::rt_bf<K,Q>& coefficient,kt::st_bf<128,Channels>& keys,
        kt::st_bf<K,Q> (&scratch)[8],const RecomputeArgs& a,int bh,int q0,
        const Args& args, Shared& shared,
        float* row_lse_dest=nullptr,const float (*row_lse)[Q]=nullptr,
        int packed_lse_offset=0) {
    // Restore logical query columns before the shared transpose. This is the
    // same necessary GEMM layout staging as v2; no full score is stored globally.
    // Output ownership is then reassigned across all8 consumers. There are no
    // shared floating-point atomics: each output tile accumulates128 keys in
    // registers and emits one FP32 global reduction per element.
    int owner=kt::warpid();
#pragma unroll
    for (int r=0;r<2;++r) {
#pragma unroll
        for (int c=0;c<4;++c) {
            auto pos=Scalar::layout(r,c,0);
            auto value=coefficient.tiles[0][c/2].data[r+2*(c&1)];
            scratch[owner][int2{pos.first,pos.second}] = __low2bfloat16(value);
            scratch[owner][int2{pos.first,pos.second+4}] = __high2bfloat16(value);
        }
    }
    kt::group<8>::sync(4);
    int lane = kt::warp::laneid();
    if (row_lse_dest && owner==0) {
        float sum=0.f;
#pragma unroll
        for (int source=0;source<8;++source) sum+=row_lse[source][lane];
        if (a.n%4==0) {
            kt::tma::store_async_read_wait<0>();
            __syncwarp();
            shared.lse_output[lane]=sum;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (kt::warp::elect_leader()) {
                if constexpr (PackedLSE)
                    kt::tma::store_add_async(args.lse_output,shared.lse_output,
                        kt::coord<>{0,0,0,packed_lse_offset});
                else
                    kt::tma::store_add_async(args.lse_output,shared.lse_output,
                        kt::coord<>{bh/a.heads,bh%a.heads,0,q0});
                kt::tma::store_commit_group();
            }
        } else
        if (q0+lane<a.n) atomicAdd(row_lse_dest+int64_t(bh)*a.n+q0+lane,sum);
    }
#pragma unroll
    for (int tile=owner;tile<2*(Channels/16);tile+=8) {
        int qt=tile/(Channels/16),c=tile%(Channels/16);
        kt::rt_fl<16,16> part{0.f};
#pragma unroll
        for (int source=0;source<8;++source) {
            kt::rt_bf<16,16,kt::ducks::rt_layout::col> grad;
            kt::warp::load(grad,scratch[source].template subtile<16,16>({0,qt}));
            kt::rt_bf<16,16,kt::ducks::rt_layout::col> column;
            kt::warp::load(column,keys.template subtile<16,16>({2*(source%4)+source/4,c}));
            [[clang::always_inline]] kt::warp::wmma::mma_AtB(part,grad,column,part);
        }
        {
            kt::tma::store_async_read_wait<0>();
            __syncwarp();
            kt::warp::store(shared.query_output[owner],part);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (kt::warp::elect_leader()) {
                kt::tma::store_add_async(args.query_output,shared.query_output[owner],
                    kt::coord<>{bh/a.heads,bh%a.heads,q0+qt*16,c*16});
                kt::tma::store_commit_group();
            }
        }
    }
    kt::group<8>::sync(4); // All readers finish before any writer reuses its slot.
}
} // namespace dism_backward

} // namespace DISM_VARIANT
