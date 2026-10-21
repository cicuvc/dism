#pragma once
#include "../../group.cuh"

#ifdef KITTENS_FEATURE_MMA_BLOCK_SCALE
namespace kittens {

/** Native SM120 m16n8k32, FP8 x E8M0 with FP32 accumulation.
 * The six FP8x4 operands use the same register ordering as BF16x2 m16n8k16.
 * Scale selectors follow PTX's {byte-id, thread-id} convention. All 32 lanes
 * must execute the instruction. D may alias C; separate D may be uninitialized.
 * https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
 */
template<int ByteA=0, int ThreadA=0, int ByteB=0, int ThreadB=0,
         typename PA, typename PB>
__device__ __forceinline__ void mma16832_block_scaled(
    float2 &d0, float2 &d1,
    const PA &a0, const PA &a1, const PA &a2, const PA &a3,
    const PB &b0, const PB &b1, const float2 &c0, const float2 &c1,
    uint32_t sa, uint32_t sb) {
    static_assert(ByteA>=0 && ByteA<4 && ByteB>=0 && ByteB<4);
    static_assert(ThreadA>=0 && ThreadA<2 && ThreadB>=0 && ThreadB<4);
    static_assert(std::is_same_v<PA,fp8e4m3_4> || std::is_same_v<PA,fp8e5m2_4>);
    static_assert(std::is_same_v<PB,fp8e4m3_4> || std::is_same_v<PB,fp8e5m2_4>);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
    static_assert(sizeof(PA)==0, "Block-scaled mma.sync requires SM120 or later");
#endif
#define KITTENS_MXF8_MMA(FA, FB) \
    asm volatile( \
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32." FA "." FB ".f32.ue8m0 " \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, " \
        "%14, {%16,%17}, %15, {%18,%19};" \
        : "=f"(d0.x), "=f"(d0.y), "=f"(d1.x), "=f"(d1.y) \
        : "r"(std::bit_cast<uint32_t>(a0)), "r"(std::bit_cast<uint32_t>(a1)), \
          "r"(std::bit_cast<uint32_t>(a2)), "r"(std::bit_cast<uint32_t>(a3)), \
          "r"(std::bit_cast<uint32_t>(b0)), "r"(std::bit_cast<uint32_t>(b1)), \
          "f"(c0.x), "f"(c0.y), "f"(c1.x), "f"(c1.y), "r"(sa), "r"(sb), \
          "n"(ByteA), "n"(ThreadA), "n"(ByteB), "n"(ThreadB))
    if constexpr (std::is_same_v<PA,fp8e4m3_4> && std::is_same_v<PB,fp8e4m3_4>) {
        KITTENS_MXF8_MMA("e4m3", "e4m3");
    } else if constexpr (std::is_same_v<PA,fp8e4m3_4>) {
        KITTENS_MXF8_MMA("e4m3", "e5m2");
    } else if constexpr (std::is_same_v<PB,fp8e4m3_4>) {
        KITTENS_MXF8_MMA("e5m2", "e4m3");
    } else {
        KITTENS_MXF8_MMA("e5m2", "e5m2");
    }
#undef KITTENS_MXF8_MMA
}
} // namespace kittens

template<int GW>
template<typename TA, typename TB>
__device__ inline void group<GW>::wmma::mma_ABt_block_scaled_base(
    rt_base<float, ducks::rt_layout::row> &d,
    const rt_base<TA, ducks::rt_layout::row> &a,
    const rt_base<TB, ducks::rt_layout::row> &b,
    const rt_base<float, ducks::rt_layout::row> &c, uint32_t sa, uint32_t sb) {
    KITTENS_CHECK_WARP
    kittens::mma16832_block_scaled<0,0,0,0>(
        d.data[0],d.data[1],a.data[0],a.data[1],a.data[2],a.data[3],
        b.data[0],b.data[2],c.data[0],c.data[1],sa,sb);
    kittens::mma16832_block_scaled<0,0,0,1>(
        d.data[2],d.data[3],a.data[0],a.data[1],a.data[2],a.data[3],
        b.data[1],b.data[3],c.data[2],c.data[3],sa,sb);
}

template<int GW>
template<ducks::rt::row_layout D, ducks::rt::row_layout A,
         ducks::rt::row_layout B, ducks::rt::row_layout C>
__device__ inline void group<GW>::wmma::mma_ABt_block_scaled(
    D &d, const A &a, const B &b, const C &c,
    const kittens::mxfp8_scale<A::rows,A::cols> &sa,
    const kittens::mxfp8_scale<B::rows,B::cols> &sb) {
    KITTENS_CHECK_WARP
    static_assert(std::is_same_v<typename D::T,float> && std::is_same_v<typename C::T,float>);
    static_assert(D::rows==A::rows && D::cols==B::rows && A::cols==B::cols);
    static_assert(D::rows==C::rows && D::cols==C::cols);
#pragma unroll
    for (int m=0; m<D::height; ++m) {
#pragma unroll
        for (int n=0; n<D::width; ++n) {
            mma_ABt_block_scaled_base(d.tiles[m][n],a.tiles[m][0],b.tiles[n][0],
                                     c.tiles[m][n],sa.data[m][0],sb.data[n][0]);
#pragma unroll
            for (int k=1; k<A::width; ++k)
                mma_ABt_block_scaled_base(d.tiles[m][n],a.tiles[m][k],b.tiles[n][k],
                                         d.tiles[m][n],sa.data[m][k],sb.data[n][k]);
        }
    }
}
// AB overload: B is a KxN col-layout register tile. Scale metadata remains
// indexed [N/16][K/32], independent of the shared/global orientation.
template<int GW>
template<ducks::rt::row_layout D, ducks::rt::row_layout A,
         ducks::rt::col_layout B, ducks::rt::row_layout C>
__device__ inline void group<GW>::wmma::mma_AB(
    D &d,const A &a,const B &b,const C &c,
    const kittens::mxfp8_scale<A::rows,A::cols> &sa,
    const kittens::mxfp8_scale<B::cols,B::rows> &sb) {
    KITTENS_CHECK_WARP
    static_assert(D::rows==A::rows && D::cols==B::cols && A::cols==B::rows);
    static_assert(D::rows==C::rows && D::cols==C::cols);
#pragma unroll
    for(int m=0;m<D::height;++m) {
#pragma unroll
        for(int n=0;n<D::width;++n) {
#pragma unroll
            for(int k=0;k<A::width;++k) {
                const auto &at=a.tiles[m][k];
                const auto &bt=b.tiles[k][n];
                auto &dt=d.tiles[m][n];
                const auto &ct=k==0?c.tiles[m][n]:dt;
                kittens::mma16832_block_scaled<0,0,0,0>(
                    dt.data[0],dt.data[1],at.data[0],at.data[1],at.data[2],at.data[3],
                    bt.data[0],bt.data[2],ct.data[0],ct.data[1],sa.data[m][k],sb.data[n][k]);
                kittens::mma16832_block_scaled<0,0,0,1>(
                    dt.data[2],dt.data[3],at.data[0],at.data[1],at.data[2],at.data[3],
                    bt.data[1],bt.data[3],ct.data[2],ct.data[3],sa.data[m][k],sb.data[n][k]);
            }
        }
    }
}
#endif
