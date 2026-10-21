#pragma once
#include "../../../../types/types.cuh"

#ifdef KITTENS_FEATURE_MMA_BLOCK_SCALE
namespace kittens {
namespace detail {
__device__ __forceinline__ uint32_t online_bf16_bits(const bf16_2 &x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, sizeof(bits));
    return bits;
}
__device__ __forceinline__ uint32_t online_max_u16x2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("max.u16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ uint32_t online_shift_e8m0_u16x2(uint32_t e) {
    constexpr uint32_t packed_offset=0x03800380u;
    uint32_t clamped;
    asm("max.u16x2 %0, %1, %2;" : "=r"(clamped) : "r"(e), "r"(packed_offset));
    return __vsub2(clamped,packed_offset);
}

template<typename F8>
__device__ __forceinline__ uint16_t online_cvt_fp8x2(const bf16_2 &x, uint32_t rcp) {
    // SM120 supports native BF16x2 conversion, but not PTX 9.4's
    // scaled::n1::ue8m0 down-conversion (restricted to the SM107 family).
    uint32_t scaled;
    asm("mul.bf16x2 %0, %1, %2;" : "=r"(scaled)
        : "r"(online_bf16_bits(x)), "r"(rcp));
    uint16_t d;
    if constexpr (std::is_same_v<F8, fp8e4m3>)
        asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(d) : "r"(scaled));
    else
        asm("cvt.rn.satfinite.e5m2x2.bf16x2 %0, %1;" : "=h"(d) : "r"(scaled));
    return d;
}
} // namespace detail

/** MXFP8 quantization for finite BF16 row-layout register tiles.
 * Scale is the maximum BF16 exponent per row/K32 minus the E4M3 exponent
 * offset 7. Output has the usual TK FP8x4 MMA layout and mxfp8_scale metadata.
 * All 32 lanes must participate. No shared memory or block synchronization.
 *
 * Deliberately preserves the existing reciprocal bit trick, including its
 * exponent-254 limitation: rcp becomes zero for a block with |x| >= 2^127.
 * NaN/Inf blocks are outside this API's contract. Scale zero means 2^-127.
 * E5M2 uses exactly the same normalization, changing only the final cast.
 */
template<typename F8, int M, int K>
__device__ __forceinline__ void online_quantize_mxfp8(
    rt<F8,M,K> &dst, mxfp8_scale<M,K> &scales, const rt<bf16,M,K> &src) {
    static_assert(std::is_same_v<F8,fp8e4m3> || std::is_same_v<F8,fp8e5m2>);
    static_assert(K % 32 == 0);
    const int lane = threadIdx.x % 32;
    const int scale_shift = 7 + 16*(lane & 1);
    const int pair = (lane & ~3) + 2*(lane & 1);
    const uint32_t perm = (lane & 2) ? 0x7632 : 0x5410;
#pragma unroll
    for (int i=0; i<M/16; ++i) {
#pragma unroll
        for (int k=0; k<K/32; ++k) {
            uint32_t rows[2];
#pragma unroll
            for (int r=0; r<2; ++r) {
                uint32_t v = 0;
#pragma unroll
                for (int j=0; j<2; ++j) {
#pragma unroll
                    for (int p=r; p<4; p+=2)
                        v = detail::online_max_u16x2(v,
                            detail::online_bf16_bits(src.tiles[i][2*k+j].data[p]) & 0x7f807f80u);
                }
                rows[r] = max(v & 0xffffu, v >> 16);
            }
            // TK row_max's quad XOR reduction, with two rows packed as u16.
            uint32_t e = rows[0] | (rows[1] << 16);
            e = detail::online_max_u16x2(e, __shfl_xor_sync(0xffffffff, e, 1));
            e = detail::online_max_u16x2(e, __shfl_xor_sync(0xffffffff, e, 2));
            e = detail::online_shift_e8m0_u16x2(e);
            // MMA's byte selector consumes only byte 0. Keep the other row's
            // exponent in unused upper bytes instead of spending an AND to
            // clear it. Even/odd lanes supply rows g/g+8 to the A operand.
            scales.data[i][k] = e >> scale_shift;
            uint32_t rcp_bits = (0x7f007f00u - e) & 0x7f807f80u;
#pragma unroll
            for (int r=0; r<2; ++r) {
                uint32_t h = (rcp_bits >> (16*r)) & 0xffff;
                uint32_t rcp = h | (h << 16);
#pragma unroll
                for (int j=0; j<2; ++j) {
                    const auto &s = src.tiles[i][2*k+j];
                    uint32_t q = uint32_t(detail::online_cvt_fp8x2<F8>(s.data[r],rcp)) |
                        (uint32_t(detail::online_cvt_fp8x2<F8>(s.data[r+2],rcp)) << 16);
                    // Two BF16 K16 tiles become one FP8 K32 tile. Neighboring
                    // lanes own pairs of adjacent elements; combine into x4.
                    uint32_t lo = __shfl_sync(0xffffffff,q,pair);
                    uint32_t hi = __shfl_sync(0xffffffff,q,pair+1);
                    dst.tiles[i][k].data[r+2*j] =
                        std::bit_cast<typename rt<F8,M,K>::T2>(__byte_perm(lo,hi,perm));
                }
            }
        }
    }
}
} // namespace kittens
#endif
