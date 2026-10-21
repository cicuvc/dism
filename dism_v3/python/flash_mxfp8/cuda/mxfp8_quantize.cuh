#pragma once

#include <cstdint>

namespace fp8gemm::quant {

constexpr int kE4M3E8M0Offset=7;

__device__ __forceinline__ uint32_t shift_e8m0_byte(
    uint32_t exponent_byte) {
    return max(exponent_byte,uint32_t(kE4M3E8M0Offset))
        -uint32_t(kE4M3E8M0Offset);
}

// Shift an unpacked BF16 exponent field into the E8M0 scale exponent.
// Zero and tiny blocks clamp to the smallest addressable E8M0 scale.
__device__ __forceinline__ uint32_t shift_e8m0(uint32_t exponent_bits) {
    return shift_e8m0_byte(exponent_bits>>7)<<7;
}

// Shift two BF16 exponent fields packed in u16x2 with no cross-half borrow.
// SM120 lowers this to VIMNMX.U16x2 plus VIADD.16x2.
__device__ __forceinline__ uint32_t shift_e8m0_pair(uint32_t exponent_bits) {
    constexpr uint32_t offset=uint32_t(kE4M3E8M0Offset)<<7;
    constexpr uint32_t packed_offset=offset|(offset<<16);
    uint32_t clamped;
    asm("max.u16x2 %0, %1, %2;"
        : "=r"(clamped) : "r"(exponent_bits), "r"(packed_offset));
    return __vsub2(clamped,packed_offset);
}

} // namespace fp8gemm::quant
