#pragma once

#include <cstdint>

namespace fp8gemm::silu_bwd {

__device__ __forceinline__ uint32_t tanh_bf16x2(uint32_t x) {
    uint32_t out;
    asm("tanh.approx.bf16x2 %0, %1;" : "=r"(out) : "r"(x));
    return out;
}

__device__ __forceinline__ uint32_t fma_bf16x2(
    uint32_t a,uint32_t b,uint32_t c) {
    uint32_t out;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;"
        : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a,uint32_t b) {
    uint32_t out;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

__device__ __forceinline__ uint16_t cvt_e4m3x2_bf16x2(uint32_t x) {
    uint16_t out;
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;"
        : "=h"(out) : "r"(x));
    return out;
}

struct PairGrad {
    uint32_t gate,up;
};

__device__ __forceinline__ PairGrad backward_pair(
    uint32_t dy,uint32_t half_gate,uint32_t up) {
    constexpr uint32_t half=0x3f003f00u; // BF16x2 0.5
    constexpr uint32_t neg_half=0xbf00bf00u;
    uint32_t tanh=tanh_bf16x2(half_gate);
    uint32_t a=fma_bf16x2(half_gate,tanh,half_gate); // silu(gate)
    uint32_t b=fma_bf16x2(neg_half,tanh,half);       // 0.5*(1-tanh)
    uint32_t c=fma_bf16x2(half,tanh,half);           // sigmoid(gate)
    uint32_t dsilu=fma_bf16x2(a,b,c);
    uint32_t up_dsilu=mul_bf16x2(up,dsilu);
    return {mul_bf16x2(dy,up_dsilu),mul_bf16x2(dy,a)};
}

__device__ __forceinline__ uint32_t bf16x2_scale(
    uint32_t values,uint8_t exponent) {
    uint16_t reciprocal=uint16_t((0x7f00u-(uint32_t(exponent)<<7))&0x7f80u);
    return mul_bf16x2(values,uint32_t(reciprocal)|(uint32_t(reciprocal)<<16));
}

__device__ __forceinline__ uint32_t bf16x2_scale_pair(
    uint32_t values,uint8_t exponent_low,uint8_t exponent_high) {
    uint32_t reciprocal_low=
        (0x7f00u-(uint32_t(exponent_low)<<7))&0x7f80u;
    uint32_t reciprocal_high=
        (0x7f00u-(uint32_t(exponent_high)<<7))&0x7f80u;
    return mul_bf16x2(values,reciprocal_low|(reciprocal_high<<16));
}

} // namespace fp8gemm::silu_bwd
