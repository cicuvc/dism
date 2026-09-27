#pragma once
#include <glx/diagonal_scan.cuh>
#include <cmath>

namespace dism_v2 {
constexpr float LOG2E = 1.4426950408889634f;
#if defined(DISM_FINITE_SENTINEL) && DISM_FINITE_SENTINEL
constexpr float LOG_ZERO = -1.e6f;
#else
constexpr float LOG_ZERO = -INFINITY;
#endif
// Explicit FTZ avoids libdevice exp2f's subnormal rescaling path. This is
// intentional: exponential outputs below the normal FP32 range become zero.
__host__ __device__ __forceinline__ float exp2_ftz(float x) {
#if defined(__CUDA_ARCH__)
    float value;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(value) : "f"(x));
    return value;
#else
    float value=exp2f(x);
    return value<0x1p-126f?0.f:value;
#endif
}
__host__ __device__ __forceinline__ float logadd2(float x, float y) {
    if (x == -INFINITY) return y;
    if (y == -INFINITY) return x;
    return fmaxf(x, y) + log1pf(exp2_ftz(-fabsf(x-y))) * LOG2E;
}
// The approx() formula from /home/cicuvc/cs/projects/rl/lse.cu, NOT approx2.
// Keep the source's rounding/order (maximum + amplitude, then FMA).
// Exact-infinity modes keep guards. The opt-in bounded finite mode deliberately
// uses an approximate log-zero/identity and removes these tile-only branches.
__host__ __device__ __forceinline__ float tile_logadd2(float x, float y) {
#if defined(DISM_TILE_LSE_TANH) && DISM_TILE_LSE_TANH
#if !defined(DISM_FINITE_SENTINEL) || !DISM_FINITE_SENTINEL
    if (x == -INFINITY) return y;
    if (y == -INFINITY) return x;
#endif
    constexpr float amplitude = 1.81089463f;
    float p = fmaf(fabsf(x-y), 0.34114549f, 0.48232999f);
    float value;
#if defined(__CUDA_ARCH__)
    asm("tanh.approx.f32 %0, %1;" : "=f"(value) : "f"(p));
#else
    value = tanhf(p);
#endif
    float base = fmaxf(x,y) + amplitude;
    return fmaf(value, -amplitude, base);
#else
    return logadd2(x,y);
#endif
}
struct LogAffine {
    template<class T> __host__ __device__ __forceinline__ static glx::BinaryElement<T> identity() {
        return {T{0.f}, T{LOG_ZERO}};
    }
    __host__ __device__ __forceinline__ static glx::BinaryElement<float> apply(
            glx::BinaryElement<float> x, glx::BinaryElement<float> y) {
        return {x.first+y.first, tile_logadd2(x.second+y.first, y.second)};
    }
    __device__ __forceinline__ static glx::BinaryElement<glx::F32x1> apply(
            glx::BinaryElement<glx::F32x1> x, glx::BinaryElement<glx::F32x1> y) {
        return {{x.first.u0+y.first.u0}, {tile_logadd2(x.second.u0+y.first.u0,y.second.u0)}};
    }
    __device__ __forceinline__ static glx::BinaryElement<glx::F32x2> apply(
            glx::BinaryElement<glx::F32x2> x, glx::BinaryElement<glx::F32x2> y) {
        return {x.first+y.first,
                {tile_logadd2(x.second.u0+y.first.u0,y.second.u0),
                 tile_logadd2(x.second.u1+y.first.u1,y.second.u1)}};
    }
};
using Buffer = glx::MMABuffer<16,64,glx::BinaryElement,LogAffine,glx::F32x2>;
using Scalar = glx::MMABuffer<16,64,glx::UnaryElement,glx::AddOp,glx::F32x2>;
} // namespace dism_v2
