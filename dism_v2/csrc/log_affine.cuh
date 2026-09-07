#pragma once
#include <glx/diagonal_scan.cuh>
#include <cmath>

namespace dism_v2 {
constexpr float LOG2E = 1.4426950408889634f;
__host__ __device__ __forceinline__ float logadd2(float x, float y) {
    if (x == -INFINITY) return y;
    if (y == -INFINITY) return x;
    return fmaxf(x, y) + log1pf(exp2f(-fabsf(x-y))) * LOG2E;
}
struct LogAffine {
    template<class T> __host__ __device__ __forceinline__ static glx::BinaryElement<T> identity() {
        return {T{0.f}, T{-INFINITY}};
    }
    __host__ __device__ __forceinline__ static glx::BinaryElement<float> apply(
            glx::BinaryElement<float> x, glx::BinaryElement<float> y) {
        return {x.first+y.first, logadd2(x.second+y.first, y.second)};
    }
    __device__ __forceinline__ static glx::BinaryElement<glx::F32x1> apply(
            glx::BinaryElement<glx::F32x1> x, glx::BinaryElement<glx::F32x1> y) {
        return {{x.first.u0+y.first.u0}, {logadd2(x.second.u0+y.first.u0,y.second.u0)}};
    }
    __device__ __forceinline__ static glx::BinaryElement<glx::F32x2> apply(
            glx::BinaryElement<glx::F32x2> x, glx::BinaryElement<glx::F32x2> y) {
        return {x.first+y.first,
                {logadd2(x.second.u0+y.first.u0,y.second.u0),
                 logadd2(x.second.u1+y.first.u1,y.second.u1)}};
    }
};
using Buffer = glx::MMABuffer<16,64,glx::BinaryElement,LogAffine,glx::F32x2>;
using Scalar = glx::MMABuffer<16,64,glx::UnaryElement,glx::AddOp,glx::F32x2>;
} // namespace dism_v2
