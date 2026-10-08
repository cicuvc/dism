#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ float geometric_weight(int length, float tau) {
    // length <= INT_MAX. Below this tau, tau*length < 2.2e-11:
    // the geometric correction is far below FP32 precision. This also avoids
    // reciprocal overflow for subnormal tau. Otherwise denominator >= 1e-20.
    if (tau < 1e-20f)
        return float(length);
    return __fdividef(-expm1f(-tau * length), -expm1f(-tau));
}
