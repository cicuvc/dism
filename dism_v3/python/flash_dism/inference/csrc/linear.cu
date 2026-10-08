// Exact hard-label, linear-history CUDA control for timing and cross-validation.
#include "device_math.cuh"
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
__device__ float cvt(float x) { return x; }
__device__ float cvt(__nv_bfloat16 x) { return __bfloat162float(x); }
struct Maximum {
    __device__ int operator()(int a, int b) const { return a > b ? a : b; }
};

__global__ void prime_lengths(const int *labels, int *keys, int *previous, int h, int n, int capacity) {
    int p = blockIdx.x * blockDim.x + threadIdx.x, head = blockIdx.y;
    if (p >= n)
        return;
    keys[int64_t(head) * capacity + p] = labels[(p * 3) * h + head];
    int length = 0;
    while (p - length >= 0 && labels[((p - length) * 3) * h + head] == labels[((n - 1 - length) * 3 + 1) * h + head]) {
        ++length;
        if (labels[((n - length) * 3 + 2) * h + head])
            break;
    }
    previous[int64_t(head) * capacity + p] = length;
}

template <class T>
__global__ void linear_step(T *sk, T *v, int *key_labels, const int *previous, int *lengths, const int *labels,
                            const T *nk, const T *nv, const T *sq, const float *taus, float *out, int h, int capacity,
                            int position, int r, int d) {
    int head = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    sk += int64_t(head) * capacity * r;
    v += int64_t(head) * capacity * d;
    key_labels += int64_t(head) * capacity;
    previous += int64_t(head) * capacity;
    lengths += int64_t(head) * capacity;
    for (int c = tid; c < r; c += 128)
        sk[int64_t(position) * r + c] = nk[head * r + c];
    for (int c = tid; c < d; c += 128)
        v[int64_t(position) * d + c] = nv[head * d + c];
    if (tid == 0)
        key_labels[position] = labels[head];
    __syncthreads();
    int maximum = 0;
    for (int p = tid; p <= position; p += 128) {
        int len = key_labels[p] == labels[h + head] ? 1 + ((p && !labels[2 * h + head]) ? previous[p - 1] : 0) : 0;
        lengths[p] = len;
        maximum = max(maximum, len);
    }
    using Reduce = cub::BlockReduce<int, 128>;
    __shared__ typename Reduce::TempStorage storage;
    __shared__ int max_length;
    maximum = Reduce(storage).Reduce(maximum, Maximum{});
    if (tid == 0)
        max_length = maximum;
    __syncthreads();
    float tau = taus[head], sum0 = 0, sum1 = 0, den = 0;
    for (int p = warp; p <= position; p += 4) {
        int len = lengths[p];
        if (!len)
            continue;
        float weight = expf(tau * (len - max_length)) * geometric_weight(len, tau);
        float dot = lane < r ? cvt(sk[int64_t(p) * r + lane]) * cvt(sq[head * r + lane]) : 0.f;
        for (int shift = 16; shift; shift /= 2)
            dot += __shfl_down_sync(0xffffffff, dot, shift);
        dot = __shfl_sync(0xffffffff, dot, 0) * weight;
        sum0 = fmaf(dot, cvt(v[int64_t(p) * d + lane]), sum0);
        if (d == 64)
            sum1 = fmaf(dot, cvt(v[int64_t(p) * d + lane + 32]), sum1);
        den += weight;
    }
    __shared__ float sums[4][65];
    sums[warp][lane] = sum0;
    if (d == 64)
        sums[warp][lane + 32] = sum1;
    if (lane == 0)
        sums[warp][64] = den;
    __syncthreads();
    if (warp == 0) {
        sum0 = sum1 = 0;
        den = expf(-tau * max_length);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            sum0 += sums[w][lane];
            if (d == 64)
                sum1 += sums[w][lane + 32];
            den += sums[w][64];
        }
        out[head * d + lane] = __fdividef(sum0, den);
        if (d == 64)
            out[head * d + lane + 32] = __fdividef(sum1, den);
    }
}
extern "C" void launch_linear_prime(const int *labels, int *keys, int *prev, int h, int n, int capacity,
                                    cudaStream_t stream) {
    prime_lengths<<<dim3((n + 127) / 128, h), 128, 0, stream>>>(labels, keys, prev, h, n, capacity);
}
extern "C" void launch_linear(void *k, void *v, int *keys, const int *prev, int *next, const int *labels,
                              const void *nk, const void *nv, const void *sq, const float *tau, float *out, int h,
                              int capacity, int position, int r, int d, bool bf16, cudaStream_t stream) {
    if (bf16)
        linear_step<<<h, 128, 0, stream>>>((__nv_bfloat16 *)k, (__nv_bfloat16 *)v, keys, prev, next, labels,
                                           (const __nv_bfloat16 *)nk, (const __nv_bfloat16 *)nv,
                                           (const __nv_bfloat16 *)sq, tau, out, h, capacity, position, r, d);
    else
        linear_step<<<h, 128, 0, stream>>>((float *)k, (float *)v, keys, prev, next, labels, (const float *)nk,
                                           (const float *)nv, (const float *)sq, tau, out, h, capacity, position, r, d);
}
