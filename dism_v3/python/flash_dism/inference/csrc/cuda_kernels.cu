#include "device_math.cuh"
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
__device__ __forceinline__ float fp32(float x) { return x; }
__device__ __forceinline__ float fp32(__nv_bfloat16 x) { return __bfloat162float(x); }

// No Torch headers in device translation unit.
template <typename T>
__global__ void rebuild_kernel(const T *keys, const T *values, const int *topology, float *scratch, float *summaries,
                               int nodes, int r, int d, int begin, int width, int pitch, float tau) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width)
        return;
    int coordinate = begin + c, stride = r * d + 1;
    const int *parent = topology, *length = parent + nodes, *position = length + nodes;
    const int *order = position + nodes, *mat_id = order + nodes, *sample_id = mat_id + nodes;
    // One coordinate per thread, bounded node-major scratch; no atomics.
    for (int n = 0; n < nodes; ++n) {
        int p = position[n];
        scratch[int64_t(n) * pitch + c] =
            p < 0 ? 0.f
                  : (coordinate == r * d
                         ? 1.f
                         : fp32(values[int64_t(p) * d + coordinate % d]) * fp32(keys[int64_t(p) * r + coordinate / d]));
    }
    for (int i = nodes - 1; i > 0; --i) {
        int n = order[i];
        scratch[int64_t(parent[n]) * pitch + c] += scratch[int64_t(n) * pitch + c];
    }
    // Subtree sums are replaced by normalized ancestor sums in preorder.
    for (int i = 0; i < nodes; ++i) {
        int n = order[i];
        float subtree = scratch[int64_t(n) * pitch + c];
        if (mat_id[n] >= 0)
            summaries[int64_t(mat_id[n]) * stride + coordinate] = subtree;
        float prefix = 0.f;
        if (n) {
            int gap = length[n] - length[parent[n]];
            float rho = expf(-tau * gap);
            float weight = geometric_weight(gap, tau);
            prefix = rho * scratch[int64_t(parent[n]) * pitch + c] + weight * subtree;
        }
        scratch[int64_t(n) * pitch + c] = prefix;
        if (sample_id[n] >= 0)
            summaries[int64_t(sample_id[n]) * stride + coordinate] = prefix;
    }
}

template <typename T, bool CpuDenominator>
__global__ void query_kernel(T *keys, T *values, const T *new_keys, const T *new_values, const T *query,
                             const float *summaries, const int *task_ids, const float *coefficients,
                             const float *fallback, float *output, int capacity, int position, int r, int d,
                             int tasks) {
    int h = blockIdx.x, lane = threadIdx.x;
    T *key = keys + int64_t(h) * capacity * r;
    T *value = values + int64_t(h) * capacity * d;
    const T *x = query + h * r;
    for (int c = lane; c < r; c += blockDim.x)
        key[int64_t(position) * r + c] = new_keys[h * r + c];
    for (int c = lane; c < d; c += blockDim.x)
        value[int64_t(position) * d + c] = new_values[h * d + c];
    __syncthreads();
    // Four warps partition tasks. Raw dot products are computed cooperatively
    // once, while matrix layout [R,DV] coalesces output-coordinate accesses.
    int warp = lane / 32, l = lane % 32;
    float sum0 = 0.f, sum1 = 0.f, den = 0.f;
    __shared__ float partial[4][65]; // output reduction / necessary warp communication
    for (int t = warp; t < tasks; t += 4) {
        int64_t task = int64_t(h) * tasks + t;
        float weight = coefficients[task];
        if (weight == 0.f)
            continue;
        int kind = task_ids[task * 2], index = task_ids[task * 2 + 1];
        if (kind == 0) {
            float dot = l < r ? fp32(key[int64_t(index) * r + l]) * fp32(x[l]) : 0.f;
            for (int shift = 16; shift; shift /= 2)
                dot += __shfl_down_sync(0xffffffff, dot, shift);
            dot = __shfl_sync(0xffffffff, dot, 0) * weight;
            sum0 = fmaf(dot, fp32(value[int64_t(index) * d + l]), sum0);
            if (d == 64)
                sum1 = fmaf(dot, fp32(value[int64_t(index) * d + l + 32]), sum1);
            if constexpr (!CpuDenominator)
                den += weight;
        } else {
            const float *matrix = summaries + int64_t(index) * (r * d + (CpuDenominator ? 0 : 1));
            float a0 = 0.f, a1 = 0.f;
            for (int j = 0; j < r; ++j) {
                float qj = fp32(x[j]);
                a0 = fmaf(matrix[j * d + l], qj, a0);
                if (d == 64)
                    a1 = fmaf(matrix[j * d + l + 32], qj, a1);
            }
            sum0 = fmaf(weight, a0, sum0);
            sum1 = fmaf(weight, a1, sum1);
            if constexpr (!CpuDenominator)
                den = fmaf(weight, matrix[r * d], den);
        }
    }
    partial[warp][l] = sum0;
    if (d == 64)
        partial[warp][l + 32] = sum1;
    if (!CpuDenominator && l == 0)
        partial[warp][64] = den;
    __syncthreads();
    if (warp == 0) {
        float denominator = fallback[h];
        sum0 = sum1 = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if constexpr (!CpuDenominator)
                denominator += partial[w][64];
            sum0 += partial[w][l];
            if (d == 64)
                sum1 += partial[w][l + 32];
        }
        output[h * d + l] = CpuDenominator ? sum0 * denominator : __fdividef(sum0, denominator);
        if (d == 64)
            output[h * d + l + 32] = CpuDenominator ? sum1 * denominator : __fdividef(sum1, denominator);
    }
}

extern "C" void launch_rebuild(const void *k, const void *v, const int *topo, float *scratch, float *out, int n, int r,
                               int d, int begin, int width, int pitch, float tau, bool bf16, cudaStream_t stream) {
    if (bf16)
        rebuild_kernel<<<(width + 127) / 128, 128, 0, stream>>>((const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v,
                                                                topo, scratch, out, n, r, d, begin, width, pitch, tau);
    else
        rebuild_kernel<<<(width + 127) / 128, 128, 0, stream>>>((const float *)k, (const float *)v, topo, scratch, out,
                                                                n, r, d, begin, width, pitch, tau);
}
extern "C" void launch_query(void *k, void *v, const void *nk, const void *nv, const void *q, const float *summaries,
                             const int *ids, const float *coeff, const float *fallback, float *output, int h,
                             int capacity, int position, int r, int d, int tasks, bool bf16, bool cpu_denominator,
                             cudaStream_t stream) {
#define LAUNCH_QUERY(T, CPU)                                                                                           \
    query_kernel<T, CPU><<<h, 128, 0, stream>>>((T *)k, (T *)v, (const T *)nk, (const T *)nv, (const T *)q, summaries, \
                                                ids, coeff, fallback, output, capacity, position, r, d, tasks)
    if (bf16) {
        if (cpu_denominator) {
            LAUNCH_QUERY(__nv_bfloat16, true);
        } else {
            LAUNCH_QUERY(__nv_bfloat16, false);
        }
    } else {
        if (cpu_denominator) {
            LAUNCH_QUERY(float, true);
        } else {
            LAUNCH_QUERY(float, false);
        }
    }
#undef LAUNCH_QUERY
}
