// Parallel tree rebuild with bounded channel slabs. Euler interval sums recover
// subtree matrices; pointer doubling evaluates normalized ancestor affine sums.
// FP32 by default; FP64 prefix sums are a build-time numerical diagnostic.
#include "device_math.cuh"
#include "prefix_type.hpp"
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ float load_float(float x) { return x; }
__device__ float load_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <class T>
__global__ void euler_scan(const T *k, const T *v, const int *topo, DismPrefix *prefix, DismPrefix *totals, int n,
                           int r, int d, int begin) {
    using Scan = cub::BlockScan<DismPrefix, 256>;
    __shared__ typename Scan::TempStorage storage;
    int i = blockIdx.x * 256 + threadIdx.x, c = blockIdx.y, coordinate = begin + c;
    const int *position = topo + 2 * n, *order = topo + 3 * n;
    DismPrefix value = 0;
    if (i < n) {
        int p = position[order[i]];
        if (p >= 0)
            value = DismPrefix(load_float(v[int64_t(p) * d + coordinate % d]) *
                               load_float(k[int64_t(p) * r + coordinate / d]));
    }
    DismPrefix aggregate;
    Scan(storage).InclusiveSum(value, value, aggregate);
    if (i < n)
        prefix[int64_t(c) * n + i] = value;
    if (threadIdx.x == 0)
        totals[int64_t(c) * gridDim.x + blockIdx.x] = aggregate;
}

__global__ void scan_totals(DismPrefix *totals, int blocks, int width) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width)
        return;
    DismPrefix sum = 0;
    for (int b = 0; b < blocks; ++b) {
        DismPrefix x = totals[int64_t(c) * blocks + b];
        totals[int64_t(c) * blocks + b] = sum;
        sum += x;
    }
}

__device__ DismPrefix prefix_at(const DismPrefix *prefix, const DismPrefix *totals, int i, int c, int n, int blocks) {
    return i < 0 ? DismPrefix(0) : prefix[int64_t(c) * n + i] + totals[int64_t(c) * blocks + i / 256];
}

__global__ void subtree_init(const int *topo, const float *weights, const DismPrefix *prefix, const DismPrefix *totals,
                             float *b, float *out, int n, int r, int d, int begin, int width, float tau) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width)
        return;
    int node = i / width, c = i % width, coordinate = begin + c;
    const int *parent = topo, *length = topo + n, *mat_id = topo + 4 * n, *tin = topo + 6 * n, *tout = topo + 7 * n;
    int blocks = (n + 255) / 256;
    DismPrefix subtree = prefix_at(prefix, totals, tout[node] - 1, c, n, blocks) -
                         prefix_at(prefix, totals, tin[node] - 1, c, n, blocks);
    if (mat_id[node] >= 0)
        out[int64_t(mat_id[node]) * (r * d) + coordinate] = float(subtree);
    b[i] = weights[node] * float(subtree);
}

__global__ void ancestor_step(const int *topo, const int *ancestor, const float *rho, const float *input, float *output,
                              int n, int width, float tau) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width)
        return;
    int node = i / width, c = i % width, a = ancestor[node];
    output[i] = a < 0 ? input[i] : fmaf(rho[node], input[int64_t(a) * width + c], input[i]);
}

__global__ void save_samples(const int *topo, const float *b, float *out, int n, int r, int d, int begin, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width)
        return;
    int node = i / width, c = i % width, index = topo[5 * n + node];
    if (index >= 0)
        out[int64_t(index) * (r * d) + begin + c] = b[i];
}

extern "C" void launch_parallel_rebuild(const void *k, const void *v, const int *topo, const float *coefficients,
                                        DismPrefix *prefix, DismPrefix *totals, float *ping, float *pong, float *out,
                                        int n, int r, int d, int begin, int width, int levels, float tau, bool bf16,
                                        cudaStream_t stream) {
    dim3 grid((n + 255) / 256, width);
    if (bf16)
        euler_scan<<<grid, 256, 0, stream>>>((const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v, topo, prefix, totals,
                                             n, r, d, begin);
    else
        euler_scan<<<grid, 256, 0, stream>>>((const float *)k, (const float *)v, topo, prefix, totals, n, r, d, begin);
    scan_totals<<<(width + 127) / 128, 128, 0, stream>>>(totals, grid.x, width);
    int blocks = (n * width + 255) / 256;
    subtree_init<<<blocks, 256, 0, stream>>>(topo, coefficients, prefix, totals, ping, out, n, r, d, begin, width, tau);
    for (int level = 0; level < levels; ++level) {
        ancestor_step<<<blocks, 256, 0, stream>>>(topo, topo + (8 + level) * n, coefficients + (level + 1) * n, ping,
                                                  pong, n, width, tau);
        float *swap = ping;
        ping = pong;
        pong = swap;
    }
    if (levels)
        save_samples<<<blocks, 256, 0, stream>>>(topo, ping, out, n, r, d, begin, width);
}
