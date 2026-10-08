#include "device_math.cuh"
#include "gpu_planner_params.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

template <class T> __device__ __forceinline__ float scalar(T x) { return float(x); }
template <> __device__ __forceinline__ float scalar(__nv_bfloat16 x) { return __bfloat162float(x); }

// Frozen CSR transitions. No hash allocation and no mutable SAM on ordinary steps.
__device__ __forceinline__ int transition(const int *t, const int *e, int stride, int edges, int node,
                                          int label) {
    int lo = t[3 * stride + node], hi = t[4 * stride + node], end = hi;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (e[m] < label)
            lo = m + 1;
        else
            hi = m;
    }
    return lo < end && e[lo] == label ? e[edges + lo] : -1;
}

struct TaskWriter {
    int *marks, *slots, *ids;
    float *weights;
    int epoch, count = 0;
    __device__ __forceinline__ void add(int id, float weight) {
        if (weight == 0.f)
            return;
        if (marks[id] != epoch) {
            marks[id] = epoch;
            slots[id] = count;
            ids[count] = id;
            weights[count++] = weight;
        } else
            weights[slots[id]] += weight;
    }
};

// Euler intervals remove the dynamic DFS stack. Materialized subtrees are
// emitted once and skipped. Epoch/slot arrays coalesce overlapping ranges.
__device__ __forceinline__ void range_tasks(const int *t, int stride, int root, int capacity, float weight,
                                            TaskWriter &w) {
    if (weight == 0.f)
        return;
    int begin = t[5 * stride + root], end = t[6 * stride + root];
    while (begin < end) {
        int2 entry = ((const int2 *)(t + 11 * stride))[begin];
        if (entry.x >= 0)
            w.add(entry.x, weight);
        begin = entry.y;
    }
}

template <class T>
__global__ void gpu_planner_kernel(GpuPlannerArgs a, const int *labels, const T *sk, const T *sq,
                                   const T *v) {
    int h = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int *state = a.state + h * 8;
    int pos = state[0];
    // Graph callers must rebuild outside capture before exhausting the horizon.
    // Errors never write history/KV out of bounds and remain sticky for checking.
    if (state[6] || pos >= a.capacity || pos - a.snapshot >= a.interval ||
        (labels[2 * a.heads + h] != 0 && labels[2 * a.heads + h] != 1)) {
        if (tid == 0)
            state[6] = 1;
        for (int i = tid; i < a.d; i += blockDim.x)
            a.output[h * a.d + i] = __int_as_float(0x7fc00000);
        return;
    }
    int query = labels[a.heads + h], key = labels[h], reset = labels[2 * a.heads + h];
    T *keys = (T *)a.keys + int64_t(h) * a.capacity * a.r;
    T *values = (T *)a.values + int64_t(h) * a.capacity * a.d;
    for (int i = tid; i < a.r; i += blockDim.x)
        keys[int64_t(pos) * a.r + i] = sk[h * a.r + i];
    for (int i = tid; i < a.d; i += blockDim.x)
        values[int64_t(pos) * a.d + i] = v[h * a.d + i];
    if (tid < 3)
        a.history[pos * 3 * a.heads + tid * a.heads + h] = labels[tid * a.heads + h];
    const int *old_band = a.band + (h * 2 + (pos & 1)) * a.interval;
    int *next_band = a.band + (h * 2 + ((pos + 1) & 1)) * a.interval;
    int start = max(0, pos + 1 - a.interval), maximum = 0;
    for (int p = start + tid; p <= pos; p += blockDim.x) {
        int symbol = p == pos ? key : a.history[p * 3 * a.heads + h];
        int length = symbol == query ? 1 + ((!reset && p > 0) ? old_band[(p - 1) % a.interval] : 0) : 0;
        next_band[p % a.interval] = length;
        if (p >= a.snapshot)
            maximum = max(maximum, length);
    }
    for (int shift = 16; shift; shift /= 2)
        maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, shift));
    // Necessary warp communication only; no shared staging of traversal nodes.
    __shared__ int warp_max[4], match_length, maximum_length, task_count;
    __shared__ float partial[4][65];
    if (lane == 0)
        warp_max[warp] = maximum;
    const int *t = a.topology + h * 13 * a.nodes, *edges = a.edges + h * 2 * a.edge_capacity;
    int *ids = a.tasks + int64_t(h) * a.task_capacity;
    float *coeff = a.coefficients + int64_t(h) * a.task_capacity;
    float tau = a.tau[h];
    if (tid == 0) {
        int node = reset ? 0 : state[1], length = reset ? 0 : state[2];
        int next = transition(t, edges, a.nodes, a.edge_capacity, node, query);
        while (node && next < 0) {
            node = t[node];
            length = min(length, t[a.nodes + node]);
            next = transition(t, edges, a.nodes, a.edge_capacity, node, query);
        }
        if (next < 0) {
            node = 0;
            length = 0;
        } else {
            node = next;
            ++length;
        }
        state[1] = node;
        state[2] = length;
        match_length = length;
        if (node != state[3] || length != state[4]) {
            TaskWriter w{a.marks + int64_t(h) * a.task_capacity, a.slots + int64_t(h) * a.task_capacity, ids,
                         coeff, pos};
            if (length) {
                int parent = t[node], sample = t[8 * a.nodes + parent];
                if (sample)
                    w.add(a.capacity + t[9 * a.nodes + sample], expf(tau * (t[a.nodes + sample] - length)));
                for (int u = parent; u != sample; u = t[u]) {
                    int gap = t[a.nodes + u] - t[a.nodes + t[u]];
                    range_tasks(t, a.nodes, u, a.capacity,
                                expf(tau * (t[a.nodes + u] - length)) * geometric_weight(gap, tau), w);
                }
                range_tasks(t, a.nodes, node, a.capacity, geometric_weight(length - t[a.nodes + parent], tau),
                            w);
            }
            state[3] = node;
            state[4] = length;
            state[5] = w.count;
        }
        task_count = state[5];
    }
    __syncthreads();
    if (tid == 0)
        maximum_length = max(match_length, max(max(warp_max[0], warp_max[1]), max(warp_max[2], warp_max[3])));
    __syncthreads();
    float scale = expf(tau * (match_length - maximum_length));
    float sum0 = 0.f, sum1 = 0.f, den = 0.f;
    int total = task_count + pos + 1 - a.snapshot;
    for (int i = warp; i < total; i += 4) {
        int id;
        float weight;
        if (i < task_count) {
            id = ids[i];
            weight = coeff[i] * scale;
        } else {
            id = a.snapshot + i - task_count;
            int length = next_band[id % a.interval];
            weight = length ? expf(tau * (length - maximum_length)) * geometric_weight(length, tau) : 0.f;
        }
        if (weight == 0.f)
            continue;
        if (id < a.capacity) {
            float dot =
                lane < a.r ? scalar(keys[int64_t(id) * a.r + lane]) * scalar(sq[h * a.r + lane]) : 0.f;
            for (int shift = 16; shift; shift /= 2)
                dot += __shfl_down_sync(0xffffffff, dot, shift);
            dot = __shfl_sync(0xffffffff, dot, 0) * weight;
            sum0 = fmaf(dot, scalar(values[int64_t(id) * a.d + lane]), sum0);
            if (a.d == 64)
                sum1 = fmaf(dot, scalar(values[int64_t(id) * a.d + lane + 32]), sum1);
            den += weight;
        } else {
            int matrix = id - a.capacity;
            const float *p = a.summaries + int64_t(matrix) * a.r * a.d;
            float x0 = 0.f, x1 = 0.f;
            for (int j = 0; j < a.r; ++j) {
                float q = scalar(sq[h * a.r + j]);
                x0 = fmaf(q, p[j * a.d + lane], x0);
                if (a.d == 64)
                    x1 = fmaf(q, p[j * a.d + lane + 32], x1);
            }
            sum0 = fmaf(weight, x0, sum0);
            sum1 = fmaf(weight, x1, sum1);
            den = fmaf(weight, a.counts[matrix], den);
        }
    }
    partial[warp][lane] = sum0;
    if (a.d == 64)
        partial[warp][lane + 32] = sum1;
    if (lane == 0)
        partial[warp][64] = den;
    __syncthreads();
    if (warp == 0) {
        float denominator = expf(-tau * maximum_length);
        sum0 = sum1 = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            sum0 += partial[w][lane];
            if (a.d == 64)
                sum1 += partial[w][lane + 32];
            denominator += partial[w][64];
        }
        float inverse = __fdividef(1.f, denominator);
        a.output[h * a.d + lane] = sum0 * inverse;
        if (a.d == 64)
            a.output[h * a.d + lane + 32] = sum1 * inverse;
    }
    if (tid == 0)
        state[0] = pos + 1;
}

extern "C" void launch_gpu_planner(GpuPlannerArgs a, const int *labels, const void *sk, const void *sq,
                                   const void *v, bool bf16, cudaStream_t stream) {
    if (bf16)
        gpu_planner_kernel<<<a.heads, 128, 0, stream>>>(a, labels, (const __nv_bfloat16 *)sk,
                                                        (const __nv_bfloat16 *)sq, (const __nv_bfloat16 *)v);
    else
        gpu_planner_kernel<<<a.heads, 128, 0, stream>>>(a, labels, (const float *)sk, (const float *)sq,
                                                        (const float *)v);
}
