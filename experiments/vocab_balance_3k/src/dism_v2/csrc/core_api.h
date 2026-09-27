#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace dism_v2 {
struct Args {
    const void *a, *b, *v;
    const float *lse, *tau;
    const int64_t *q_label, *k_label;
    float *summary, *boundary, *normalizer;
    void* output;
    int batch_heads, heads, n, padded_n, checkpoints;
    float scale;
    bool column_lse;
    float hard_prob;
    uint64_t seed, offset;
    float *vertical = nullptr, *horizontal = nullptr;
    const uint32_t* hard_bits = nullptr;
    bool label32 = false;
    __host__ __device__ __forceinline__ int64_t query_label(int64_t i) const {
        return label32?reinterpret_cast<const int32_t*>(q_label)[i]:q_label[i];
    }
    __host__ __device__ __forceinline__ int64_t key_label(int64_t i) const {
        return label32?reinterpret_cast<const int32_t*>(k_label)[i]:k_label[i];
    }
};
void launch_summary(const Args&, int d, cudaStream_t);
void launch_passing(const Args&, cudaStream_t);
void launch_output(const Args&, int d, int dv, cudaStream_t);
} // namespace dism_v2
