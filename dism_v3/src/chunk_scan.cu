#include <climits>
#include <kittens.cuh>

#include "variant.cuh"
namespace DISM_VARIANT {

namespace {
constexpr int CHECKPOINT_ROWS = 32;
constexpr int WORKLOAD_ROWS = 256;
constexpr float LOG_ZERO = -1e6f;
constexpr float LOG2E = 1.4426950408889634f;
using Global = kittens::gl<float, -1, -1, -1, -1>;

__device__ __forceinline__ float logadd2_full(float x, float y) {
    float exponential;
    float distance = -fabsf(x - y);
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(distance));
    return fmaxf(x, y) + log1pf(exponential) * LOG2E;
}

// Explicit async input staging requested for chunk passing. Each warp owns its
// ring slots; warp waits order the eight copying lanes before all32 readers.
// Slots hold inputs only, never recurrence intermediates.
template <int Pending>
__device__ __forceinline__ void wait_input(int remaining) {
    if constexpr (Pending == 0) {
        kittens::warp::load_async_wait<0>();
    } else {
        if (remaining > Pending) kittens::warp::load_async_wait<Pending>();
        else wait_input<Pending - 1>(remaining);
    }
}

template<int Stages, class Configuration = ActiveConfig>
__global__ void chunk_scan_async_kernel(Global a, Global b,
                                        float* boundary, int checkpoints, int padded_n) {
    __shared__ kittens::sv_fl<32> staged_a[4][Stages];
    __shared__ kittens::sv_fl<32> staged_b[4][Stages];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int diagonal = int(blockIdx.x * blockDim.x) + warp * 32
                   - (checkpoints - 1) * CHECKPOINT_ROWS;
    if (diagonal >= padded_n) return; // Entire warp is inactive.
    int64_t base = int64_t(blockIdx.y) * checkpoints * padded_n;

    // Entire noncausal warps initialize output without touching summary inputs.
    if (diagonal > 0) {
        int end = min(checkpoints, (padded_n - diagonal) / CHECKPOINT_ROWS);
#pragma unroll 1
        for (int checkpoint = 0; checkpoint < end; ++checkpoint) {
            int column = diagonal + checkpoint * CHECKPOINT_ROWS + lane;
            boundary[base + int64_t(checkpoint) * padded_n + column] = LOG_ZERO;
        }
        return;
    }

    int first = -diagonal / CHECKPOINT_ROWS;
    auto prefetch = [&](int slot, int checkpoint) {
        int column = diagonal + checkpoint * CHECKPOINT_ROWS;
        kittens::warp::load_async(staged_a[warp][slot], a, {int(blockIdx.y), 0, checkpoint, column});
        kittens::warp::load_async(staged_b[warp][slot], b, {int(blockIdx.y), 0, checkpoint, column});
        kittens::warp::load_async_commit_group();
    };
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
        if (first + stage < checkpoints) prefetch(stage, first + stage);
    }
    float state = LOG_ZERO;
    int slot = 0;
#pragma unroll 1
    for (int checkpoint = first; checkpoint < checkpoints; ++checkpoint) {
        wait_input<Stages - 1>(checkpoints - checkpoint);
        float affine_a = staged_a[warp][slot].data[lane];
        float affine_b = staged_b[warp][slot].data[lane];
        // All readers must finish before refilling this slot Stages steps ahead.
        // No CTA-wide synchronization is needed.
        kittens::warp::sync();
        if (checkpoint + Stages < checkpoints) prefetch(slot, checkpoint + Stages);
        slot = (slot + 1) % Stages;
        state = logadd2_full(state + affine_a, affine_b);
        int column = diagonal + checkpoint * CHECKPOINT_ROWS + lane;
        boundary[base + int64_t(checkpoint) * padded_n + column] = state;
    }
}
} // namespace

const void* chunk_scan_address_0() { return reinterpret_cast<const void*>(chunk_scan_async_kernel<4>); }
} // namespace DISM_VARIANT
