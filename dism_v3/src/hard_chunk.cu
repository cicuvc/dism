#include "common/graph_route.cuh"
#include <kittens.cuh>
#include "summary/hard_chunk.cuh"

namespace DISM_VARIANT {
namespace {
template<int Pending>
__device__ __forceinline__ void wait_hard_chunk(int remaining) {
    if constexpr (Pending == 0) kittens::warp::load_async_wait<0>();
    else {
        if (remaining > Pending) kittens::warp::load_async_wait<Pending>();
        else wait_hard_chunk<Pending - 1>(remaining);
    }
}
} // namespace

// Same diagonal ownership and four-stage, warp-local async input ring as soft
// chunk passing. Only one int32 state and one staged input replace affine A/B.
template<class Configuration = ActiveConfig>
__global__ __launch_bounds__(128) void hard_chunk_kernel(
        const __grid_constant__ HardChunkArgs args, const uint32_t* choice) {
    if (!dism_graph_route::quadratic(choice)) return;
    constexpr int Stages = 4;
    __shared__ __align__(16) int staged[4][Stages][32];
    int n = args.n, diagonal_block = blockIdx.x;
    int64_t base;
    if (args.tasks) {
        auto task = args.tasks[blockIdx.x];
        n = task.n;
        diagonal_block = task.diagonal_block;
        base = task.offset;
    } else {
        base = int64_t(blockIdx.y) * ((n - 1) / 32) * n;
    }
    int checkpoints = (n - 1) / 32;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int diagonal = diagonal_block * 128 + warp * 32 - (checkpoints - 1) * 32;
    if (diagonal >= n) return; // warp-uniform
    if (diagonal > 0) {
        // Fully initialize unused entries without reading noncausal inputs.
        int end = min(checkpoints, (n - diagonal) / 32);
        for (int c = 0; c < end; ++c)
            args.boundary[base + int64_t(c) * n + diagonal + c * 32 + lane] = -1;
        return;
    }
    int first = -diagonal / 32;
    auto prefetch = [&](int slot, int c) {
        if (lane < 8) {
            uint32_t dst = uint32_t(__cvta_generic_to_shared(&staged[warp][slot][lane * 4]));
            const int* src = args.summary + base + int64_t(c) * n + diagonal + c * 32 + lane * 4;
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                         :: "r"(dst), "l"(src) : "memory");
        }
        kittens::warp::load_async_commit_group();
    };
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage)
        if (first + stage < checkpoints) prefetch(stage, first + stage);
    int state = -1, slot = 0;
    for (int c = first; c < checkpoints; ++c) {
        wait_hard_chunk<Stages - 1>(checkpoints - c);
        int local = staged[warp][slot][lane];
        kittens::warp::sync(); // all readers finish before slot reuse
        if (c + Stages < checkpoints) prefetch(slot, c + Stages);
        slot = (slot + 1) % Stages;
        state = max(state, local);
        args.boundary[base + int64_t(c) * n + diagonal + c * 32 + lane] = state;
    }
}

const void* hard_chunk_address() { return reinterpret_cast<const void*>(hard_chunk_kernel<ActiveConfig>); }
} // namespace DISM_VARIANT
