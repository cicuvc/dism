#pragma once
#include "forward/types.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_forward {
__device__ __forceinline__ float exp2_ftz(float value) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

// Normalize P before applying signed soft readout. The fallback starts with
// maximum=0, denominator=1, output=0 and is rescaled along with all old terms.
__device__ __forceinline__ void readout(
        Scalar& scores, const kt::rt_fl<16,WarpKSize>& similarity,
        kt::rt_bf<16,WarpKSize>& weights, kt::rt_fl<16,DIM>& accum,
        float (&maximum)[2], float (&denominator)[2], int q_start, int key_start, int n) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float next_max = maximum[r];
#pragma unroll
        for (int c = 0; c < Scalar::COL_BLOCKS; ++c) {
            auto pos = Scalar::layout(r, c, 0);
            auto& value = scores.data[r][c].value;
            int i = q_start + pos.first;
            int j = key_start + pos.second;
            if (i >= n || j >= n || j > i) value.u0 = LOG_ZERO;
            if (i >= n || j + Scalar::COL_BLOCKS >= n || j + Scalar::COL_BLOCKS > i) value.u1 = LOG_ZERO;
            next_max = fmaxf(next_max, fmaxf(value.u0, value.u1));
        }
        next_max = fmaxf(next_max, __shfl_xor_sync(0xffffffff, next_max, 1));
        next_max = fmaxf(next_max, __shfl_xor_sync(0xffffffff, next_max, 2));
        float alpha = exp2_ftz(maximum[r] - next_max);
        float sum = 0.f;
#pragma unroll
        for (int c = 0; c < Scalar::COL_BLOCKS; ++c) {
            auto value = scores.data[r][c].value;
            float px = exp2_ftz(value.u0 - next_max);
            float py = exp2_ftz(value.u1 - next_max);
            sum += px + py;
            auto s = similarity.tiles[0][c / 2].data[r + 2 * (c & 1)];
            weights.tiles[0][c / 2].data[r + 2 * (c & 1)] =
                __floats2bfloat162_rn(px * s.x, py * s.y);
        }
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        sum += __shfl_xor_sync(0xffffffff, sum, 2);
        denominator[r] = denominator[r] * alpha + sum;
        maximum[r] = next_max;
#pragma unroll
        for (int c = 0; c < DIM / 16; ++c) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                auto& value = accum.tiles[0][c].data[r + 2 * half];
                value.x *= alpha;
                value.y *= alpha;
            }
        }
    }
}

template<bool PackedStatistics=false, bool FP32Output>
__device__ __forceinline__ void store_output(const ArgsT<FP32Output>& args, const Scheduler::Task& task,
        int q_start, kt::rt_fl<16,DIM>& accum, const float (&maximum)[2],
        const float (&denominator)[2], SharedT<FP32Output>& shared, int64_t statistics_offset=0) {
    int lane = kt::warp::laneid();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        int row = q_start + 8 * r + lane / 4;
        if (row < args.n) {
            // Online normalization keeps this in [1, N+1]. No exceptional or
            // subnormal reciprocal path is needed; refine the hardware result.
            float inverse;
            asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(denominator[r]));
            inverse = fmaf(inverse, fmaf(-denominator[r], inverse, 1.f), inverse);
#pragma unroll
            for (int c = 0; c < DIM / 16; ++c) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    auto& value = accum.tiles[0][c].data[r + 2 * half];
                    value.x *= inverse;
                    value.y *= inverse;
                }
            }
            if ((lane & 3) == 0) {
                int64_t index;
                if constexpr (PackedStatistics) index=statistics_offset+8*r+lane/4;
                else index=(int64_t(task.batch)*args.heads+task.head)*args.n+row;
                args.lse2[index] =
                    maximum[r] + log2f(denominator[r]);
            }
        }
    }
    auto& output = shared.output[kt::warpid()];
    kt::warp::store(output,accum);
    __syncwarp(); // Register/shared and shared/global stores use different lanes.
    kt::warp::store(args.output_map,output,{task.batch,task.head,q_start,0});
    __syncwarp(); // Finish shared reads before this warp reuses its slot.
}
} // namespace dism_forward

} // namespace DISM_VARIANT
