#pragma once

#include "kittens.cuh"

namespace fp8gemm {

__device__ __forceinline__ float softcap_tanh(float x, float s) {
    float result;
    // Volatile keeps the wide-tile sum pass from being CSE'd into a second
    // live FP32 tile, which would spill alongside the retained raw logits.
    asm volatile("{ .reg .f32 ratio; div.rn.f32 ratio, %1, %2; "
                 "tanh.approx.f32 %0, ratio; }"
                 : "=f"(result) : "f"(x), "f"(s));
    return result;
}

// Keep raw BF16 logits for CE's chain rule. The optional FP32 transform is
// local to the reduction; no extra global tensor or logits reload is needed.
template<int R, int C, bool Softcap=false>
__device__ __forceinline__ void store_partial_lse(
    kt::rt_fl<R,C> &z, float *lse, int mr, int nc, int m, int n, float softcap) {
    const int lane = kt::laneid(), parts = (n+C-1)/C;
    __syncwarp();
#pragma unroll
    for (int i=0; i<R/16; ++i) {
        float maxima[2] = {-INFINITY, -INFINITY};
        // A 256-column tile cannot keep both raw and transformed tiles live
        // within the register budget. Recompute tanh in its sum pass instead.
        float2 capped[C <= 128 ? C/16 : 1][4];
#pragma unroll
        for (int j=0; j<C/16; ++j) {
#pragma unroll
            for (int p=0; p<4; ++p) {
                auto &v = z.tiles[i][j].data[p];
                v.x = __bfloat162float(__float2bfloat16_rn(v.x));
                v.y = __bfloat162float(__float2bfloat16_rn(v.y));
                float2 value = v;
                if constexpr(Softcap) {
                    value.x = softcap * softcap_tanh(v.x, softcap);
                    value.y = softcap * softcap_tanh(v.y, softcap);
                    if constexpr(C <= 128) capped[j][p] = value;
                }
                int col = nc+j*16+(lane%4)*2+(p/2)*8;
                maxima[p%2] = fmaxf(maxima[p%2], col<n ? value.x : -INFINITY);
                maxima[p%2] = fmaxf(maxima[p%2], col+1<n ? value.y : -INFINITY);
            }
        }
#pragma unroll
        for (int r=0; r<2; ++r) {
            maxima[r] = fmaxf(maxima[r], __shfl_xor_sync(0xffffffff, maxima[r], 1));
            maxima[r] = fmaxf(maxima[r], __shfl_xor_sync(0xffffffff, maxima[r], 2));
        }
        float sums[2] = {0.f, 0.f};
#pragma unroll
        for (int j=0; j<C/16; ++j) {
#pragma unroll
            for (int p=0; p<4; ++p) {
                auto v = z.tiles[i][j].data[p];
                if constexpr(Softcap) {
                    if constexpr(C <= 128) v = capped[j][p];
                    else {
                        v.x = softcap * softcap_tanh(v.x, softcap);
                        v.y = softcap * softcap_tanh(v.y, softcap);
                    }
                }
                int col = nc+j*16+(lane%4)*2+(p/2)*8;
                sums[p%2] += col<n ? __expf(v.x-maxima[p%2]) : 0.f;
                sums[p%2] += col+1<n ? __expf(v.y-maxima[p%2]) : 0.f;
            }
        }
#pragma unroll
        for (int r=0; r<2; ++r) {
            sums[r] += __shfl_xor_sync(0xffffffff, sums[r], 1);
            sums[r] += __shfl_xor_sync(0xffffffff, sums[r], 2);
            int row = mr+i*16+lane/4+r*8;
            if (lane%4==0 && row<m)
                lse[size_t(row)*parts+nc/C] = maxima[r]+__logf(sums[r]);
        }
    }
}
} // namespace fp8gemm
