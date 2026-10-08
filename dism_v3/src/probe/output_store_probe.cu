#include "forward/types.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace {
constexpr int DIM = dism_forward::DIM;

template<bool FP32Output>
__global__ void output_store_probe_kernel(const __grid_constant__ dism_forward::OutputGlobal<FP32Output> output) {
    __shared__ dism_forward::OutputTile<FP32Output> tiles[8];
    int warp = kt::warpid();
    int lane = kt::warp::laneid();
    int logical_warp = 2 * (warp % 4) + warp / 4;
    int blocks = (output.rows()+127)/128;
    int tasks = blocks * output.depth() * output.batch();
#pragma unroll 1
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        int bh = task / blocks;
        int head = bh % output.depth();
        int batch = bh / output.depth();
        int start = task % blocks * 128;
        kt::rt_fl<16,DIM> values;
#pragma unroll
        for (int c = 0; c < DIM / 16; ++c) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    int row = start + logical_warp * 16 + lane / 4 + 8 * r;
                    int col = c * 16 + half * 8 + 2 * (lane % 4);
                    int index = (bh * output.rows() + row) * DIM + col;
                    values.tiles[0][c].data[r + 2 * half] =
                        {float(index % 251 - 125), float((index + 1) % 251 - 125)};
                }
            }
        }
        kt::warp::store(tiles[warp],values);
        __syncwarp();
        kt::warp::store(output,tiles[warp],{batch,head,start+logical_warp*16,0});
        __syncwarp();
    }
}
}

template<bool FP32> const void* output_probe_address() { return reinterpret_cast<const void*>(output_store_probe_kernel<FP32>); }
template const void* output_probe_address<false>();
#if DISM_ENABLE_FP32
template const void* output_probe_address<true>();
#endif
} // namespace DISM_VARIANT
