#include "summary/primitives.cuh"
#include "summary/key_metadata.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {



__global__ void key_metadata_probe_kernel(
    decltype(TmaSummarizationKernelArgs::IdxK) input_labels,
    decltype(TmaSummarizationKernelArgs::KLseVec) input_bias,
    int* output_labels, float* output_bias, bool query_lse) {
    __shared__ LoadSharedMemoryLayouts::VectorRegion::DefaultLayout shared;
    load_key_vector_async(shared.k_idx, input_labels, {0, 0, 0, 0});
    if (!query_lse) load_key_vector_async(shared.klse, input_bias, {0, 0, 0, 0});
    kt::load_async_commit_group();
    kt::load_async_wait();
    kt::rv<int, 64, kt::ducks::rv_layout::align> labels;
    kt::rv_fl<64, kt::ducks::rv_layout::align> bias;
    kt::warp::rv_maps::zero(bias);
    load_key_metadata(labels, bias, shared, query_lse);
#pragma unroll
    for (int block = 0; block < 4; ++block) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int offset = kt::warp::laneid() * 16 + 4 * block + 2 * half;
            output_labels[offset] = labels.data[block][half].x;
            output_labels[offset + 1] = labels.data[block][half].y;
            output_bias[offset] = bias.data[block][half].x;
            output_bias[offset + 1] = bias.data[block][half].y;
        }
    }
}




 // Only the legacy layout remains.




__global__ void lse_probe_kernel(const float2* inputs, float* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = LogAffineOp::softplus_approx(inputs[i].x, inputs[i].y);
}



// A/B for Clang's lowering of TK's runtime-indexed rv::load. Both kernels
// perform the same shared->register->global transfer, independent of scan.
template<bool DIRECT>
__global__ void rv_load_probe_kernel(const float* source, float* output) {
    __shared__ kt::sv_fl<32> shared;
    int lane = kt::warp::laneid();
    shared.data[lane] = source[lane];
    __syncthreads();
    kt::rv_fl<32, kt::ducks::rv_layout::ortho> values;
    if constexpr (DIRECT) {
        #pragma unroll
        for (int block = 0; block < 2; ++block) {
            values.data[block][0].x = shared.data[block * 16 + lane / 4];
            values.data[block][0].y = shared.data[block * 16 + lane / 4 + 8];
        }
    } else {
        kt::warp::load(values, shared);
    }
    if (lane % 4 == 0) {
        #pragma unroll
        for (int block = 0; block < 2; ++block) {
            output[block * 16 + lane / 4] = values.data[block][0].x;
            output[block * 16 + lane / 4 + 8] = values.data[block][0].y;
        }
    }
}



// Independent probes: no GEMM/TMA in scan_probe and no scan/GEMM in pipe_probe.
// These make failures attributable before composing the full pipeline.
__global__ void scan_probe_kernel(const float *values, float2 *output, int columns) {
    using Scalar = pscore::AltLayoutSplitScanBuffer<32, 64, pscore::UnaryElement>;
    using Scan =
        pscore::AltLayoutSplitScanBuffer<32, 64, pscore::BinaryElement, LogAffineOp>;
    typename Scan::VState left;
    int lane = kt::warp::laneid();
#pragma unroll 1
    for (int start = 0; start < columns; start += 64) {
        Scalar scalar;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                int row = r * 8 + lane / 4;
                int col = start + c + 16 * (lane & 3);
                scalar.data[r][c].value = {values[row * columns + col],
                                           values[row * columns + col + 8]};
            }
        }
        scalar.roll();
        Scan scan;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                scan.data[r][c] = {scalar.data[r][c].value, scalar.data[r][c].value};
            }
        }
        auto result = scan.reduce_forward(left, {});
        left = result.first;
        int col = start + 16 * (lane & 3) + 7 - lane / 4;
        auto bottom = result.second.init[0];
        output[col] = make_float2(bottom.first.u0, bottom.second.u0);
        output[col + 8] = make_float2(bottom.first.u1, bottom.second.u1);
    }
}



struct PipeProbeStorage {
    int4 slots[3];
    int metadata[1];
};

__global__ void pipe_probe_kernel(const int *source, int *output, int tasks) {
    extern __shared__ int probe_smem[];
    kt::shared_allocator<128> alloc{probe_smem};
    int group = kt::warpgroup::groupid();
    MbarrierRingPipe pipe(alloc, BufferSet{&PipeProbeStorage::slots}, group, ProducerWarp<true, 2>,
                          ConsumerWarpGroup<false, 0, 1>);
    MbarrierRingPipe prefetch(alloc, BufferSet{&PipeProbeStorage::metadata}, group,
                              ProducerWarp<true, 2>, ConsumerWarpGroup<false, 0, 1>);
    auto &data = alloc.allocate<PipeProbeStorage>();
    bool leader = kt::warp::elect_leader();
    if (group == 2) {
        if (kt::warpgroup::warpid() == 0) {
            for (int task = 0; task < tasks; ++task) {
                prefetch.setup();
                auto first = prefetch.waitBuffer(0, data.metadata);
                if (leader)
                    first.template get<0>() = task;
                first.submitToNextAndTrigger();
                prefetch.moveNext();
                prefetch.waitSlot(0);
                pipe.setup();
                int last = 0;
                for (int tile = 0; tile <= task % 7; ++tile) {
                    auto packet = pipe.waitBuffer(0, data.slots);
                    load_async_any(packet.template get<0>(),
                                   reinterpret_cast<uint8_t *>(
                                       const_cast<int *>(source + (task * 7 + tile) * 4)));
                    kt::warp::load_async_commit_group(packet.getBarrier());
                    last = packet.SlotIdx;
                    packet.submitToNextAndTrigger();
                    pipe.moveNext();
                }
                pipe.waitSlot(last);
            }
        }
    } else {
        for (int task = 0; task < tasks; ++task) {
            prefetch.setup();
            auto first = prefetch.waitBuffer(2, data.metadata);
            int metadata = first.template get<0>();
            kt::warpgroup::sync(group + 1);
            first.submitToNextAndTrigger();
            prefetch.moveNext();
            pipe.setup();
            for (int tile = 0; tile <= task % 7; ++tile) {
                auto packet = pipe.waitBuffer(2, data.slots);
                int value = packet.template get<0>().x;
                kt::warpgroup::sync(group + 1);
                packet.submitToNextAndTrigger();
                pipe.moveNext();
                if (leader)
                    output[(task * 8 + kt::warp::groupid()) * 7 + tile] =
                        metadata == task ? value : -1;
            }
        }
    }
    kt::warpgroup::sync(group + 1);
}



const void* key_metadata_probe_kernel_address0() { return reinterpret_cast<const void*>(key_metadata_probe_kernel); }
const void* lse_probe_kernel_address0() { return reinterpret_cast<const void*>(lse_probe_kernel); }
const void* rv_load_probe_kernel_address0() { return reinterpret_cast<const void*>(rv_load_probe_kernel<true>); }
const void* rv_load_probe_kernel_address1() { return reinterpret_cast<const void*>(rv_load_probe_kernel<false>); }
const void* scan_probe_kernel_address0() { return reinterpret_cast<const void*>(scan_probe_kernel); }
const void* pipe_probe_kernel_address0() { return reinterpret_cast<const void*>(pipe_probe_kernel); }

} // namespace DISM_VARIANT
