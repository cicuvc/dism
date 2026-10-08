#include "summary/primitives.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

// Isolate cp.async -> ready -> shared read -> free, without TMA, MMA, scan,
// slot unions or register redistribution. Use the production pipe primitive.
struct MetadataProbeStorage {
    int values[1][256];
};

__global__ void metadata_pipe_probe_kernel(const int *source, int *output,
                                          int tasks, bool local_wait, bool unit_arrivals) {
    extern __shared__ int scratch[];
    kt::shared_allocator<128> allocator{scratch};
    int group = kt::warpgroup::groupid();
    MbarrierRingPipe pipe(allocator, BufferSet{&MetadataProbeStorage::values},
                         group, ProducerWarp<true, 2>,
                         ConsumerWarpGroup<false, 0, 1>);
    // Diagnostic alternative: ready32/free256 instead of weighted ready256.
    if (unit_arrivals) {
        if (kt::warpid() == 0 && kt::warp::elect_leader()) {
            kt::init_semaphore(pipe.Barriers[1][0], 32);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();
        pipe.ArriveCount = 1;
    }
    __syncthreads();
    auto &storage = allocator.allocate<MetadataProbeStorage>();
    if (group == 2) {
        if (kt::warpgroup::warpid() == 0) {
            for (int task = 0; task < tasks; ++task) {
                auto packet = pipe.waitBuffer(0, storage.values);

                load_async_any(packet.template get<0>(),
                               reinterpret_cast<uint8_t *>(
                                   const_cast<int *>(source + task * 256)));
                if (local_wait) {
                    kt::warp::load_async_commit_group();
                    kt::warp::load_async_wait<0>();
                }
                kt::warp::load_async_commit_group(packet.getBarrier());

                packet.submitToNextAndTrigger();
                pipe.moveNext();
            }
            pipe.waitSlot(0);
        }
    } else {
        for (int task = 0; task < tasks; ++task) {
            auto packet = pipe.waitBuffer(2, storage.values);
            int value = packet.template get<0>()[threadIdx.x];
            kt::warpgroup::sync(group + 1);
            packet.submitToNextAndTrigger();
            pipe.moveNext();
            output[task * 256 + threadIdx.x] = value;
        }
    }
    kt::warpgroup::sync(group + 1);
}



const void* metadata_pipe_probe_kernel_address0() { return reinterpret_cast<const void*>(metadata_pipe_probe_kernel); }

} // namespace DISM_VARIANT
