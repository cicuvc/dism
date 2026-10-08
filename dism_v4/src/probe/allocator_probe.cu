#include "summary/primitives.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

struct alignas(128) AllocatorProbeAligned {
    static constexpr int required_alignment = 128;
    int values[32];
};

__global__ void allocator_probe_kernel(int *output, int offset) {
    __shared__ __align__(1024) unsigned char storage[4096];
    kt::shared_allocator<16> alloc(reinterpret_cast<int *>(storage + offset));
    auto &small = alloc.allocate<-1, uint8_t, 3>();
    auto &next = alloc.allocate<-1, uint8_t, 5>();
    auto &matrix = alloc.allocate<int, 2, 3>();
    auto &aligned = alloc.allocate<64, AllocatorProbeAligned>();
    auto &wide = alloc.allocate<1024, int, 32>();
    auto &last = alloc.allocate<-1, uint8_t>();
    const int lane = threadIdx.x;
    aligned.values[lane] = lane + 100;
    wide[lane] = lane + 200;
    if (lane == 0) {
        small[0] = 7;
        next[4] = 9;
        matrix[1][2] = 11;
        last = 13;
        void *pointers[] = {small, next, matrix, &aligned, wide, &last};
        uint32_t base = uint32_t(__cvta_generic_to_shared(storage));
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            output[i] = uint32_t(__cvta_generic_to_shared(pointers[i])) - base;
            output[6 + i] = __isShared(pointers[i]);
        }
    }
    __syncthreads();
    output[12 + lane] = aligned.values[31 - lane] + wide[31 - lane];
    if (lane == 0) output[44] = small[0] + next[4] + matrix[1][2] + last;
}



const void* allocator_probe_kernel_address0() { return reinterpret_cast<const void*>(allocator_probe_kernel); }

} // namespace DISM_VARIANT
