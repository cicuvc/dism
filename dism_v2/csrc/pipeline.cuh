#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace dism_v2 {
__device__ inline unsigned smaddr(const void* p) { return unsigned(__cvta_generic_to_shared(p)); }
__device__ inline void init_bar(uint64_t* p, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smaddr(p)), "r"(count) : "memory");
}
__device__ inline void arrive(uint64_t* p) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smaddr(p)) : "memory");
}
__device__ inline void wait(uint64_t* p, int phase) {
    asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }"
                 :: "r"(smaddr(p)), "r"(phase) : "memory");
}
__device__ inline void expect(uint64_t* p, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smaddr(p)), "r"(bytes) : "memory");
}
__device__ inline void tma5(const CUtensorMap* map, void* dst, uint64_t* bar, int row, int segment) {
    // On our sm120a toolchain shared::cluster lowers through an external
    // call and suppresses setmaxnreg; shared::cta emits native UTMALDG.5D.
    asm volatile("cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {0, 0, 0, %3, %4}], [%2];"
                 :: "r"(smaddr(dst)), "l"(map), "r"(smaddr(bar)), "r"(row), "r"(segment) : "memory");
}
__host__ __device__ constexpr int logical_row(int physical) {
    return physical/8 + ((physical/2)&3)*8 + (physical&1)*32;
}
template<int D> CUtensorMap permuted_map(const void* p, int rows) {
    constexpr int S = D==32?32:64;
    const cuuint64_t dims[]{S,2,4,cuuint64_t(rows),D/S};
    const cuuint64_t strides[]{32*D*2,8*D*2,D*2,S*2};
    const cuuint32_t box[]{S,2,4,8,1}, elem[]{1,1,1,1,1};
    CUtensorMap map{};
    auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,5,const_cast<void*>(p),
        dims,strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,
        D==32?CU_TENSOR_MAP_SWIZZLE_64B:CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(status!=CUDA_SUCCESS) throw std::runtime_error("Dism TMA descriptor encoding failed");
    return map;
}
} // namespace dism_v2
