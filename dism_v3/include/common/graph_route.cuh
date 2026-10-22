#pragma once
#include <cstdint>

namespace dism_graph_route {
// Every CTA acquires the immutable decision before any indirect read or TMA.
// nullptr preserves eager execution. Only the sampler writes this word, once
// per replay; never pass the concurrently changing ready_epoch here.
__device__ __forceinline__ bool quadratic(const uint32_t* choice) {
    if (!choice) return true;
    uint32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(choice) : "memory");
    return value == 0;
}
}
