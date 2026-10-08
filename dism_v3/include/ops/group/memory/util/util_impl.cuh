#pragma once

#include "../../group.cuh"

using kittens::group;
using kittens::semaphore;

/**
 * @brief Commits all preceding cp.async loads issued by each thread as one group.
 */
template <int GW>
__device__ inline void group<GW>::load_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int GW>
__device__ inline void group<GW>::load_async_commit_group(semaphore& sem) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
    asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n" ::"r"(addr): "memory");
}

template <int GW>
template <int N>
__device__ inline void group<GW>::load_async_wait(int bar_id) { // for completing (non-TMA) async loads
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
    sync(bar_id);
}

template <int GW>
template <int N>
__device__ inline void group<GW>::load_async_wait() { // for completing (non-TMA) async loads
    KITTENS_CHECK_WARP
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
    __syncwarp();
}

template <int GW>
__device__ inline void group<GW>::arrive(barrier<GROUP_WARPS> bar) {
    asm volatile("bar.arrive %0, %1;\n" ::"r"(bar.barrier_id), "n"(GROUP_WARPS * WARP_THREADS) : "memory");
}

template <int GW>
__device__ inline void group<GW>::arrive_and_wait(barrier<GROUP_WARPS> bar) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(bar.barrier_id), "n"(GROUP_WARPS * WARP_THREADS) : "memory");
}

template <int GW>
__device__ inline void group<GW>::init_semaphore(semaphore &bar, int thread_count, int transaction_count) {
    void const *const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    if (elect_leader()) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr), "r"(thread_count + transaction_count));
    }
}
/**
 * @brief Invalidate an mbarrier
 *
 * @param[out] semaphore The semaphore variable to initialize.
 * @param[in] tc The thread counter for the semaphore.
 */
template <int GW>
__device__ inline void group<GW>::invalidate_semaphore(semaphore &bar) {
    if (laneid() == 0) {
        void const *const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" ::"r"(bar_ptr));
    }
}
template <int GW>
__device__ inline void group<GW>::arrive(semaphore &sem) {
    if (laneid() == 0) {
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" : : "r"(mbar_ptr) : "memory");
    }
}
template <int GW>
template <int num_warps>
__device__ inline void group<GW>::arrive(barrier<num_warps> bar) {
    asm volatile("bar.arrive %0, %1;\n" ::"r"(bar.barrier_id), "n"(num_warps * WARP_THREADS) : "memory");
}

#if (defined(KITTENS_FEATURE_MBARRIER))
template <int GW>
__device__ inline void group<GW>::arrive(semaphore &sem, uint32_t count) {
    if (elect_leader()) {
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
                     :
                     : "r"(mbar_ptr), "r"(count)
                     : "memory");
    }
}
#endif

template <int GW>
__device__ inline void group<GW>::wait(semaphore &sem, int kPhaseBit) {
    void const *const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

#if (defined(KITTENS_FEATURE_MBARRIER))
    asm volatile("{\n"
                 ".reg .pred                P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1                       bra.uni DONE;\n"
                 "bra.uni                   LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n" ::"r"(mbar_ptr),
                 "r"(kPhaseBit));
#else
    asm volatile("{\n"
                 ".reg .pred                P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1                       bra.uni DONE;\n"
                 "nanosleep.u32 5;\n" // wait a few nanoseconds on pre-Hopper
                                      // architectures to save instruction issue slots
                 "bra.uni                   LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n" ::"r"(mbar_ptr),
                 "r"(kPhaseBit));
#endif
}

template <int GW>
__device__ inline int group<GW>::test_wait(semaphore &sem, int kPhaseBit) {
    void const *const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    int result;
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
                 "selp.u32 %0,1,0,P1;"
                 "}\n"
                 : "=r"(result)
                 : "r"(mbar_ptr), "r"(kPhaseBit));
    return result;
}
