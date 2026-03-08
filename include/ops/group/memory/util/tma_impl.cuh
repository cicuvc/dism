#pragma once

#include "../../group.cuh"

#ifdef KITTENS_FEATURE_TMA
/* ----------   Barrier functions for async load  ---------- */

/**
 * @brief Sets the number of bytes expected at the semaphore.
 *
 * This function sets the number of bytes expected at the semaphore for the
 * first thread in the warp. It converts the semaphore pointer to a generic
 * shared memory pointer and uses an inline assembly instruction to set the
 * expected number of bytes.
 *
 * @param semaphore Reference to the semaphore variable.
 * @param bytes The number of bytes expected at the semaphore.
 */
template <int GW>
__device__ inline void group<GW>::tma::expect_bytes(semaphore &bar, uint32_t bytes) {
    if (laneid() == 0) {
        ::kittens::tma::expect_bytes(bar, bytes);
    }
}
/**
 * @brief Sets the number of bytes expected at the semaphore.
 *
 * This function sets the number of bytes expected at the mbarrier before the
 * transaction arrives.
 */
template <int GW>
template <typename T, typename... args>
__device__ inline void group<GW>::tma::expect(semaphore &bar, const T &_1, const args &..._2) {
    expect_bytes(bar, size_bytes<T, args...>);
}

/* ----------   Synchronization functions for async store  ---------- */

/**
 * @brief Commits previous asynchronous TMA stores to a group and performs them.
 */
template <int GW>
__device__ inline void group<GW>::tma::store_commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
}
/**
 * @brief Waits for previous committed TMA store groups to complete.
 *
 * @tparam N The maximum number of remaining TMA store groups. Defaults to 0.
 */
template <int GW>
template <int N>
__device__ inline void group<GW>::tma::store_async_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
}
/**
 * @brief Waits for previous committed TMA store groups to finish reading from
 * shared memory.
 *
 * @tparam N The maximum number of remaining TMA store groups. Defaults to 0.
 */
template <int GW>
template <int N>
__device__ inline void group<GW>::tma::store_async_read_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(N) : "memory");
}
#endif