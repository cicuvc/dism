#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

#include <cuda.h>
#include <iostream>

#ifdef KITTENS_FEATURE_TMA

namespace kittens {
namespace tma {

namespace atoms {

// Prefetch atom
template <cache_policy policy> __device__ static inline void prefetch_atom(uint64_t tma_ptr, const int5 &tma_coords) {
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile"
                     " [%0, {%1, %2, %3, %4, %5}];"
                     :
                     : "l"(tma_ptr), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "r"(tma_coords.h)
                     : "memory");
    } else {
        asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint"
                     " [%0, {%1, %2, %3, %4, %5}], %6;"
                     :
                     : "l"(tma_ptr), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "r"(tma_coords.h),
                       "l"(make_cache_policy<policy>())
                     : "memory");
    }
}

// Store async atom
template <cache_policy policy>
__device__ static inline void store_async_atom(uint64_t tma_ptr, uint32_t src_ptr, const int5 &tma_coords) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
                     " [%0, {%2, %3, %4, %5, %6}], [%1];"
                     :
                     : "l"(tma_ptr), "r"(src_ptr), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z),
                       "r"(tma_coords.w), "r"(tma_coords.h)
                     : "memory");
    } else {
        asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_"
                     "group.L2::cache_hint"
                     " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                     :
                     : "l"(tma_ptr), "r"(src_ptr), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z),
                       "r"(tma_coords.w), "r"(tma_coords.h), "l"(make_cache_policy<policy>())
                     : "memory");
    }
}

// Store add async atom
template <cache_policy policy>
__device__ static inline void store_add_async_atom(uint64_t tma_ptr, uint32_t src_ptr, const int5 &tma_coords) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile("cp.reduce.async.bulk.tensor.5d.global.shared::cta.add."
                     "tile.bulk_group"
                     " [%0, {%2, %3, %4, %5, %6}], [%1];"
                     :
                     : "l"(tma_ptr), "r"(src_ptr), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z),
                       "r"(tma_coords.w), "r"(tma_coords.h)
                     : "memory");
    } else {
        asm volatile("cp.reduce.async.bulk.tensor.5d.global.shared::cta.add."
                     "tile.bulk_group.L2::cache_hint"
                     " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                     :
                     : "l"(tma_ptr), "r"(src_ptr), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z),
                       "r"(tma_coords.w), "r"(tma_coords.h), "l"(make_cache_policy<policy>())
                     : "memory");
    }
}

// Store min async atom
template <cache_policy policy>
__device__ static inline void store_min_async_atom(uint64_t tma_ptr, uint32_t src_ptr, const int5 &tma_coords) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile("cp.reduce.async.bulk.tensor.5d.global.shared::cta.min."
                     "tile.bulk_group"
                     " [%0, {%2, %3, %4, %5, %6}], [%1];"
                     :
                     : "l"(tma_ptr), "r"(src_ptr), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z),
                       "r"(tma_coords.w), "r"(tma_coords.h)
                     : "memory");
    } else {
        asm volatile("cp.reduce.async.bulk.tensor.5d.global.shared::cta.min."
                     "tile.bulk_group.L2::cache_hint"
                     " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                     :
                     : "l"(tma_ptr), "r"(src_ptr), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z),
                       "r"(tma_coords.w), "r"(tma_coords.h), "l"(make_cache_policy<policy>())
                     : "memory");
    }
}

// Store max async atom
template <cache_policy policy>
__device__ static inline void store_max_async_atom(uint64_t tma_ptr, uint32_t src_ptr, const int5 &tma_coords) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile("cp.reduce.async.bulk.tensor.5d.global.shared::cta.max."
                     "tile.bulk_group"
                     " [%0, {%2, %3, %4, %5, %6}], [%1];"
                     :
                     : "l"(tma_ptr), "r"(src_ptr), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z),
                       "r"(tma_coords.w), "r"(tma_coords.h)
                     : "memory");
    } else {
        asm volatile("cp.reduce.async.bulk.tensor.5d.global.shared::cta.max."
                     "tile.bulk_group.L2::cache_hint"
                     " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                     :
                     : "l"(tma_ptr), "r"(src_ptr), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z),
                       "r"(tma_coords.w), "r"(tma_coords.h), "l"(make_cache_policy<policy>())
                     : "memory");
    }
}

// Load async atom
template <cache_policy policy>
__device__ static inline void load_async_atom(uint32_t dst_ptr, uint64_t tma_ptr, const int5 &tma_coords,
                                              semaphore &bar) {

    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));

    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile("cp.async.bulk.tensor.5d.shared::cta.global.tile."
                     "mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
                     :
                     : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(tma_coords.x), "r"(tma_coords.y),
                       "r"(tma_coords.z), "r"(tma_coords.w), "r"(tma_coords.h)
                     : "memory");
    } else {
        asm volatile("cp.async.bulk.tensor.5d.shared::cta.global.tile."
                     "mbarrier::complete_tx::bytes.L2::cache_hint"
                     " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
                     :
                     : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(tma_coords.x), "r"(tma_coords.y),
                       "r"(tma_coords.z), "r"(tma_coords.w), "r"(tma_coords.h), "l"(make_cache_policy<policy>())
                     : "memory");
    }
}

} // namespace atoms

/* ----------   Prefetch Tensor Map  ---------- */

/**
 * @brief Prefetches data from global memory into a shared memory tile, along
 * with the tensormap.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in] idx Logical B/D/R/C element offset; alignment is the caller's
 * responsibility.
 */
template <cache_policy policy, tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src.template get_tma<ST>());
    int5 tma_coords = buffer_traits<ST>::coordinates(idx, src);

    atoms::prefetch_atom<policy>(tma_ptr, tma_coords);
}
template <tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx) {
    prefetch<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

/* ----------   Async load and store data from gmem/smem  ---------- */

// TMA stores below only issue bulk-group operations. The caller chooses the
// group boundary by calling tma::store_commit_group(). TMA loads use mbarriers
// and therefore do not have a commit-group operation.

/**
 * @brief Asynchronously stores data into global memory from a shared memory
 * tile.
 *
 * This function performs an asynchronous copy operation using CUDA's
 * cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] idx Logical B/D/R/C element offset; alignment is the caller's
 * responsibility.
 */
template <cache_policy policy, tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST>());
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(buffer_traits<ST>::shared_address(src)));
    int5 tma_coords = buffer_traits<ST>::coordinates(idx, dst);

    atoms::store_async_atom<policy>(tma_ptr, src_ptr, tma_coords);
}
template <tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    store_async<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

/* ----------   Async reduction + store data from gmem/smem  ---------- */

/**
 * @brief Asynchronously performs an add reduction and stores the result into
 * global memory from a shared memory tile.
 *
 * This function performs an asynchronous add reduction and copy operation using
 * CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] idx Logical B/D/R/C element offset; alignment is the caller's
 * responsibility.
 */
template <cache_policy policy, tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {
#ifdef KITTENS_FEATURE_FP8
    using dtype = typename buffer_traits<ST>::dtype;
    static_assert(!(std::is_same_v<dtype, fp8e4m3> || std::is_same_v<dtype, fp8e5m2>),
                  "TMA does not support async add reductions for fp8 types.");
#endif
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST>());
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(buffer_traits<ST>::shared_address(src)));
    int5 tma_coords = buffer_traits<ST>::coordinates(idx, dst);

    atoms::store_add_async_atom<policy>(tma_ptr, src_ptr, tma_coords);
}
template <tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {
    store_add_async<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief Asynchronously performs an min reduction and stores the result into
 * global memory from a shared memory tile.
 *
 * This function performs an asynchronous min reduction and copy operation using
 * CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] idx Logical B/D/R/C element offset; alignment is the caller's
 * responsibility.
 */
template <cache_policy policy, tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx) {
    using dtype = typename buffer_traits<ST>::dtype;
    static_assert(!std::is_same_v<dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
#ifdef KITTENS_FEATURE_FP8
    static_assert(!(std::is_same_v<dtype, fp8e4m3> || std::is_same_v<dtype, fp8e5m2>),
                  "TMA does not support async add reductions for fp8 types.");
#endif
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST>());
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(buffer_traits<ST>::shared_address(src)));
    int5 tma_coords = buffer_traits<ST>::coordinates(idx, dst);

    atoms::store_min_async_atom<policy>(tma_ptr, src_ptr, tma_coords);
}
template <tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx) {
    store_min_async<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief Asynchronously performs an max reduction and stores the result into
 * global memory from a shared memory tile.
 *
 * This function performs an asynchronous max reduction and copy operation using
 * CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] idx Logical B/D/R/C element offset; alignment is the caller's
 * responsibility.
 */
template <cache_policy policy, tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx) {
    using dtype = typename buffer_traits<ST>::dtype;
    static_assert(!std::is_same_v<dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
#ifdef KITTENS_FEATURE_FP8
    static_assert(!(std::is_same_v<dtype, fp8e4m3> || std::is_same_v<dtype, fp8e5m2>),
                  "TMA does not support async add reductions for fp8 types.");
#endif
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST>());
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(buffer_traits<ST>::shared_address(src)));
    int5 tma_coords = buffer_traits<ST>::coordinates(idx, dst);

    atoms::store_max_async_atom<policy>(tma_ptr, src_ptr, tma_coords);
}
template <tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx) {
    store_max_async<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

/**
 * @brief Asynchronously loads data from global memory into a shared memory
 * tile.
 *
 * This function performs an asynchronous copy operation using CUDA's
 * cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The semaphore used for synchronization of the asynchronous
 * copy.
 * @param[in] idx Logical B/D/R/C element offset; alignment is the caller's
 * responsibility.
 * @return The number of transaction bytes reported to the semaphore.
 */
template <cache_policy policy, tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline uint32_t load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar) {
    int5 tma_coords = buffer_traits<ST>::coordinates(idx, src);
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src.template get_tma<ST>());
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(buffer_traits<ST>::shared_address(dst)));

    atoms::load_async_atom<policy>(dst_ptr, tma_ptr, tma_coords, bar);
    return buffer_traits<ST>::transfer_bytes;
}
template <tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline uint32_t load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar) {
    return load_async<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar);
}

namespace cluster {

/**
 * @brief Asynchronously loads data from global memory into a shared memory
 * tile, across a threadblock cluster
 *
 * This function performs an asynchronous copy operation using CUDA's
 * cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The semaphore used for synchronization of the asynchronous
 * copy.
 * @param[in] idx Logical B/D/R/C element offset; alignment is the caller's
 * responsibility.
 * @param[in] cluster_mask The mask of the clusters to broadcast to.
 */
#ifdef KITTENS_FEATURE_SELECTIVE_MBAR_TRIGGER
template <cache_policy policy, tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                         uint16_t cluster_mask, int dst_mbar_cta = -1)
#else
template <cache_policy policy, tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                         uint16_t cluster_mask)
#endif
{
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src.template get_tma<ST>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(buffer_traits<ST>::shared_address(dst)));
    int5 tma_coords = buffer_traits<ST>::coordinates(idx, src);

#ifdef KITTENS_FEATURE_SELECTIVE_MBAR_TRIGGER
    if (dst_mbar_cta != -1) {
        uint32_t neighbor_mbar_ptr;
        asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
                     : "=r"(neighbor_mbar_ptr)
                     : "r"(mbar_ptr), "r"(dst_mbar_cta));
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::"
                         "complete_tx::bytes.cta_group::2.multicast::cluster"
                         " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
                         :
                         : "r"(dst_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr), "r"(tma_coords.x),
                           "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "r"(tma_coords.h), "h"(cluster_mask)
                         : "memory");
        } else {
            asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile."
                         "mbarrier::complete_tx::bytes.cta_group::2.multicast::"
                         "cluster.L2::cache_hint"
                         " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8, %9;"
                         :
                         : "r"(dst_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr), "r"(tma_coords.x),
                           "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "r"(tma_coords.h), "h"(cluster_mask),
                           "l"(make_cache_policy<policy>())
                         : "memory");
        }
    } else
#endif
        if constexpr (policy == cache_policy::NORMAL) {
        asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile."
                     "mbarrier::complete_tx::bytes.multicast::cluster"
                     " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
                     :
                     : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(tma_coords.x), "r"(tma_coords.y),
                       "r"(tma_coords.z), "r"(tma_coords.w), "r"(tma_coords.h), "h"(cluster_mask)
                     : "memory");
    } else {
        asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::"
                     "complete_tx::bytes.multicast::cluster.L2::cache_hint"
                     " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8, %9;"
                     :
                     : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(tma_coords.x), "r"(tma_coords.y),
                       "r"(tma_coords.z), "r"(tma_coords.w), "r"(tma_coords.h), "h"(cluster_mask), "l"(make_cache_policy<policy>())
                     : "memory");
    }
}
#ifdef KITTENS_FEATURE_SELECTIVE_MBAR_TRIGGER
template <tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                         uint16_t cluster_mask, int dst_mbar_cta = -1) {
    load_async<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask, dst_mbar_cta);
}
#else
template <tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD = buffer_coord_t<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                         uint16_t cluster_mask) {
    load_async<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask);
}
#endif

} // namespace cluster
} // namespace tma

} // namespace kittens

#endif
