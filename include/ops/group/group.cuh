/**
 * @file
 * @brief An aggregate header of all group (multi-warp) operations defined by
 * ThunderKittens
 */

#pragma once

// #include <cuda/pipeline>

#include "../../common/common.cuh"
#include "../../types/types.cuh"
#include "../thread/thread.cuh" // several group memory ops rely on underlying warp-scope ops

#define KITTENS_CHECK_WARP                                                                                             \
    static_assert(GROUP_WARPS == 1, "Warp (GROUP_WARPS=1) function called from a non-warp group.");
// A "warpgroup" is a special group of 4 consecutive warps defined by NVIDIA for
// certain SM_90+ operations.
#define KITTENS_CHECK_WARPGROUP                                                                                        \
    static_assert(GROUP_WARPS == 4, "Warpgroup (GROUP_WARPS=4) function "                                              \
                                    "called from a non-warpgroup group.");

// WGMMA relies on some template structures that cannot be specialized within
// the group struct, so we declare them in advance.
#if defined(KITTENS_FEATURE_WGMMA)
#include "mma/warpgroup/base/base.cuh"
#endif

namespace kittens {
/*
This is meant to be used with a `using group_N = kittens::group<NUM_WORKERS>;`
at the start of every kernel.
*/
template <int _GROUP_WARPS>
struct group {
    static constexpr int GROUP_WARPS = _GROUP_WARPS;                          // This alias produces nice parallelism.
    static constexpr int GROUP_THREADS = GROUP_WARPS * kittens::WARP_THREADS; // This alias produces nice parallelism.
    __device__ static inline int laneid() { return threadIdx.x % GROUP_THREADS; }
    __device__ static inline int warpid() { return laneid() / kittens::WARP_THREADS; }
    __device__ static inline int groupid() { return threadIdx.x / GROUP_THREADS; }

    __device__ static inline void sync(int id) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "n"(GROUP_THREADS)); }
    template <uint32_t MASK = 0xFFFFFFFF>
    __device__ static inline void sync() {
        static_assert(GROUP_WARPS == 1, "barrier-less sync() can only be called by a single warp!");
        asm volatile("bar.warp.sync %0;\n" ::"n"(MASK));
    }
    __device__ static inline void arrive(int id) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "n"(GROUP_THREADS)); }

    template <ducks::rt::all RT, ducks::st::all ST>
    __device__ static inline void load(RT &dst, const ST &src);
    template <ducks::st::all ST, ducks::rt::all RT>
    __device__ static inline void store(ST &dst, const RT &src);
    template <ducks::rv::naive_layout RV, ducks::st::all ST>
    __device__ static inline auto load(RV &dst, const ST &src, int2 row_col);
    template <ducks::rv::naive_layout RV, ducks::st::all ST>
    __device__ static inline auto store(ST &dst, const RV &src, int2 row_col);

    template <
        int axis, ducks::rt::row_layout RT, ducks::gl::all GL,
        ducks::coord::tile COORD = coord<rt<typename RT::T, GROUP_WARPS * RT::rows, RT::cols, typename RT::layout>>>
    __device__ inline static void load(RT &dst, const GL &src, const COORD &idx);
    template <
        int axis, ducks::rt::col_layout RT, ducks::gl::all GL,
        ducks::coord::tile COORD = coord<rt<typename RT::T, GROUP_WARPS * RT::rows, RT::cols, typename RT::layout>>>
    __device__ inline static void load(RT &dst, const GL &src, const COORD &idx);
    template <
        ducks::rt::all RT, ducks::gl::all GL,
        ducks::coord::tile COORD = coord<rt<typename RT::T, GROUP_WARPS * RT::rows, RT::cols, typename RT::layout>>>
    __device__ inline static void load(RT &dst, const GL &src, const COORD &idx);

    template <
        int axis, ducks::rt::row_layout RT, ducks::gl::all GL,
        ducks::coord::tile COORD = coord<rt<typename RT::T, GROUP_WARPS * RT::rows, RT::cols, typename RT::layout>>>
    __device__ inline static void store(const GL &dst, const RT &src, const COORD &idx);
    template <
        int axis, ducks::rt::col_layout RT, ducks::gl::all GL,
        ducks::coord::tile COORD = coord<rt<typename RT::T, GROUP_WARPS * RT::rows, RT::cols, typename RT::layout>>>
    __device__ inline static void store(const GL &dst, const RT &src, const COORD &idx);
    template <
        ducks::rt::all RT, ducks::gl::all GL,
        ducks::coord::tile COORD = coord<rt<typename RT::T, GROUP_WARPS * RT::rows, RT::cols, typename RT::layout>>>
    __device__ inline static void store(const GL &dst, const RT &src, const COORD &idx);

    template <int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
    __device__ static inline void load(ST &dst, const GL &src, const COORD &idx);

    template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
    __device__ static inline void load(ST &dst, const GL &src, const COORD &idx);
    template <int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
    __device__ static inline void store(const GL &dst, const ST &src, const COORD &idx);

    template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
    __device__ static inline void store(const GL &dst, const ST &src, const COORD &idx);

    template <int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
    __device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx);

    template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
    __device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx);

    template <ducks::rv::all RV, ducks::sv::all SV>
    __device__ inline static void load(RV &dst, const SV &src);
    template <ducks::sv::all SV, ducks::rv::all RV>
    __device__ inline static void store(SV &dst, const RV &src);

    template <ducks::rv::all RV, ducks::gl::all GL>
    __device__ inline static void
    load(RV &dst, const GL &src, const coord<rv<typename RV::T, GROUP_WARPS * RV::length, typename RV::layout>> &idx);

    template <ducks::rv::all RV, ducks::gl::all GL>
    __device__ inline static void
    store(GL &dst, const RV &src, const coord<rv<typename RV::T, GROUP_WARPS * RV::length, typename RV::layout>> &idx);

    template <ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD = coord<SV>>
    __device__ static inline void load(SV &dst, const GL &src, const COORD &idx);
    template <ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD = coord<SV>>
    __device__ static inline void store(GL &dst, const SV &src, const COORD &idx);
    template <ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD = coord<SV>>
    __device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx);

#ifdef KITTENS_FEATURE_TMA
    struct tma {
        __device__ static inline void expect_bytes(semaphore &bar, uint32_t bytes);
        template <typename T, typename... args>
        __device__ static inline void expect(semaphore &bar, const T &_1, const args &..._2);
        __device__ static inline void store_commit_group();
        template <int N = 0>
        __device__ static inline void store_async_wait();
        template <int N = 0>
        __device__ static inline void store_async_read_wait();

        template <int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL,
                  ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx);

        template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx);

        template <cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD = coord<SV>>
        __device__ static inline void prefetch(SV &dst, const GL &src, const COORD &idx);

        template <cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD = coord<SV>>
        __device__ static inline void store_async(const GL &dst, const SV &src, const COORD &idx);

        template <int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL,
                  ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx);

        template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx);

        template <ducks::tma::wrapper::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<typename ST::T_>>
        __device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx);

        template <int axis, ducks::tma::wrapper::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<typename ST::T_>>
        __device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx);

        template <int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL,
                  ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx);

        template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx);

        template <ducks::tma::wrapper::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<typename ST::T_>>
        __device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx);

        template <cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD = coord<SV>>
        __device__ static inline void store_add_async(const GL &dst, const SV &src, const COORD &idx);

        template <int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL,
                  ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx);

        template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx);

        template <ducks::tma::wrapper::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<typename ST::T_>>
        __device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx);

        template <cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD = coord<SV>>
        __device__ static inline void store_min_async(const GL &dst, const SV &src, const COORD &idx);

        template <int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL,
                  ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx);

        template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx);

        template <ducks::tma::wrapper::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<typename ST::T_>>
        __device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx);

        template <cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD = coord<SV>>
        __device__ static inline void store_max_async(const GL &dst, const SV &src, const COORD &idx);

        template <int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL,
                  ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar);

        template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
        __device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar);

        template <int axis, cache_policy policy, ducks::tma::wrapper::all ST, ducks::gl::all GL,
                  ducks::coord::tile COORD = coord<typename ST::T_>>
        __device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar);

        template <ducks::tma::wrapper::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<typename ST::T_>>
        __device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar);

        template <cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD = coord<SV>>
        __device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore &bar);

        struct cluster {
            __device__ static inline void wait(semaphore &bar, int kPhaseBit);
            __device__ static inline void expect_bytes(semaphore &bar, uint32_t bytes, int dst_cta);
            template <typename T, typename... args>
            __device__ static inline void expect(semaphore &bar, int dst_cta, const T &_1, const args &..._2);
            __device__ static inline void arrive(semaphore &bar, int dst_cta, uint32_t count = 1);
            __device__ static inline void store_async(void *dst, void *src, int dst_cta, uint32_t size_bytes,
                                                      semaphore &bar);
            template <typename T>
            __device__ static inline void store_async(T &dst_, T &src_, int dst_cta, semaphore &bar);

            template <cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD = coord<SV>>
            __device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore &bar,
                                                     uint16_t cluster_mask, int dst_mbar_cta = -1);

#ifdef KITTENS_FEATURE_SELECTIVE_MBAR_TRIGGER
            template <int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL,
                      ducks::coord::tile COORD = coord<ST>>
            __device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                                     uint16_t cluster_mask, int dst_cta = -1);

            template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
            __device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                                     uint16_t cluster_mask, int dst_cta = -1);
#else
            template <int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL,
                      ducks::coord::tile COORD = coord<ST>>
            __device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                                     uint16_t cluster_mask);

            template <ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
            __device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                                     uint16_t cluster_mask);
#endif
        };
    };
#endif

    struct st_maps;
    struct st_reductions;
    struct sv_maps;
    struct sv_reductions;

    template <ducks::st::all ST1, ducks::st::all ST2>
    __device__ static inline void copy(ST1 &dst, const ST2 &src);

    template <ducks::sv::all SV1, ducks::sv::all SV2>
    __device__ static inline void copy(SV1 &dst, const SV2 &src);

    template <ducks::rv::all RV1, ducks::rv::all RV2>
    __device__ static inline void copy(RV1 &dst, const RV2 &src);

    template <typename T, typename U, ducks::rt_layout::all layout>
    __device__ static inline void copy(rt_base<T, layout> &dst, const rt_base<U, layout> &src);
#if (defined(KITTENS_FEATURE_FP8))
    template <typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
    __device__ static inline void copy(rt<T2, _height, _width, layout> &dst,
                                       const rt<U2, _height, _width, layout> &src);
#else
    template <typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
    __device__ static inline void copy(rt<T2, _height, _width, layout> &dst,
                                       const rt<U2, _height, _width, layout> &src);
#endif

    struct rt_conversions;
    struct rt_maps;
    struct rt_reductions;
    struct rv_maps;
    struct rv_reductions;

    struct vec_conversion_detail {
        __device__ static inline int row_from_indices_dim2(int laneid, int inner_dim, int x_or_y);
        __device__ static inline int row_from_indices_dim1(int laneid, int x_or_y);
        __device__ static inline int canonical_src_lane_dim2(int row);
        __device__ static inline int canonical_src_lane_dim1(int row);
    };

    struct wmma;
    struct emulated_wgmma;

#ifdef KITTENS_FEATURE_WGMMA
    struct wgmma;
#endif

    template <int N = 0>
    __device__ static inline void load_async_wait(int bar_id);
    template <int N = 0>
    __device__ static inline void load_async_wait();

    __device__ static inline void arrive(barrier<GROUP_WARPS> bar);
    __device__ static inline void arrive_and_wait(barrier<GROUP_WARPS> bar);

    __device__ static inline void init_semaphore(semaphore &bar, int thread_count, int transaction_count = 0);
    __device__ static inline void invalidate_semaphore(semaphore &bar);

    __device__ static inline void arrive(semaphore &sem);

    template <int num_warps>
    __device__ static inline void arrive(barrier<num_warps> bar);

    __device__ static inline int test_wait(semaphore &sem, int kPhaseBit);

#if (defined(KITTENS_FEATURE_MBARRIER))
    __device__ static inline void arrive(semaphore &sem, uint32_t count);
#endif

    __device__ static inline void wait(semaphore &sem, int kPhaseBit);

#ifdef KITTENS_FEATURE_REG_INCDEC
    template <int n_reg>
    __device__ static inline void increase_registers() {
        static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(n_reg));
    }
    template <int n_reg>
    __device__ static inline void decrease_registers() {
        static_assert(n_reg % 8 == 0, "n_reg must be a multiple of 8");
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(n_reg));
    }
    __device__ static inline void producer_registers() { decrease_registers<24>(); }
    template <int NCWG>
    __device__ static inline void consumer_registers() {
        increase_registers<480 / NCWG - 8 * (NCWG > 3) - 224 * (NCWG == 1)>();
    }

#endif
};

namespace everyone {
// Block-level synchronization
__device__ static inline void sync(int id) { asm volatile("bar.sync %0;\n" ::"r"(id)); }

// Cluster-level synchronization functions
namespace tma {
namespace cluster {
__device__ static inline void arrive_aligned() { // All threads in the cluster must call this
    asm volatile("barrier.cluster.arrive.release.aligned;\n");
}
__device__ static inline void wait_aligned() { asm volatile("barrier.cluster.wait.acquire.aligned;\n"); }
__device__ static inline void sync() {
    arrive_aligned();
    wait_aligned();
}
} // namespace cluster
} // namespace tma

}; // namespace everyone

using warp = group<1>;      // scope used by most pre-Hopper GPUs, and also for most
                            // register operations.
using warpgroup = group<4>; // special scope commonly used by Hopper and later.

} // namespace kittens

#include "memory/tile/g2r_impl.cuh"
#include "memory/tile/g2s_impl.cuh"
#include "memory/tile/s2r_impl.cuh"
#include "memory/util/util_impl.cuh"

#include "memory/vec/g2r_impl.cuh"
#include "memory/vec/g2s_impl.cuh"
#include "memory/vec/s2r_impl.cuh"

#include "memory/tile/tma_cluster_impl.cuh"
#include "memory/tile/tma_impl.cuh"
#include "memory/util/tma_cluster_impl.cuh"
#include "memory/util/tma_impl.cuh"
#include "memory/vec/tma_cluster_impl.cuh"
#include "memory/vec/tma_impl.cuh"

#include "shared/tile/conversions_impl.cuh"
#include "shared/tile/maps_impl.cuh"
#include "shared/tile/reductions_impl.cuh"

#include "shared/vec/conversions_impl.cuh"
#include "shared/vec/maps_impl.cuh"
#include "shared/vec/reductions_impl.cuh"

#include "register/tile/conversions_impl.cuh"
#include "register/tile/maps_impl.cuh"
#include "register/tile/reductions_impl.cuh"

#include "register/vec/conversions_impl.cuh"
#include "register/vec/maps_impl.cuh"
#include "register/vec/reductions_impl.cuh"

#include "mma/warp/warp_impl.cuh"
#include "mma/warpgroup/emulated_impl.cuh"
#include "mma/warpgroup/warpgroup_impl.cuh"
