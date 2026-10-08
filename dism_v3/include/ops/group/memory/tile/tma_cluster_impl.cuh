#pragma once
#include "../../group.cuh"
/**
 * @file
 * @brief Functions for a group scope to call tile TMA cluster functions.
 */
#ifdef KITTENS_FEATURE_TMA

#ifdef KITTENS_FEATURE_SELECTIVE_MBAR_TRIGGER
template <int GW>
template <cache_policy policy, kittens::tma::tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::tma::cluster::load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                                           uint16_t cluster_mask, int dst_mbar_cta) {
    if (laneid() == 0) {
        ::kittens::tma::cluster::load_async<policy, ST, GL, COORD>(dst, src, idx, bar, cluster_mask, dst_mbar_cta);
    }
}
template <int GW>
template <kittens::tma::tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::tma::cluster::load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                                           uint16_t cluster_mask, int dst_mbar_cta) {
    if (laneid() == 0) {
        ::kittens::tma::cluster::load_async<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask,
                                                                                 dst_mbar_cta);
    }
}
#else
template <int GW>
template <kittens::cache_policy policy, kittens::tma::tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::tma::cluster::load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                                           uint16_t cluster_mask) {
    if (laneid() == 0) {
        ::kittens::tma::cluster::load_async<policy, ST, GL, COORD>(dst, src, idx, bar, cluster_mask);
    }
}
template <int GW>
template <kittens::tma::tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::tma::cluster::load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar,
                                                           uint16_t cluster_mask) {
    if (laneid() == 0) {
        ::kittens::tma::cluster::load_async<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask);
    }
}
#endif

#endif
