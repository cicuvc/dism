#pragma once

#include "../../group.cuh"

using kittens::cache_policy;
using kittens::group;

#ifdef KITTENS_FEATURE_TMA

/** @file Group-scope forwarding functions for tile-like TMA buffers. */

template <int GW>
template <cache_policy policy, kittens::tma::tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::tma::prefetch(ST &dst, const GL &src, const COORD &idx) {
    if (elect_leader())
        ::kittens::tma::prefetch<policy, ST, GL, COORD>(dst, src, idx);
}

template <int GW>
template <kittens::tma::tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline void group<GW>::tma::prefetch(ST &dst, const GL &src, const COORD &idx) {
    if (elect_leader())
        ::kittens::tma::prefetch<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

#define KITTENS_DEFINE_GROUP_TMA_STORE(function_name)                                                                  \
    template <int GW>                                                                                                  \
    template <cache_policy policy, kittens::tma::tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD>          \
    __device__ inline void group<GW>::tma::function_name(const GL &dst, const ST &src, const COORD &idx) {             \
        if (elect_leader())                                                                                            \
            ::kittens::tma::function_name<policy, ST, GL, COORD>(dst, src, idx);                                       \
    }                                                                                                                  \
    template <int GW>                                                                                                  \
    template <kittens::tma::tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD>                               \
    __device__ inline void group<GW>::tma::function_name(const GL &dst, const ST &src, const COORD &idx) {             \
        if (elect_leader())                                                                                            \
            ::kittens::tma::function_name<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);                         \
    }

KITTENS_DEFINE_GROUP_TMA_STORE(store_async)
KITTENS_DEFINE_GROUP_TMA_STORE(store_add_async)
KITTENS_DEFINE_GROUP_TMA_STORE(store_min_async)
KITTENS_DEFINE_GROUP_TMA_STORE(store_max_async)

#undef KITTENS_DEFINE_GROUP_TMA_STORE

template <int GW>
template <cache_policy policy, kittens::tma::tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline uint32_t group<GW>::tma::load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar) {
    if (elect_leader())
        ::kittens::tma::load_async<policy, ST, GL, COORD>(dst, src, idx, bar);
    return kittens::tma::buffer_traits<ST>::transfer_bytes;
}

template <int GW>
template <kittens::tma::tile_buffer ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ inline uint32_t group<GW>::tma::load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar) {
    if (elect_leader())
        ::kittens::tma::load_async<cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar);
    return kittens::tma::buffer_traits<ST>::transfer_bytes;
}

#endif
