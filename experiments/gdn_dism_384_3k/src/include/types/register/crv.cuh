/**
 * @file
 * @brief Register vectors for computations on axes.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

/**
 * @brief Register vector structure.
 *
 * @tparam _T The packed data type used for the vector elements.
 * @tparam _outer_dim The size of the tile, in units of TILE_DIM (16).
 * @tparam _inner_dim This controls the layout of the tile in terms of which
 * axis it maps on the register tile layout.
 *
 * Register vectors are used to accumulate and map values across tiles. You can
 * do computation on them directly if you want, but they're not designed to be
 * maximally efficient vectors as they have substantial duplication and strange
 * layouts to help them work efficiently with the register layouts used by the
 * tensor cores. ThunderKittens wants you working with tiles where possible!
 */

template <typename _T, size_t _length, ducks::rv_layout::all _layout = ducks::rv_layout::naive>
struct crv {
    using identifier = ducks::crv::identifier;
    using component = rv<_T, _length, _layout>; /// Data type of each internal tile.
    using layout = component::layout;           ///< Layout of the matrix tile, ensures
                                                ///< compatibility with the rv concepts

    using T = component::T;
    using T2 = component::T2;
    using dtype = component::dtype; ///< Data type of the elements in the tile.

    static constexpr int length = component::length;
    static constexpr int tiles = component::tiles;

    // Real/imag tiles have same internal layout and size
    component real;
    component imag;
};

template <int _l, ducks::rv_layout::all layout = ducks::rv_layout::naive>
using crv_fl = crv<float, _l, layout>;
template <int _l, ducks::rv_layout::all layout = ducks::rv_layout::naive>
using crv_bf = crv<bf16, _l, layout>;
template <int _l, ducks::rv_layout::all layout = ducks::rv_layout::naive>
using crv_hf = crv<half, _l, layout>;

} // namespace kittens