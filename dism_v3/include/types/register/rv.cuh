/**
 * @file
 * @brief Register vectors for computations on axes.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/base_concepts.cuh"
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
struct rv {
    using identifier = ducks::rv::identifier;          ///< Type identifier for the rv structure.
    static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
    using layout = _layout;
    static constexpr bool is_naive = std::is_same_v<layout, ducks::rv_layout::naive>;
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = std::conditional_t<is_naive, T,
                                     T2>; ///< Data type of the vector elements

    static constexpr int length = _length; ///< Length in elements.
    static_assert(length % kittens::TILE_ROW_DIM<T> == 0, "Length must be divisible by the tile dimension");
    static constexpr int tiles = _length / kittens::TILE_ROW_DIM<T>; ///< Length in subtiles, aliased for
                                                                     ///< consistency with sv type
    static constexpr int inner_dim = layout::inner_dim; ///< Internal layout within a subtile. Either 1 or 2.
    static constexpr int outer_dim = is_naive ? (tiles + 1) / 2 : tiles; ///< Outer dim (also length in tiles)
#if (defined(KITTENS_FEATURE_FP8))
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for fp8");
#endif

    dtype data[outer_dim][inner_dim]; ///< The actual register vector data.

    __device__ inline dtype *operator[](size_t idx) {
        return &data[idx][0];
    } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype *operator[](size_t idx) const {
        return &data[idx][0];
    } ///< A wrapper for indexing into vector data.
    __device__ inline dtype &operator[](int2 outin) {
        return data[outin.x][outin.y];
    } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype &operator[](int2 outin) const {
        return data[outin.x][outin.y];
    } ///< A wrapper for indexing into vector data.

    __device__ inline void operator=(const T &value) {
        dtype value2;
        if constexpr (is_naive) {
            value2 = value;
        } else {
            value2 = base_types::packing<T>::pack(value);
        }
#pragma unroll
        for (int i = 0; i < outer_dim; i++) {
#pragma unroll
            for (int j = 0; j < inner_dim; j++) {
                data[i][j] = value2;
            }
        }
    }
    template <typename U>
    __device__ inline void operator=(const rv<U, length, layout> &other) {
        using U2 = base_types::packing<U>::packed_type;
#pragma unroll
        for (int i = 0; i < outer_dim; i++) {
#pragma unroll
            for (int j = 0; j < inner_dim; j++) {
                data[i][j] = base_types::convertor<T2, U2>::convert(other.data[i][j]);
            }
        }
    }
};

template <int _l, ducks::rv_layout::all layout = ducks::rv_layout::naive>
using rv_fl = rv<float, _l, layout>;
template <int _l, ducks::rv_layout::all layout = ducks::rv_layout::naive>
using rv_bf = rv<bf16, _l, layout>;
template <int _l, ducks::rv_layout::all layout = ducks::rv_layout::naive>
using rv_hf = rv<half, _l, layout>;

} // namespace kittens