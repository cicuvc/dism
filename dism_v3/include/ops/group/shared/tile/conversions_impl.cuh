#pragma once

#include "../../group.cuh"
/**
 * @file
 * @brief Group conversions between different shared memory tile types.
 */

/* ----------  COPIES  ---------- */

template <int GW>
template <ducks::st::all ST1, ducks::st::all ST2>
__device__ inline void group<GW>::copy(ST1 &dst, const ST2 &src) {
    static_assert(ST1::height == ST2::height && ST1::width == ST2::width, "Tiles must have the same height and width");
#pragma unroll
    for (int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        int row = i / dst.cols, col = i % dst.cols;
        dst[{row, col}] = base_types::convertor<typename ST1::dtype, typename ST2::dtype>::convert(src[{row, col}]);
    }
}