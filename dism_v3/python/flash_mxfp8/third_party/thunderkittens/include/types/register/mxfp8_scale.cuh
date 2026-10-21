#pragma once
#include "../../common/common.cuh"

#ifdef KITTENS_FEATURE_MMA_BLOCK_SCALE
namespace kittens {
/** E8M0 metadata for a row-major FP8 register tile, one scale per row/K32.
 * Each 16x32 base tile uses one uint32_t per lane. Its low byte contains
 * the scale for row lane/4 + 8*(lane%2); upper bytes are ignored.
 * Lanes 2/3 may duplicate lanes 0/1. A uses thread selector 0 (lanes 0/1);
 * B's first and second N8 halves use selectors 0 and 1 respectively.
 * A raw byte e represents 2^(e-127); 0xff is NaN, not a zero scale.
 * Metadata is already in registers: MMA performs no loads or shuffles.
 */
template<int Rows, int K>
struct mxfp8_scale {
    static_assert(Rows > 0 && Rows % 16 == 0 && K > 0 && K % 32 == 0);
    static constexpr int rows = Rows, cols = K;
    uint32_t data[Rows/16][K/32];
};
} // namespace kittens
#endif
