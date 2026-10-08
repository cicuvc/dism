#pragma once

#include "summary/primitives.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

// Metadata follows the SAME column permutation as the score MMA. Loading it
// directly in this layout avoids a register transpose after LDSM.
template <class KeyIndex, class KeyBias, class Shared>
__device__ __forceinline__ void load_key_metadata(KeyIndex &labels, KeyBias &bias,
                                                  const Shared &shared, bool query_lse) {
    int strip = 2 * (kt::warp::laneid() & 3);
#pragma unroll
    for (int block = 0; block < CONFIG.WarpKSize / 16; ++block) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int column = 2 * block + half + strip * 8;
            labels.data[block][half].x = shared.k_idx.data[column];
            labels.data[block][half].y = shared.k_idx.data[column + 8];
            if (!query_lse) {
                bias.data[block][half].x = -shared.klse.data[column];
                bias.data[block][half].y = -shared.klse.data[column + 8];
            }
        }
    }
}

template <class SharedVector, class GlobalVector>
__device__ __forceinline__ void load_key_vector_async(
    SharedVector &dst, const GlobalVector &src, const kt::coord<> &coordinate) {
    kt::warp::load_async(dst, src, coordinate);
}

} // namespace DISM_VARIANT
