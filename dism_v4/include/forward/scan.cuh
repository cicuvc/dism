#pragma once

#include "summary/kernel_common.cuh"
#include "forward/affine.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_forward {
constexpr int WarpKSize = 64;
using Scan = pscore::AltLayoutSplitScanBuffer<16, WarpKSize, pscore::BinaryElement, ForwardLogAffineOp>;
using Scalar = pscore::AltLayoutSplitScanBuffer<16, WarpKSize, pscore::UnaryElement>;

// Forward metadata follows this tile's TMA permutation, independently of the
// summary's64-column tiles. Half-elements are K/8 logical columns apart.
template <class KeyIndex, class KeyBias, class Shared>
__device__ __forceinline__ void load_forward_key_metadata(
        KeyIndex& labels, KeyBias& bias, const Shared& shared, bool query_lse) {
    int strip = 2 * (kt::warp::laneid() & 3);
#pragma unroll
    for (int block = 0; block < WarpKSize / 16; ++block) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int column = 2 * block + half + strip * Scalar::COL_BLOCKS;
            labels.data[block][half].x = shared.k_idx.data[column];
            labels.data[block][half].y = shared.k_idx.data[column + Scalar::COL_BLOCKS];
            if (!query_lse) {
                bias.data[block][half].x = -shared.klse.data[column];
                bias.data[block][half].y = -shared.klse.data[column + Scalar::COL_BLOCKS];
            }
        }
    }
}

__device__ __forceinline__ Scan make_forward_scan(const kt::rt_fl<16,WarpKSize>& score, const float* row_gate = nullptr) {
    Scalar scalar;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int c = 0; c < Scalar::COL_BLOCKS; ++c) {
            auto value = score.tiles[0][c / 2].data[r + 2 * (c & 1)];
            scalar.data[r][c].value = {value.x, value.y};
        }
    }
    scalar.roll();
    Scan scan;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int c = 0; c < Scalar::COL_BLOCKS; ++c) {
            scan.data[r][c].first = scan.data[r][c].second = scalar.data[r][c].value;
            if (row_gate) {
                int row=Scan::physical_row(r,c,kt::warp::laneid()/4);
                scan.data[r][c].first = scan.data[r][c].first + pscore::F32x2{-row_gate[row]};
            }
        }
    }
    return scan;
}

// Convert only the surviving W component back to the MMA accumulator layout.
// Do not unroll/shuffle the dead affine first component.
template <class ScanType>
__device__ __forceinline__ Scalar finish_scan(ScanType& scan, const typename ScanType::HState& incoming,
                                             const typename ScanType::ImmState& intermediate) {
    scan.inclusive_postscan(incoming, intermediate);
    Scalar value;
#pragma unroll
    for (int row = 0; row < Scalar::ROW_BLOCKS; ++row) {
#pragma unroll
        for (int col = 0; col < Scalar::COL_BLOCKS; ++col) {
            value.data[row][col].value = scan.data[row][col].second;
        }
    }
    value.roll<false>();
    return value;
}

__device__ __forceinline__ int boundary_column() {
    int lane = kt::warp::laneid();
    return 2 * Scalar::COL_BLOCKS * (lane & 3) + Scalar::COL_BLOCKS - 1 - lane / 4;
}
__device__ __forceinline__ bool boundary_owner() {
    return kt::warp::laneid() / 4 < Scalar::COL_BLOCKS;
}
} // namespace dism_forward

} // namespace DISM_VARIANT
