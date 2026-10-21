#pragma once
#include "kittens.cuh"

namespace fp8gemm {
// Complete FP8 shared tiles only. The actual allocation must honor ST's
// 8*swizzle_bytes alignment, so the uniform base can stay outside the XOR.
template<int GW, bool Swap, class RT, class ST>
__device__ inline void load_fp8_factored(RT &dst, const ST &src) {
    namespace kt = kittens;
    static_assert(sizeof(typename ST::dtype) == 1);
    static_assert(std::is_same_v<typename RT::layout, kt::ducks::rt_layout::row>);
    static_assert(ST::rows == ST::underlying_rows && ST::cols == ST::underlying_cols);
    static_assert(ST::cols == RT::cols && ST::rows % (GW * RT::rows) == 0);
    constexpr int SB = ST::swizzle_bytes;
    static_assert(alignof(ST) >= 8 * SB && ST::required_alignment >= 8 * SB);
    constexpr int panel_tiles = SB / RT::tile_size_col;
    int lane = kt::laneid(), warp = kt::group<GW>::warpid();
    if constexpr (GW % 4 == 0) warp = warp / 4 + (warp % 4) * (GW / 4);
    int row = Swap ? (lane % 8) + (lane / 16) * 8 : lane % 16;
    int col = (Swap ? (lane / 8) % 2 : lane / 16) * 16;
    // Disjoint row-group/lane bits: XOR prevents ptxas from reassociating
    // the warp offset into the changing uniform ring-buffer base.
    uint32_t lane_offset = ST::idx(0u, {row, col}) ^ (warp * RT::rows * SB);
    uint32_t shared_base = __cvta_generic_to_shared(src.data);
#pragma unroll
    for (int i = 0; i < RT::height; ++i) {
#pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            uint32_t offset = lane_offset ^ ((j % panel_tiles) * RT::tile_size_col);
            uint32_t addr = shared_base + offset + i * RT::tile_size_row * SB
                + (j / panel_tiles) * ST::rows * SB;
            auto &d = dst.tiles[i][j].data;
            using U2 = typename kt::base_types::packing<typename ST::dtype>::packed_type;
            if constexpr (Swap) kt::move<U2>::ldsm4(d[0], d[2], d[1], d[3], addr);
            else kt::move<U2>::ldsm4(d[0], d[1], d[2], d[3], addr);
        }
    }
}
}
