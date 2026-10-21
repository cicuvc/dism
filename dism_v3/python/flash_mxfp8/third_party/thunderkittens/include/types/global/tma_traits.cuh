/**
 * @file
 * @brief Customization point for shared-memory buffers used by TMA.
 */

#pragma once

#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "tma.cuh"
#include "util.cuh"

namespace kittens {
namespace tma {

/**
 * Specialize this template for a custom shared-memory buffer type.
 * Specializations must provide:
 *
 *   dtype, coord_type, default_swizzle, rank, transfer_bytes,
 *   encode<swizzle>(), shared_address(), and coordinates(). The
 *   encoder receives shape and element strides in logical B/D/R/C order.
 *
 * Tensor maps created by a custom specialization must currently be 5D. The
 * low-level Hopper TMA atoms in this library issue tensor.5d instructions.
 */
template <typename Buffer> struct buffer_traits;

/** Default adapter for ThunderKittens shared tiles. */
template <ducks::st::all ST> struct buffer_traits<ST> {
    using dtype = typename ST::dtype;
    using coord_type = coord<>;

    static constexpr bool default_swizzle = true;
    static constexpr int rank = 5;
    static constexpr uint32_t transfer_bytes = ST::num_elements * sizeof(dtype);

    template <bool enable_swizzle, typename GlobalT>
    __host__ static inline void encode(CUtensorMap *map, const global_tensor_view<GlobalT> &view) {
        static_assert(std::is_same_v<dtype, GlobalT>, "The TMA buffer dtype must match the global layout dtype.");
        detail::tma::create_tensor_map<ST, enable_swizzle>(map, view);
    }

    __device__ static inline const void *shared_address(const ST &tile) { return &tile; }
    __device__ static inline void *shared_address(ST &tile) { return &tile; }

    template <typename COORD, typename GL> __device__ static inline int5 coordinates(const COORD &idx, const GL &) {
        constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);
        coord<ducks::default_type> unit = coord<>(idx);
        return {0, unit.r, unit.c / swizzle_elements, unit.d, unit.b};
    }
};

template <typename Buffer>
concept custom_buffer = !ducks::st::all<Buffer> && requires {
    typename buffer_traits<Buffer>::dtype;
    typename buffer_traits<Buffer>::coord_type;
    { buffer_traits<Buffer>::default_swizzle } -> std::convertible_to<bool>;
    { buffer_traits<Buffer>::rank } -> std::convertible_to<int>;
    { buffer_traits<Buffer>::transfer_bytes } -> std::convertible_to<uint32_t>;
    requires(buffer_traits<Buffer>::rank == 5);
};

template <typename Buffer>
concept tile_buffer = ducks::st::all<Buffer> || custom_buffer<Buffer>;

template <tile_buffer Buffer> using buffer_coord_t = typename buffer_traits<Buffer>::coord_type;

} // namespace tma
} // namespace kittens
