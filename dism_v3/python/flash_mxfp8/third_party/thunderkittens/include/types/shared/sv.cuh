/**
 * @file
 * @brief The ThunderKittens shared vector struct.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/base_concepts.cuh"
#include "../../common/common.cuh"

namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

/**
 * @brief Shared vector structure.
 *
 * @tparam _T The packed data type used for the vector elements.
 * @tparam _tiles The size of the tile, in units of TILE_ROW_DIM (16 for fp16,
 * bf16, fp32).
 *
 * Shared vectors are used to accumulate and map values across shared tiles.
 * Unlike every other structure present in ThunderKittens, these have a simple
 * uniform layout which is just an array in memory. EZ!
 */
template <typename _T, size_t _length>
struct KITTENS_DEFAULT_ALIGN sv {
    using identifier = ducks::sv::identifier;
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.

    static constexpr int length = _length; ///< Length in elements.
    static_assert(length % TILE_ROW_DIM<T> == 0, "Length must be divisible by the tile dimension");
    static constexpr int tiles = length / TILE_ROW_DIM<T>; ///< Length in subtiles.'
#if (defined(KITTENS_FEATURE_FP8))
    static_assert(!std::is_same_v<T2, fp8e4m3_4> && !std::is_same_v<T2, fp8e5m2_4>, "Unsupported type for fp8");
#endif

#if (defined(KITTENS_FEATURE_FP8))
    static constexpr int num_alloc_elements =
        ((length * sizeof(dtype) + 127) / 128) * (128 / sizeof(dtype)); // round up to the nearest 128-byte boundary
#else
    static constexpr int num_alloc_elements = length;
#endif
    dtype data[num_alloc_elements]; ///< The actual shared vector data.

    __device__ static inline T *idx(T *ptr, int idx) { // useful for computations in shared address space,
                                                       // as silly as it sounds.
        return ptr[idx];
    }

    __device__ inline dtype &operator[](size_t idx) { return data[idx]; }
    __device__ inline const dtype &operator[](size_t idx) const { return data[idx]; }

    template <int sub_length>
    __device__ inline sv<_T, sub_length> &subvec(int idx) {
        return *(sv<dtype, sub_length> *)&data[idx * sub_length];
    }
    template <int sub_length>
    __device__ inline const sv<_T, sub_length> &subvec(int idx) const {
        return *(sv<dtype, sub_length> *)&data[idx * sub_length];
    }

    __device__ inline void operator=(const dtype &value) { // runs at warp scope by default
#pragma unroll
        for (int i = kittens::laneid(); i < length; i += WARP_THREADS) {
            data[i] = value;
        }
    }
};

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// vector types
template <size_t _length>
using sv_bf = sv<bf16, _length>;
template <size_t _length>
using sv_hf = sv<half, _length>;
template <size_t _length>
using sv_fl = sv<float, _length>;

/* ----------  PRINTOUTS  ---------- */

template <int colsep_interval = 1, ducks::sv::all SV>
__device__ inline void print(const SV &sv) {
    static_assert(colsep_interval > 0, "Column separator interval must be positive.");

    printf("     ");
    for (int c = 0; c < SV::length; c++) {
        printf(" %9d", c);
    }
    printf("\n      ");

    for (int c = 0; c < SV::length; c++) {
        printf("%s─────────", c % colsep_interval == 0 ? (c ? "┬" : "┌") : "─");
    }
    printf("┐\n");

    printf("%4d ", 0);
    for (int c = 0; c < SV::length; c++) {
        printf(c % colsep_interval == 0 ? " │" : "  ");
        if constexpr (std::is_same_v<typename SV::dtype, float>) {
            printf("%8.3f", sv[c]);
#ifdef KITTENS_FEATURE_UE8
        } else if constexpr (std::is_same_v<typename SV::dtype, fp8e8m0>) {
            printf("%8.3f", static_cast<float>(sv[c]));
#endif
#if (defined(KITTENS_FEATURE_FP8))
        } else if constexpr (std::is_same_v<typename SV::dtype, fp8e4m3>) {
            printf("%8.3f", static_cast<float>(sv[c]));
#endif
        } else if constexpr (std::is_same_v<typename SV::dtype, __nv_bfloat16>) {
            printf("%8.3f", __bfloat162float(sv[c]));
        } else if constexpr (std::is_integral_v<typename SV::dtype>) {
            printf("%8d", static_cast<int>(sv[c]));
        } else {
            printf("%8.3f", static_cast<float>(sv[c]));
        }
    }
    printf(" │\n      ");

    for (int c = 0; c < SV::length; c++) {
        printf("%s─────────", c % colsep_interval == 0 ? (c ? "┴" : "└") : "─");
    }
    printf("┘\n");
}

} // namespace kittens
