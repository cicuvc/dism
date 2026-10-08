/**
 * @file
 * @brief Host-side construction of tiled CUtensorMap descriptors.
 *
 * CUtensorMap is opaque in the CUDA API.  The encoding below is the layout
 * used by Hopper and Blackwell drivers (verified against CUDA 13).  Keeping
 * the encoder here lets host code construct a descriptor without calling the
 * CUDA driver or requiring an active CUDA context.
 */

#pragma once

#include <cuda.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace kittens {
namespace tma {

/**
 * Encode a tiled tensor map directly into host memory.
 *
 * The arguments have the same meaning as cuTensorMapEncodeTiled.  On success
 * the complete 128-byte descriptor is written to @p tensor_map.  This routine
 * checks that every argument is representable in the descriptor.  As a
 * low-level encoder it does not reproduce all cross-field policy checks made
 * by the CUDA driver, and deliberately does not query pointer attributes:
 * @p global_address may be a numerical device address even when no CUDA
 * context exists on the calling thread.
 */
__host__ static inline CUresult
encode_tiled(CUtensorMap *tensor_map, CUtensorMapDataType data_type, uint32_t rank, void *global_address,
             const uint64_t *global_dim, const uint64_t *global_stride, const uint32_t *box_dim,
             const uint32_t *element_stride, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
             CUtensorMapL2promotion l2_promotion, CUtensorMapFloatOOBfill oob_fill) noexcept {
    static_assert(sizeof(CUtensorMap) == 128, "Unexpected CUtensorMap size");

    if (tensor_map == nullptr || global_dim == nullptr || box_dim == nullptr || element_stride == nullptr ||
        rank < 1 || rank > 5 || (rank > 1 && global_stride == nullptr)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const uint32_t dtype = static_cast<uint32_t>(data_type);
    const uint32_t interleave_value = static_cast<uint32_t>(interleave);
    const uint32_t swizzle_value = static_cast<uint32_t>(swizzle);
    const uint32_t l2_value = static_cast<uint32_t>(l2_promotion);
    const uint32_t oob_value = static_cast<uint32_t>(oob_fill);
    if (dtype > 15 || interleave_value > 2 || swizzle_value > 6 || l2_value > 3 || oob_value > 1) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // API enum -> the four-bit hardware type code in descriptor word 2.
    constexpr uint8_t dtype_code[16] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 8, 7, 8, 11, 12, 13};
    constexpr uint8_t element_bytes[16] = {1, 2, 4, 4, 8, 8, 2, 4, 8, 2, 4, 4, 4, 0, 0, 0};

    uint32_t words[32] = {};
    const uint64_t address = reinterpret_cast<uint64_t>(global_address);
    words[0] = static_cast<uint32_t>(address);
    words[1] = static_cast<uint32_t>(address >> 32);

    uint32_t flags = ((rank - 1) << 4) | (static_cast<uint32_t>(dtype_code[dtype]) << 7) |
                     (interleave_value << 11) | (std::min(swizzle_value, 3u) << 13) | (oob_value << 15) |
                     (l2_value << 17) | ((swizzle_value > 3 ? swizzle_value - 3 : 0) << 19);
    if (dtype == 11 || dtype == 12) {
        flags |= 1u << 16; // TFLOAT32 uses the FLOAT32 code plus a separate flag.
    }

    uint64_t total_elements = 1;
    uint64_t box_elements = 1;
    uint64_t strided_box_elements = 1;
    uint32_t packed_element_strides = 0;
    for (uint32_t i = 0; i < rank; ++i) {
        if (global_dim[i] == 0 || global_dim[i] > (uint64_t{1} << 32) || box_dim[i] == 0 || box_dim[i] > 256 ||
            element_stride[i] == 0 || element_stride[i] > 8) {
            return CUDA_ERROR_INVALID_VALUE;
        }

        words[8 + i] = static_cast<uint32_t>(global_dim[i] - 1);
        // boxDim[0] occupies word 13's high byte; the rest occupy word 14.
        const uint32_t box_shift = i == 0 ? 24 : 8 * (i - 1);
        words[i == 0 ? 13 : 14] |= (box_dim[i] - 1) << box_shift;
        packed_element_strides |= (element_stride[i] - 1) << (3 * i);

        // Only the >= 2^16 predicate is needed, so saturate to avoid overflow.
        if (total_elements < 65536) {
            total_elements = std::min<uint64_t>(65536, total_elements * global_dim[i]);
        }
        box_elements *= box_dim[i];
        strided_box_elements *= box_dim[i] / element_stride[i];
    }
    words[13] |= packed_element_strides;

    for (uint32_t i = 0; i + 1 < rank; ++i) {
        if ((global_stride[i] & 0xf) != 0 || global_stride[i] >= (uint64_t{1} << 40)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        const uint64_t stride_div_16 = global_stride[i] >> 4;
        words[3 + i] = static_cast<uint32_t>(stride_div_16);
        words[7] |= static_cast<uint32_t>((stride_div_16 >> 32) & 0xf) << (4 * i);
    }

    uint64_t transfer_bytes;
    if (dtype >= 13) {
        transfer_bytes = 0; // Packed U4/U6 encodings do not store a transfer size.
    } else if (interleave_value == 0) {
        transfer_bytes = strided_box_elements * element_bytes[dtype];
    } else {
        transfer_bytes = box_elements * (interleave_value == 1 ? 16 : 32);
    }
    words[16] = static_cast<uint32_t>(transfer_bytes);
    words[18] = swizzle_value == 0 ? 0x10 : (1u << (7 + std::min(swizzle_value, 3u)));

    const bool wide = interleave_value == 0 ? total_elements >= 65536 : transfer_bytes >= 65536;
    words[2] = flags | (static_cast<uint32_t>(wide) << 21);

    std::memcpy(tensor_map, words, sizeof(words));
    return CUDA_SUCCESS;
}

} // namespace tma
} // namespace kittens
