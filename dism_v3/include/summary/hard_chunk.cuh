#pragma once
#include <cstdint>
#include "variant.cuh"

namespace DISM_VARIANT {
struct HardChunkTask {
    int64_t offset;
    int n, diagonal_block;
};
struct HardChunkArgs {
    const int* summary;
    int* boundary;
    const HardChunkTask* tasks; // null for fixed length
    int n;
};
} // namespace DISM_VARIANT
