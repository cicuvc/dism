#pragma once
// Shared types for the direct top-k training packer. No Torch/CUDA dependency.
#include "topk_prefill.hpp"
#include "planning_workers.hpp"
#include <array>
#include <memory>
namespace topk_prefill {
struct TrainingJob { int batch, head, start, end; double tau; };
constexpr int compact_widths[3] = {16,32,64};
inline int compact_bucket(int length) {
    int bucket=0;
    while(bucket<2 && length>compact_widths[bucket])++bucket;
    return bucket;
}
}
