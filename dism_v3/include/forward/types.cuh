#pragma once
#include "forward/scan.cuh"
#include "dimensions.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_forward {
#ifndef DISM_KC_HEAD_DIM
#define DISM_KC_HEAD_DIM 64
#endif
static_assert(DISM_KC_HEAD_DIM == 32 || DISM_KC_HEAD_DIM == 64 || DISM_KC_HEAD_DIM == 128);
template<bool FP32Output>
using OutputDType = std::conditional_t<FP32Output, float, kt::bf16>;
constexpr int QROWS = 128;
constexpr int DIM = ActiveConfig::DV; // KcHeadDim / value channels DV.
constexpr int READOUT_DIM = ActiveConfig::R;
// K64 D128/DV128 double buffering exceeds99KiB; K32 fits two slots.
constexpr int STAGES = WarpKSize == 64 && CONFIG.KcKeyDim == 128 && DIM == 128 ? 1 : 2;
using QTile = kt::st_bf<QROWS, CONFIG.KcKeyDim>;
using SQTile = kt::st_bf<QROWS, READOUT_DIM>;
using KTile = KPermutationSharedBuffer<kt::bf16, WarpKSize, CONFIG.KcKeyDim>;
using VTile = KPermutationSharedBuffer<kt::bf16, WarpKSize, DIM>;
using SKTile = KPermutationSharedBuffer<kt::bf16, WarpKSize, READOUT_DIM>;
template<bool FP32Output> using OutputTile = kt::st<OutputDType<FP32Output>,16,DIM>;
template<bool FP32Output> using OutputGlobal = kt::gl<OutputDType<FP32Output>,-1,-1,-1,DIM>;

template<bool FP32Output> struct SharedT {
    union Tiles {
        struct Prefetch {
            QTile q;
            SQTile sq;
        } prefetch[1];
        struct KeyValue {
            KTile k;
            SKTile sk;
            VTile v;
        } kv[STAGES];
    } tiles;
    union Metadata {
        struct Prefetch {
            kt::sv_fl<QROWS> lse;
            kt::sv<int, QROWS> labels;
            uint8_t hard[QROWS];
        } prefetch[1];
        struct KeyValue {
            kt::sv_fl<WarpKSize> klse;
            kt::sv<int, WarpKSize> k_idx;
        } kv[STAGES];
    } metadata;
    // Necessary WG communication only. All128 writers/readers participate.
    struct Mail { float2 value[4][32]; } mail[2];
    bool query_lse;
    float tau2;
    // Warp-private output-layout staging; no cross-warp communication.
    OutputTile<FP32Output> output[8];
};
template<bool FP32Output> constexpr size_t shared_bytes = 4 * 1024 + sizeof(SharedT<FP32Output>);

template<bool FP32Output> struct ArgsT {
    int batch, heads, n, padded, checkpoints;
    kt::gl<kt::bf16, -1, -1, -1, CONFIG.KcKeyDim, QTile> q;
    kt::gl<kt::bf16, -1, -1, -1, READOUT_DIM, SQTile> sq;
    kt::gl<kt::bf16, -1, -1, -1, CONFIG.KcKeyDim, KTile> k;
    kt::gl<kt::bf16, -1, -1, -1, READOUT_DIM, SKTile> sk;
    kt::gl<kt::bf16, -1, -1, -1, DIM, VTile> v;
    kt::gl<float, -1, -1, 1, -1> q_lse, k_lse;
    kt::gl<int, -1, -1, 1, -1> q_label, k_label;
    uint8_t *hard, *direction;
    float *tau, *boundary;
    OutputDType<FP32Output> *output;
    float *lse2;
    // Optional training checkpoint: S2[q,16*(c+1)-1], query contiguous.
    float *vertical;
    OutputGlobal<FP32Output> output_map;
};

// Same persistent task order as summary; output includes every128-row block.
struct Scheduler {
    struct Task { int batch, head, q_start, key_blocks; };
    int index, blocks, heads, total, n;
    template<bool FP32Output>
    __device__ Scheduler(const ArgsT<FP32Output>& args)
        : index(blockIdx.x), blocks((args.n + QROWS - 1) / QROWS), heads(args.heads),
          total(blocks * args.heads * args.batch), n(args.n) {}
    __device__ __forceinline__ bool next(Task& task) {
        if (index >= total) return false;
        task.batch = index / (heads * blocks);
        task.head = index / blocks % heads;
        task.q_start = index % blocks * QROWS;
        task.key_blocks = (min(task.q_start + QROWS, n) + WarpKSize - 1) / WarpKSize;
        index += gridDim.x;
        return true;
    }
};
} // namespace dism_forward

} // namespace DISM_VARIANT
