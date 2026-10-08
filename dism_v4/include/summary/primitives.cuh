#pragma once

#include "common/base_concepts.cuh"
#define KITTENS_FEATURE_TMA
#define KITTENS_FEATURE_MBARRIER
#define KITTENS_FEATURE_REG_INCDEC
#include <kittens.cuh>

#include <ds_alt.cuh>

namespace kt = kittens;

template <typename T, int ROWS, int COLS> struct KPermutationSharedBuffer {
    using tile_type = kt::st<T, ROWS, COLS>;

    static constexpr int required_alignment = 1024;

    alignas(1024) tile_type payload;

    __device__ tile_type &tile() {
        return payload;
    }
    __device__ const tile_type &tile() const {
        return payload;
    }
};

namespace kittens::tma {
template <typename T, int ROWS, int COLS>
struct buffer_traits<::KPermutationSharedBuffer<T, ROWS, COLS>> {
    static_assert(sizeof(T) == 2, "Element must be 2bytes");

    using buffer_type = ::KPermutationSharedBuffer<T, ROWS, COLS>;
    using tile_type = typename buffer_type::tile_type;
    using dtype = T;

    // Coordinates passed to TMA operations are logical element offsets.
    using coord_type = kt::coord<>;

    static constexpr bool default_swizzle = true;
    static constexpr int rank = 5; // The current TMA atoms issue tensor.5d PTX.
    static constexpr uint32_t transfer_bytes = ROWS * COLS * sizeof(dtype);

    // Host customization point used while constructing gl. This example
    // reuses TK's descriptor encoder, but it may instead call
    // cuTensorMapEncodeTiled directly and define a different 5D layout.
    template <bool EnableSwizzle, typename GlobalT>
    __host__ static void encode(CUtensorMap *map,
                                const kt::tma::global_tensor_view<GlobalT> &view) {
        static_assert(std::is_same_v<GlobalT, dtype>);

        // Known limitations: TMA OOB detection can be broken when seqlen % COL_ELEMENTS != 0.
        constexpr int COL_ELEMENTS = ROWS / 8;
        assert(view.shape[1] % COL_ELEMENTS == 0);

        auto swizzle_elements = std::min(COLS, 64);

        auto siwzzle = (CUtensorMapSwizzle)__builtin_ffs(swizzle_elements / 16);

        uint64_t gmem_shape[5] = {0, 0, 0, 0, 0};
        uint64_t gmem_stride[4] = {0, 0, 0, 0};
        uint32_t smem_shape[5] = {0, 0, 0, 0, 0};
        uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

        gmem_shape[0] = view.shape[3];
        gmem_shape[1] = 8;
        gmem_shape[2] = view.shape[1];
        // Axis3 selects channel/swizzle panels, NOT attention heads. With
        // D128/H1 a head-sized extent would zero-fill the second64 channels.
        gmem_shape[3] = COLS / swizzle_elements;
        gmem_shape[4] = view.shape[0] * view.shape[1] * view.shape[2] * view.shape[3];

        gmem_stride[0] = sizeof(T) * COL_ELEMENTS * view.stride[1];
        gmem_stride[1] = sizeof(T) * 1 * view.stride[1];
        gmem_stride[2] = sizeof(T) * swizzle_elements;
        gmem_stride[3] = sizeof(T) * swizzle_elements;

        smem_shape[0] = swizzle_elements;
        smem_shape[1] = 8;
        smem_shape[2] = COL_ELEMENTS;
        smem_shape[3] = COLS / swizzle_elements;
        smem_shape[4] = 1;

        kt::tma::encode_tiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, view.data, gmem_shape,
                              gmem_stride, smem_shape, smem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                              siwzzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    // Device customization point: only payload participates in TMA.
    __device__ static void *shared_address(buffer_type &buffer) {
        return &buffer.payload;
    }
    __device__ static const void *shared_address(const buffer_type &buffer) {
        return &buffer.payload;
    }

    // Device customization point: descriptor coordinates must agree with
    // encode(). This example adopts the normal swizzled-st coordinate mapping.
    template <typename Coord, typename GL>
    __device__ static int5 coordinates(const Coord &idx, const GL &gl) {
        auto swizzle_elements = std::min(COLS, 64);
        auto offset = idx.b * gl.template stride<0>() + idx.r * gl.template stride<2>();
        auto coord = int5{0, 0, idx.d, idx.c / swizzle_elements, int(offset) / swizzle_elements};
        return coord;
    }
};
} // namespace kittens::tma

template <int WARPS, bool INIT, int... ENDS_IDX> struct CommEndInfo {
    static constexpr bool INITIAL = INIT;

    static constexpr int N_WARPS = WARPS * sizeof...(ENDS_IDX);
    static constexpr int LOG2_WARPS = __builtin_ffs(N_WARPS) - 1;
    static constexpr int ENDS[] = {ENDS_IDX...};

    static_assert(N_WARPS == (1 << LOG2_WARPS), "Warp number must be power of 2");

    static constexpr bool hasEnd(int end_idx) {
        for (int i = 0; i < (int)sizeof...(ENDS_IDX); i++)
            if (ENDS[i] == end_idx)
                return true;
        return false;
    }
};

template <bool INIT, int... ENDIDX> constexpr CommEndInfo<1, INIT, ENDIDX...> ProducerWarp = {};

template <bool INIT, int... ENDIDX>
constexpr CommEndInfo<4, INIT, ENDIDX...> ConsumerWarpGroup = {};

template <typename... Ts> struct CommEndCollection {
    static constexpr int MAX_BITS = std::max({Ts::LOG2_WARPS...});
    inline static constexpr int findEnd(int end_idx) {
        bool masks[] = {Ts::hasEnd(end_idx)...};
        for (int i = 0; i < (int)sizeof...(Ts); i++) {
            if (masks[i])
                return i;
        }
        return -1;
    }
    static constexpr int getWarpBits(int end_idx) {
        int warps[] = {Ts::LOG2_WARPS...};
        return warps[findEnd(end_idx)];
    }
    static constexpr bool isInitial(int end_idx) {
        return ((Ts::hasEnd(end_idx) && Ts::INITIAL) || ...);
    }
};

template <int N_BUFFERS, typename... TBuffer> struct BufferSet {
    using BufferTuple = std::tuple<std::remove_cvref_t<TBuffer> &...>;
    template <typename... TContainer>
    __forceinline__ __device__ BufferSet(TBuffer (TContainer::*...ps)[N_BUFFERS]) {}

    __forceinline__ __device__ static BufferTuple
    make_tuple(std::remove_cvref_t<TBuffer> &...buffers) {
        return {buffers...};
    }
};

template <int N_BUFFERS, typename... TBuffer, typename... TContainer>
BufferSet(TBuffer (TContainer::*...ps)[N_BUFFERS]) -> BufferSet<N_BUFFERS, TBuffer...>;

template <int N_BUFFERS, int N_ENDS, typename TBufferSet, typename... Ts> struct MbarrierRingPipe {
    kt::semaphore (&Barriers)[N_ENDS][N_BUFFERS];
    uint32_t PhaseState;
    uint32_t Slot, CurrIdx, ArriveCount;
    CommEndCollection<Ts...> EndDesc;

    template <int align, typename... TBuffers>
    __forceinline__ __device__ MbarrierRingPipe(kt::shared_allocator<align> &alloc,
                                                BufferSet<N_BUFFERS, TBuffers...>,
                                                uint32_t role_idx, Ts... ends)
        : Barriers(alloc.template allocate<kt::semaphore, N_ENDS, N_BUFFERS>()), PhaseState(0),
          Slot(0) {
        static_assert(sizeof...(Ts) == N_ENDS, "The number of communication ends mismatches!");

        uint32_t wait_count = kt::WARP_THREADS << EndDesc.MAX_BITS;

        CurrIdx = EndDesc.findEnd(role_idx);
        ArriveCount = 1u << (EndDesc.MAX_BITS - EndDesc.getWarpBits(role_idx));

        PhaseState = EndDesc.isInitial(role_idx) ? ((1u << N_BUFFERS) - 1) : 0;
        if (kt::warpid() == 0 && kt::warp::elect_leader()) {
#pragma unroll
            for (int i = 0; i < N_BUFFERS; i++) {
#pragma unroll
                for (int j = 0; j < N_ENDS; j++) {
                    kt::init_semaphore(Barriers[j][i], wait_count);
                }
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
    }

    __forceinline__ __device__ void setup(uint32_t slot = 0) {
        Slot = slot;
    }

    struct Handle {
        MbarrierRingPipe *PipeRef;
        typename TBufferSet::BufferTuple Container;
        uint32_t SlotIdx, NextEnd;

        template <int N> __forceinline__ __device__ auto &get() const {
            return std::get<N>(Container);
        }

        __forceinline__ __device__ kt::semaphore &getBarrier() const {
            return PipeRef->Barriers[PipeRef->EndDesc.findEnd(NextEnd)][SlotIdx];
        }

        __forceinline__ __device__ void submitToNextAndTrigger() {
            // Converge the participating warp before publishing its releases.
            kt::warp::sync();
            // Establish release ordering for every participating reader. Keep
            // representative-only arrival as a future independently tested
            // optimization, not an assumption of the correctness baseline.
            uint32_t address = uint32_t(__cvta_generic_to_shared(&getBarrier()));
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                         :: "r"(address), "r"(PipeRef->ArriveCount) : "memory");
        }
        __forceinline__ __device__ int getArrivalCount() const {
            return PipeRef->ArriveCount;
        }
    };

    __forceinline__ __device__ Handle getNullHandle() {
        return {this, 0, 0};
    }

    template <typename... TBuffers>
    __forceinline__ __device__ Handle waitBuffer(uint32_t next_end,
                                                 TBuffers (&...buffers)[N_BUFFERS]) {
        waitSlot(Slot);
        // Consume exactly one incoming generation; publication does not consume it.
        PhaseState ^= 1u << Slot;
        return {this, TBufferSet::make_tuple(buffers[Slot]...), Slot, next_end};
    }

    __forceinline__ __device__ void waitSlot(uint32_t slot) {
        uint32_t address = uint32_t(__cvta_generic_to_shared(&Barriers[CurrIdx][slot]));
        uint32_t phase = (PhaseState >> slot) & 1;
        asm volatile("{ .reg .pred done;\n"
                     "wait_loop: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
                     "done, [%0], %1;\n"
                     "@done bra.uni wait_done;\n"
                     "bra.uni wait_loop;\n"
                     "wait_done: }" :: "r"(address), "r"(phase) : "memory");
        // Non-consuming peek/drain. A subsequent acquire waits the same generation.
    }
    __forceinline__ __device__ void moveNext() { Slot = (Slot + 1) % N_BUFFERS; }
};

template <int align, int N_BUFFERS, typename... TBuffers, typename... Ts>
MbarrierRingPipe(kt::shared_allocator<align> &alloc, BufferSet<N_BUFFERS, TBuffers...>,
                 uint32_t role_idx, Ts... endWarps)
    -> MbarrierRingPipe<N_BUFFERS, sizeof...(Ts), BufferSet<N_BUFFERS, TBuffers...>, Ts...>;


#include "variant.cuh"
namespace DISM_VARIANT {
#ifndef DISM_KC_KEY_DIM
#define DISM_KC_KEY_DIM 64
#endif
static_assert(DISM_KC_KEY_DIM == 32 || DISM_KC_KEY_DIM == 64 || DISM_KC_KEY_DIM == 128);

struct TmaSummarizationKernelConfig {
    int KcKeyDim = ActiveConfig::D;
    int WarpQSize = 32;
    int WarpKSize = 64;
    int WarpGroups = 2;
    int KStages = 3;
    int KPrefetch = 1;

    constexpr int getCheckpointSize() const {
        return WarpQSize;
    }
    constexpr int getQBlockSize() const {
        return getCheckpointSize() * kt::WARPGROUP_WARPS * WarpGroups;
    }
};

static constexpr TmaSummarizationKernelConfig CONFIG = {};

struct TmaKernelTraits {
    using QBuffer = kt::st<kt::bf16, CONFIG.WarpQSize * kt::WARPGROUP_WARPS * CONFIG.WarpGroups,
                           CONFIG.KcKeyDim>;
    using KBuffer = KPermutationSharedBuffer<kt::bf16, CONFIG.WarpKSize, CONFIG.KcKeyDim>;
};

struct TmaSummarizationKernelArgs {
    int Batch, Seqlen, Head;
    int PaddedSeqlen, Checkpoints;
    // Q's descriptor iterates sequence rows; its physical tensor is B,N,H,D.
    kt::gl<kt::bf16, -1, -1, -1, CONFIG.KcKeyDim, TmaKernelTraits::QBuffer> QVec; // logical B,H,N,D
    kt::gl<kt::bf16, -1, -1, -1, CONFIG.KcKeyDim, TmaKernelTraits::KBuffer>
        KVec; // [batch, seq, head, channel]

    // Natural-log inputs. Conversion to base2 happens once per metadata value.
    kt::gl<float, -1, -1, 1, -1> QLseVec; // [batch, head, seq]
    kt::gl<float, -1, -1, 1, -1> KLseVec; // [batch, head, seq]
    kt::gl<int, -1, -1, 1, -1> IdxQ;      // [batch, head, seq]
    kt::gl<int, -1, -1, 1, -1> IdxK;      // [batch, head, seq]
    uint8_t *hard_flags;                  // [batch, head, seq] must be contiguous
    uint8_t *direction;                   // [batch, head] must be contiguous
    float *rtau;                          // [head]

    // Independent contiguous SoA buffers: [B,H,floor((N-1)/32),padded_N].
    float *SummaryA; // log2 affine first component
    float *SummaryB; // log2 affine second component; retained after passing
    const float *gate_delta = nullptr; // natural-log [B,H,N] attenuation
};

struct LoadSharedMemoryLayouts {
    union TileRegion {
        struct DefaultLayout {
            TmaKernelTraits::KBuffer kbuffer;
        } Default[CONFIG.KStages];
        struct PrefetchLayout {
            DefaultLayout preflight_k[CONFIG.KPrefetch];
            TmaKernelTraits::QBuffer qbuffer;
        } Prefetch[1];
    } Tiles;
    union VectorRegion {
        struct DefaultLayout {
            kt::sv_fl<CONFIG.WarpKSize> klse;
            kt::sv<int, CONFIG.WarpKSize> k_idx;
        } Default[CONFIG.KStages];
        struct PrefetchLayout {
            DefaultLayout preflight_k[CONFIG.KPrefetch];
            kt::sv_fl<CONFIG.getQBlockSize()> qlse;
            kt::sv<int, CONFIG.getQBlockSize()> q_idx;
            uint8_t hard_flags[CONFIG.getQBlockSize()];
        } Prefetch[1];
    } Vectors;
    float row_gate[CONFIG.getQBlockSize()]; // task lifetime; never aliases the K ring
    bool direction;
    float tau2; // Producer converts natural-log rtau to log2 for hard matches.
};

// Include allocator alignment and both barrier rings, not just tile storage.
inline constexpr size_t SUMMARY_SHARED_BYTES = 4 * 1024 + sizeof(LoadSharedMemoryLayouts);

struct FixedLengthScheduler {

    struct TaskInfo {
        int Batch, QStart, KStart, Head;
        int KBlocks;
    };

    int CurrentTaskIdx;
    const int Batch, Seqlen, SeqlenQBlocks, Head;

    __device__ FixedLengthScheduler(int batch, int seqlen, int head)
        : Batch(batch), Head(head), Seqlen(seqlen),
          SeqlenQBlocks(((seqlen - 1) / CONFIG.WarpQSize +
                         CONFIG.getQBlockSize() / CONFIG.WarpQSize - 1) /
                        (CONFIG.getQBlockSize() / CONFIG.WarpQSize)) {
        CurrentTaskIdx = blockIdx.x;
    }

    __forceinline__ __device__ bool getNextTask(TaskInfo &out) {
        auto Idx = CurrentTaskIdx;
        auto Total = SeqlenQBlocks * Batch * Head;
        if (Idx >= Total)
            return false;
        out.Batch = Idx / (Head * SeqlenQBlocks);
        Idx -= out.Batch * (Head * SeqlenQBlocks);
        out.Head = Idx / SeqlenQBlocks;
        Idx -= out.Head * SeqlenQBlocks;

        out.QStart = CONFIG.getQBlockSize() * Idx;
        out.KStart = 0;
        int summary_end = (Seqlen - 1) / CONFIG.WarpQSize * CONFIG.WarpQSize;
        int query_end = min(out.QStart + CONFIG.getQBlockSize(), summary_end);
        out.KBlocks = (query_end + CONFIG.WarpKSize - 1) / CONFIG.WarpKSize;

        CurrentTaskIdx += gridDim.x;
        return true;
    }
};

template <typename T> __device__ __forceinline__ void load_async_any(T &dst, uint8_t *src) {
    constexpr int atom_bytes = 16;
    constexpr int total_bytes = sizeof(T);
    static_assert(total_bytes % atom_bytes == 0, "size must be multiply of 16");
    constexpr int total_calls = total_bytes / atom_bytes;
    constexpr int waves = (total_calls - kt::WARP_THREADS + 1) / kt::WARP_THREADS + 1;
    int tid = kt::warp::laneid();

    uint32_t dst_ptr = uint32_t(__cvta_generic_to_shared(&dst)) + atom_bytes * tid;
    src += atom_bytes * tid;
#pragma unroll
    for (int i = 0; i < waves; i++) {
        if (i * kt::WARP_THREADS + tid < total_calls) {
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                             dst_ptr + atom_bytes * i * kt::WARP_THREADS),
                         "l"((uint64_t)(src + atom_bytes * i * kt::WARP_THREADS))
                         : "memory");
        }
    }
}

// Bounded-domain sentinel, deliberately not an exact log-affine zero/identity.
inline constexpr float LOG_ZERO = -1e6f;

// A producer warp stages one query-row attenuation vector, including optional
// halo. The caller publishes its normal pipe packet only after this completes.
template<int Count>
__device__ __forceinline__ void load_row_gate(float (&dst)[Count], const float* src,
                                             int valid) {
    for (int i=kt::warp::laneid(); i<Count; i+=32)
        dst[i]=(src && i<valid) ? src[i]*1.4426950408889634f : 0.f;
    __syncwarp();
}

struct LogAffineOp {
    template <typename T>
    __forceinline__ __host__ __device__ static pscore::BinaryElement<T> identity() {
        // All summary paths use finite arithmetic, including padding identities.
        return {T{0.0f}, T{LOG_ZERO}};
    }

    __forceinline__ __device__ static float softplus_approx(float x, float y) {
        float value;
        constexpr float amplitude = 1.81089463f;
        float p = fmaf(fabsf(x - y), 0.34114549f, 0.48232999f);
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(value) : "f"(p));
        return fmaf(value, -amplitude, fmaxf(x, y) + amplitude);
    }

    __forceinline__ __device__ static pscore::F32x1
    combine_second(pscore::F32x1 lhs_b, pscore::F32x1 rhs_a, pscore::F32x1 rhs_b) {
        return {softplus_approx(lhs_b.u0 + rhs_a.u0, rhs_b.u0)};
    }

    __forceinline__ __device__ static pscore::F32x2
    combine_second(pscore::F32x2 lhs_b, pscore::F32x2 rhs_a, pscore::F32x2 rhs_b) {
        return {
            softplus_approx(lhs_b.u0 + rhs_a.u0, rhs_b.u0),
            softplus_approx(lhs_b.u1 + rhs_a.u1, rhs_b.u1),
        };
    }

    template <typename T>
    __forceinline__ __device__ static pscore::BinaryElement<T>
    apply(const pscore::BinaryElement<T> &lhs, const pscore::BinaryElement<T> &rhs) {
        return {lhs.first + rhs.first, combine_second(lhs.second, rhs.first, rhs.second)};
    }
};

template <int ROWS, int COLS>
__device__ __forceinline__ auto make_scan(const kt::rt_fl<ROWS, COLS> &buffer, const float* row_gate = nullptr) {
    pscore::AltLayoutSplitScanBuffer<ROWS, COLS, pscore::UnaryElement> single;
#pragma unroll
    for (int i = 0; i < buffer.height; i++) {
#pragma unroll
        for (int j = 0; j < buffer.width; j++) {
#pragma unroll
            for (int ti = 0; ti < 2; ti++) {
#pragma unroll
                for (int tj = 0; tj < 2; tj++) {
                    single.data[i * 2 + ti][j * 2 + tj].value.u0 =
                        buffer.tiles[i][j].data[ti + tj * 2].x;
                    single.data[i * 2 + ti][j * 2 + tj].value.u1 =
                        buffer.tiles[i][j].data[ti + tj * 2].y;
                }
            }
        }
    }
    single.roll();
    pscore::AltLayoutSplitScanBuffer<ROWS, COLS, pscore::BinaryElement, LogAffineOp>
        res;
#pragma unroll
    for (int i = 0; i < res.ROW_BLOCKS; i++) {
#pragma unroll
        for (int j = 0; j < res.COL_BLOCKS; j++) {
            res.data[i][j].first = res.data[i][j].second = single.data[i][j].value;
            if (row_gate) {
                int row=decltype(res)::physical_row(i,j,kt::warp::laneid()/4);
                res.data[i][j].first = res.data[i][j].first + pscore::F32x2{-row_gate[row]};
            }
        }
    }
    // Caller skips the final warp block; every scanned query row is valid.
    return res;
}

} // namespace DISM_VARIANT
