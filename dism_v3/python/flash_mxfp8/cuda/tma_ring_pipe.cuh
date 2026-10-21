#pragma once
// Generic ring-pipe portion used by dism_v3/src/forward.cu, copied from
// dism_v3/include/summary/primitives.cuh
// (MIT; see ../third_party/thunderkittens/LICENSE.dism-core).
// The DISM custom shared-buffer traits and model-specific types are omitted.
// hasEnd uses a parameter-pack fold to avoid runtime device indexing of a
// host constexpr array (CUDA 13.1 otherwise eliminates the kernel as undefined).
#include "kittens.cuh"
#include <algorithm>
#include <tuple>
namespace fp8gemm {
namespace kt = kittens;
template <int WARPS, bool INIT, int... ENDS_IDX> struct CommEndInfo {
    static constexpr bool INITIAL = INIT;

    static constexpr int N_WARPS = WARPS * sizeof...(ENDS_IDX);
    static constexpr int LOG2_WARPS = __builtin_ffs(N_WARPS) - 1;
    static constexpr int ENDS[] = {ENDS_IDX...};

    static_assert(N_WARPS == (1 << LOG2_WARPS), "Warp number must be power of 2");

    static constexpr bool hasEnd(int end_idx) {
        return ((end_idx == ENDS_IDX) || ...);
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
        // Avoid a dynamically indexed automatic array and the resulting
        // local-memory load when more than two logical role IDs are used.
        return ((Ts::hasEnd(end_idx) ? Ts::LOG2_WARPS : 0) | ...);
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
    __forceinline__ __device__ Handle currentBuffer(uint32_t next_end,
                                                    TBuffers (&...buffers)[N_BUFFERS]) {
        // Non-acquiring view of the current slot. The caller must already
        // have acquired this generation and must not have released it.
        return {this, TBufferSet::make_tuple(buffers[Slot]...), Slot, next_end};
    }

    template <typename... TBuffers>
    __forceinline__ __device__ Handle waitBuffer(uint32_t next_end,
                                                 TBuffers (&...buffers)[N_BUFFERS]) {
        waitSlot(Slot);
        // Consume exactly one incoming generation; publication does not consume it.
        PhaseState ^= 1u << Slot;
        return currentBuffer(next_end,buffers...);
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

} // namespace fp8gemm
