#pragma once
#include "summary/kernel_common.cuh"
#include "varlen/operands.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_varlen {
using SummaryArgs=PackedArgs<TmaSummarizationKernelArgs>;
struct SummaryTask : FixedLengthScheduler::TaskInfo {
    int begin,n,checkpoints;
    int64_t metadata,forward;
};
struct SummaryOutput {
    int Head,PaddedSeqlen,Checkpoints;
    float *SummaryA,*SummaryB;
};
__device__ __forceinline__ SummaryTask summary_task(const SummaryArgs& packed) {
    auto work=packed.tasks[blockIdx.x];
    const auto* row=packed.documents+int64_t(work.x)*Fields;
    int n=row[Length],begin=row[Begin],blocks=n/CONFIG.getQBlockSize();
    int head=work.y/blocks,q0=(work.y%blocks)*CONFIG.getQBlockSize();
    int checkpoints=(n-1)/32;
    int end=min(q0+CONFIG.getQBlockSize(),checkpoints*32);
    return {{0,q0,0,head,(end+CONFIG.WarpKSize-1)/CONFIG.WarpKSize},
            begin,n,checkpoints,int64_t(head)*packed.common.Seqlen+begin,
            row[ForwardOffset]*packed.common.Head};
}

template <class LoadPipe, class PrefetchPipe>
__device__ __forceinline__ void
produce_packed_summary(const TmaSummarizationKernelArgs &args, LoadPipe &load_pipe,
                      PrefetchPipe &prefetch_pipe, LoadSharedMemoryLayouts &shared,
                      const SummaryTask &task, uint32_t warp_in_group) {
    constexpr uint32_t CONSUMER_A = 0;
    constexpr float LOG2E = 1.4426950408889634f;
    kt::warpgroup::decrease_registers<40>();
    // Keep the whole warp for coalesced cp.async metadata loads.
    if (warp_in_group == 0) {
        bool leader = kt::warp::elect_leader();
        {
            bool query_lse = args.direction[task.Batch * args.Head + task.Head];

            prefetch_pipe.setup();
            auto prefetch = prefetch_pipe.waitBuffer(CONSUMER_A, shared.Tiles.Prefetch,
                                                     shared.Vectors.Prefetch);
            if (leader) {
                shared.direction = query_lse;
                shared.tau2 = args.rtau[task.Head] * LOG2E;
            }
            auto &tiles = prefetch.template get<0>();
            auto &vectors = prefetch.template get<1>();
            if (leader) {
                kt::tma::expect_bytes(prefetch.getBarrier(),
                                      sizeof(tiles.qbuffer) + sizeof(tiles.preflight_k[0]));
                kt::tma::load_async(tiles.qbuffer, args.QVec,
                                    {0, task.Head, task.begin+task.QStart, 0}, prefetch.getBarrier());
                kt::tma::load_async(tiles.preflight_k[0].kbuffer, args.KVec,
                                    {0, task.begin, task.Head, 0}, prefetch.getBarrier());
            }
            load_async_any(vectors.hard_flags,
                           args.hard_flags +
                               task.metadata +
                               task.QStart);
            kt::warp::load_async(vectors.q_idx, args.IdxQ, {0, 0, 0, int(task.metadata)+task.QStart});
            load_key_vector_async(vectors.preflight_k[0].k_idx, args.IdxK,
                                 {0, 0, 0, int(task.metadata)});
            if (query_lse) {
                kt::warp::load_async(vectors.qlse, args.QLseVec,
                                     {0, 0, 0, int(task.metadata)+task.QStart});
            } else {
                load_key_vector_async(vectors.preflight_k[0].klse, args.KLseVec,
                                     {0, 0, 0, int(task.metadata)});
            }
            load_row_gate(shared.row_gate,args.gate_delta ? args.gate_delta+task.metadata+task.QStart : nullptr,CONFIG.getQBlockSize());
            kt::warp::load_async_commit_group(prefetch.getBarrier());
            prefetch.submitToNextAndTrigger();
            prefetch_pipe.moveNext();

            // Q must be in registers before the union changes to K ring.
            prefetch_pipe.waitSlot(0);
            load_pipe.setup();
            int last_slot = 0;
#pragma unroll 1
            for (int tile = 0; tile < task.KBlocks; ++tile) {
                auto packet =
                    load_pipe.waitBuffer(CONSUMER_A, shared.Tiles.Default, shared.Vectors.Default);
                if (tile != 0) { // K0 is already in slot0 from prefetch.
                    auto &key = packet.template get<0>().kbuffer;
                    auto &meta = packet.template get<1>();
                    if (leader) {
                        kt::tma::expect_bytes(packet.getBarrier(), sizeof(key));
                        kt::tma::load_async(key, args.KVec,
                                            {0, task.begin+tile * CONFIG.WarpKSize, task.Head, 0},
                                            packet.getBarrier());
                    }
                    load_key_vector_async(meta.k_idx, args.IdxK,
                                         {0, 0, 0, int(task.metadata)+tile * CONFIG.WarpKSize});
                    if (!query_lse) {
                        load_key_vector_async(meta.klse, args.KLseVec,
                                             {0, 0, 0, int(task.metadata)+tile * CONFIG.WarpKSize});
                    }
                    kt::warp::load_async_commit_group(packet.getBarrier());
                }
                last_slot = packet.SlotIdx;
                packet.submitToNextAndTrigger();
                load_pipe.moveNext();
            }
            // Both groups release in order. Draining the last tile protects
            // the entire union; consumers can still be reducing/storing it.
            load_pipe.waitSlot(last_slot);
        }
    }
}

} // namespace dism_varlen

} // namespace DISM_VARIANT
