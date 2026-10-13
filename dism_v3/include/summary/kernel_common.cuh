#pragma once

#include "summary/primitives.cuh"

#include "summary/key_metadata.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

// LSE already includes -rtau. Seed the accumulator in natural-log units;
// LOG2E is applied only after MMA, never to the seed alone.
template <class Score, class QBias, class KBias>
__device__ __forceinline__ void initialize_score(Score &score, const QBias &q_bias,
                                                 const KBias &k_bias, bool query_lse) {
    if (query_lse)
        kt::warp::rt_maps::add_row(score, {0.f}, q_bias);
    else
        kt::warp::rt_maps::add_col(score, {0.f}, k_bias);
}

// Soft-only conversion. Hard training rows are skipped at the task boundary.
// Noncausal scores cannot influence a causal bottom-edge summary.
template <class Score>
__device__ __forceinline__ void finish_score(Score &score) {
#pragma unroll
    for (int rb = 0; rb < score.height; ++rb) {
#pragma unroll
        for (int cb = 0; cb < score.width; ++cb) {
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
#pragma unroll
                for (int ch = 0; ch < 2; ++ch) {
                    auto &value = score.tiles[rb][cb].data[rh + 2 * ch];
                    value.x *= 1.4426950408889634f;
                    value.y *= 1.4426950408889634f;
                }
            }
        }
    }
}

// HState now contains every bottom-edge column, including the corner. No
// special last-lane store from VState, unlike v2's shifted horizontal state.
template <class HState, class SummaryArgs>
__device__ __forceinline__ void store_summary(const SummaryArgs &args,
                                              const HState &bottom, int batch, int head,
                                              int checkpoint, int key_start) {
    if (checkpoint >= args.Checkpoints)
        return;
    int lane = kt::warp::laneid();
    int column = 16 * (lane & 3) + 7 - lane / 4;
    int64_t offset =
        ((int64_t(batch) * args.Head + head) * args.Checkpoints + checkpoint) * args.PaddedSeqlen +
        key_start;
    auto value = bottom.init[0];
    args.SummaryA[offset + column] = value.first.u0;
    args.SummaryA[offset + column + 8] = value.first.u1;
    args.SummaryB[offset + column] = value.second.u0;
    args.SummaryB[offset + column + 8] = value.second.u1;
}

// Producer protocol for the persistent summary kernel.
template <class LoadPipe, class PrefetchPipe>
__device__ __forceinline__ void
produce_summary_tasks(const TmaSummarizationKernelArgs &args, LoadPipe &load_pipe,
                      PrefetchPipe &prefetch_pipe, LoadSharedMemoryLayouts &shared,
                      FixedLengthScheduler &scheduler, uint32_t warp_in_group) {
    constexpr uint32_t CONSUMER_A = 0;
    kt::warpgroup::decrease_registers<40>();
    // Keep the whole warp for coalesced cp.async metadata loads.
    if (warp_in_group == 0) {
        bool leader = kt::warp::elect_leader();
        FixedLengthScheduler::TaskInfo task;
#pragma unroll 1
        while (scheduler.getNextTask(task)) {
            if (args.hard_flags[task.Batch * args.Head + task.Head]) continue;
            bool query_lse = args.direction[task.Batch * args.Head + task.Head];

            prefetch_pipe.setup();
            auto prefetch = prefetch_pipe.waitBuffer(CONSUMER_A, shared.Tiles.Prefetch,
                                                     shared.Vectors.Prefetch);
            if (leader) {
                shared.direction = query_lse;
            }
            auto &tiles = prefetch.template get<0>();
            auto &vectors = prefetch.template get<1>();
            if (leader) {
                kt::tma::expect_bytes(prefetch.getBarrier(),
                                      sizeof(tiles.qbuffer) + sizeof(tiles.preflight_k[0]));
                kt::tma::load_async(tiles.qbuffer, args.QVec,
                                    {task.Batch, task.Head, task.QStart, 0}, prefetch.getBarrier());
                kt::tma::load_async(tiles.preflight_k[0].kbuffer, args.KVec,
                                    {task.Batch, 0, task.Head, 0}, prefetch.getBarrier());
            }
            if (query_lse) {
                kt::warp::load_async(vectors.qlse, args.QLseVec,
                                     {task.Batch, task.Head, 0, task.QStart});
            } else {
                load_key_vector_async(vectors.preflight_k[0].klse, args.KLseVec,
                                     {task.Batch, task.Head, 0, 0});
            }
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
                                            {task.Batch, tile * CONFIG.WarpKSize, task.Head, 0},
                                            packet.getBarrier());
                    }
                    if (!query_lse) {
                        load_key_vector_async(meta.klse, args.KLseVec,
                                             {task.Batch, task.Head, 0, tile * CONFIG.WarpKSize});
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

} // namespace DISM_VARIANT
