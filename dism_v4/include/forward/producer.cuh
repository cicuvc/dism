#pragma once
#include "forward/types.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_forward {
template <bool FP32Output, class InputPipe, class PrefetchPipe>
__device__ __forceinline__ void produce(const ArgsT<FP32Output>& args, SharedT<FP32Output>& shared,
                                        InputPipe& input, PrefetchPipe& prefetch,
                                        Scheduler& scheduler) {
    kt::warpgroup::decrease_registers<40>();
    if (kt::warpgroup::warpid() != 0) return;
    bool leader = kt::warp::elect_leader();
    Scheduler::Task task;
#pragma unroll 1
    while (scheduler.next(task)) {
        prefetch.setup();
        auto first = prefetch.waitBuffer(0, shared.tiles.prefetch, shared.metadata.prefetch);
        auto& tiles = first.template get<0>();
        auto& metadata = first.template get<1>();
        bool query_lse = args.direction[task.batch * args.heads + task.head];
        if (leader) {
            shared.query_lse = query_lse;
            shared.tau2 = args.tau[task.head] * 1.4426950408889634f;
            kt::tma::expect_bytes(first.getBarrier(), sizeof(QTile) + sizeof(SQTile));
            kt::tma::load_async(tiles.q, args.q, {task.batch, task.head, task.q_start, 0}, first.getBarrier());
            kt::tma::load_async(tiles.sq, args.sq, {task.batch, task.head, task.q_start, 0}, first.getBarrier());
        }
        kt::warp::load_async(metadata.labels, args.q_label, {task.batch, task.head, 0, task.q_start});
        if (query_lse) {
            kt::warp::load_async(metadata.lse, args.q_lse, {task.batch, task.head, 0, task.q_start});
        }
        load_async_any(metadata.hard, args.hard +
            (int64_t(task.batch) * args.heads + task.head) * args.padded + task.q_start);
        load_row_gate(shared.row_gate,args.gate_delta ? args.gate_delta+(int64_t(task.batch)*args.heads+task.head)*args.padded+task.q_start : nullptr,QROWS);
        kt::warp::load_async_commit_group(first.getBarrier());
        first.submitToNextAndTrigger();
        prefetch.moveNext();
        // Q/SQ must be resident in consumer registers before reusing their union.
        prefetch.waitSlot(0);
        input.setup();
        int last_slot = 0;
#pragma unroll 1
        for (int tile = 0; tile < task.key_blocks; ++tile) {
            auto packet = input.waitBuffer(0, shared.tiles.kv, shared.metadata.kv);
            auto& kv = packet.template get<0>();
            auto& meta = packet.template get<1>();
            if (leader) {
                kt::tma::expect_bytes(packet.getBarrier(), sizeof(KTile) + sizeof(SKTile) + sizeof(VTile));
                kt::tma::load_async(kv.k, args.k, {task.batch, tile * WarpKSize, task.head, 0}, packet.getBarrier());
                kt::tma::load_async(kv.sk, args.sk, {task.batch, tile * WarpKSize, task.head, 0}, packet.getBarrier());
                kt::tma::load_async(kv.v, args.v, {task.batch, tile * WarpKSize, task.head, 0}, packet.getBarrier());
            }
            load_key_vector_async(meta.k_idx, args.k_label, {task.batch, task.head, 0, tile * WarpKSize});
            if (!query_lse) {
                load_key_vector_async(meta.klse, args.k_lse, {task.batch, task.head, 0, tile * WarpKSize});
            }
            kt::warp::load_async_commit_group(packet.getBarrier());
            last_slot = packet.SlotIdx;
            packet.submitToNextAndTrigger();
            input.moveNext();
        }
        // All K/SK/V readers finish before prefetching the next Q/SQ into the union.
        input.waitSlot(last_slot);
    }
}
} // namespace dism_forward

} // namespace DISM_VARIANT
