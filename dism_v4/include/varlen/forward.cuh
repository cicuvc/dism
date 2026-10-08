#pragma once
#include "forward/producer.cuh"
#include "varlen/operands.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_varlen {
template<bool FP32Output> using ForwardArgs=PackedArgs<dism_forward::ArgsT<FP32Output>>;
struct ForwardTask : dism_forward::Scheduler::Task {
    int begin,n,checkpoints;
    int64_t metadata,forward,vertical;
};
template<bool FP32Output>
__device__ __forceinline__ ForwardTask forward_task(const ForwardArgs<FP32Output>& packed) {
    auto work=packed.tasks[blockIdx.x];
    const auto* row=packed.documents+int64_t(work.x)*Fields;
    int n=row[Length],begin=row[Begin],blocks=n/dism_forward::QROWS;
    int head=work.y/blocks,q0=(work.y%blocks)*dism_forward::QROWS;
    int h=packed.common.heads;
    return {{0,head,q0,(q0+dism_forward::QROWS)/dism_forward::WarpKSize},
            begin,n,(n-1)/32,int64_t(head)*packed.common.n+begin,
            row[ForwardOffset]*h,row[VerticalOffset]*h};
}
using namespace dism_forward;

template <bool FP32Output, class InputPipe, class PrefetchPipe>
__device__ __forceinline__ void produce_packed_forward(const ArgsT<FP32Output>& args, SharedT<FP32Output>& shared,
                                        InputPipe& input, PrefetchPipe& prefetch,
                                        const ForwardTask& task) {
    kt::warpgroup::decrease_registers<40>();
    if (kt::warpgroup::warpid() != 0) return;
    bool leader = kt::warp::elect_leader();
    {
        prefetch.setup();
        auto first = prefetch.waitBuffer(0, shared.tiles.prefetch, shared.metadata.prefetch);
        auto& tiles = first.template get<0>();
        auto& metadata = first.template get<1>();
        bool query_lse = args.direction[task.batch * args.heads + task.head];
        if (leader) {
            shared.query_lse = query_lse;
            shared.tau2 = args.tau[task.head] * 1.4426950408889634f;
            kt::tma::expect_bytes(first.getBarrier(), sizeof(QTile) + sizeof(SQTile));
            kt::tma::load_async(tiles.q, args.q, {0, task.head, task.begin+task.q_start, 0}, first.getBarrier());
            kt::tma::load_async(tiles.sq, args.sq, {0, task.head, task.begin+task.q_start, 0}, first.getBarrier());
        }
        kt::warp::load_async(metadata.labels, args.q_label, {0, 0, 0, int(task.metadata)+task.q_start});
        if (query_lse) {
            kt::warp::load_async(metadata.lse, args.q_lse, {0, 0, 0, int(task.metadata)+task.q_start});
        }
        load_async_any(metadata.hard, args.hard +
            task.metadata + task.q_start);
        load_row_gate(shared.row_gate,args.gate_delta ? args.gate_delta+task.metadata+task.q_start : nullptr,QROWS);
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
                kt::tma::load_async(kv.k, args.k, {0, task.begin+tile * WarpKSize, task.head, 0}, packet.getBarrier());
                kt::tma::load_async(kv.sk, args.sk, {0, task.begin+tile * WarpKSize, task.head, 0}, packet.getBarrier());
                kt::tma::load_async(kv.v, args.v, {0, task.begin+tile * WarpKSize, task.head, 0}, packet.getBarrier());
            }
            load_key_vector_async(meta.k_idx, args.k_label, {0, 0, 0, int(task.metadata)+tile * WarpKSize});
            if (!query_lse) {
                load_key_vector_async(meta.klse, args.k_lse, {0, 0, 0, int(task.metadata)+tile * WarpKSize});
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

} // namespace dism_varlen

} // namespace DISM_VARIANT
