#include "common/graph_route.cuh"
#include "summary/hard.cuh"

namespace DISM_VARIANT {

template<class Configuration = ActiveConfig>
__global__ __launch_bounds__(384, 1) void hard_summary_kernel(
        const __grid_constant__ HardSummaryArgs args, const uint32_t* choice) {
    if (!dism_graph_route::quadratic(choice)) return;
    extern __shared__ __align__(1024) unsigned char smem[];
    kt::shared_allocator<1024> alloc{reinterpret_cast<int*>(smem)};
    const uint32_t group = kt::warpgroup::groupid();
    const uint32_t warp = kt::warpgroup::warpid();
    constexpr uint32_t A = 0, B = 1, P = 2;
    MbarrierRingPipe load_pipe(alloc, BufferSet{&HardSummaryShared::keys}, group,
        ProducerWarp<true, P>, ConsumerWarpGroup<false, A, B>);
    MbarrierRingPipe prefetch_pipe(alloc, BufferSet{&HardSummaryShared::prefetch}, group,
        ProducerWarp<true, P>, ConsumerWarpGroup<false, A, B>);
    auto& shared = alloc.allocate<HardSummaryShared>();
    HardSummaryScheduler scheduler(args);

    if (group == P) {
        kt::warpgroup::decrease_registers<40>();
        if (warp == 0) {
            HardSummaryTask task;
            while (scheduler.next(task)) {
                if (!args.hard[task.hard_index]) continue;
                prefetch_pipe.setup();
                auto prefetch = prefetch_pipe.waitBuffer(A, shared.prefetch);
                load_async_any(prefetch.template get<0>(),
                    reinterpret_cast<uint8_t*>(const_cast<int*>(args.query + task.metadata + task.q_start)));
                load_async_any(shared.keys[0],
                    reinterpret_cast<uint8_t*>(const_cast<int*>(args.key + task.metadata)));
                // Prefetch is immediately followed by a consumer handshake,
                // so there is no producer lookahead here. Explicit completion
                // also avoids the documented ARRIVES [UR+offset] racecheck
                // ambiguity. The steady-state K ring below remains async.
                kt::warp::load_async_commit_group();
                kt::warp::load_async_wait<0>();
                prefetch.submitToNextAndTrigger();
                prefetch_pipe.moveNext();
                prefetch_pipe.waitSlot(0);

                load_pipe.setup();
                int end = min(task.q_start + 256, (task.n - 1) / 32 * 32);
                int tiles = (end + 63) / 64, last_slot = 0;
                for (int tile = 0; tile < tiles; ++tile) {
                    auto packet = load_pipe.waitBuffer(A, shared.keys);
                    if (tile != 0) {
                        load_async_any(packet.template get<0>(),
                            reinterpret_cast<uint8_t*>(const_cast<int*>(args.key + task.metadata + tile * 64)));
                        kt::warp::load_async_commit_group(packet.getBarrier());
                    }
                    last_slot = packet.SlotIdx;
                    packet.submitToNextAndTrigger();
                    load_pipe.moveNext();
                }
                load_pipe.waitSlot(last_slot);
            }
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        HardSummaryTask task;
        while (scheduler.next(task)) {
            if (!args.hard[task.hard_index]) continue;
            int query_block = group + 2 * warp; // unchanged interleaved WG placement
            int q_start = task.q_start + query_block * 32;
            int checkpoint = q_start / 32, checkpoints = (task.n - 1) / 32;
            HardSummaryScan::VState left;
            prefetch_pipe.setup();
            auto prefetch = prefetch_pipe.waitBuffer(P, shared.prefetch);
            kt::warpgroup::sync(group + 1);
            prefetch.submitToNextAndTrigger();
            prefetch_pipe.moveNext();

            load_pipe.setup();
            int end = min(task.q_start + 256, checkpoints * 32);
            int tiles = (end + 63) / 64;
            for (int tile = 0; tile < tiles; ++tile) {
                auto packet = load_pipe.waitBuffer(P, shared.keys);
                if (checkpoint >= checkpoints) {
                    packet.submitToNextAndTrigger();
                    load_pipe.moveNext();
                    continue;
                }
                auto scan = make_hard_summary_scan(shared.prefetch[0].labels + query_block * 32,
                    packet.template get<0>().labels, tile * 64);
                // Every Q/K shared read must precede this release, including Q
                // reads on the last K tile. Subsequent work is register-only.
                packet.submitToNextAndTrigger();
                load_pipe.moveNext();
                auto result = scan.reduce_forward(left, {});
                left = result.first;
                int lane = kt::warp::laneid();
                int col = tile * 64 + 16 * (lane & 3) + 7 - lane / 4;
                int64_t offset = task.output + int64_t(checkpoint) * task.n;
                auto bottom = result.second.init[0].value;
                if (col <= q_start + 31) args.summary[offset + col] = bottom.u0;
                if (col + 8 <= q_start + 31) args.summary[offset + col + 8] = bottom.u1;
            }
        }
    }
    kt::warpgroup::sync(group + 1);
}

const void* hard_summary_address() {
    return reinterpret_cast<const void*>(hard_summary_kernel<ActiveConfig>);
}
} // namespace DISM_VARIANT
