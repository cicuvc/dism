#include "summary/kernel_common.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

// Original 32x64 MMA, score and scan.
template<class Configuration = ActiveConfig>
__global__ __launch_bounds__(384, 1) void wmma_tma_wasp_persistent_summarization_kernel(
    const __grid_constant__ TmaSummarizationKernelArgs args) {
    extern __shared__ __align__(1024) unsigned char smem[];
    kt::shared_allocator<1024> alloc{reinterpret_cast<int *>(smem)};
    const uint32_t group = kt::warpgroup::groupid();
    const uint32_t warp_in_group = kt::warpgroup::warpid();
    constexpr uint32_t CONSUMER_A = 0, CONSUMER_B = 1, PRODUCER = 2;

    MbarrierRingPipe load_pipe(alloc,
                               BufferSet{&LoadSharedMemoryLayouts::TileRegion::Default,
                                         &LoadSharedMemoryLayouts::VectorRegion::Default},
                               group, ProducerWarp<true, PRODUCER>,
                               ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>);
    MbarrierRingPipe prefetch_pipe(alloc,
                                   BufferSet{&LoadSharedMemoryLayouts::TileRegion::Prefetch,
                                             &LoadSharedMemoryLayouts::VectorRegion::Prefetch},
                                   group, ProducerWarp<true, PRODUCER>,
                                   ConsumerWarpGroup<false, CONSUMER_A, CONSUMER_B>);
    auto &shared = alloc.allocate<LoadSharedMemoryLayouts>();
    FixedLengthScheduler scheduler(args.Batch, args.Seqlen, args.Head);

    if (group == PRODUCER) {
        produce_summary_tasks(args, load_pipe, prefetch_pipe, shared, scheduler, warp_in_group);
    } else {
        kt::warpgroup::increase_registers<232>();
        using Scan = pscore::AltLayoutSplitScanBuffer<CONFIG.WarpQSize, CONFIG.WarpKSize,
                                                      pscore::BinaryElement, LogAffineOp>;
        FixedLengthScheduler::TaskInfo task;
#pragma unroll 1
        while (scheduler.getNextTask(task)) {
            // TK tile group-load interleaves the two WGs: 0,4,1,5,2,6,3,7.
            int query_block = group + 2 * warp_in_group;
            int q_start = task.QStart + query_block * CONFIG.WarpQSize;
            int checkpoint = q_start / CONFIG.WarpQSize;
            typename Scan::VState left;

            prefetch_pipe.setup();
            auto prefetch =
                prefetch_pipe.waitBuffer(PRODUCER, shared.Tiles.Prefetch, shared.Vectors.Prefetch);
            bool query_lse = shared.direction;
            float tau2 = shared.tau2;
            auto &vectors = prefetch.template get<1>();
            kt::rt_bf<CONFIG.WarpQSize, CONFIG.KcKeyDim> query;
            kt::rv<int, CONFIG.WarpQSize, kt::ducks::rv_layout::ortho> query_labels;
            kt::rv_fl<CONFIG.WarpQSize, kt::ducks::rv_layout::ortho> query_bias;
            kt::group<8>::load(query, prefetch.template get<0>().qbuffer);
            kt::warp::load(query_labels,
                           vectors.q_idx.template subvec<CONFIG.WarpQSize>(query_block));
            kt::warp::rv_maps::zero(query_bias);
            if (query_lse) {
                kt::warp::load(query_bias,
                               vectors.qlse.template subvec<CONFIG.WarpQSize>(query_block));
            }
#pragma unroll
            for (int row = 0; row < CONFIG.WarpQSize / 16; ++row) {
                query_bias.data[row][0].x *= -1.f;
                query_bias.data[row][0].y *= -1.f;
            }
            uint32_t hard_rows = 0;
#pragma unroll
            for (int row = 0; row < CONFIG.WarpQSize / 8; ++row) {
                int index = query_block * CONFIG.WarpQSize + row * 8 + kt::warp::laneid() / 4;
                hard_rows |= uint32_t(vectors.hard_flags[index] != 0) << row;
            }
            // Keep each compute WG in step before releasing the shared input.
            kt::warpgroup::sync(group + 1);
            prefetch.submitToNextAndTrigger();
            prefetch_pipe.moveNext();

            load_pipe.setup();
#pragma unroll 1
            for (int tile = 0; tile < task.KBlocks; ++tile) {
                auto packet =
                    load_pipe.waitBuffer(PRODUCER, shared.Tiles.Default, shared.Vectors.Default);
                // Inactive warps still release every input slot. They cannot
                // return: prefetch/WG retirement and future tasks need them.
                if (checkpoint >= args.Checkpoints) {
                    packet.submitToNextAndTrigger();
                    load_pipe.moveNext();
                    continue;
                }
                kt::rt_bf<CONFIG.WarpKSize, CONFIG.KcKeyDim> key;
                kt::rv<int, CONFIG.WarpKSize, kt::ducks::rv_layout::align> key_labels;
                kt::rv_fl<CONFIG.WarpKSize, kt::ducks::rv_layout::align> key_bias;
                kt::rt_fl<CONFIG.WarpQSize, CONFIG.WarpKSize> score;
                kt::warp::rv_maps::zero(key_bias);
                load_key_metadata(key_labels, key_bias, packet.template get<1>(), query_lse);
                // swap_axis=true: direct RHS LDSM -> HMMA layout.
                kt::warp::load<true>(key, packet.template get<0>().kbuffer.payload);
                initialize_score(score, query_bias, key_bias, query_lse);
                [[clang::always_inline]] kt::warp::wmma::mma_ABt(score, query, key, score);
                // Every reader publishes its own release arrival. The producer
                // waits for all256; no additional WG rendezvous is required.
                packet.submitToNextAndTrigger();
                load_pipe.moveNext();

                finish_score(score, query_labels, key_labels, hard_rows, tau2);
                auto scan = make_scan(score);
                auto result = scan.reduce_forward(left, {});
                left = result.first;
                auto bottom = result.second;
                store_summary(args, bottom, task.Batch, task.Head, checkpoint,
                              tile * CONFIG.WarpKSize);
            }
            // The scheduler omits key tiles strictly above this CTA's diagonal.
            if (checkpoint < args.Checkpoints) {
                int64_t offset = ((int64_t(task.Batch) * args.Head + task.Head) * args.Checkpoints +
                                  checkpoint) *
                                 args.PaddedSeqlen;
                for (int column = task.KBlocks * CONFIG.WarpKSize + kt::warp::laneid();
                     column < args.PaddedSeqlen; column += 32) {
                    args.SummaryA[offset + column] = LOG_ZERO;
                    args.SummaryB[offset + column] = LOG_ZERO;
                }
            }
        }
    }
    // Each register-allocation group retires together, never a 128-thread
    // barrier shared ambiguously by all384 threads.
    kt::warpgroup::sync(group + 1);
}

const void* summary_address() { return reinterpret_cast<const void*>(wmma_tma_wasp_persistent_summarization_kernel<ActiveConfig>); }

} // namespace DISM_VARIANT
