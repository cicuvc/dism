#include "forward/producer.cuh"
#include "forward/readout.cuh"
#include "varlen/forward.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

using namespace dism_forward;

template<bool FP32Output, class Configuration = ActiveConfig>
__global__ __launch_bounds__(384, 1) void varlen_forward_kernel(
        const __grid_constant__ dism_varlen::ForwardArgs<FP32Output> packed) {
    using Shared = SharedT<FP32Output>;
    const auto& args=packed.common;
    const auto task=dism_varlen::forward_task(packed);
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> allocator(reinterpret_cast<int*>(storage));
    int group = kt::warpgroup::groupid();
    int warp = kt::warpgroup::warpid();
    MbarrierRingPipe input(allocator, BufferSet{&Shared::Tiles::kv, &Shared::Metadata::kv},
                          group, ProducerWarp<true, 2>, ConsumerWarpGroup<false, 0, 1>);
    MbarrierRingPipe prefetch(allocator,
        BufferSet{&Shared::Tiles::prefetch, &Shared::Metadata::prefetch},
        group, ProducerWarp<true, 2>, ConsumerWarpGroup<false, 0, 1>);
    // Producer WG initializes the object collectively but never uses this pipe.
    MbarrierRingPipe mail(allocator, BufferSet{&Shared::mail}, min(group, 1),
                          ConsumerWarpGroup<true, 0>, ConsumerWarpGroup<false, 1>);
    auto& shared = allocator.allocate<Shared>();
    if (group == 2) {
        dism_varlen::produce_packed_forward(args, shared, input, prefetch, task);
    } else {
        kt::warpgroup::increase_registers<232>();
        {
            int query_block = 2 * warp + group;
            int q_start = task.q_start + 16 * query_block;
            kt::rt_bf<16, CONFIG.KcKeyDim> query;
            kt::rt_bf<16, READOUT_DIM> soft_query;
            kt::rv<int,16,kt::ducks::rv_layout::ortho> labels;
            kt::rv_fl<16,kt::ducks::rv_layout::ortho> bias;
            kt::warp::rv_maps::zero(bias);
            prefetch.setup();
            auto first = prefetch.waitBuffer(2, shared.tiles.prefetch, shared.metadata.prefetch);
            bool query_lse = shared.query_lse;
            float tau2 = shared.tau2;
            auto& meta = first.template get<1>();
            kt::group<8>::load(query, first.template get<0>().q);
            kt::group<8>::load(soft_query, first.template get<0>().sq);
            kt::warp::load(labels, meta.labels.template subvec<16>(query_block));
            if (query_lse) kt::warp::load(bias, meta.lse.template subvec<16>(query_block));
            bias.data[0][0].x *= -1.f;
            bias.data[0][0].y *= -1.f;
            uint32_t hard = 0;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                int index = 16 * query_block + 8 * r + kt::warp::laneid() / 4;
                hard |= uint32_t(meta.hard[index] != 0) << r;
            }
            kt::warpgroup::sync(group + 1);
            first.submitToNextAndTrigger();
            prefetch.moveNext();

            Scan::VState left;
            kt::rt_fl<16,DIM> accum;
            kt::warp::rt_maps::zero(accum);
            float maximum[2]{0.f, 0.f}, denominator[2]{1.f, 1.f};
            input.setup();
            // Mail ring retains its slot/phase across tasks. Each pair performs
            // exactly one publish/consume per key tile, including padded warps.
#pragma unroll 1
            for (int tile = 0; tile < task.key_blocks; ++tile) {
                auto packet = input.waitBuffer(2, shared.tiles.kv, shared.metadata.kv);
                auto& kv = packet.template get<0>();
                kt::rt_fl<16,WarpKSize> score;
                {
                    kt::rt_bf<WarpKSize,CONFIG.KcKeyDim> key;
                    kt::rv<int,WarpKSize,kt::ducks::rv_layout::align> key_labels;
                    kt::rv_fl<WarpKSize,kt::ducks::rv_layout::align> key_bias;
                    kt::warp::rv_maps::zero(key_bias);
                    load_forward_key_metadata(key_labels, key_bias, packet.template get<1>(), query_lse);
                    kt::warp::load<true>(key, kv.k.payload);
                    initialize_score(score, bias, key_bias, query_lse);
                    [[clang::always_inline]] kt::warp::wmma::mma_ABt(score, query, key, score);
                    finish_score(score, labels, key_labels, hard, tau2);
                }
                auto scan = make_forward_scan(score);
                Scan::HState incoming;
                if (group == 0) {
                    int checkpoint = q_start / 32 - 1;
                    if (checkpoint >= 0 && checkpoint < task.checkpoints && boundary_owner()) {
                        int column = tile * WarpKSize + boundary_column();
                        int64_t base = task.forward+((int64_t(task.batch) * args.heads + task.head) *
                                        task.checkpoints + checkpoint) * task.n;
                        incoming.init[0].second = {args.boundary[base + column],
                                                   args.boundary[base + column + Scalar::COL_BLOCKS]};
                    }
                } else {
                    auto received = mail.waitBuffer(0, shared.mail);
                    auto value = received.template get<0>().value[warp][kt::warp::laneid()];
                    incoming.init[0].second = {value.x, value.y};
                    received.submitToNextAndTrigger();
                    mail.moveNext();
                }
                auto state = scan.inclusive_prescan(left, incoming);
                left = state.vertical;
                if (group == 0) {
                    auto sent = mail.waitBuffer(1, shared.mail);
                    auto value = state.horizontal.init[0].second;
                    sent.template get<0>().value[warp][kt::warp::laneid()] =
                        make_float2(value.u0, value.u1);
                    sent.submitToNextAndTrigger();
                    mail.moveNext();
                }
                // Publication is complete before the deferred downsweep.
                auto values = finish_scan(scan, incoming, state.intermediate);
                if (args.vertical) {
                    constexpr int C = Scalar::COL_BLOCKS;
#pragma unroll
                    for (int r = 0; r < Scalar::ROW_BLOCKS; ++r) {
#pragma unroll
                        for (int c = 0; c < C; ++c) {
                            auto pos = Scalar::layout(r, c, 0);
#pragma unroll
                            for (int e = 0; e < 2; ++e) {
                                int column = tile * WarpKSize + pos.second + e * C;
                                int row = q_start + pos.first;
                                if ((column & 15) == 15 && column < task.n - 1 &&
                                    row < task.n && column <= row) {
                                    int64_t offset = task.vertical+((int64_t(task.batch) * args.heads + task.head) *
                                        ((task.n - 1) / 16) + column / 16) * task.n + row;
                                    args.vertical[offset] = e ? values.data[r][c].value.u1 :
                                                               values.data[r][c].value.u0;
                                }
                            }
                        }
                    }
                }
                kt::rt_fl<16,WarpKSize> similarity;
                kt::warp::rt_maps::zero(similarity);
                {
                    kt::rt_bf<WarpKSize,READOUT_DIM> soft_key;
                    kt::warp::load<true>(soft_key, kv.sk.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_ABt(
                        similarity, soft_query, soft_key, similarity);
                }
                kt::rt_bf<16,WarpKSize> weights;
                readout(values, similarity, weights, accum, maximum, denominator,
                        q_start, tile * WarpKSize, task.n);
                {
                    kt::rt_bf<WarpKSize,DIM,kt::ducks::rt_layout::col> value;
                    kt::warp::load(value, kv.v.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_AB(accum, weights, value, accum);
                }
                // K, SK and V all share a slot. Release only after the final PV.
                packet.submitToNextAndTrigger();
                input.moveNext();
            }
            store_output<true>(args, task, task.begin+q_start, accum, maximum, denominator,
                               shared,task.metadata+q_start);
        }
    }
    kt::warpgroup::sync(group + 1);
}

template<bool FP32> const void* varlen_forward_address() { return reinterpret_cast<const void*>(varlen_forward_kernel<FP32>); }
template const void* varlen_forward_address<false>();
#if DISM_ENABLE_FP32
template const void* varlen_forward_address<true>();
#endif

} // namespace DISM_VARIANT
