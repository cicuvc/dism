#include "common/graph_route.cuh"
#include "forward/hard.cuh"

namespace DISM_VARIANT {
namespace hf = dism_hard_forward;

// Stable geometric weights in log2 units. The table is invocation-owned and
// rebuilt from current per-head tau, never cached across optimizer updates.
template<class Configuration = ActiveConfig>
__global__ void hard_log_weights_kernel(const float* tau,float* output,int stride,const uint32_t* choice) {
    if (!dism_graph_route::quadratic(choice)) return;
    int length = blockIdx.x*blockDim.x+threadIdx.x;
    if (length>=stride) return;
    float t = tau[blockIdx.y];
    float result = LOG_ZERO;
    if (length>0) {
        if (t==0.f) result = log2f(float(length));
        else {
            float a = fabsf(t);
            result = (t>0.f ? float(length)*t : t)*1.4426950408889634f
                +log2f(-expm1f(-float(length)*a))-log2f(-expm1f(-a));
        }
    }
    output[int64_t(blockIdx.y)*stride+length] = result;
}

template<class Configuration = ActiveConfig>
__global__ __launch_bounds__(384,1) void hard_forward_kernel(
        const __grid_constant__ hf::Args args, const uint32_t* choice) {
    if (!dism_graph_route::quadratic(choice)) return;
    using namespace dism_forward;
    using hf::IntScan;
    using Shared = hf::Shared;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> allocator(reinterpret_cast<int*>(storage));
    int group=kt::warpgroup::groupid(),warp=kt::warpgroup::warpid();
    MbarrierRingPipe input(allocator,BufferSet{&Shared::Tiles::kv,&Shared::Metadata::kv},
        group,ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    MbarrierRingPipe prefetch(allocator,BufferSet{&Shared::Tiles::prefetch,&Shared::Metadata::prefetch},
        group,ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    MbarrierRingPipe mail(allocator,BufferSet{&Shared::mail},min(group,1),
        ConsumerWarpGroup<true,0>,ConsumerWarpGroup<false,1>);
    auto& shared=allocator.allocate<Shared>();
    hf::Scheduler scheduler(args);
    if (group==2) hf::produce(args,shared,input,prefetch,scheduler);
    else {
        kt::warpgroup::increase_registers<232>();
        hf::Task task;
        while (scheduler.next(task)) {
            if (!args.hard[task.hard_index]) continue;
            int query_block=2*warp+group;
            int q_start=task.q_start+16*query_block;
            kt::rt_bf<16,READOUT_DIM> soft_query;
            prefetch.setup();
            auto first=prefetch.waitBuffer(2,shared.tiles.prefetch,shared.metadata.prefetch);
            kt::group<8>::load(soft_query,first.template get<0>().sq);
            kt::warpgroup::sync(group+1);
            first.submitToNextAndTrigger();
            prefetch.moveNext();
            IntScan::VState left;
            kt::rt_fl<16,DIM> accum;
            kt::warp::rt_maps::zero(accum);
            float maximum[2]{0.f,0.f},denominator[2]{1.f,1.f};
            input.setup();
            for (int tile=0;tile<(task.q_start+QROWS)/WarpKSize;++tile) {
                auto packet=input.waitBuffer(2,shared.tiles.kv,shared.metadata.kv);
                auto& kv=packet.template get<0>();
                auto scan=hf::make_scan(shared.metadata.prefetch[0].labels+query_block*16,
                    packet.template get<1>().labels,tile*WarpKSize);
                IntScan::HState incoming;
                if (group==0) {
                    int checkpoint=q_start/32-1;
                    if (checkpoint>=0 && boundary_owner()) {
                        int col=tile*WarpKSize+boundary_column();
                        int64_t offset=task.boundary+int64_t(checkpoint)*task.n+col;
                        incoming.init[0].value={args.boundary[offset],args.boundary[offset+Scalar::COL_BLOCKS]};
                    }
                } else {
                    auto received=mail.waitBuffer(0,shared.mail);
                    auto value=received.template get<0>().value[warp][kt::warp::laneid()];
                    incoming.init[0].value={value.x,value.y};
                    received.submitToNextAndTrigger();
                    mail.moveNext();
                }
                auto state=scan.inclusive_prescan(left,incoming);
                left=state.vertical;
                if (group==0) {
                    auto sent=mail.waitBuffer(1,shared.mail);
                    auto value=state.horizontal.init[0].value;
                    sent.template get<0>().value[warp][kt::warp::laneid()]=make_int2(value.u0,value.u1);
                    sent.submitToNextAndTrigger();
                    mail.moveNext();
                }
                auto values=hf::finish_scan(scan,incoming,state.intermediate,
                    args.log_weights+int64_t(task.head)*args.lut_stride,q_start,tile*WarpKSize);
                kt::rt_fl<16,WarpKSize> similarity;
                kt::warp::rt_maps::zero(similarity);
                {
                    kt::rt_bf<WarpKSize,READOUT_DIM> soft_key;
                    kt::warp::load<true>(soft_key,kv.sk.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_ABt(similarity,soft_query,soft_key,similarity);
                }
                kt::rt_bf<16,WarpKSize> weights;
                readout(values,similarity,weights,accum,maximum,denominator,q_start,tile*WarpKSize,task.n);
                {
                    kt::rt_bf<WarpKSize,DIM,kt::ducks::rt_layout::col> value;
                    kt::warp::load(value,kv.v.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_AB(accum,weights,value,accum);
                }
                packet.submitToNextAndTrigger();
                input.moveNext();
            }
            store_output<true>(args,task,task.begin+q_start,accum,maximum,denominator,
                shared,task.metadata+q_start);
        }
    }
    kt::warpgroup::sync(group+1);
}

const void* hard_forward_address() { return reinterpret_cast<const void*>(hard_forward_kernel<ActiveConfig>); }
const void* hard_log_weights_address() { return reinterpret_cast<const void*>(hard_log_weights_kernel<ActiveConfig>); }
} // namespace DISM_VARIANT
