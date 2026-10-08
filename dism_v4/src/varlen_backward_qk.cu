#include "backward/gradients.cuh"
#include "varlen/backward.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

using namespace dism_backward;

template<bool FP32Output, class Configuration = ActiveConfig>
__global__ __launch_bounds__(384,1) void varlen_backward_qk_kernel(
        const __grid_constant__ dism_varlen::BackwardArgs packed) {
    const auto& args=packed.common;
    const auto task=dism_varlen::backward_task(packed);
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> allocator(reinterpret_cast<int*>(storage));
    int group=kt::warpgroup::groupid(),warp=kt::warpgroup::warpid();
    int lane=kt::warp::laneid();
    MbarrierRingPipe input(allocator,BufferSet{&Shared::input},group,
                          ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    MbarrierRingPipe held(allocator,BufferSet{&Shared::held},group,
                         ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    MbarrierRingPipe mail(allocator,BufferSet{&Shared::mail},1-min(group,1),
                         ConsumerWarpGroup<true,0>,ConsumerWarpGroup<false,1>);
    auto& shared=allocator.allocate<Shared>();
    if (group==2) {
        dism_varlen::produce_backward(args,shared,held,input,task);
    } else {
        kt::warpgroup::increase_registers<232>();
        const auto score_args=dism_varlen::local_score(args,task);
        {
            int k0=task.k0+16*(2*warp+group);
            kt::rt_bf<K,D> keys;
            kt::rt_bf<K,R> soft_keys;
            kt::rt_bf<K,DV> values;
            auto first=held.waitBuffer(2,shared.held);
            kt::group<8>::load(keys,first.template get<0>().k);
            kt::group<8>::load(soft_keys,first.template get<0>().sk);
            kt::group<8>::load(values,first.template get<0>().v);
            kt::warpgroup::sync(group+1);
            kt::rt_fl<K,D> dk{0.f};
            float key_lse[2]{0.f,0.f};
            float tau_sum=0.f;
            Reverse::VState right;
#pragma unroll 1
            for (int q0=((score_args.n+Q-1)/Q-1)*Q;q0>=task.k0;q0-=Q) {
                auto packet=input.waitBuffer(2,shared.input);
                auto& data=packet.template get<0>();
                auto score=recompute(score_args,keys,data.q,task.bh,k0,q0,
                    score_args.gate_delta ? data.row_gate : nullptr);
                kt::rt_fl<K,Q> a{0.f},b{0.f};
                {
                    kt::rt_bf<Q,R> soft_queries;
                    kt::warp::load<true>(soft_queries,data.sq.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_ABt(a,soft_keys,soft_queries,a);
                }
                {
                    kt::rt_bf<Q,DV> derivative;
                    kt::warp::load<true>(derivative,data.dout.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_ABt(b,values,derivative,b);
                }
                Reverse reverse;
                Scalar local_beta;
#pragma unroll
                for (int r=0;r<2;++r) {
#pragma unroll
                    for (int c=0;c<4;++c) {
                        auto pos=Scalar::layout(r,c,0);
                        auto av=a.tiles[0][c/2].data[r+2*(c&1)];
                        auto bv=b.tiles[0][c/2].data[r+2*(c&1)];
                        float alpha[2],beta[2];
#pragma unroll
                        for (int e=0;e<2;++e) {
                            int query=q0+pos.second+4*e;
                            bool valid=query<score_args.n && k0+pos.first<score_args.n;
                            int64_t off=task.metadata+query;
                            float norm=valid?args.normalizer[off]:INFINITY;
                            float delta=valid?args.delta[off]:0.f;
                            float w=e?score.data[r][c].value.u1:score.data[r][c].value.u0;
                            alpha[e]=valid ? ((query+1<score_args.n) ?
                                sigmoid2(w-(score_args.gate_delta ? data.row_gate[pos.second+4*e+1] : 0.f)) : 0.f) : 1.f;
                            beta[e]=exp2_ftz(w-norm)*fmaf(e?av.y:av.x,e?bv.y:bv.x,-delta);
                        }
                        reverse.data[r][c]={{alpha[0],alpha[1]},{beta[0],beta[1]}};
                        if(args.dgate) local_beta.data[r][c].value={beta[0],beta[1]};
                    }
                }
                reverse.reverse_roll();
                Reverse::HState bottom;
                if (group==1 && k0+16<score_args.n && lane/4>=4) {
                    int query=q0+8*(lane&3)+7-lane/4;
                    int chunks=(score_args.n+SummaryK-1)/SummaryK;
                    int64_t off=task.reverse+(int64_t(task.bh)*chunks+(k0+16)/SummaryK)*score_args.padded+query;
                    bottom.init[0].second={args.g_boundary[off],args.g_boundary[off+4]};
                }
                if (group==0) {
                    auto received=mail.waitBuffer(0,shared.mail);
                    auto pair=received.template get<0>().pairs[warp][lane];
                    bottom.init[0]={{pair.x,pair.y},{pair.z,pair.w}};
                    received.submitToNextAndTrigger();
                    mail.moveNext();
                }
                auto state=reverse.reverse_inclusive_prescan(right,bottom);
                right=state.vertical;
                if (group==1) {
                    auto sent=mail.waitBuffer(1,shared.mail);
                    auto pair=state.horizontal.init[0];
                    sent.template get<0>().pairs[warp][lane]=
                        make_float4(pair.first.u0,pair.first.u1,pair.second.u0,pair.second.u1);
                    sent.submitToNextAndTrigger();
                    mail.moveNext();
                }
                // Release the other WG before the postscan and gradient MMAs.
                reverse.reverse_inclusive_postscan(bottom,state.intermediate);
                Scalar gradient;
#pragma unroll
                for (int r=0;r<2;++r) {
#pragma unroll
                    for (int c=0;c<4;++c) gradient.data[r][c].value=reverse.data[r][c].second;
                }
                gradient.reverse_roll<false>();
                kt::rt_bf<K,Q> soft_gradient;
                bool query_lse=score_args.direction[task.bh];
                float row_lse[4][2]{};
                float gate_gradient[4][2]{};
#pragma unroll
                for (int r=0;r<2;++r) {
#pragma unroll
                    for (int c=0;c<4;++c) {
                        auto pos=Scalar::layout(r,c,0);
                        float soft[2];
#pragma unroll
                        for (int e=0;e<2;++e) {
                            int query=q0+pos.second+4*e;
                            bool valid=query<score_args.n && k0+pos.first<score_args.n;
                            float g=valid?(e?gradient.data[r][c].value.u1:gradient.data[r][c].value.u0):0.f;
#if DISM_BACKWARD_DEBUG
                            if (args.debug_g && valid)
                                args.debug_g[(int64_t(task.bh)*score_args.n+query)*score_args.n+k0+pos.first]=g;
#endif
                            tau_sum+=g;
                            // G_ij-beta_ij = alpha_(i+1,j+1)*G_(i+1,j+1).
                            // Sum this flow into the NEXT query's gate gradient.
                            // This uses the actual reverse recurrence and avoids
                            // recovering alpha from approximate forward logadd.
                            if(args.dgate && query+1<score_args.n)
                                gate_gradient[c][e]-=g-(e ? local_beta.data[r][c].value.u1 : local_beta.data[r][c].value.u0);
                            bool hard=score_args.hard[int64_t(task.bh)*score_args.padded+query];
                            soft[e]=hard?0.f:g;
                            if (query_lse) row_lse[c][e]-=soft[e];
                            if (!query_lse) key_lse[r]-=soft[e];
                        }
                        soft_gradient.tiles[0][c/2].data[r+2*(c&1)]=
                            __floats2bfloat162_rn(soft[0],soft[1]);
                    }
                }
                if(args.dgate) {
                    for(int c=0;c<4;++c) for(int e=0;e<2;++e) {
                        float sum=gate_gradient[c][e];
                        sum+=__shfl_xor_sync(0xffffffff,sum,4);
                        sum+=__shfl_xor_sync(0xffffffff,sum,8);
                        sum+=__shfl_xor_sync(0xffffffff,sum,16);
                        if(lane/4==0 && q0+c+(2*(lane&3)+e)*4+1<score_args.n) atomicAdd(args.dgate+task.metadata+q0+c+(2*(lane&3)+e)*4+1,sum);
                    }
                }
                if (query_lse) {
#pragma unroll
                    for (int c=0;c<4;++c) {
#pragma unroll
                        for (int e=0;e<2;++e) {
                            float sum=row_lse[c][e];
                            sum+=__shfl_xor_sync(0xffffffff,sum,4);
                            sum+=__shfl_xor_sync(0xffffffff,sum,8);
                            sum+=__shfl_xor_sync(0xffffffff,sum,16);
                            if (lane/4==0)
                                shared.row_lse[4*group+warp][c+(2*(lane&3)+e)*4]=sum;
                        }
                    }
                }
                {
                    kt::rt_bf<Q,D,kt::ducks::rt_layout::col> queries;
                    kt::warp::load(queries,data.q.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_AB(dk,soft_gradient,queries,dk);
                }
                packet.submitToNextAndTrigger();
                input.moveNext();
                query_gradient<D,true>(soft_gradient,first.template get<0>().k,shared.transpose,
                               args.score,task.bh,task.begin+q0,args,shared,query_lse?args.dlq:nullptr,
                               shared.row_lse,int(task.metadata+q0));
            }
            first.submitToNextAndTrigger();
            held.moveNext();
            store_key_gradient<D,2,FP32Output>(dk,args.dk,args.score,task.bh,task.begin+k0,args,shared);
#pragma unroll
            for (int r=0;r<2;++r) {
                float x=key_lse[r];
                x+=__shfl_xor_sync(0xffffffff,x,1);
                x+=__shfl_xor_sync(0xffffffff,x,2);
                int key=k0+r*8+lane/4;
                if ((lane&3)==0 && key<score_args.n) {
                    int64_t offset=task.metadata+key;
                    if constexpr (FP32Output) static_cast<float*>(args.dlk)[offset]=x;
                    else static_cast<__nv_bfloat16*>(args.dlk)[offset]=__float2bfloat16_rn(x);
                }
            }
#pragma unroll
            for (int shift=16;shift>0;shift/=2)
                tau_sum+=__shfl_down_sync(0xffffffff,tau_sum,shift);
            if (lane==0) atomicAdd(args.dtau+task.head,tau_sum);
        }
    }
    if (group<2) kt::tma::store_async_wait<0>();
    kt::warpgroup::sync(group+1);
}

template<bool FP32> const void* varlen_backward_qk_address() { return reinterpret_cast<const void*>(varlen_backward_qk_kernel<FP32>); }
template const void* varlen_backward_qk_address<false>();
#if DISM_ENABLE_FP32
template const void* varlen_backward_qk_address<true>();
#endif

} // namespace DISM_VARIANT
