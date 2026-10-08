#include "backward/gradients.cuh"
#include "varlen/backward.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

using namespace dism_backward;

template<bool FP32Output, class Configuration = ActiveConfig>
__global__ __launch_bounds__(384,1) void varlen_backward_summary_kernel(
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
            kt::rt_fl<K,DV> dv{0.f};
            kt::rt_fl<K,R> dsk{0.f};
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
                // dO is consumed in two register layouts, never live together.
                {
                    kt::rt_bf<Q,DV> derivative;
                    kt::warp::load<true>(derivative,data.dout.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_ABt(b,values,derivative,b);
                }
                kt::rt_bf<K,Q> probability_times_a;
                Scalar probability;
#pragma unroll
                for (int r=0;r<2;++r) {
#pragma unroll
                    for (int c=0;c<4;++c) {
                        auto pos=Scalar::layout(r,c,0);
                        auto av=a.tiles[0][c/2].data[r+2*(c&1)];
                        float p[2];
#pragma unroll
                        for (int e=0;e<2;++e) {
                            int query=q0+pos.second+4*e;
                            float norm=query<score_args.n?args.normalizer[task.metadata+query]:INFINITY;
                            float w=e?score.data[r][c].value.u1:score.data[r][c].value.u0;
                            p[e]=k0+pos.first<score_args.n?exp2_ftz(w-norm):0.f;
                        }
                        probability.data[r][c].value={p[0],p[1]};
                        probability_times_a.tiles[0][c/2].data[r+2*(c&1)]=
                            __floats2bfloat162_rn(p[0]*av.x,p[1]*av.y);
#if DISM_BACKWARD_DEBUG
                        if (args.debug_ca) {
#pragma unroll
                            for (int e=0;e<2;++e) {
                                int query=q0+pos.second+4*e,key=k0+pos.first;
                                if (query<score_args.n && key<score_args.n)
                                    args.debug_ca[(int64_t(task.bh)*score_args.n+query)*score_args.n+key]=
                                        p[e]*(e?av.y:av.x);
                            }
                        }
#endif
                    }
                }
                {
                    kt::rt_bf<Q,DV,kt::ducks::rt_layout::col> derivative;
                    kt::warp::load(derivative,data.dout.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_AB(dv,probability_times_a,derivative,dv);
                }
                Reverse reverse;
                kt::rt_bf<K,Q> soft_gradient;
#pragma unroll
                for (int r=0;r<2;++r) {
#pragma unroll
                    for (int c=0;c<4;++c) {
                        auto pos=Scalar::layout(r,c,0);
                        auto av=a.tiles[0][c/2].data[r+2*(c&1)];
                        auto bv=b.tiles[0][c/2].data[r+2*(c&1)];
                        auto p=probability.data[r][c].value;
                        float alpha[2],beta[2];
#pragma unroll
                        for (int e=0;e<2;++e) {
                            int query=q0+pos.second+4*e;
                            bool valid=query<score_args.n && k0+pos.first<score_args.n;
                            float delta=valid?args.delta[task.metadata+query]:0.f;
                            float w=e?score.data[r][c].value.u1:score.data[r][c].value.u0;
                            alpha[e]=valid ? ((query+1<score_args.n) ?
                                sigmoid2(w-(score_args.gate_delta ? data.row_gate[pos.second+4*e+1] : 0.f)) : 0.f) : 1.f;
                            beta[e]=(e?p.u1:p.u0)*fmaf(e?av.y:av.x,e?bv.y:bv.x,-delta);
                        }
                        reverse.data[r][c]={{alpha[0],alpha[1]},{beta[0],beta[1]}};
                        soft_gradient.tiles[0][c/2].data[r+2*(c&1)]=
                            __floats2bfloat162_rn(p.u0*bv.x,p.u1*bv.y);
#if DISM_BACKWARD_DEBUG
                        if (args.debug_cb) {
#pragma unroll
                            for (int e=0;e<2;++e) {
                                int query=q0+pos.second+4*e,key=k0+pos.first;
                                if (query<score_args.n && key<score_args.n)
                                    args.debug_cb[(int64_t(task.bh)*score_args.n+query)*score_args.n+key]=
                                        (e?p.u1:p.u0)*(e?bv.y:bv.x);
                            }
                        }
#endif
                    }
                }
                {
                    kt::rt_bf<Q,R,kt::ducks::rt_layout::col> soft_queries;
                    kt::warp::load(soft_queries,data.sq.payload);
                    [[clang::always_inline]] kt::warp::wmma::mma_AB(dsk,soft_gradient,soft_queries,dsk);
                }
                packet.submitToNextAndTrigger();
                input.moveNext();
                query_gradient(soft_gradient,first.template get<0>().sk,shared.transpose,
                               args.score,task.bh,task.begin+q0,args,shared);
                reverse.reverse_roll();
                Reverse::HState bottom;
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
                } else
                if (k0<score_args.n && lane/4>=4) {
                    int query=q0+8*(lane&3)+7-lane/4;
                    int chunks=(score_args.n+SummaryK-1)/SummaryK;
                    int64_t off=task.reverse+(int64_t(task.bh)*chunks+k0/SummaryK)*score_args.padded+query;
                    auto pair=state.horizontal.init[0];
                    args.summary_a[off]=pair.first.u0;
                    args.summary_a[off+4]=pair.first.u1;
                    args.summary_b[off]=pair.second.u0;
                    args.summary_b[off+4]=pair.second.u1;
                }
            }
            first.submitToNextAndTrigger();
            held.moveNext();
            store_key_gradient<DV,0,FP32Output>(dv,args.dv,args.score,task.bh,task.begin+k0,args,shared);
            store_key_gradient<R,1,FP32Output>(dsk,args.dsk,args.score,task.bh,task.begin+k0,args,shared);
        }
    }
    if (group<2) kt::tma::store_async_wait<0>();
    kt::warpgroup::sync(group+1);
}

template<bool FP32> const void* varlen_backward_summary_address() { return reinterpret_cast<const void*>(varlen_backward_summary_kernel<FP32>); }
template const void* varlen_backward_summary_address<false>();
#if DISM_ENABLE_FP32
template const void* varlen_backward_summary_address<true>();
#endif

} // namespace DISM_VARIANT
