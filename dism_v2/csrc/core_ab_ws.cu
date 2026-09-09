// B3: paired reverse scan, producer input ring and asynchronous dA reduction.
#include <cuda.h>
#if DISM_BWD_OPT >= 3
#define KITTENS_FEATURE_TMA
#define KITTENS_FEATURE_FP8
#define KITTENS_FEATURE_REG_INCDEC
#endif
#include <kittens.cuh>
#include "core_api.h"
#include "log_affine.cuh"
#include "pipeline.cuh"
#include "row_rng.cuh"
#include "backward_metadata.cuh"
#include "backward_input.cuh"
namespace dism_v2 {
namespace ab_ws {
namespace kt=kittens;
#include "scalar_grad.cuh"
using Reverse=glx::MMABuffer<16,64,glx::BinaryElement,glx::AffineComposeOp,glx::F32x2>;
template<int D,int DV> struct Slot { kt::st_bf<64,D> query; kt::st_bf<64,DV> dout; };
#if DISM_BWD_OPT >= 14
struct Scratch {
    kt::st_bf<16,64> soft;
    // Output-layout storage, independent of the next tile's Gsoft transpose.
    alignas(128) float da[1][16][16];
};
#else
union Scratch {
    kt::st_bf<16,64> soft;
#if DISM_BWD_OPT == 7 || DISM_BWD_OPT == 8
    // Output-layout-only experiment: same two1-KiB dA slots, now64B swizzled.
    kt::st_fl<16,16> da[2];
#else
    alignas(128) float da[2][16][16];
#endif
};
#endif
template<int D,int DV,bool ENABLE> struct ScratchBuffer { Scratch scratch[8]; };
template<int D,int DV> struct ScratchBuffer<D,DV,true> {
    // Async first A/dO input aliases existing Gsoft/dA scratch. Each task
    // releases old TMA reads before input arrives; all256 first-input readers
    // finish before ANY warp starts overwriting this union with Gsoft.
    union { Scratch scratch[8]; Slot<D,DV> first; };
    uint64_t first_ready,first_free,first_consumed;
};
template<int D,int DV> struct Shared {
    static constexpr bool PREFETCH=DISM_BWD_OPT==6 && sizeof(Slot<D,DV>)<=8*sizeof(Scratch);
    static constexpr int REQUESTED=DISM_BWD_STAGES;
    static constexpr int INPUT_BYTES=128*(D+DV);
    static constexpr int CANDIDATE_BYTES=(REQUESTED>2?REQUESTED:2)*INPUT_BYTES
        +8*sizeof(Scratch)+4*REQUESTED*sizeof(Reverse::HState::SharedStorage)+512;
    // Keep the old two-stage allocation if a requested variant exceeds the
    // conservative63-KiB budget; do not enlarge already-large baseline shapes.
    static constexpr int SLOTS=DISM_BWD_OPT>=4 && CANDIDATE_BYTES<=63*1024?REQUESTED:2;
    union {
#if DISM_BWD_OPT >= 3
        struct { kt::st_bf<128,D> key; kt::st_bf<128,DV> value; } initial;
#else
        struct { kt::st_bf<16,D> key[8]; kt::st_bf<16,DV> value[8]; } initial;
#endif
        Slot<D,DV> slot[SLOTS];
    };
    ScratchBuffer<D,DV,PREFETCH> work;
#if DISM_BWD_OPT >= 14
    uint64_t output_ready[2],output_free[2];
#endif
    uint64_t ready[SLOTS],free[SLOTS];
#if DISM_BWD_OPT >= 4
    uint64_t mail_ready[SLOTS],mail_free[SLOTS];
#else
    uint64_t mail_ready[4][SLOTS],mail_free[4][SLOTS];
#endif
    __device__ __forceinline__ uint64_t* mail_ready_at(int w,int s) {
#if DISM_BWD_OPT >= 4
        return &mail_ready[s];
#else
        return &mail_ready[w][s];
#endif
    }
    __device__ __forceinline__ uint64_t* mail_free_at(int w,int s) {
#if DISM_BWD_OPT >= 4
        return &mail_free[s];
#else
        return &mail_free[w][s];
#endif
    }
#if DISM_BWD_OPT >= 3
    uint64_t held_ready,held_free;
#endif
    Reverse::HState::SharedStorage mail[4][SLOTS];
    __device__ __forceinline__ Slot<D,DV>& input_slot(bool first_tile,int s) {
        if constexpr(PREFETCH) { if(first_tile) return work.first; }
        return slot[s];
    }
};
// OPT0 baseline only. Optimized kernels use bwd_metadata::score below.
__device__ __forceinline__ float transposed_score(const Args& p,float dot,int bh,int q,int k,bool hard) {
    if(q>=p.n || k>=p.n || k>q) return LOG_ZERO;
    float tau=p.tau[bh%p.heads];
    if(hard) return p.query_label(int64_t(bh)*p.n+q)==p.key_label(int64_t(bh)*p.n+k)?tau*LOG2E:LOG_ZERO;
    return (dot*p.scale-p.lse[int64_t(bh)*p.n+(p.column_lse?k:q)]+tau)*LOG2E;
}

__device__ __forceinline__ float2 reverse_coefficient(const Args& p,const float* delta,
        float w,float dot,int bh,int q,int k) {
    if(k>=p.n || q>=p.n) return {1,0};
    if(w==LOG_ZERO) return {0,0};
    // sigmoid(W_natural) = (1 + tanh(W2 * ln(2) / 2)) / 2.
    float half_natural=w*0.3465735902799726547f,t;
    asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(half_natural));
    float a=fmaf(0.5f,t,0.5f);
    return {a,exp2f(w-p.normalizer[int64_t(bh)*p.n+q])*(dot-delta[int64_t(bh)*p.n+q])};
}


template<int D,int DV,int SPEC=-1>
__global__ __launch_bounds__(384,1) void run(
        __grid_constant__ const Args p,__grid_constant__ const CUtensorMap qm,
        __grid_constant__ const CUtensorMap dm,
#if DISM_BWD_OPT >= 3
        __grid_constant__ const CUtensorMap bm,__grid_constant__ const CUtensorMap vm,
#endif
        const __nv_bfloat16* dout,
        const float* delta,const float* boundary,
        __grid_constant__ const CUtensorMap dam,float* db,float* dlse,float* tau_partial) {
    extern __shared__ __align__(128) unsigned char bytes[];
    auto& shared=*reinterpret_cast<Shared<D,DV>*>(bytes);
    constexpr int SLOTS=Shared<D,DV>::SLOTS;
    constexpr bool PREFETCH=Shared<D,DV>::PREFETCH;
    constexpr bool HARD_ONLY=DISM_BWD_OPT>=11 && SPEC>=0 && bwd_metadata::Policy<SPEC>::mode==1;
    int input_tile=0;
    int warp=threadIdx.x/32,lane=threadIdx.x&31,g=lane&3,l=lane/4;
    int bh=blockIdx.y,chunk=blockIdx.x*4+(warp&3);
    int kb=chunk*32+(warp/4)*16;
    // Uniform CTA bound: all q below the smallest key are noncausal.
    const int query_begin=DISM_BWD_OPT>=2?int(blockIdx.x)*128:0;
    const int blocks=(p.padded_n+127)/128;
    int tile=0;
    if(warp==0 && kt::warp::elect_leader()) {
#if DISM_BWD_OPT >= 14
        #pragma unroll
        for(int wg=0;wg<2;++wg) {
            init_bar(&shared.output_ready[wg],128);
            init_bar(&shared.output_free[wg],1);
            arrive(&shared.output_free[wg]); // Phase0: initially empty.
        }
#endif
        if constexpr(PREFETCH) {
            init_bar(&shared.work.first_ready,32); init_bar(&shared.work.first_free,256);
            init_bar(&shared.work.first_consumed,256);
        }
#if DISM_BWD_OPT >= 3
        init_bar(&shared.held_ready,1); init_bar(&shared.held_free,256);
#endif
        #pragma unroll
        for(int s=0;s<SLOTS;++s) {
            init_bar(&shared.ready[s],32); init_bar(&shared.free[s],256);
#if DISM_BWD_OPT >= 4
            init_bar(&shared.mail_ready[s],128);
            init_bar(&shared.mail_free[s],128);
#else
            #pragma unroll
            for(int w=0;w<4;++w) {
                init_bar(&shared.mail_ready[w][s],32);
                init_bar(&shared.mail_free[w][s],32);
            }
#endif
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
#if DISM_BWD_OPT < 3
    if(warp<8) {
        for(int x=lane;x<16*D;x+=32) {
            int k=kb+x/D;
            shared.initial.key[warp][int2{x/D,x%D}]=k<p.n?
                static_cast<const __nv_bfloat16*>(p.b)[(int64_t(bh)*p.n+k)*D+x%D]:__float2bfloat16(0);
        }
        for(int x=lane;x<16*DV;x+=32) {
            int k=kb+x/DV;
            shared.initial.value[warp][int2{x/DV,x%DV}]=k<p.n?
                static_cast<const __nv_bfloat16*>(p.v)[(int64_t(bh)*p.n+k)*DV+x%DV]:__float2bfloat16(0);
        }
    }
    __syncthreads();
    kt::rt_bf<16,D> keys;
    kt::rt_bf<16,DV> values;
    if(warp<8) {
        kt::warp::load(keys,shared.initial.key[warp]);
        kt::warp::load(values,shared.initial.value[warp]);
    }
    __syncthreads(); // Held keys/values are now registers; initial staging may be reused.
#else
    __syncthreads(); // Initialization only; tasks use mbarrier epochs.
#endif
    if(warp>=8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        bool leader=kt::warp::elect_leader();
#if DISM_BWD_OPT >= 14
        if constexpr(!HARD_ONLY) {
            if(warp==9 && leader) {
                int epoch=0;
                #pragma unroll 1
                for(int task=blockIdx.x;task<blocks*p.batch_heads;task+=gridDim.x) {
                    int bh=task/blocks,query_begin=(task%blocks)*128;
                    #pragma unroll 1
                    for(int t=0;t*64<p.padded_n-query_begin;++t) {
                        int qb=p.padded_n-64-t*64;
                        // Reverse scan publishes WG1 before WG0. Service the
                        // entire WG1 tile first; no all-eight-warp rendezvous.
                        #pragma unroll
                        for(int group=0;group<2;++group) {
                            int wg=1-group;
                            #pragma unroll
                            for(int qt=0;qt<4;++qt) {
                                #pragma unroll
                                for(int f=0;f<D/16;++f) {
                                    int phase=(epoch+qt*(D/16)+f)&1;
                                    wait(&shared.output_ready[wg],phase);
                                    #pragma unroll
                                    for(int w=0;w<4;++w) {
                                        unsigned addr=smaddr(shared.work.scratch[wg*4+w].da[0]);
                                        asm volatile("cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group "
                                            "[%0, {%2, %3, %4}], [%1];" ::
                                            "l"(&dam),"r"(addr),"r"(f*16),"r"(qb+qt*16),"r"(bh):"memory");
                                    }
                                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                                    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
                                    arrive(&shared.output_free[wg]);
                                }
                            }
                        }
                        epoch+=D/4;
                    }
                }
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            }
        }
#endif
        if(warp==8) {
        int task_round=0;
#if DISM_BWD_OPT >= 3
        #pragma unroll 1
        for(int task=blockIdx.x;task<blocks*p.batch_heads;task+=gridDim.x,++task_round) {
            int bh=task/blocks,query_begin=(task%blocks)*128;
            // Reuse the existing input union only after every old A/dO reader
            // finishes. Held B/V can overlap old reverse/dA and final writes.
            #pragma unroll
            for(int s=0;s<SLOTS;++s) if(tile>s) wait(&shared.free[s],((tile-1-s)/SLOTS)&1);
            if(leader) {
                expect(&shared.held_ready,sizeof(shared.initial));
                bwd_input::load(&bm,shared.initial.key.data,&shared.held_ready,{query_begin,bh,0,0});
                bwd_input::load(&vm,shared.initial.value.data,&shared.held_ready,{query_begin,bh,0,0});
            }
            if constexpr(PREFETCH) {
                if(task_round) wait(&shared.work.first_free,(task_round-1)&1);
                bwd_input::issue_query<D,DV>(p,&qm,&dm,dout,shared.work.first,
                    &shared.work.first_ready,bh,p.padded_n-64,leader,lane);
            }
            wait(&shared.held_free,task_round&1);
#endif
        #pragma unroll 1
        for(int t=PREFETCH?1:0;t*64<p.padded_n-query_begin;++t,++tile) {
            int s=tile%SLOTS,qb=p.padded_n-64-t*64;
            if(tile>=SLOTS) wait(&shared.free[s],((tile/SLOTS)-1)&1);
            auto& slot=shared.slot[s];
            if(qb+64<=p.n) {
                if(leader) {
                    expect(&shared.ready[s],sizeof(slot.query)+sizeof(slot.dout));
#if DISM_BWD_OPT >= 3
                    bwd_input::load(&qm,slot.query.data,&shared.ready[s],{0,0,bh*p.n+qb,0});
                    bwd_input::load(&dm,slot.dout.data,&shared.ready[s],{0,0,bh*p.n+qb,0});
#else
                    constexpr int Q=D==32?32:64,V=DV==32?32:64;
                    #pragma unroll
                    for(int c=0;c<D/Q;++c) tma5(&qm,slot.query.data+c*64*Q,&shared.ready[s],bh*p.n+qb,c);
                    #pragma unroll
                    for(int c=0;c<DV/V;++c) tma5(&dm,slot.dout.data+c*64*V,&shared.ready[s],bh*p.n+qb,c);
#endif
                } else arrive(&shared.ready[s]);
            } else {
                for(int x=lane;x<64*D;x+=32) {
                    int q=qb+logical_row(x/D);
                    slot.query[int2{x/D,x%D}]=q<p.n?
                        static_cast<const __nv_bfloat16*>(p.a)[(int64_t(bh)*p.n+q)*D+x%D]:__float2bfloat16(0);
                }
                for(int x=lane;x<64*DV;x+=32) {
                    int q=qb+logical_row(x/DV);
                    slot.dout[int2{x/DV,x%DV}]=q<p.n?dout[(int64_t(bh)*p.n+q)*DV+x%DV]:__float2bfloat16(0);
                }
                __syncwarp();
                arrive(&shared.ready[s]); // Every writer publishes its tail stores.
            }
        }
#if DISM_BWD_OPT >= 3
        }
#endif
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
        int output_epoch=0;
        int task_round=0;
#if DISM_BWD_OPT >= 3
        #pragma unroll 1
        for(int task=blockIdx.x;task<blocks*p.batch_heads;task+=gridDim.x,++task_round) {
            int bh=task/blocks,key_cta=task%blocks,chunk=key_cta*4+(warp&3);
            int kb=chunk*32+(warp/4)*16,query_begin=key_cta*128;
#if DISM_BWD_OPT >= 12
            // Issue held-key metadata before waiting on its asynchronous B/V input.
            bwd_metadata::KeyFor<SPEC> key_meta(p,bh,kb);
#endif
            wait(&shared.held_ready,task_round&1);
            kt::rt_bf<16,D> keys;
            kt::rt_bf<16,DV> values;
            auto key_view=shared.initial.key.template subtile<16,D>({(warp&3)*2+warp/4,0});
            auto value_view=shared.initial.value.template subtile<16,DV>({(warp&3)*2+warp/4,0});
            kt::warp::load(keys,key_view); kt::warp::load(values,value_view);
            arrive(&shared.held_free);
#else
        int key_cta=blockIdx.x;
#endif
        kt::rt_fl<16,D> accumulated{0.f};
        int issued=0;
        float tau_sum=0,key_lse[2]={0,0};
        Reverse::VState right;
#if DISM_BWD_OPT && DISM_BWD_OPT < 12
        bwd_metadata::KeyFor<SPEC> key_meta(p,bh,kb);
#endif
        #pragma unroll 1
        for(int t=0;t*64<p.padded_n-query_begin;++t,++tile) {
            int s=tile%SLOTS,phase=(tile/SLOTS)&1,qb=p.padded_n-64-t*64;
            int input_s=PREFETCH?input_tile%SLOTS:s;
            int input_phase=PREFETCH?(input_tile/SLOTS)&1:phase;
            auto& input=shared.input_slot(t==0,input_s);
#if DISM_BWD_OPT >= 12
            // LSE/label/norm/delta loads precede RNG and the A/dO ready wait.
            bwd_metadata::QueryFor<SPEC> query_meta(p,delta,bh,qb,key_meta.tau);
#endif
uint32_t hard0,hard1;
            if(p.hard_bits) {
                const int words=(p.n+31)/32;
                hard0=qb<p.n?p.hard_bits[int64_t(bh)*words+qb/32]:0;
                hard1=qb+32<p.n?p.hard_bits[int64_t(bh)*words+qb/32+1]:0;
            } else {
                hard0=__ballot_sync(0xffffffff,qb+lane<p.n &&
                row_hard(p.seed,p.offset,uint64_t(bh)*p.n+qb+lane,p.hard_prob));
            hard1=__ballot_sync(0xffffffff,qb+lane+32<p.n &&
                row_hard(p.seed,p.offset,uint64_t(bh)*p.n+qb+lane+32,p.hard_prob));
            }
#if DISM_BWD_OPT && DISM_BWD_OPT < 12
            bwd_metadata::QueryFor<SPEC> query_meta(p,delta,bh,qb);
#endif
            if constexpr(PREFETCH) {
                if(t==0) wait(&shared.work.first_ready,task_round&1);
                else wait(&shared.ready[input_s],input_phase);
            } else wait(&shared.ready[input_s],input_phase);
        Scalar scalar;
        {
            // Keys remain in registers across query tiles.
            kt::rt_fl<16,64> dot{0.f};
            if constexpr(SPEC<0 || bwd_metadata::Policy<SPEC>::mode!=1) {
#if DISM_BWD_OPT >= 5
            kt::rt_bf<D,64,kt::ducks::rt_layout::col> queries;
            bwd_input::load_rhs(queries,input.query);
            kt::warp::wmma::mma_AB(dot,keys,queries,dot);
#else
            kt::rt_bf<64,D> queries;
            kt::warp::load(queries,input.query);
            kt::warp::wmma::mma_ABt(dot,keys,queries,dot);
#endif
            }
            #pragma unroll
            for(int r=0;r<2;++r) {
                #pragma unroll
                for(int c=0;c<8;++c) {
                    auto x=dot.tiles[0][c/2].data[r+2*(c&1)];
                    auto pos=Buffer::layout(r,c,0);
#if DISM_BWD_OPT
                    scalar.data[r][c].value={
                        bwd_metadata::score<0>(p,key_meta,query_meta,x.x,r,pos.second,qb+pos.second,kb+pos.first,(hard0>>pos.second)&1),
                        bwd_metadata::score<1>(p,key_meta,query_meta,x.y,r,pos.second,qb+pos.second+32,kb+pos.first,(hard1>>pos.second)&1)};
#else
                    scalar.data[r][c].value={transposed_score(p,x.x,bh,qb+pos.second,kb+pos.first,(hard0>>pos.second)&1),
                        transposed_score(p,x.y,bh,qb+pos.second+32,kb+pos.first,(hard1>>pos.second)&1)};
#endif
                }
            }
        }
        scalar.roll(); Buffer data;
        #pragma unroll
        for(int r=0;r<2;++r) {
            #pragma unroll
            for(int c=0;c<8;++c) {
                auto x=scalar.data[r][c].value; data.data[r][c]={x,x};
                int k=kb+(r*8+l-(7-c)+16)%16,q=qb+c+8*g;
                if(k>=p.n || q>=p.n) { data.data[r][c].first.u0=0; data.data[r][c].second.u0=LOG_ZERO; }
                if(k>=p.n || q+32>=p.n) { data.data[r][c].first.u1=0; data.data[r][c].second.u1=LOG_ZERO; }
            }
        }
        Buffer::HState top;
        if(kb>0 && kb<p.padded_n) {
            int q=qb+8*g+6-l;
            int64_t off=(int64_t(bh)*(p.padded_n/16)+kb/16-1)*p.padded_n;
            top.init[0].first={0,0};
            top.init[0].second={q>=0?p.vertical[off+q]:LOG_ZERO,p.vertical[off+q+32]};
        }
        Buffer::VState left;
        if(qb>0 && g==3 && kb<p.padded_n) {
            int64_t off=(int64_t(bh)*(p.padded_n/64)+qb/64-1)*p.padded_n+kb;
            #pragma unroll
            for(int r=0;r<2;++r) {
                left.init[r].first.u0=0;
                left.init[r].second.u0=p.horizontal[off+8*r+l];
            }
        }
        data.inclusive_scan(left,top);
        #pragma unroll
        for(int r=0;r<2;++r) {
            #pragma unroll
            for(int c=0;c<8;++c) scalar.data[r][c].value=data.data[r][c].second;
        }
        scalar.template roll<false>();

            Reverse reverse;
            {
                kt::rt_fl<16,64> dp{0.f};
                {
#if DISM_BWD_OPT >= 5
                    kt::rt_bf<DV,64,kt::ducks::rt_layout::col> derivatives;
                    bwd_input::load_rhs(derivatives,input.dout);
                    kt::warp::wmma::mma_AB(dp,values,derivatives,dp);
#else
                    kt::rt_bf<64,DV> derivatives;
                    kt::warp::load(derivatives,input.dout);
                    kt::warp::wmma::mma_ABt(dp,values,derivatives,dp);
#endif
                }
                #pragma unroll
                for(int r=0;r<2;++r) {
                    #pragma unroll
                    for(int c=0;c<8;++c) {
                        auto pos=Buffer::layout(r,c,0);
                        auto w=scalar.data[r][c].value;
                        auto dot=dp.tiles[0][c/2].data[r+2*(c&1)];
                        int k=kb+pos.first,q=qb+pos.second;
#if DISM_BWD_OPT
                        auto x=bwd_metadata::coefficient<0>(p,query_meta,w.u0,dot.x,pos.second,q,k);
                        auto y=bwd_metadata::coefficient<1>(p,query_meta,w.u1,dot.y,pos.second,q+32,k);
#else
                        auto x=reverse_coefficient(p,delta,w.u0,dot.x,bh,q,k);
                        auto y=reverse_coefficient(p,delta,w.u1,dot.y,bh,q+32,k);
#endif
                        reverse.data[r][c]={{x.x,y.x},{x.y,y.y}};
                    }
                }
            }
            reverse.reverse_roll();
            Reverse::HState bottom;
            if(warp>=4 && chunk+1<p.padded_n/32) {
                int q=qb+8*g+8-l;
                int64_t off=(int64_t(bh)*(p.padded_n/32)+chunk+1)*p.padded_n;
                bottom.init[0].first={1,1};
                bottom.init[0].second={q<p.padded_n?boundary[off+q]:0,
                    q+32<p.padded_n?boundary[off+q+32]:0};
            }
            if(warp<4) {
                wait(shared.mail_ready_at(warp,s),phase);
                bottom=Reverse::HState::load_shared(shared.mail[warp][s]);
                arrive(shared.mail_free_at(warp,s));
            }
            auto state=reverse.reverse_inclusive_scan(right,bottom);
            right=state.first;
            if(warp>=4) {
                if(tile>=SLOTS) wait(shared.mail_free_at(warp-4,s),phase^1);
                state.second.store_shared(shared.mail[warp-4][s]);
                arrive(shared.mail_ready_at(warp-4,s));
            }
            // Publish paired boundary before gradient GEMMs.
            #pragma unroll
            for(int r=0;r<2;++r) {
                #pragma unroll
                for(int c=0;c<8;++c) scalar.data[r][c].value=reverse.data[r][c].second;
            }
            scalar.template reverse_roll<false>();
            scalar_gradients<HARD_ONLY,bwd_metadata::finite_zero>(p,scalar,bh,kb,qb,hard0,hard1,dlse,tau_sum,key_lse);
            if constexpr(HARD_ONLY) {
                // Labels are stop-gradient. Keep G and tau accumulation above;
                // dA/dB/dLSE are zero in pure hard mode. Final dB stores still
                // initialize its output; atomic dA/dLSE were host-zeroed.
                arrive(&shared.free[input_s]); ++input_tile;
            } else {
            auto& scratch=shared.work.scratch[warp];
            // Last iteration's output TMA must finish reading before union reuse.
#if DISM_BWD_OPT < 14
            if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
#endif
            __syncwarp();
            {
                kt::rt_bf<16,64> grad;
                #pragma unroll
                for(int r=0;r<2;++r) {
                    #pragma unroll
                    for(int c=0;c<8;++c) {
                        auto pos=Buffer::layout(r,c,0);auto x=scalar.data[r][c].value;
                        int k=kb+pos.first,q=qb+pos.second;
                        float a,b;
                        if constexpr(bwd_metadata::finite_zero) {
                            // Reverse recurrence has zero terminal G; all padding
                            // and noncausal diagonals have beta0, hence exact G0.
                            a=bwd_metadata::select(!((hard0>>pos.second)&1),x.u0,0.f);
                            b=bwd_metadata::select(!((hard1>>pos.second)&1),x.u1,0.f);
                        } else {
                            a=k<p.n && q<p.n && !((hard0>>pos.second)&1)?x.u0:0.f;
                            b=k<p.n && q+32<p.n && !((hard1>>pos.second)&1)?x.u1:0.f;
                        }
                        // Register G matches physically permuted query; shared G is logical.
                        grad.tiles[0][c/2].data[r+2*(c&1)]=__floats2bfloat162_rn(a,b);
                        if(!PREFETCH || t!=0) {
                            scratch.soft[int2{pos.first,pos.second}]=__float2bfloat16_rn(a);
                            scratch.soft[int2{pos.first,pos.second+32}]=__float2bfloat16_rn(b);
                        }
                    }
                }
                kt::rt_bf<64,D,kt::ducks::rt_layout::col> queries;
                kt::warp::load(queries,input.query);
                kt::warp::wmma::mma_AB(accumulated,grad,queries,accumulated);
                if constexpr(PREFETCH) {
                    if(t==0) {
                        arrive(&shared.work.first_consumed);
                        wait(&shared.work.first_consumed,task_round&1);
                        #pragma unroll
                        for(int r=0;r<2;++r) {
                            #pragma unroll
                            for(int c=0;c<8;++c) {
                                auto pos=Buffer::layout(r,c,0);
                                auto x=grad.tiles[0][c/2].data[r+2*(c&1)];
                                scratch.soft[int2{pos.first,pos.second}]=__low2bfloat16(x);
                                scratch.soft[int2{pos.first,pos.second+32}]=__high2bfloat16(x);
                            }
                        }
                    }
                }
            }
            if(!PREFETCH || t!=0) {
                arrive(&shared.free[input_s]); ++input_tile;
            } // First-input storage remains scratch-owned until task end.
            __syncwarp();
            kt::rt_bf<16,64,kt::ducks::rt_layout::col> transposed;
            kt::warp::load(transposed,scratch.soft);
            __syncwarp(); // All Gsoft read into registers; union becomes dA ring.
            #pragma unroll
            for(int qt=0;qt<4;++qt) {
                kt::rt_bf<16,16,kt::ducks::rt_layout::col> grad;
                grad.tiles[0][0]=transposed.tiles[0][qt];
                #pragma unroll
                for(int f=0;f<D/16;++f) {
                    kt::rt_bf<16,16> keyrow;
                    keyrow.tiles[0][0]=keys.tiles[0][f];
                    kt::rt_bf<16,16,kt::ducks::rt_layout::col> keycol;
                    kt::warp::rt_conversions::swap_layout(keycol,keyrow);
                    kt::rt_fl<16,16> part{0.f};
                    kt::warp::wmma::mma_AtB(part,grad,keycol,part);
#if DISM_BWD_OPT >= 14
                    constexpr int out_slot=0;
                    wait(&shared.output_free[warp/4],output_epoch&1);
#else
                    int out_slot=issued%2;
                    if(lane==0 && issued>=2)
                        asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
#endif
                    __syncwarp();
#if DISM_BWD_OPT == 9
                    // Four lanes own one complete16-float row. Each8-lane
                    // STS.128 wavefront covers two full rows, with unique banks.
                    #pragma unroll
                    for(int h=0;h<2;++h) {
                        auto low=part.tiles[0][0].data[h];
                        auto high=part.tiles[0][0].data[h+2];
                        float2 send={bwd_metadata::select(g&1,low.x,high.x),
                                     bwd_metadata::select(g&1,low.y,high.y)};
                        float2 peer={__shfl_xor_sync(0xffffffff,send.x,1),
                                     __shfl_xor_sync(0xffffffff,send.y,1)};
                        float2 a={bwd_metadata::select(g&1,peer.x,low.x),
                                  bwd_metadata::select(g&1,peer.y,low.y)};
                        float2 b={bwd_metadata::select(g&1,high.x,peer.x),
                                  bwd_metadata::select(g&1,high.y,peer.y)};
                        int q=h*8+l,j=(g/2)*4+(g&1)*8;
                        kt::move<float4>::sts(smaddr(&scratch.da[out_slot][q][j]),
                            make_float4(a.x*p.scale,a.y*p.scale,b.x*p.scale,b.y*p.scale));
                    }
#else
                    #pragma unroll
                    for(int r=0;r<4;++r) {
                        auto x=part.tiles[0][0].data[r];
                        int q=(r%2)*8+l,j=(r/2)*8+2*g;
#if DISM_BWD_OPT == 8
                        // j is even and the64B swizzle preserves low4 address
                        // bits: the pair remains adjacent and8-byte aligned.
                        auto dst=kt::st_fl<16,16>::idx(smaddr(scratch.da[out_slot].data),{q,j});
                        kt::move<float2>::sts(dst,make_float2(x.x*p.scale,x.y*p.scale));
#elif DISM_BWD_OPT == 7
                        scratch.da[out_slot][int2{q,j}]=x.x*p.scale;
                        scratch.da[out_slot][int2{q,j+1}]=x.y*p.scale;
#else
                        scratch.da[out_slot][q][j]=x.x*p.scale;
                        scratch.da[out_slot][q][j+1]=x.y*p.scale;
#endif
                    }
#endif
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
#if DISM_BWD_OPT >= 14
                    arrive(&shared.output_ready[warp/4]); // All128 writers publish.
                    ++output_epoch;
#else
                    if(lane==0) {
#if DISM_BWD_OPT == 7 || DISM_BWD_OPT == 8
                        unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(scratch.da[out_slot].data));
#else
                        unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(scratch.da[out_slot]));
#endif
                        asm volatile("cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group "
                            "[%0, {%2, %3, %4}], [%1];" ::
                            "l"(&dam),"r"(addr),"r"(f*16),"r"(qb+qt*16),"r"(bh):"memory");
                        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                    }
#endif
                    ++issued;
                }
            }
            }
        }
        if constexpr(PREFETCH) {
            if(lane==0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            __syncwarp();
            arrive(&shared.work.first_free); // Before final global writes; source reads done.
        }
        #pragma unroll
        for(int c=0;c<D/16;++c) {
            #pragma unroll
            for(int r=0;r<4;++r) {
                int k=kb+(r%2)*8+l,j=c*16+(r/2)*8+g*2;
                if(k<p.n) {
                    auto x=accumulated.tiles[0][c].data[r];
                    int64_t off=(int64_t(bh)*p.n+k)*D+j;
                    db[off]=x.x*p.scale; db[off+1]=x.y*p.scale;
                }
            }
        }
        if constexpr(!HARD_ONLY) store_key_lse(p,bh,kb,dlse,key_lse);
        float total=warp_sum(tau_sum);
        if(lane==0) tau_partial[(int64_t(bh)*blocks+key_cta)*8+warp]=total;
#if DISM_BWD_OPT < 14
        if(lane==0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
#endif
        __syncwarp();
    }
#if DISM_BWD_OPT >= 3
    }
    kt::warpgroup::sync(1+warp/4); // Retire register groups independently.
#else
    __syncthreads(); // Invalid compute warps also drain the full protocol.
#endif
}


template<int D,int DV,int SPEC=-1> void launch_case(const Args& p,const void* dout,const float* delta,
        const float* boundary,const CUtensorMap& dam,float* db,float* dlse,float* tau_partial,cudaStream_t stream) {
    auto qm=permuted_map<D,(DISM_BWD_OPT>=3)>(p.a,p.batch_heads*p.n);
    auto dm=permuted_map<DV,(DISM_BWD_OPT>=3)>(dout,p.batch_heads*p.n);
    constexpr int sm=sizeof(Shared<D,DV>);
    auto err=cudaFuncSetAttribute(run<D,DV,SPEC>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm);
    if(err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
#if DISM_BWD_OPT >= 3
    auto bm=bwd_input::held_map<D>(p,p.b),vm=bwd_input::held_map<DV>(p,p.v);
    int device,sms;
    auto status=cudaGetDevice(&device);
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    int tasks=((p.padded_n+127)/128)*p.batch_heads;
    run<D,DV,SPEC><<<std::min(tasks,sms),384,sm,stream>>>(
        p,qm,dm,bm,vm,static_cast<const __nv_bfloat16*>(dout),delta,boundary,dam,db,dlse,tau_partial);
#else
    run<D,DV,SPEC><<<dim3((p.padded_n+127)/128,p.batch_heads),384,sm,stream>>>(
        p,qm,dm,static_cast<const __nv_bfloat16*>(dout),delta,boundary,dam,db,dlse,tau_partial);
#endif
}
template<int D,int DV> void launch(const Args& p,const void* dout,const float* delta,
        const float* boundary,const CUtensorMap& dam,float* db,float* dlse,float* tau_partial,cudaStream_t stream) {
#if DISM_BWD_OPT >= 10
    bwd_metadata::dispatch(p,[&]<int S>() { launch_case<D,DV,S>(p,dout,delta,boundary,dam,db,dlse,tau_partial,stream); });
#else
    launch_case<D,DV>(p,dout,delta,boundary,dam,db,dlse,tau_partial,stream);
#endif
}
template<int D> void dispatch(const Args& p,int dv,const void* dout,const float* delta,
        const float* boundary,const CUtensorMap& dam,float* db,float* dlse,float* tau_partial,cudaStream_t stream) {
    switch(dv) {
        case 32: launch<D,32>(p,dout,delta,boundary,dam,db,dlse,tau_partial,stream); break;
        case 64: launch<D,64>(p,dout,delta,boundary,dam,db,dlse,tau_partial,stream); break;
        case 128: launch<D,128>(p,dout,delta,boundary,dam,db,dlse,tau_partial,stream); break;
    }
}
} // namespace ab_ws
void launch_operand_backward_ws(const Args& p,int d,int dv,const void* dout,const float* delta,
        const float* boundary,float* da,float* db,float* dlse,float* tau_partial,cudaStream_t stream) {
    CUtensorMap dam;
    cuuint64_t dims[3]={cuuint64_t(d),cuuint64_t(p.n),cuuint64_t(p.batch_heads)};
    cuuint64_t strides[2]={cuuint64_t(d)*4,cuuint64_t(d)*p.n*4};
    cuuint32_t box[3]={16,16,1},steps[3]={1,1,1};
    constexpr auto swizzle=(DISM_BWD_OPT==7 || DISM_BWD_OPT==8)?CU_TENSOR_MAP_SWIZZLE_64B:CU_TENSOR_MAP_SWIZZLE_NONE;
    static_assert(kittens::st_fl<16,16>::swizzle_bytes==64);
    auto status=cuTensorMapEncodeTiled(&dam,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,3,da,dims,strides,box,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(status!=CUDA_SUCCESS) throw std::runtime_error("WS dA tensor map encoding failed");
    switch(d) {
        case 32:ab_ws::dispatch<32>(p,dv,dout,delta,boundary,dam,db,dlse,tau_partial,stream);break;
        case 64:ab_ws::dispatch<64>(p,dv,dout,delta,boundary,dam,db,dlse,tau_partial,stream);break;
        case 128:ab_ws::dispatch<128>(p,dv,dout,delta,boundary,dam,db,dlse,tau_partial,stream);break;
    }
}
} // namespace dism_v2
