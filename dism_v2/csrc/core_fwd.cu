#include <cuda.h>
// This vendored TK does not auto-enable SM120 features. TMA declarations
// also reference FP8 types; enabling those types does not change BF16 math.
#define KITTENS_FEATURE_TMA
#define KITTENS_FEATURE_FP8
#define KITTENS_FEATURE_REG_INCDEC
#include <kittens.cuh>
#include "core_api.h"
#include "log_affine.cuh"
#include "pipeline.cuh"
#include "row_rng.cuh"

namespace dism_v2 {
namespace kt=kittens;
__device__ __forceinline__ float normal_reciprocal(float x) {
    // With finite scan maxima, the online denominator is in [1,N+1].
    // Both x and its reciprocal are normal FP32 numbers at supported N.
    // Use the hardware reciprocal followed by one FP32 Newton correction.
    // Neither rcp.rn nor rcp.rn.ftz avoids ptxas's outlined slow path here.
    // This is only output normalization; log-affine LSE is unchanged.
    float result;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return fmaf(result,fmaf(-x,result,1.f),result);
}
template<int D,int DV,bool OUTPUT> struct Slot { kt::st_bf<64,D> k; kt::st_bf<64,DV> v; };
template<int D,int DV> struct Slot<D,DV,false> { kt::st_bf<64,D> k; };
template<int D,int DV,bool OUTPUT> struct Shared {
    static_assert(OUTPUT);
    static constexpr int QROWS=(D==128 && DV==128)?64:128;
    static constexpr int QPHASES=128/QROWS;
    static constexpr int PER_STAGE=128*(D+DV)+4*sizeof(Buffer::HState::SharedStorage)+32;
    static constexpr int CAPACITY=(63*1024-128-QROWS*D*2)/PER_STAGE;
    static constexpr int SLOTS=CAPACITY<3?CAPACITY:3;
    static_assert(SLOTS>=1);
    kt::st_bf<QROWS,D> q;
    Slot<D,DV,true> slot[SLOTS];
    uint64_t qready,qfree;
    uint64_t ready[SLOTS],free[SLOTS];
    uint64_t mail_ready[SLOTS],mail_free[SLOTS];
    Buffer::HState::SharedStorage mail[4][SLOTS];
};
template<int D,int QROWS> CUtensorMap output_q_map(const Args& p) {
    constexpr int S=D==32?32:64;
    const cuuint64_t dims[]{S,cuuint64_t(p.n),cuuint64_t(p.batch_heads),D/S,1};
    const cuuint64_t strides[]{D*2,cuuint64_t(p.n)*D*2,S*2,S*2};
    const cuuint32_t box[]{S,QROWS,1,D/S,1},elem[]{1,1,1,1,1};
    CUtensorMap map{};
    auto e=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,5,
        const_cast<void*>(p.a),dims,strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,
        D==32?CU_TENSOR_MAP_SWIZZLE_64B:CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(e!=CUDA_SUCCESS) throw std::runtime_error("output Q TMA map failed");
    return map;
}

// Shared with the validated summary implementation included below.
__device__ __forceinline__ float summary_select(bool pred, float yes, float no);
template<int ROWS,int COLS>
__device__ __forceinline__ void load_rhs_tile(
    kt::rt_bf<COLS,ROWS,kt::ducks::rt_layout::col>& dst,
    const kt::st_bf<ROWS,COLS>& src);
__device__ __forceinline__ void summary_tma(const CUtensorMap *map, void *dst,
                                          uint64_t *bar, int4 coord);

template<int D,int DV,bool OUTPUT,bool COLUMN_LSE,int MODE,typename Label>
__global__ __launch_bounds__(384,1) void core(__grid_constant__ const Args p, __grid_constant__ const CUtensorMap km,
                     __grid_constant__ const CUtensorMap vm,
                     __grid_constant__ const CUtensorMap qm) {
    constexpr int STAGES=Shared<D,DV,OUTPUT>::SLOTS;
    constexpr int QROWS=Shared<D,DV,OUTPUT>::QROWS;
    constexpr int QPHASES=Shared<D,DV,OUTPUT>::QPHASES;
    extern __shared__ __align__(128) unsigned char bytes[];
    auto& shared=*reinterpret_cast<Shared<D,DV,OUTPUT>*>(bytes);
    int warp=threadIdx.x/32,lane=threadIdx.x&31;
    int blocks=(p.n+127)/128,total=blocks*p.batch_heads;
    // Single elected producer, including safe scalar tails. A cooperative
    // ready32 variant failed persistent replay and is not the selected path.
    if(warp==0 && kt::warp::elect_leader()) {
        init_bar(&shared.qready,1); init_bar(&shared.qfree,256);
#pragma unroll
        for(int s=0;s<STAGES;++s) {
            init_bar(&shared.ready[s],1);
            init_bar(&shared.free[s],256);
            init_bar(&shared.mail_ready[s],128);
            init_bar(&shared.mail_free[s],128);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    int tile=0,task_round=0;
    if(warp>=8) {
        kt::warpgroup::decrease_registers<40>();
        bool leader=kt::warp::elect_leader();
        if(warp==8 && leader) {
#pragma unroll 1
            for(int task=blockIdx.x;task<total;task+=gridDim.x,++task_round) {
                int bh=task/blocks,base=(task%blocks)*128;
                int key_end=min(p.padded_n,base+128);
                int qepoch=task_round*QPHASES;
                if(leader) {
                    if(qepoch) wait(&shared.qfree,(qepoch-1)&1);
                    expect(&shared.qready,sizeof(shared.q));
                    summary_tma(&qm,shared.q.data,&shared.qready,{base,bh,0,0});
                }
#pragma unroll 1
                for(int t=0;t*64<key_end;++t,++tile) {
                    int s=tile%STAGES;
                    if(tile>=STAGES) wait(&shared.free[s],((tile/STAGES)-1)&1);
                    auto& slot=shared.slot[s];
                    if(t*64+64<=p.n) {
                        if(leader) {
                            expect(&shared.ready[s],sizeof(slot.k)+sizeof(slot.v));
                            summary_tma(&km,slot.k.data,&shared.ready[s],{0,0,bh*p.n+t*64,0});
                            summary_tma(&vm,slot.v.data,&shared.ready[s],{0,0,bh*p.n+t*64,0});
                        }
                    } else {
                        int valid=min(64,max(0,p.n-t*64));
#pragma unroll 1
                        for(int r=0;r<valid;++r) {
                            int pr=(r&7)*8+((r>>3)&3)*2+(r>>5);
#pragma unroll
                            for(int c=0;c<D;++c)
                                slot.k[int2{pr,c}]=static_cast<const __nv_bfloat16*>(p.b)[
                                    (int64_t(bh)*p.n+t*64+r)*D+c];
#pragma unroll
                            for(int c=0;c<DV;++c)
                                slot.v[int2{pr,c}]=static_cast<const __nv_bfloat16*>(p.v)[
                                    (int64_t(bh)*p.n+t*64+r)*DV+c];
                        }
#pragma unroll 1
                        for(int r=valid;r<64;++r) {
                            int pr=(r&7)*8+((r>>3)&3)*2+(r>>5);
#pragma unroll
                            for(int c=0;c<D;++c) slot.k[int2{pr,c}]=__float2bfloat16(0);
#pragma unroll
                            for(int c=0;c<DV;++c) slot.v[int2{pr,c}]=__float2bfloat16(0);
                        }
                        arrive(&shared.ready[s]);
                    }
                    // Largest shape uses two Q phases, but K0/V0 is issued
                    // before waiting for Q0 consumption. This keeps next-task
                    // Q0/K0/V0 prefetch possible while the old task drains.
                    if constexpr(QPHASES==2) {
                        if(t==0 && leader) {
                            wait(&shared.qfree,qepoch&1);
                            expect(&shared.qready,sizeof(shared.q));
                            summary_tma(&qm,shared.q.data,&shared.qready,{base+QROWS,bh,0,0});
                        }
                    }
                }
            }
        }
    } else {
        kt::warpgroup::increase_registers<232>();
#pragma unroll 1
        for(int task=blockIdx.x;task<total;task+=gridDim.x,++task_round) {
            int bh=task/blocks,base=(task%blocks)*128;
            int key_end=min(p.padded_n,base+128);
            int checkpoint=(task%blocks)*4+(warp&3);
            int qbase=base+(warp&3)*32+(warp/4)*16;
            kt::rt_bf<16,D> qreg;
#pragma unroll
            for(int phase=0;phase<QPHASES;++phase) {
                wait(&shared.qready,(task_round*QPHASES+phase)&1);
                if((qbase-base)/QROWS==phase) {
                    auto view=shared.q.template subtile<16,D>({((qbase-base)%QROWS)/16,0});
                    kt::warp::load(qreg,view);
                }
                arrive(&shared.qfree);
            }
        // One lane per query row generates a decision, before the key loop.
bool hard[2];
            if constexpr(MODE!=2) {hard[0]=hard[1]=MODE==1;}
            else if(p.hard_bits) {
                uint32_t bits=qbase<p.n?p.hard_bits[int64_t(bh)*((p.n+31)/32)+qbase/32]:0;
                hard[0]=(bits>>((qbase%32)+lane/4))&1;
                hard[1]=(bits>>((qbase%32)+8+lane/4))&1;
            } else {
                int decision=0;
                if(lane<16 && qbase+lane<p.n)
                    decision=row_hard(p.seed,p.offset,uint64_t(bh)*p.n+qbase+lane,p.hard_prob);
                hard[0]=__shfl_sync(0xffffffff,decision,lane/4);
                hard[1]=__shfl_sync(0xffffffff,decision,8+lane/4);
            }
        float tau=p.tau[bh%p.heads], tau2=tau*LOG2E, scale2=p.scale*LOG2E;
        float row_bias2[2];
        Label cached_label[2];
        #pragma unroll
        for(int r=0;r<2;++r) {
            int i=qbase+r*8+lane/4;
            if constexpr(MODE!=1 && !COLUMN_LSE)
                row_bias2[r]=(tau-(i<p.n?p.lse[int64_t(bh)*p.n+i]:0.f))*LOG2E;
            if constexpr(MODE!=0)
                cached_label[r]=i<p.n?reinterpret_cast<const Label*>(p.q_label)[int64_t(bh)*p.n+i]:Label(-1);
        }
        Buffer::VState left;
        kt::rt_fl<16,DV> out{0.f};
        float maximum[2]{0,0}, denominator[2]{1,1};
        #pragma unroll 1
        for(int t=0;t*64<key_end;++t,++tile) {
            int s=tile%STAGES, phase=(tile/STAGES)&1;
            Scalar scalar;
            {
                Label key_label[2];
                float key_bias2[2];
                #pragma unroll
                for(int e=0;e<2;++e) {
                    int j=t*64+lane+e*32;
                    if constexpr(MODE!=0)
                        key_label[e]=j<p.n?reinterpret_cast<const Label*>(p.k_label)[int64_t(bh)*p.n+j]:Label(-1);
                    if constexpr(MODE!=1 && COLUMN_LSE)
                        key_bias2[e]=(tau-(j<p.n?p.lse[int64_t(bh)*p.n+j]:0.f))*LOG2E;
                }
                wait(&shared.ready[s],phase);
                kt::rt_bf<D,64,kt::ducks::rt_layout::col> kreg;
                kt::rt_fl<16,64> accum{0.f};
                load_rhs_tile(kreg,shared.slot[s].k);
                kt::warp::wmma::mma_AB(accum,qreg,kreg,accum);
                #pragma unroll
                for(int r=0;r<2;++r) {
                    #pragma unroll
                    for(int c=0;c<8;++c) {
                        auto x=accum.tiles[0][c/2].data[r+2*(c&1)];
                        int i=qbase+r*8+lane/4, col=c+8*(lane&3);
                        float values[2];
                        #pragma unroll
                        for(int e=0;e<2;++e) {
                            float score;
                            if constexpr(MODE!=1) {
                                float bias;
                                if constexpr(COLUMN_LSE) bias=__shfl_sync(0xffffffff,key_bias2[e],col);
                                else bias=row_bias2[r];
                                score=fmaf(e==0?x.x:x.y,scale2,bias);
                            }
                            if constexpr(MODE!=0) {
                                Label label=__shfl_sync(0xffffffff,key_label[e],col);
                                float hs=summary_select(cached_label[r]==label,tau2,LOG_ZERO);
                                if constexpr(MODE==1) score=hs;
                                else score=summary_select(hard[r],hs,score);
                            }
                            int j=t*64+col+e*32;
                            values[e]=summary_select(i<p.n && j<p.n && j<=i,score,LOG_ZERO);
                        }
                        scalar.data[r][c].value={values[0],values[1]};
                    }
                }
            }
            if constexpr(!OUTPUT) arrive(&shared.free[s]);
            scalar.roll(); Buffer data;
            #pragma unroll
            for(int r=0;r<2;++r) {
                #pragma unroll
                for(int c=0;c<8;++c) {
                    auto x=scalar.data[r][c].value; data.data[r][c]={x,x};
                    int i=qbase+(r*8+lane/4-(7-c)+16)%16;
                    int j=t*64+c+(lane&3)*8;
                    if(i>=p.n || j>=p.n) { data.data[r][c].first.u0=0; data.data[r][c].second.u0=LOG_ZERO; }
                    if(i>=p.n || j+32>=p.n) { data.data[r][c].first.u1=0; data.data[r][c].second.u1=LOG_ZERO; }
                }
            }
            Buffer::HState top;
            if(warp>=4) {
                wait(&shared.mail_ready[s],phase);
                top=Buffer::HState::load_shared(shared.mail[warp-4][s]);
                arrive(&shared.mail_free[s]);
            } else if constexpr(OUTPUT) {
                if(checkpoint>0 && checkpoint<=p.checkpoints) {
                    int j=t*64+8*(lane&3)+6-lane/4;
                    int64_t off=(int64_t(bh)*p.checkpoints+checkpoint-1)*p.padded_n;
                    top.init[0].first={0,0};
                    top.init[0].second={j>=0?p.boundary[off+j]:LOG_ZERO,p.boundary[off+j+32]};
                }
            }
            Buffer::StatePair result;
            if constexpr(OUTPUT) result=data.inclusive_scan(left,top);
            else result=data.reduce_forward(left,top);
            left=result.first;
            if(warp<4) {
                if(tile>=STAGES) wait(&shared.mail_free[s],phase^1);
                result.second.store_shared(shared.mail[warp][s]);
                arrive(&shared.mail_ready[s]);
            } else if constexpr(!OUTPUT) {
                if(checkpoint<p.checkpoints) {
                    int j=8*(lane&3)+6-lane/4;
                    auto h=result.second.init[0];
                    auto* dest=reinterpret_cast<float2*>(p.summary)+(int64_t(bh)*p.checkpoints+checkpoint)*p.padded_n+t*64;
                    if(j>=0) dest[j]=make_float2(h.first.u0,h.second.u0);
                    dest[j+32]=make_float2(h.first.u1,h.second.u1);
                    if(lane==31) dest[63]=make_float2(left.init[1].first.u0,left.init[1].second.u0);
                }
            }
            if constexpr(OUTPUT) {
                #pragma unroll
                for(int r=0;r<2;++r) {
                    #pragma unroll
                    for(int c=0;c<8;++c) scalar.data[r][c].value=data.data[r][c].second;
                }
                // c=7 has zero roll. Odd lane groups own columns 15/31/47/63.
                // Affine first is already dead; export from the scalar tile.
                if(p.vertical && (lane&1)) {
                    #pragma unroll
                    for(int r=0;r<2;++r) {
                        int i=qbase+8*r+lane/4;
                        if(i<p.padded_n) {
                            auto x=scalar.data[r][7].value;
                            int edge=t*4+(lane&3)/2;
                            int64_t off=int64_t(bh)*(p.padded_n/16)*p.padded_n;
                            p.vertical[off+int64_t(edge)*p.padded_n+i]=x.u0;
                            p.vertical[off+int64_t(edge+2)*p.padded_n+i]=x.u1;
                        }
                    }
                }
                scalar.template roll<false>();
                if(p.horizontal && qbase%64==48 && qbase<p.padded_n && lane/4==7) {
                    int64_t off=(int64_t(bh)*(p.padded_n/64)+qbase/64)*p.padded_n+t*64;
                    #pragma unroll
                    for(int c=0;c<8;++c) {
                        auto x=scalar.data[1][c].value;
                        int j=c+8*(lane&3);
                        p.horizontal[off+j]=x.u0;
                        p.horizontal[off+j+32]=x.u1;
                    }
                }
                kt::rt_bf<16,64> weights;
                #pragma unroll
                for(int r=0;r<2;++r) {
                    float m=maximum[r];
                    #pragma unroll
                    for(int c=0;c<8;++c) {
                        auto x=scalar.data[r][c].value;
                        auto pos=Buffer::layout(r,c,0);
                        // Padding identity transports state, but is not an attention weight.
                        if(qbase+pos.first>=p.n || t*64+pos.second>=p.n) x.u0=LOG_ZERO;
                        if(qbase+pos.first>=p.n || t*64+pos.second+32>=p.n) x.u1=LOG_ZERO;
                        scalar.data[r][c].value=x; m=fmaxf(m,fmaxf(x.u0,x.u1));
                    }
                    m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));
                    m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));
                    float alpha=exp2f(maximum[r]-m), sum=0;
                    #pragma unroll
                    for(int c=0;c<8;++c) {
                        auto x=scalar.data[r][c].value;
                        float a=exp2f(x.u0-m),b=exp2f(x.u1-m); sum+=a+b;
                        weights.tiles[0][c/2].data[r+2*(c&1)]=__floats2bfloat162_rn(a,b);
                    }
                    sum+=__shfl_xor_sync(0xffffffff,sum,1); sum+=__shfl_xor_sync(0xffffffff,sum,2);
                    denominator[r]=denominator[r]*alpha+sum; maximum[r]=m;
                    #pragma unroll
                    for(int c=0;c<DV/16;++c) {
                        #pragma unroll
                        for(int k=r;k<4;k+=2) { out.tiles[0][c].data[k].x*=alpha; out.tiles[0][c].data[k].y*=alpha; }
                    }
                }
                kt::rt_bf<64,DV,kt::ducks::rt_layout::col> vreg;
                kt::warp::load(vreg,shared.slot[s].v);
                kt::warp::wmma::mma_AB(out,weights,vreg,out);
                arrive(&shared.free[s]);
            }
        }
        // Initialize omitted strictly upper-triangular state: passing and backward
        // read dense checkpoint arrays, including these empty regions.
        int end=base+128;
        if constexpr(!OUTPUT) {
            if(warp>=4 && checkpoint<p.checkpoints)
                for(int j=end+lane;j<p.padded_n;j+=32) {
                    // A 32-row diagonal wholly beyond N is identity, not zero.
                    // Earlier padded endpoints still cross a masked valid cell.
                    int64_t off=(int64_t(bh)*p.checkpoints+checkpoint)*p.padded_n+j;
                    reinterpret_cast<float2*>(p.summary)[off]=make_float2(j>=p.n+31?0.f:LOG_ZERO,LOG_ZERO);
                }
        } else {
            if(p.vertical)
                for(int e=end/16;e<p.padded_n/16;++e)
                    for(int r=lane;r<16;r+=32) {
                        int i=qbase+r;
                        if(i<p.padded_n) p.vertical[(int64_t(bh)*(p.padded_n/16)+e)*p.padded_n+i]=LOG_ZERO;
                    }
            if(p.horizontal && qbase%64==48 && qbase<p.padded_n)
                for(int j=end+lane;j<p.padded_n;j+=32)
                    p.horizontal[(int64_t(bh)*(p.padded_n/64)+qbase/64)*p.padded_n+j]=LOG_ZERO;
        }
        if constexpr(OUTPUT) {
            // One Newton-refined FP32 reciprocal per row, rather than per-element
            // IEEE division (ptxas outlines its exceptional slow path as CALL).
            float inverse[2]{normal_reciprocal(denominator[0]),normal_reciprocal(denominator[1])};
            #pragma unroll
            for(int c=0;c<DV/16;++c) {
                #pragma unroll
                for(int k=0;k<4;++k) {
                    int i=qbase+(k%2)*8+lane/4,j=c*16+(k/2)*8+(lane%4)*2;
                    if(i<p.n) {
                        auto x=out.tiles[0][c].data[k];
                        auto* dest=static_cast<__nv_bfloat16*>(p.output)+(int64_t(bh)*p.n+i)*DV+j;
                        // j is even: pack the adjacent BF16 pair into one
                        // aligned32-bit store instead of two half-width writes.
                        *reinterpret_cast<__nv_bfloat162*>(dest)=__floats2bfloat162_rn(
                            x.x*inverse[k%2],x.y*inverse[k%2]);
                    }
                }
            }
            #pragma unroll
            for(int r=0;r<2;++r) {
                int i=qbase+r*8+lane/4;
                if((lane&3)==0 && i<p.n) p.normalizer[int64_t(bh)*p.n+i]=maximum[r]+log2f(denominator[r]);
            }
        }
    }
    }
    kt::warpgroup::sync(1+warp/4); // Independent, collective role exit.
}

__global__ void passing(Args p) {
    int d=int(blockIdx.x*blockDim.x+threadIdx.x)-(p.checkpoints-1)*32;
    if(d>=p.padded_n) return;
    int bh=blockIdx.y; float x=LOG_ZERO;
    for(int s=0;s<p.checkpoints;++s) {
        int j=d+s*32;
        if(j>=0 && j<p.padded_n) {
            int64_t off=(int64_t(bh)*p.checkpoints+s)*p.padded_n+j;
            auto a=reinterpret_cast<const float2*>(p.summary)[off];
            x=logadd2(x+a.x,a.y); p.boundary[off]=x;
        }
    }
}

template<int D,int DV,bool OUTPUT,bool COLUMN_LSE,int MODE,typename Label>
void launch(const Args& p,cudaStream_t stream) {
    auto km=permuted_map<D,true>(p.b,p.batch_heads*p.n);
    CUtensorMap vm{};
    if constexpr(OUTPUT) vm=permuted_map<DV,true>(p.v,p.batch_heads*p.n);
    auto qm=output_q_map<D,Shared<D,DV,OUTPUT>::QROWS>(p);
    constexpr int sm=sizeof(Shared<D,DV,OUTPUT>);
    static_assert(sm<=63*1024,"output shared budget exceeded");
    auto err=cudaFuncSetAttribute(core<D,DV,OUTPUT,COLUMN_LSE,MODE,Label>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm);
    if(err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    int device,sms;
    auto status=cudaGetDevice(&device);
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    int tasks=((p.n+127)/128)*p.batch_heads;
    core<D,DV,OUTPUT,COLUMN_LSE,MODE,Label><<<std::min(tasks,sms),384,sm,stream>>>(p,km,vm,qm);
}
#include "summary_persistent.cuh"
void launch_summary(const Args& p,int d,cudaStream_t stream) {
    switch(d) { case 32: launch_persistent_summary<32>(p,stream);break; case 64: launch_persistent_summary<64>(p,stream);break; case 128: launch_persistent_summary<128>(p,stream);break; }
}
void launch_passing(const Args& p,cudaStream_t stream) {
    passing<<<dim3((p.padded_n+(p.checkpoints-1)*32+127)/128,p.batch_heads),128,0,stream>>>(p);
}
template<int D,int DV,bool C,typename L> void output_mode(const Args& p,cudaStream_t s) {
    if(p.hard_prob==0) launch<D,DV,true,C,0,int>(p,s);
    else if(p.hard_prob==1) launch<D,DV,true,C,1,L>(p,s);
    else launch<D,DV,true,C,2,L>(p,s);
}
template<int D,int DV,bool C> void output_labels(const Args& p,cudaStream_t s) {
    if(p.label32) output_mode<D,DV,C,int>(p,s);
    else output_mode<D,DV,C,long long>(p,s);
}
template<int D,int DV> void output_direction(const Args& p,cudaStream_t s) {
    if(p.column_lse) output_labels<D,DV,true>(p,s);
    else output_labels<D,DV,false>(p,s);
}
template<int D> void output_dim(const Args& p,int dv,cudaStream_t stream) {
    switch(dv) { case 32: output_direction<D,32>(p,stream);break; case 64: output_direction<D,64>(p,stream);break; case 128: output_direction<D,128>(p,stream);break; }
}
void launch_output(const Args& p,int d,int dv,cudaStream_t stream) {
    switch(d) { case 32: output_dim<32>(p,dv,stream);break; case 64: output_dim<64>(p,dv,stream);break; case 128: output_dim<128>(p,dv,stream);break; }
}
} // namespace dism_v2
