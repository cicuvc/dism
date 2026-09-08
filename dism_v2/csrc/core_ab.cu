// B3 compile-gated implementation: no global W/P/E/G materialization.
#include <cuda.h>
#include <kittens.cuh>
#include <stdexcept>
#include "core_api.h"
#include "log_affine.cuh"
#include "pipeline.cuh"
#include "row_rng.cuh"
namespace dism_v2 {
namespace ab {
namespace kt=kittens;
#include "ab_rescan.cuh"
#include "scalar_grad.cuh"
using Reverse=glx::MMABuffer<16,64,glx::BinaryElement,glx::AffineComposeOp,glx::F32x2>;
template<int D,int DV> struct Shared {
    kt::st_bf<16,D> key;
    kt::st_bf<16,DV> value;
    kt::st_bf<64,D> query;
    kt::st_bf<64,DV> dout;
    kt::st_bf<16,64> soft;
    kt::st_bf<64,32> query_gemm;
    alignas(128) float da[2][16][32];
    uint64_t ready;
};
__device__ __forceinline__ float2 coefficient(const Args& p,const float* delta,float w,float dot,
                                              int bh,int q,int k) {
    if(q>=p.n || k>=p.n) return {1,0};
    if(w==-INFINITY) return {0,0};
    float t,x=w*0.3465735902799726547f;
    asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(x));
    return {fmaf(.5f,t,.5f),exp2f(w-p.normalizer[int64_t(bh)*p.n+q])*(dot-delta[int64_t(bh)*p.n+q])};
}
template<int D,int DV>
__global__ void run(__grid_constant__ const Args p,__grid_constant__ const CUtensorMap qm,
                   __grid_constant__ const CUtensorMap dm,const __nv_bfloat16* dout,
                   const float* delta,const float* boundary,
                   __grid_constant__ const CUtensorMap dam,float* db,float* dlse,float* tau_partial) {
    extern __shared__ __align__(128) unsigned char memory[];
    auto& shared=*reinterpret_cast<Shared<D,DV>*>(memory);
    int lane=threadIdx.x,chunk=blockIdx.x,bh=blockIdx.y,phase=0;
    if(lane==0) {init_bar(&shared.ready,1);asm volatile("fence.proxy.async.shared::cta;" ::: "memory");}
    __syncwarp();
    Reverse::VState right[2];
    kt::rt_fl<16,D> accumulated[2]={kt::rt_fl<16,D>{0.f},kt::rt_fl<16,D>{0.f}};
    int issued=0;
    float tau_sum=0,key_lse[2][2]={{0,0},{0,0}};
    for(int qb=p.padded_n-64;qb>=0;qb-=64) {
        if(qb+64<=p.n) {
            if(lane==0) {
                expect(&shared.ready,sizeof(shared.query)+sizeof(shared.dout));
                constexpr int Q=D==32?32:64,V=DV==32?32:64;
                #pragma unroll
                for(int c=0;c<D/Q;++c) tma5(&qm,shared.query.data+c*64*Q,&shared.ready,bh*p.n+qb,c);
                #pragma unroll
                for(int c=0;c<DV/V;++c) tma5(&dm,shared.dout.data+c*64*V,&shared.ready,bh*p.n+qb,c);
            }
            wait(&shared.ready,phase);phase^=1;
        } else {
            for(int x=lane;x<64*D;x+=32) {
                int q=qb+logical_row(x/D);
                shared.query[int2{x/D,x%D}]=q<p.n?static_cast<const __nv_bfloat16*>(p.a)[(int64_t(bh)*p.n+q)*D+x%D]:__float2bfloat16(0);
            }
            for(int x=lane;x<64*DV;x+=32) {
                int q=qb+logical_row(x/DV);
                shared.dout[int2{x/DV,x%DV}]=q<p.n?dout[(int64_t(bh)*p.n+q)*DV+x%DV]:__float2bfloat16(0);
            }
            __syncwarp();
        }
        uint32_t hard0=__ballot_sync(0xffffffff,qb+lane<p.n && row_hard(p.seed,p.offset,uint64_t(bh)*p.n+qb+lane,p.hard_prob));
        uint32_t hard1=__ballot_sync(0xffffffff,qb+lane+32<p.n && row_hard(p.seed,p.offset,uint64_t(bh)*p.n+qb+lane+32,p.hard_prob));
        Reverse::HState bottom;
        if(chunk+1<p.padded_n/32) {
            int q=qb+8*(lane&3)+8-lane/4;
            int64_t off=(int64_t(bh)*(p.padded_n/32)+chunk+1)*p.padded_n;
            bottom.init[0].first={1,1};
            bottom.init[0].second={q<p.padded_n?boundary[off+q]:0,q+32<p.padded_n?boundary[off+q+32]:0};
        }
        // Same 4->0 ordering as the planned paired warps, sequential within one warp.
        #pragma unroll
        for(int half=1;half>=0;--half) {
            int kb=chunk*32+half*16;
            for(int x=lane;x<16*D;x+=32) {
                int k=kb+x/D;
                shared.key[int2{x/D,x%D}]=k<p.n?static_cast<const __nv_bfloat16*>(p.b)[(int64_t(bh)*p.n+k)*D+x%D]:__float2bfloat16(0);
            }
            for(int x=lane;x<16*DV;x+=32) {
                int k=kb+x/DV;
                shared.value[int2{x/DV,x%DV}]=k<p.n?static_cast<const __nv_bfloat16*>(p.v)[(int64_t(bh)*p.n+k)*DV+x%DV]:__float2bfloat16(0);
            }
            __syncwarp();
            Scalar scalar;
            reconstruct<D>(p,shared,bh,kb,qb,hard0,hard1,scalar);
            Reverse rev;
            {
                kt::rt_fl<16,64> dp{0.f};
                #pragma unroll
                for(int f=0;f<DV/32;++f) {
                    auto vs=shared.value.template subtile<16,32>({0,f});
                    auto ds=shared.dout.template subtile<64,32>({0,f});
                    kt::rt_bf<16,32> v;
                    kt::rt_bf<64,32> d;
                    kt::warp::load(v,vs);kt::warp::load(d,ds);
                    kt::warp::wmma::mma_ABt(dp,v,d,dp);
                }
                #pragma unroll
                for(int r=0;r<2;++r) {
                    #pragma unroll
                    for(int c=0;c<8;++c) {
                        auto pos=Buffer::layout(r,c,0);auto w=scalar.data[r][c].value;
                        auto dot=dp.tiles[0][c/2].data[r+2*(c&1)];
                        auto x=coefficient(p,delta,w.u0,dot.x,bh,qb+pos.second,kb+pos.first);
                        auto y=coefficient(p,delta,w.u1,dot.y,bh,qb+pos.second+32,kb+pos.first);
                        rev.data[r][c]={{x.x,y.x},{x.y,y.y}};
                    }
                }
            }
            rev.reverse_roll();
            auto state=rev.reverse_inclusive_scan(right[half],bottom);
            right[half]=state.first;bottom=state.second;
            #pragma unroll
            for(int r=0;r<2;++r) {
                #pragma unroll
                for(int c=0;c<8;++c) scalar.data[r][c].value=rev.data[r][c].second;
            }
            scalar.template reverse_roll<false>();
            scalar_gradients(p,scalar,bh,kb,qb,hard0,hard1,dlse,tau_sum,key_lse[half]);
            #pragma unroll
            for(int r=0;r<2;++r) {
                #pragma unroll
                for(int c=0;c<8;++c) {
                    auto pos=Buffer::layout(r,c,0);auto x=scalar.data[r][c].value;
                    int k=kb+pos.first,q=qb+pos.second;
                    bool valid=k<p.n && q<p.n;
                    bool valid1=k<p.n && q+32<p.n;
                    shared.soft[int2{pos.first,pos.second}]=__float2bfloat16_rn(
                        valid && !((hard0>>pos.second)&1)?x.u0:0.f);
                    shared.soft[int2{pos.first,pos.second+32}]=__float2bfloat16_rn(
                        valid1 && !((hard1>>pos.second)&1)?x.u1:0.f);
                }
            }
            __syncwarp();
            // Logical Gsoft shared layout serves both gradient GEMMs.
            // dB is owned by this warp across every query tile.
            {
                kt::rt_bf<16,64> grad;
                kt::warp::load(grad,shared.soft);
                #pragma unroll
                for(int f=0;f<D/32;++f) {
                    for(int x=lane;x<64*32;x+=32) {
                        int q=x/32,physical=8*(q&7)+2*((q>>3)&3)+(q>>5);
                        shared.query_gemm[int2{q,x%32}]=shared.query[int2{physical,f*32+x%32}];
                    }
                    __syncwarp();
                    kt::rt_bf<64,32,kt::ducks::rt_layout::col> query;
                    kt::warp::load(query,shared.query_gemm);
                    kt::rt_fl<16,32> part;
                    #pragma unroll
                    for(int c=0;c<2;++c) part.tiles[0][c]=accumulated[half].tiles[0][f*2+c];
                    kt::warp::wmma::mma_AB(part,grad,query,part);
                    #pragma unroll
                    for(int c=0;c<2;++c) accumulated[half].tiles[0][f*2+c]=part.tiles[0][c];
                    __syncwarp();
                }
            }
            // dA: transpose reload Gsoft, stage 16x32 FP32 results, async TMA add.
            #pragma unroll
            for(int qt=0;qt<4;++qt) {
                auto gs=shared.soft.template subtile<16,16>({0,qt});
                kt::rt_bf<16,16,kt::ducks::rt_layout::col> grad;
                kt::warp::load(grad,gs);
                #pragma unroll
                for(int f=0;f<D/32;++f) {
                    auto ks=shared.key.template subtile<16,32>({0,f});
                    kt::rt_bf<16,32,kt::ducks::rt_layout::col> key;
                    kt::warp::load(key,ks);
                    kt::rt_fl<16,32> part{0.f};
                    kt::warp::wmma::mma_AtB(part,grad,key,part);
                    int slot=issued%2;
                    if(lane==0 && issued>=2)
                        asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
                    __syncwarp();
                    #pragma unroll
                    for(int c=0;c<2;++c) {
                        #pragma unroll
                        for(int r=0;r<4;++r) {
                            auto x=part.tiles[0][c].data[r];
                            int q=(r%2)*8+lane/4,j=c*16+(r/2)*8+2*(lane%4);
                            shared.da[slot][q][j]=x.x*p.scale;
                            shared.da[slot][q][j+1]=x.y*p.scale;
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if(lane==0) {
                        unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(shared.da[slot]));
                        asm volatile("cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group "
                            "[%0, {%2, %3, %4}], [%1];" ::
                            "l"(&dam),"r"(addr),"r"(f*32),"r"(qb+qt*16),"r"(bh):"memory");
                        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                    }
                    ++issued;
                }
            }
            __syncwarp();
        }
    }
    #pragma unroll
    for(int half=0;half<2;++half) {
        #pragma unroll
        for(int c=0;c<D/16;++c) {
            #pragma unroll
            for(int r=0;r<4;++r) {
                int k=chunk*32+half*16+(r%2)*8+lane/4,j=c*16+(r/2)*8+2*(lane%4);
                if(k<p.n) {
                    auto x=accumulated[half].tiles[0][c].data[r];
                    int64_t off=(int64_t(bh)*p.n+k)*D+j;
                    db[off]=x.x*p.scale;db[off+1]=x.y*p.scale;
                }
            }
        }
    }
    #pragma unroll
    for(int half=0;half<2;++half) store_key_lse(p,bh,chunk*32+half*16,dlse,key_lse[half]);
    float total=warp_sum(tau_sum);
    if(lane==0) tau_partial[int64_t(bh)*gridDim.x+chunk]=total;
    if(lane==0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    __syncwarp();
}

template<int D,int DV> void launch(const Args& p,const void* dout,const float* delta,const float* boundary,
                                  const CUtensorMap& dam,float* db,float* dlse,float* tau_partial,cudaStream_t stream) {
    auto qm=permuted_map<D>(p.a,p.batch_heads*p.n),dm=permuted_map<DV>(dout,p.batch_heads*p.n);
    constexpr int bytes=sizeof(Shared<D,DV>);
    auto err=cudaFuncSetAttribute(run<D,DV>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    run<D,DV><<<dim3(p.padded_n/32,p.batch_heads),32,bytes,stream>>>(p,qm,dm,
        static_cast<const __nv_bfloat16*>(dout),delta,boundary,dam,db,dlse,tau_partial);
}
template<int D> void dispatch(const Args& p,int dv,const void* dout,const float* delta,const float* boundary,
                             const CUtensorMap& dam,float* db,float* dlse,float* tau_partial,cudaStream_t stream) {
    switch(dv) {
        case 32:launch<D,32>(p,dout,delta,boundary,dam,db,dlse,tau_partial,stream);break;
        case 64:launch<D,64>(p,dout,delta,boundary,dam,db,dlse,tau_partial,stream);break;
        case 128:launch<D,128>(p,dout,delta,boundary,dam,db,dlse,tau_partial,stream);break;
    }
}
}
void launch_operand_backward(const Args& p,int d,int dv,const void* dout,const float* delta,
                             const float* boundary,float* da,float* db,float* dlse,float* tau_partial,cudaStream_t stream) {
    CUtensorMap dam;
    cuuint64_t dims[3]={cuuint64_t(d),cuuint64_t(p.n),cuuint64_t(p.batch_heads)};
    cuuint64_t strides[2]={cuuint64_t(d)*4,cuuint64_t(d)*p.n*4};
    cuuint32_t box[3]={32,16,1},steps[3]={1,1,1};
    auto status=cuTensorMapEncodeTiled(&dam,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,3,da,dims,strides,box,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(status!=CUDA_SUCCESS) throw std::runtime_error("dA tensor map encoding failed");
    switch(d) {
        case 32:ab::dispatch<32>(p,dv,dout,delta,boundary,dam,db,dlse,tau_partial,stream);break;
        case 64:ab::dispatch<64>(p,dv,dout,delta,boundary,dam,db,dlse,tau_partial,stream);break;
        case 128:ab::dispatch<128>(p,dv,dout,delta,boundary,dam,db,dlse,tau_partial,stream);break;
    }
}
}
