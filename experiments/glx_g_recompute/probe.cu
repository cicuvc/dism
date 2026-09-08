// Diagnostic only: real independent W rescan + dP/E + G tile restoration.
#include <cuda.h>
#include <kittens.cuh>
#include "../../dism_v2/csrc/core_api.h"
#include "../../dism_v2/csrc/log_affine.cuh"
#include "../../dism_v2/csrc/pipeline.cuh"
#include "../../dism_v2/csrc/row_rng.cuh"
namespace dism_v2 {
namespace g_probe {
namespace kt=kittens;
#include "rescan.cuh"
using Reverse=glx::MMABuffer<16,64,glx::BinaryElement,glx::AffineComposeOp,glx::F32x2>;
template<int D,int DV> struct Shared {
    kt::st_bf<16,D> key;
    kt::st_bf<16,DV> value;
    kt::st_bf<64,D> query;
    kt::st_bf<64,DV> dout;
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
                   const float* delta,const float* boundary,float* output) {
    __shared__ Shared<D,DV> shared;
    int lane=threadIdx.x,chunk=blockIdx.x,bh=blockIdx.y,phase=0;
    if(lane==0) {init_bar(&shared.ready,1);asm volatile("fence.proxy.async.shared::cta;" ::: "memory");}
    __syncwarp();
    Reverse::VState right[2];
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
            #pragma unroll
            for(int r=0;r<2;++r) {
                #pragma unroll
                for(int c=0;c<8;++c) {
                    auto pos=Buffer::layout(r,c,0);auto x=scalar.data[r][c].value;
                    int64_t off=(int64_t(bh)*p.padded_n+qb+pos.second)*p.padded_n+kb+pos.first;
                    output[off]=x.u0;output[off+32*p.padded_n]=x.u1;
                }
            }
            __syncwarp();
        }
    }
}
template<int D,int DV> void launch(const Args& p,const void* dout,const float* delta,const float* boundary,float* out,cudaStream_t stream) {
    auto qm=permuted_map<D>(p.a,p.batch_heads*p.n),dm=permuted_map<DV>(dout,p.batch_heads*p.n);
    run<D,DV><<<dim3(p.padded_n/32,p.batch_heads),32,0,stream>>>(p,qm,dm,static_cast<const __nv_bfloat16*>(dout),delta,boundary,out);
}
template<int D> void dispatch(const Args& p,int dv,const void* dout,const float* delta,const float* boundary,float* out,cudaStream_t stream) {
    switch(dv) {
        case 32: launch<D,32>(p,dout,delta,boundary,out,stream);break;
        case 64: launch<D,64>(p,dout,delta,boundary,out,stream);break;
        case 128: launch<D,128>(p,dout,delta,boundary,out,stream);break;
    }
}
}
void launch_g_probe(const Args& p,int d,int dv,const void* dout,const float* delta,const float* boundary,float* out,cudaStream_t stream) {
    switch(d) {
        case 32: g_probe::dispatch<32>(p,dv,dout,delta,boundary,out,stream);break;
        case 64: g_probe::dispatch<64>(p,dv,dout,delta,boundary,out,stream);break;
        case 128: g_probe::dispatch<128>(p,dv,dout,delta,boundary,out,stream);break;
    }
}
}
