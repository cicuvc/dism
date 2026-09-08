// Initial key-owned dV: independent W rescan, no global W/P or atomics.
#include <cuda.h>
#include <kittens.cuh>
#include "core_api.h"
#include "log_affine.cuh"
#include "pipeline.cuh"
#include "row_rng.cuh"

namespace dism_v2 {
namespace kt=kittens;
using Reverse=glx::MMABuffer<16,64,glx::BinaryElement,glx::AffineComposeOp,glx::F32x2>;
struct ReverseScratch { glx::F32x2 w[2][8][32]; };
template<int D,int DV,bool SUMMARY> struct RecomputeShared {
    kt::st_bf<16,D> key;
    kt::st_bf<64,D> query;
    kt::st_bf<64,DV> dout;
    uint64_t ready;
    // Shared tile scratch bounds register lifetime between PV and reverse reduce.
    kt::st_bf<16,SUMMARY?DV:16> value;
    ReverseScratch scratch[SUMMARY?1:0];
    float2 saved[SUMMARY?DV/16:0][4][32];
};

__device__ __forceinline__ float transposed_score(const Args& p,float dot,int bh,int q,int k,bool hard) {
    if(q>=p.n || k>=p.n || k>q) return LOG_ZERO;
    float tau=p.tau[bh%p.heads];
    if(hard) return p.q_label[int64_t(bh)*p.n+q]==p.k_label[int64_t(bh)*p.n+k]?tau*LOG2E:LOG_ZERO;
    return (dot*p.scale-p.lse[int64_t(bh)*p.n+(p.column_lse?k:q)]+tau)*LOG2E;
}

__device__ __forceinline__ float2 reverse_coefficient(const Args& p,const float* delta,
        float w,float dot,int bh,int q,int k) {
    if(k>=p.n || q>=p.n) return {1,0};
    if(w==LOG_ZERO) return {0,0};
    float z=exp2f(-fabsf(w)),a;
    float denominator=1+z;
    asm("rcp.approx.ftz.f32 %0,%1;":"=f"(a):"f"(denominator));
    a=fmaf(a,fmaf(-denominator,a,1.f),a);
    if(w<0) a*=z;
    return {a,exp2f(w-p.normalizer[int64_t(bh)*p.n+q])*(dot-delta[int64_t(bh)*p.n+q])};
}

template<int D,int DV,bool SUMMARY>
__global__ void value_backward(__grid_constant__ const Args p,
        __grid_constant__ const CUtensorMap qm,
        __grid_constant__ const CUtensorMap dm,const __nv_bfloat16* dout,float* dv,const float* delta,float2* local) {
    extern __shared__ __align__(128) unsigned char bytes[];
    auto& shared=*reinterpret_cast<RecomputeShared<D,DV,SUMMARY>*>(bytes);
    int lane=threadIdx.x,g=lane&3,l=lane/4;
    int bh=blockIdx.y,kb=blockIdx.x*16;
    if(lane==0) {
        init_bar(&shared.ready,1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    for(int x=lane;x<16*D;x+=32) {
        int k=kb+x/D;
        shared.key[int2{x/D,x%D}]=k<p.n?
            static_cast<const __nv_bfloat16*>(p.b)[(int64_t(bh)*p.n+k)*D+x%D]:__float2bfloat16(0);
    }
    if constexpr(SUMMARY) for(int x=lane;x<16*DV;x+=32) {
        int k=kb+x/DV;
        shared.value[int2{x/DV,x%DV}]=k<p.n?
            static_cast<const __nv_bfloat16*>(p.v)[(int64_t(bh)*p.n+k)*DV+x%DV]:__float2bfloat16(0);
    }
    __syncwarp();
    if constexpr(SUMMARY) {
        #pragma unroll
        for(int c=0;c<DV/16;++c) {
            #pragma unroll
            for(int r=0;r<4;++r) shared.saved[c][r][lane]={0,0};
        }
    }
    __syncwarp();
    Reverse::VState right;
    kt::rt_fl<16,DV> accumulated{0.f};
    int phase=0;
    // One owning key warp streams query tiles in reverse order. No preceding
    // query tile or neighbour warp supplies recompute state.
    for(int qb=p.padded_n-64;qb>=0;qb-=64) {
        // Two decisions per lane, shared across all 16 held keys. No global mask.
        uint32_t hard0=__ballot_sync(0xffffffff,qb+lane<p.n &&
            row_hard(p.seed,p.offset,uint64_t(bh)*p.n+qb+lane,p.hard_prob));
        uint32_t hard1=__ballot_sync(0xffffffff,qb+lane+32<p.n &&
            row_hard(p.seed,p.offset,uint64_t(bh)*p.n+qb+lane+32,p.hard_prob));
        if(qb+64<=p.n) {
            if(lane==0) {
                expect(&shared.ready,sizeof(shared.query)+sizeof(shared.dout));
                constexpr int S=D==32?32:64;
                #pragma unroll
                for(int s=0;s<D/S;++s) tma5(&qm,shared.query.data+s*64*S,&shared.ready,bh*p.n+qb,s);
                constexpr int V=DV==32?32:64;
                #pragma unroll
                for(int s=0;s<DV/V;++s) tma5(&dm,shared.dout.data+s*64*V,&shared.ready,bh*p.n+qb,s);
            }
            wait(&shared.ready,phase); phase^=1;
        } else {
            for(int x=lane;x<64*D;x+=32) {
                int q=qb+logical_row(x/D);
                shared.query[int2{x/D,x%D}]=q<p.n?
                    static_cast<const __nv_bfloat16*>(p.a)[(int64_t(bh)*p.n+q)*D+x%D]:__float2bfloat16(0);
            }
            for(int x=lane;x<64*DV;x+=32) {
                int q=qb+logical_row(x/DV);
                shared.dout[int2{x/DV,x%DV}]=q<p.n?dout[(int64_t(bh)*p.n+q)*DV+x%DV]:__float2bfloat16(0);
            }
            __syncwarp();
        }
        Scalar scalar;
        {
            // Reload held key staging here so keys die before scan/PV; keeping
            // both keys and dV live across scan increases register pressure.
            kt::rt_bf<16,D> keys;
            kt::warp::load(keys,shared.key);
            kt::rt_bf<64,D> queries;
            kt::rt_fl<16,64> dot{0.f};
            kt::warp::load(queries,shared.query);
            kt::warp::wmma::mma_ABt(dot,keys,queries,dot);
            #pragma unroll
            for(int r=0;r<2;++r) {
                #pragma unroll
                for(int c=0;c<8;++c) {
                    auto x=dot.tiles[0][c/2].data[r+2*(c&1)];
                    auto pos=Buffer::layout(r,c,0);
                    scalar.data[r][c].value={transposed_score(p,x.x,bh,qb+pos.second,kb+pos.first,(hard0>>pos.second)&1),
                        transposed_score(p,x.y,bh,qb+pos.second+32,kb+pos.first,(hard1>>pos.second)&1)};
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
        if(kb>0) {
            int q=qb+8*g+6-l;
            int64_t off=(int64_t(bh)*(p.padded_n/16)+kb/16-1)*p.padded_n;
            top.init[0].first={0,0};
            top.init[0].second={q>=0?p.vertical[off+q]:LOG_ZERO,p.vertical[off+q+32]};
        }
        Buffer::VState left;
        if(qb>0 && g==3) {
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
        if constexpr(SUMMARY) {
            #pragma unroll
            for(int r=0;r<2;++r) {
                #pragma unroll
                for(int c=0;c<8;++c) shared.scratch[0].w[r][c][lane]=scalar.data[r][c].value;
            }
        }
        { // Probability operands and dO registers die before the dP GEMM.
        if constexpr(SUMMARY) {
            #pragma unroll
            for(int c=0;c<DV/16;++c) {
                #pragma unroll
                for(int r=0;r<4;++r) accumulated.tiles[0][c].data[r]=shared.saved[c][r][lane];
            }
        }
        kt::rt_bf<16,64> weights,residual;
        #pragma unroll
        for(int r=0;r<2;++r) {
            #pragma unroll
            for(int c=0;c<8;++c) {
                auto pos=Buffer::layout(r,c,0); auto x=scalar.data[r][c].value;
                int k=kb+pos.first,q=qb+pos.second;
                float a=0,b=0;
                if(k<p.n && q<p.n) a=exp2f(x.u0-p.normalizer[int64_t(bh)*p.n+q]);
                if(k<p.n && q+32<p.n) b=exp2f(x.u1-p.normalizer[int64_t(bh)*p.n+q+32]);
                auto high=__floats2bfloat162_rn(a,b);
                auto rounded=__bfloat1622float2(high);
                weights.tiles[0][c/2].data[r+2*(c&1)]=high;
                residual.tiles[0][c/2].data[r+2*(c&1)]=__floats2bfloat162_rn(a-rounded.x,b-rounded.y);
            }
        }
        kt::rt_bf<64,DV,kt::ducks::rt_layout::col> derivatives;
        kt::warp::load(derivatives,shared.dout);
        kt::warp::wmma::mma_AB(accumulated,weights,derivatives,accumulated);
        // Preserve FP32 P more closely using two BF16 terms. dO is already
        // BF16; only the probability operand needs residual correction.
        kt::warp::wmma::mma_AB(accumulated,residual,derivatives,accumulated);
        if constexpr(SUMMARY) {
            #pragma unroll
            for(int c=0;c<DV/16;++c) {
                #pragma unroll
                for(int r=0;r<4;++r) shared.saved[c][r][lane]=accumulated.tiles[0][c].data[r];
            }
        }
        }
        if constexpr(SUMMARY) {
            Reverse reverse;
            {
                kt::rt_fl<16,64> dp{0.f};
                #pragma unroll
                for(int segment=0;segment<DV/32;++segment) {
                    kt::rt_bf<16,32> values;
                    kt::rt_bf<64,32> derivatives;
                    auto vs=shared.value.template subtile<16,32>({0,segment});
                    auto ds=shared.dout.template subtile<64,32>({0,segment});
                    kt::warp::load(values,vs);
                    kt::warp::load(derivatives,ds);
                    kt::warp::wmma::mma_ABt(dp,values,derivatives,dp);
                }
                #pragma unroll
                for(int r=0;r<2;++r) {
                    #pragma unroll
                    for(int c=0;c<8;++c) {
                        auto pos=Buffer::layout(r,c,0);
                        auto w=shared.scratch[0].w[r][c][lane];
                        auto dot=dp.tiles[0][c/2].data[r+2*(c&1)];
                        int k=kb+pos.first,q=qb+pos.second;
                        auto x=reverse_coefficient(p,delta,w.u0,dot.x,bh,q,k);
                        auto y=reverse_coefficient(p,delta,w.u1,dot.y,bh,q+32,k);
                        reverse.data[r][c]={{x.x,y.x},{x.y,y.y}};
                    }
                }
            }
            reverse.reverse_roll();
            auto state=reverse.reduce_backward(right,{});
            right=state.first;
            auto x=state.second.init[0];
            int q=qb+8*g+8-l;
            int64_t off=(int64_t(bh)*(p.padded_n/16)+kb/16)*p.padded_n;
            if(q<qb+64) local[off+q]={x.first.u0,x.second.u0};
            if(q+32<qb+64) local[off+q+32]={x.first.u1,x.second.u1};
            if(lane==0) local[off+qb]={right.init[0].first.u0,right.init[0].second.u0};
        }
        __syncwarp(); // all readers finish before reusing query staging
    }
    #pragma unroll
    for(int c=0;c<DV/16;++c) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            int k=kb+(r%2)*8+l,j=c*16+(r/2)*8+g*2;
            if(k<p.n) {
                float2 x;
                if constexpr(SUMMARY) x=shared.saved[c][r][lane];
                else x=accumulated.tiles[0][c].data[r];
                int64_t off=(int64_t(bh)*p.n+k)*DV+j;
                dv[off]=x.x; dv[off+1]=x.y;
            }
        }
    }
}

template<int D,int DV,bool SUMMARY> void run_value(const Args& p,const void* dout,float* out,
        const float* delta,float2* local,cudaStream_t stream) {
    auto qm=permuted_map<D>(p.a,p.batch_heads*p.n);
    auto dm=permuted_map<DV>(dout,p.batch_heads*p.n);
    constexpr int sm=sizeof(RecomputeShared<D,DV,SUMMARY>);
    auto error=cudaFuncSetAttribute(value_backward<D,DV,SUMMARY>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    value_backward<D,DV,SUMMARY><<<dim3(SUMMARY?p.padded_n/16:(p.n+15)/16,p.batch_heads),32,sm,stream>>>(
        p,qm,dm,static_cast<const __nv_bfloat16*>(dout),out,delta,local);
}
template<int D,bool SUMMARY> void value_dim(const Args& p,int dv,const void* dout,float* out,
        const float* delta,float2* local,cudaStream_t stream) {
    switch(dv) {
        case 32: run_value<D,32,SUMMARY>(p,dout,out,delta,local,stream); break;
        case 64: run_value<D,64,SUMMARY>(p,dout,out,delta,local,stream); break;
        case 128: run_value<D,128,SUMMARY>(p,dout,out,delta,local,stream); break;
    }
}
template<bool SUMMARY> void value_dispatch(const Args& p,int d,int dv,const void* dout,float* out,
        const float* delta,float2* local,cudaStream_t stream) {
    switch(d) {
        case 32: value_dim<32,SUMMARY>(p,dv,dout,out,delta,local,stream); break;
        case 64: value_dim<64,SUMMARY>(p,dv,dout,out,delta,local,stream); break;
        case 128: value_dim<128,SUMMARY>(p,dv,dout,out,delta,local,stream); break;
    }
}
void launch_value_backward(const Args& p,int d,int dv,const void* dout,float* out,
        const float* delta,float2* local,cudaStream_t stream) {
    if(local) value_dispatch<true>(p,d,dv,dout,out,delta,local,stream);
    else value_dispatch<false>(p,d,dv,dout,out,delta,local,stream);
}

// Combine two independent16-key affine functions then pass32-key chunks.
// local[top] maps the output of local[bottom]; order is intentionally reversed.
__global__ void backward_passing(const float2* local,float2* summary,float* boundary,int np) {
    int cp=np/32,bh=blockIdx.y;
    int d=int(blockIdx.x*blockDim.x+threadIdx.x)-(cp-1)*32;
    if(d>=np) return;
    int64_t lo=int64_t(bh)*(np/16)*np,off=int64_t(bh)*cp*np;
    float state=0;
    for(int c=cp-1;c>=0;--c) {
        int q=d+c*32;
        if(q>=0 && q<np) {
            auto top=local[lo+int64_t(2*c)*np+q];
            float2 bottom{1,0};
            if(q+16<np) bottom=local[lo+int64_t(2*c+1)*np+q+16];
            float2 pair{top.x*bottom.x,fmaf(top.x,bottom.y,top.y)};
            int64_t dest=off+int64_t(c)*np+q;
            summary[dest]=pair;
            state=fmaf(pair.x,state,pair.y);
            boundary[dest]=state;
        }
    }
}
void launch_backward_passing(const float2* local,float2* summary,float* boundary,int np,int bh,cudaStream_t stream) {
    backward_passing<<<dim3((np+(np/32-1)*32+127)/128,bh),128,0,stream>>>(local,summary,boundary,np);
}
} // namespace dism_v2
