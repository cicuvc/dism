// Experimental12-warp B1. Compile/codegen gate precedes runtime validation.
#include <cuda.h>
#include <kittens.cuh>
#include "core_api.h"
#include "log_affine.cuh"
#include "pipeline.cuh"
#include "row_rng.cuh"
namespace dism_v2 {
namespace ws {
namespace kt=kittens;
using Reverse=glx::MMABuffer<16,64,glx::BinaryElement,glx::AffineComposeOp,glx::F32x2>;
template<int D,int DV> struct Slot { kt::st_bf<64,D> query; kt::st_bf<64,DV> dout; };
template<int D,int DV> struct Shared {
    union {
        struct { kt::st_bf<16,D> key[8]; kt::st_bf<16,DV> value[8]; } initial;
        Slot<D,DV> slot[2];
    };
    uint64_t ready[2],free[2],mail_ready[4][2],mail_free[4][2];
    Reverse::HState::SharedStorage mail[4][2];
};
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


template<int D,int DV>
__global__ __launch_bounds__(384,1) void value_backward(
        __grid_constant__ const Args p,__grid_constant__ const CUtensorMap qm,
        __grid_constant__ const CUtensorMap dm,const __nv_bfloat16* dout,
        const float* delta,float* dv,float2* summary) {
    extern __shared__ __align__(128) unsigned char bytes[];
    auto& shared=*reinterpret_cast<Shared<D,DV>*>(bytes);
    int warp=threadIdx.x/32,lane=threadIdx.x&31,g=lane&3,l=lane/4;
    int bh=blockIdx.y,chunk=blockIdx.x*4+(warp&3);
    int kb=chunk*32+(warp/4)*16;
    if(threadIdx.x==0) {
        for(int s=0;s<2;++s) {
            init_bar(&shared.ready[s],32); init_bar(&shared.free[s],256);
            for(int w=0;w<4;++w) {
                init_bar(&shared.mail_ready[w][s],32);
                init_bar(&shared.mail_free[w][s],32);
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
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
    if(warp>=8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if(warp==8) for(int t=0;t*64<p.padded_n;++t) {
            int s=t%2,qb=p.padded_n-64-t*64;
            if(t>=2) wait(&shared.free[s],((t/2)-1)&1);
            auto& slot=shared.slot[s];
            if(qb+64<=p.n) {
                if(lane==0) {
                    expect(&shared.ready[s],sizeof(slot.query)+sizeof(slot.dout));
                    constexpr int Q=D==32?32:64,V=DV==32?32:64;
                    #pragma unroll
                    for(int c=0;c<D/Q;++c) tma5(&qm,slot.query.data+c*64*Q,&shared.ready[s],bh*p.n+qb,c);
                    #pragma unroll
                    for(int c=0;c<DV/V;++c) tma5(&dm,slot.dout.data+c*64*V,&shared.ready[s],bh*p.n+qb,c);
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
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
        kt::rt_fl<16,DV> accumulated{0.f};
        Reverse::VState right;
        for(int t=0;t*64<p.padded_n;++t) {
            int s=t%2,phase=(t/2)&1,qb=p.padded_n-64-t*64;
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
            wait(&shared.ready[s],phase);
        Scalar scalar;
        {
            // Keys remain in registers across query tiles.
            kt::rt_bf<64,D> queries;
            kt::rt_fl<16,64> dot{0.f};
            kt::warp::load(queries,shared.slot[s].query);
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

            { // C then D: independent dV before constructing reverse pairs.
        kt::rt_bf<16,64> weights;
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
                weights.tiles[0][c/2].data[r+2*(c&1)]=high;
            }
        }
        kt::rt_bf<64,DV,kt::ducks::rt_layout::col> derivatives;
        kt::warp::load(derivatives,shared.slot[s].dout);
        kt::warp::wmma::mma_AB(accumulated,weights,derivatives,accumulated);
        // Single BF16 P MMA: accepted precision tradeoff for the WS path.
        // The single-warp baseline retains its high+residual correction.

            }
            Reverse reverse;
            {
                kt::rt_fl<16,64> dp{0.f};
                {
                    kt::rt_bf<64,DV> derivatives;
                    kt::warp::load(derivatives,shared.slot[s].dout);
                    kt::warp::wmma::mma_ABt(dp,values,derivatives,dp);
                }
                #pragma unroll
                for(int r=0;r<2;++r) {
                    #pragma unroll
                    for(int c=0;c<8;++c) {
                        auto pos=Buffer::layout(r,c,0);
                        auto w=scalar.data[r][c].value;
                        auto dot=dp.tiles[0][c/2].data[r+2*(c&1)];
                        int k=kb+pos.first,q=qb+pos.second;
                        auto x=reverse_coefficient(p,delta,w.u0,dot.x,bh,q,k);
                        auto y=reverse_coefficient(p,delta,w.u1,dot.y,bh,q+32,k);
                        reverse.data[r][c]={{x.x,y.x},{x.y,y.y}};
                    }
                }
            }
            // Both A and dO have been consumed. Producer can reuse this slot
            // independently of the subsequent reverse mailbox dependency.
            arrive(&shared.free[s]);
            reverse.reverse_roll();
            Reverse::HState bottom;
            if(warp<4) {
                wait(&shared.mail_ready[warp][s],phase);
                bottom=Reverse::HState::load_shared(shared.mail[warp][s]);
                arrive(&shared.mail_free[warp][s]);
            }
            auto state=reverse.reduce_backward(right,bottom);
            right=state.first;
            if(warp>=4) {
                if(t>=2) wait(&shared.mail_free[warp-4][s],phase^1);
                state.second.store_shared(shared.mail[warp-4][s]);
                arrive(&shared.mail_ready[warp-4][s]);
            } else if(chunk<p.padded_n/32) {
                auto x=state.second.init[0];
                int q=qb+8*g+8-l;
                int64_t off=(int64_t(bh)*(p.padded_n/32)+chunk)*p.padded_n;
                if(q<qb+64) summary[off+q]={x.first.u0,x.second.u0};
                if(q+32<qb+64) summary[off+q+32]={x.first.u1,x.second.u1};
                if(lane==0) summary[off+qb]={right.init[0].first.u0,right.init[0].second.u0};
            }
        }
        #pragma unroll
        for(int c=0;c<DV/16;++c) {
            #pragma unroll
            for(int r=0;r<4;++r) {
                int k=kb+(r%2)*8+l,j=c*16+(r/2)*8+g*2;
                if(k<p.n) {
                    auto x=accumulated.tiles[0][c].data[r];
                    int64_t off=(int64_t(bh)*p.n+k)*DV+j;
                    dv[off]=x.x; dv[off+1]=x.y;
                }
            }
        }
    }
    __syncthreads(); // Invalid compute warps also drain the full protocol.
}

__global__ void passing32(const float2* summary,float* boundary,int np) {
    int cp=np/32,bh=blockIdx.y;
    int d=int(blockIdx.x*blockDim.x+threadIdx.x)-(cp-1)*32;
    if(d>=np) return;
    int64_t off=int64_t(bh)*cp*np;
    float x=0;
    for(int c=cp-1;c>=0;--c) {
        int q=d+c*32;
        if(q>=0 && q<np) {
            int64_t i=off+int64_t(c)*np+q;
            auto pair=summary[i]; x=fmaf(pair.x,x,pair.y); boundary[i]=x;
        }
    }
}
template<int D,int DV> void launch(const Args& p,const void* dout,const float* delta,
        float* dv,float2* summary,cudaStream_t stream) {
    auto qm=permuted_map<D>(p.a,p.batch_heads*p.n);
    auto dm=permuted_map<DV>(dout,p.batch_heads*p.n);
    constexpr int sm=sizeof(Shared<D,DV>);
    auto err=cudaFuncSetAttribute(value_backward<D,DV>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm);
    if(err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    value_backward<D,DV><<<dim3((p.padded_n+127)/128,p.batch_heads),384,sm,stream>>>(
        p,qm,dm,static_cast<const __nv_bfloat16*>(dout),delta,dv,summary);
}
template<int D> void dim(const Args& p,int dv,const void* dout,const float* delta,
        float* out,float2* summary,cudaStream_t stream) {
    switch(dv) {
        case 32: launch<D,32>(p,dout,delta,out,summary,stream); break;
        case 64: launch<D,64>(p,dout,delta,out,summary,stream); break;
        case 128: launch<D,128>(p,dout,delta,out,summary,stream); break;
    }
}
} // namespace ws
void launch_value_backward_ws(const Args& p,int d,int dv,const void* dout,const float* delta,
        float* out,float2* summary,float* boundary,cudaStream_t stream) {
    switch(d) {
        case 32: ws::dim<32>(p,dv,dout,delta,out,summary,stream); break;
        case 64: ws::dim<64>(p,dv,dout,delta,out,summary,stream); break;
        case 128: ws::dim<128>(p,dv,dout,delta,out,summary,stream); break;
    }
    ws::passing32<<<dim3((p.padded_n+(p.padded_n/32-1)*32+127)/128,p.batch_heads),128,0,stream>>>(
        summary,boundary,p.padded_n);
}
} // namespace dism_v2
