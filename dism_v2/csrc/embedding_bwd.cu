#include <cuda.h>
#include <kittens.cuh>
#include <stdexcept>
#include "embedding_bwd_api.h"
#include "pipeline.cuh"
namespace kt=kittens;
namespace dism_v2::embedding_bwd {
__device__ __forceinline__ const __nv_bfloat16* bf(const void* p) {
    return static_cast<const __nv_bfloat16*>(p);
}
template<int D> __global__ void preprocess(__grid_constant__ const Args a) {
    int lane=threadIdx.x%32,row=blockIdx.x*4+threadIdx.x/32;
    if(row>=a.batch*a.heads*a.n) return;
    float sum=0;
    #pragma unroll
    for(int c=lane;c<D;c+=32) {
        int64_t i=int64_t(row)*D+c;
        float u=a.u[i];
        sum=fmaf(u,__bfloat162float(bf(a.out)[i]),sum);
        static_cast<__nv_bfloat16*>(a.packed_u)[i]=__float2bfloat16_rn(u);
    }
    #pragma unroll
    for(int s=16;s;s/=2) sum+=__shfl_down_sync(0xffffffff,sum,s);
    if(lane==0) {
        a.delta[row]=sum;
        // Public/saved LSE remains natural-log; temporary row metadata is log2.
        a.lx2[row]=a.lx[row]*1.4426950408889634f;
        a.ly2[row]=a.ly[row]*1.4426950408889634f;
    }
}
template<int R,int D> __device__ __forceinline__ void stage(
        kt::st_bf<R,D>& dst,const void* src,int64_t base,int valid,int lane) {
    for(int i=lane;i<R*D;i+=32)
        dst[int2{i/D,i%D}]=i/D<valid?bf(src)[base*D+i]:__float2bfloat16_rn(0.f);
}
template<int D,int T> __device__ __forceinline__ void dot(
        kt::rt_fl<16,T>& dst,const kt::rt_bf<16,D>& x,kt::st_bf<T,D>& y) {
    #pragma unroll
    for(int c=0;c<T/16;++c) {
        #pragma unroll
        for(int r=0;r<4;++r) dst.tiles[0][c].data[r]=float2{0.f,0.f};
    }
    #pragma unroll
    for(int f=0;f<D/32;++f) {
        kt::rt_bf<16,32> xp;
        #pragma unroll
        for(int c=0;c<2;++c) xp.tiles[0][c]=x.tiles[0][f*2+c];
        kt::rt_bf<T,32> yp;
        auto view=y.template subtile<T,32>({0,f});
        kt::warp::load(yp,view);
        kt::warp::wmma::mma_ABt(dst,xp,yp,dst);
    }
}
template<int D,int T> __device__ __forceinline__ void product(
        kt::rt_fl<16,D>& acc,const kt::rt_bf<16,T>& weights,kt::st_bf<T,D>& value,float scale) {
    #pragma unroll
    for(int f=0;f<D/32;++f) {
        kt::rt_bf<T,32,kt::ducks::rt_layout::col> vp;
        auto view=value.template subtile<T,32>({0,f});
        kt::warp::load(vp,view);
        kt::rt_fl<16,32> part{0.f};
        kt::warp::wmma::mma_AB(part,weights,vp,part);
        #pragma unroll
        for(int c=0;c<2;++c) {
            #pragma unroll
            for(int r=0;r<4;++r) {
                auto p=part.tiles[0][c].data[r];
                auto& a=acc.tiles[0][f*2+c].data[r];
                a.x=fmaf(scale,p.x,a.x); a.y=fmaf(scale,p.y,a.y);
            }
        }
    }
}
template<int D> __device__ __forceinline__ void store(
        const kt::rt_fl<16,D>& acc,float* dst,int64_t base,int valid,int lane) {
    #pragma unroll
    for(int c=0;c<D/16;++c) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            int row=r%2*8+lane/4,col=c*16+r/2*8+2*(lane%4);
            auto a=acc.tiles[0][c].data[r];
            if(row<valid) {dst[(base+row)*D+col]=a.x;dst[(base+row)*D+col+1]=a.y;}
        }
    }
}
template<int D,int T,bool FULL> __device__ __forceinline__ void token_step(
        const kt::rt_bf<16,D>& x,const kt::rt_bf<16,D>& u,kt::rt_fl<16,D>& acc,
        kt::st_bf<T,D>& key_shared,kt::st_bf<T,D>& value_shared,
        const float (&lse)[2],const float (&corr)[2],int rb,int vb,const Args& a,int lane) {
        kt::rt_fl<16,T> p;
        dot(p,x,key_shared);
        #pragma unroll
        for(int c=0;c<T/16;++c) {
            #pragma unroll
            for(int r=0;r<4;++r) {
                int col=vb+c*16+r/2*8+2*(lane%4),row=rb+r%2*8+lane/4;
                auto& v=p.tiles[0][c].data[r];
                v.x=row<a.n && col<a.voc?exp2f(fmaf(v.x,a.scale2,-lse[r%2])):0.f;
                v.y=row<a.n && col+1<a.voc?exp2f(fmaf(v.y,a.scale2,-lse[r%2])):0.f;
            }
        }
        if constexpr(FULL) {
            kt::rt_fl<16,T> dp;
            dot(dp,u,value_shared);
            #pragma unroll
            for(int c=0;c<T/16;++c) {
                #pragma unroll
                for(int r=0;r<4;++r) {
                    auto& v=p.tiles[0][c].data[r]; auto d=dp.tiles[0][c].data[r];
                    v.x*=d.x+corr[r%2]; v.y*=d.y+corr[r%2];
                }
            }
        } else {
            #pragma unroll
            for(int c=0;c<T/16;++c) {
                #pragma unroll
                for(int r=0;r<4;++r) {
                    p.tiles[0][c].data[r].x*=corr[r%2];
                    p.tiles[0][c].data[r].y*=corr[r%2];
                }
            }
        }
        kt::rt_bf<16,T> g;
        kt::warp::copy(g,p);
        product(acc,g,key_shared,a.scale);

}
template<int D,int T> struct TokenScratch {
    kt::st_bf<16,D> x,u;
    kt::st_bf<T,D> key,value;
};
template<int D,int T,bool FULL> __global__ void token(__grid_constant__ const Args a) {
    extern __shared__ __align__(128) unsigned char mem[];
    auto& s=*reinterpret_cast<TokenScratch<D,T>*>(mem);
    int lane=threadIdx.x,bh=blockIdx.y,rb=blockIdx.x*16,h=bh%a.heads;
    stage(s.x,FULL?a.x:a.y,int64_t(bh)*a.n+rb,a.n-rb,lane);
    if constexpr(FULL) stage(s.u,a.packed_u,int64_t(bh)*a.n+rb,a.n-rb,lane);
    __syncwarp();
    kt::rt_bf<16,D> x,u;
    kt::warp::load(x,s.x);
    if constexpr(FULL) kt::warp::load(u,s.u);
    float lse[2],corr[2];
    #pragma unroll
    for(int z=0;z<2;++z) {
        int row=rb+8*z+lane/4; int64_t i=int64_t(bh)*a.n+row;
        lse[z]=row<a.n?(FULL?a.lx2[i]:a.ly2[i]):0.f;
        corr[z]=row<a.n?(FULL?-a.delta[i]:a.lambda[i]):0.f;
    }
    kt::rt_fl<16,D> acc{0.f};
    for(int vb=0;vb<a.voc;vb+=T) {
        stage(s.key,FULL?a.key:a.value,int64_t(h)*a.voc+vb,a.voc-vb,lane);
        if constexpr(FULL) stage(s.value,a.value,int64_t(h)*a.voc+vb,a.voc-vb,lane);
        __syncwarp();
        token_step<D,T,FULL>(x,u,acc,s.key,s.value,lse,corr,rb,vb,a,lane);
        __syncwarp();
    }
    store(acc,FULL?a.dx:a.dy,int64_t(bh)*a.n+rb,a.n-rb,lane);
}
template<int D,int T> struct VocabScratch {
    kt::st_bf<16,D> key,value;
    kt::st_bf<T,D> x,y,u;
};
template<int T> __device__ __forceinline__ void probability(
        kt::rt_fl<16,T>& p,const float* lse,const Args& a,int bh,int rb,int vb,int lane) {
    #pragma unroll
    for(int c=0;c<T/16;++c) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            int col=rb+c*16+r/2*8+2*(lane%4),row=vb+r%2*8+lane/4;
            auto& v=p.tiles[0][c].data[r];
            float lx=col<a.n?lse[int64_t(bh)*a.n+col]:0.f;
            float ly=col+1<a.n?lse[int64_t(bh)*a.n+col+1]:0.f;
            v.x=col<a.n && row<a.voc?exp2f(fmaf(v.x,a.scale2,-lx)):0.f;
            v.y=col+1<a.n && row<a.voc?exp2f(fmaf(v.y,a.scale2,-ly)):0.f;
        }
    }
}
template<int D,int T,bool FULL> __global__ void vocabulary(__grid_constant__ const Args a) {
    extern __shared__ __align__(128) unsigned char mem[];
    auto& s=*reinterpret_cast<VocabScratch<D,T>*>(mem);
    int lane=threadIdx.x,h=blockIdx.y,vb=blockIdx.x*16;
    stage(s.key,a.key,int64_t(h)*a.voc+vb,a.voc-vb,lane);
    stage(s.value,a.value,int64_t(h)*a.voc+vb,a.voc-vb,lane);
    __syncwarp();
    kt::rt_bf<16,D> key,value;
    kt::warp::load(key,s.key); kt::warp::load(value,s.value);
    kt::rt_fl<16,D> acc{0.f};
    for(int batch=0;batch<a.batch;++batch) for(int rb=0;rb<a.n;rb+=T) {
        int bh=batch*a.heads+h;
        stage(s.x,a.x,int64_t(bh)*a.n+rb,a.n-rb,lane);
        stage(s.u,a.packed_u,int64_t(bh)*a.n+rb,a.n-rb,lane);
        if constexpr(!FULL) stage(s.y,a.y,int64_t(bh)*a.n+rb,a.n-rb,lane);
        __syncwarp();
        if constexpr(!FULL) {
            kt::rt_fl<16,T> g;
            dot(g,value,s.y);
            probability(g,a.ly2,a,bh,rb,vb,lane);
            #pragma unroll
            for(int c=0;c<T/16;++c) {
                #pragma unroll
                for(int r=0;r<4;++r) {
                    int col=rb+c*16+r/2*8+2*(lane%4);
                    auto& v=g.tiles[0][c].data[r];
                    v.x*=col<a.n?a.lambda[int64_t(bh)*a.n+col]:0.f;
                    v.y*=col+1<a.n?a.lambda[int64_t(bh)*a.n+col+1]:0.f;
                }
            }
            kt::rt_bf<16,T> gb;
            kt::warp::copy(gb,g);
            product(acc,gb,s.y,a.scale);
        }
        kt::rt_fl<16,T> p;
        dot(p,key,s.x);
        probability(p,a.lx2,a,bh,rb,vb,lane);
        if constexpr(FULL) {
            kt::rt_fl<16,T> dp;
            dot(dp,value,s.u);
            #pragma unroll
            for(int c=0;c<T/16;++c) {
                #pragma unroll
                for(int r=0;r<4;++r) {
                    int col=rb+c*16+r/2*8+2*(lane%4);
                    auto& v=p.tiles[0][c].data[r]; auto d=dp.tiles[0][c].data[r];
                    v.x*=d.x-(col<a.n?a.delta[int64_t(bh)*a.n+col]:0.f);
                    v.y*=d.y-(col+1<a.n?a.delta[int64_t(bh)*a.n+col+1]:0.f);
                }
            }
        }
        kt::rt_bf<16,T> gb;
        kt::warp::copy(gb,p);
        product(acc,gb,FULL?s.x:s.u,FULL?a.scale:1.f);
        __syncwarp();
    }
    store(acc,FULL?a.dkey:a.dvalue,int64_t(h)*a.voc+vb,a.voc-vb,lane);
}
#include "embedding_bwd_ws.cuh"
template<int D> void dispatch(Args a,cudaStream_t stream) {
    constexpr int T=D==128?32:64;
    auto set=[&](auto fn,int bytes) {
        auto e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
        if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    };
    preprocess<D><<<(a.batch*a.heads*a.n+3)/4,128,0,stream>>>(a);
    set(token<D,T,true>,sizeof(TokenScratch<D,T>));
    set(token<D,T,false>,sizeof(TokenScratch<D,T>));
    token<D,T,true><<<dim3((a.n+15)/16,a.batch*a.heads),32,sizeof(TokenScratch<D,T>),stream>>>(a);
    token<D,T,false><<<dim3((a.n+15)/16,a.batch*a.heads),32,sizeof(TokenScratch<D,T>),stream>>>(a);
    set(vocabulary<D,T,true>,sizeof(VocabScratch<D,T>));
    set(vocabulary<D,T,false>,sizeof(VocabScratch<D,T>));
    vocabulary<D,T,true><<<dim3((a.voc+15)/16,a.heads),32,sizeof(VocabScratch<D,T>),stream>>>(a);
    vocabulary<D,T,false><<<dim3((a.voc+15)/16,a.heads),32,sizeof(VocabScratch<D,T>),stream>>>(a);
}
void launch(Args a,int d,bool ws,cudaStream_t stream) {
    if(ws) {
        if(d==32) dispatch_ws<32>(a,stream);
        else if(d==64) dispatch_ws<64>(a,stream);
        else dispatch_ws<128>(a,stream);
    } else if(d==32) dispatch<32>(a,stream);
    else if(d==64) dispatch<64>(a,stream);
    else dispatch<128>(a,stream);
}
}
