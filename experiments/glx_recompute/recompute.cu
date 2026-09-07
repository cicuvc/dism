// Diagnostic only: materializes W for tests, never part of production forward.
#include <cuda.h>
#include <kittens.cuh>
#include "../../dism_v2/csrc/core_api.h"
#include "../../dism_v2/csrc/log_affine.cuh"
#include "../../dism_v2/csrc/pipeline.cuh"
#include "../../dism_v2/csrc/row_rng.cuh"

namespace dism_v2 {
namespace kt=kittens;
template<int D> struct RecomputeShared {
    kt::st_bf<16,D> key;
    kt::st_bf<64,D> query;
    uint64_t ready;
};

__device__ __forceinline__ float transposed_score(const Args& p,float dot,int bh,int q,int k,bool hard) {
    if(q>=p.n || k>=p.n || k>q) return -INFINITY;
    float tau=p.tau[bh%p.heads];
    if(hard) return p.q_label[int64_t(bh)*p.n+q]==p.k_label[int64_t(bh)*p.n+k]?tau*LOG2E:-INFINITY;
    return (dot*p.scale-p.lse[int64_t(bh)*p.n+(p.column_lse?k:q)]+tau)*LOG2E;
}

template<int D>
__global__ void recompute_probe(__grid_constant__ const Args p,
        __grid_constant__ const CUtensorMap qm,float* diagnostic) {
    __shared__ RecomputeShared<D> shared;
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
    __syncwarp();
    kt::rt_bf<16,D> keys;
    kt::warp::load(keys,shared.key);
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
                expect(&shared.ready,sizeof(shared.query));
                constexpr int S=D==32?32:64;
                #pragma unroll
                for(int s=0;s<D/S;++s) tma5(&qm,shared.query.data+s*64*S,&shared.ready,bh*p.n+qb,s);
            }
            wait(&shared.ready,phase); phase^=1;
        } else {
            for(int x=lane;x<64*D;x+=32) {
                int q=qb+logical_row(x/D);
                shared.query[int2{x/D,x%D}]=q<p.n?
                    static_cast<const __nv_bfloat16*>(p.a)[(int64_t(bh)*p.n+q)*D+x%D]:__float2bfloat16(0);
            }
            __syncwarp();
        }
        Scalar scalar;
        {
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
                if(k>=p.n || q>=p.n) { data.data[r][c].first.u0=0; data.data[r][c].second.u0=-INFINITY; }
                if(k>=p.n || q+32>=p.n) { data.data[r][c].first.u1=0; data.data[r][c].second.u1=-INFINITY; }
            }
        }
        Buffer::HState top;
        if(kb>0) {
            int q=qb+8*g+6-l;
            int64_t off=(int64_t(bh)*(p.padded_n/16)+kb/16-1)*p.padded_n;
            top.init[0].first={0,0};
            top.init[0].second={q>=0?p.vertical[off+q]:-INFINITY,p.vertical[off+q+32]};
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
        #pragma unroll
        for(int r=0;r<2;++r) {
            #pragma unroll
            for(int c=0;c<8;++c) {
                auto pos=Buffer::layout(r,c,0); auto x=scalar.data[r][c].value;
                int64_t off=(int64_t(bh)*p.padded_n+qb+pos.second)*p.padded_n+kb+pos.first;
                diagnostic[off]=x.u0; diagnostic[off+32*p.padded_n]=x.u1;
            }
        }
        __syncwarp(); // all readers finish before reusing query staging
    }
}

template<int D> void run_recompute(const Args& p,float* out,cudaStream_t stream) {
    auto qm=permuted_map<D>(p.a,p.batch_heads*p.n);
    recompute_probe<D><<<dim3(p.padded_n/16,p.batch_heads),32,0,stream>>>(p,qm,out);
}
void launch_recompute_probe(const Args& p,int d,float* out,cudaStream_t stream) {
    switch(d) {
        case 32: run_recompute<32>(p,out,stream); break;
        case 64: run_recompute<64>(p,out,stream); break;
        case 128: run_recompute<128>(p,out,stream); break;
    }
}
} // namespace dism_v2
