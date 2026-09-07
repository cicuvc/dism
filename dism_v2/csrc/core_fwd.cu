#include <cuda.h>
#include <kittens.cuh>
#include "core_api.h"
#include "log_affine.cuh"
#include "pipeline.cuh"
#include "row_rng.cuh"

namespace dism_v2 {
namespace kt=kittens;
constexpr int STAGES=2;
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
    union {
        kt::st_bf<16,D> q[8];
        Slot<D,DV,OUTPUT> slot[STAGES];
    };
    uint64_t ready[STAGES], free[STAGES];
    uint64_t mail_ready[4][STAGES], mail_free[4][STAGES];
    Buffer::HState::SharedStorage mail[4][STAGES];
};

__device__ __forceinline__ float score(const Args& p, float dot, int bh, int i, int j, bool hard) {
    if(i>=p.n || j>=p.n || j>i) return -INFINITY;
    float tau=p.tau[bh%p.heads];
    if(hard) return p.q_label[int64_t(bh)*p.n+i]==p.k_label[int64_t(bh)*p.n+j]?tau*LOG2E:-INFINITY;
    float l=p.lse[int64_t(bh)*p.n+(p.column_lse?j:i)];
    return (dot*p.scale-l+tau)*LOG2E;
}

template<int D,int DV,bool OUTPUT>
__global__ __launch_bounds__(384,1) void core(__grid_constant__ const Args p, __grid_constant__ const CUtensorMap km,
                     __grid_constant__ const CUtensorMap vm) {
    extern __shared__ __align__(128) unsigned char bytes[];
    auto& shared=*reinterpret_cast<Shared<D,DV,OUTPUT>*>(bytes);
    int warp=threadIdx.x/32, lane=threadIdx.x&31;
    int bh=blockIdx.y, base=blockIdx.x*128;
    int checkpoint=blockIdx.x*4+(warp&3);
    int qbase=base+(warp&3)*32+(warp/4)*16;
    if(threadIdx.x==0) {
        for(int s=0;s<STAGES;++s) {
            init_bar(&shared.ready[s],32); init_bar(&shared.free[s],256);
            for(int w=0;w<4;++w) { init_bar(&shared.mail_ready[w][s],32); init_bar(&shared.mail_free[w][s],32); }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    if(warp<8) for(int x=lane;x<16*D;x+=32) {
        int row=qbase+x/D;
        shared.q[warp][int2{x/D,x%D}]=row<p.n?
            static_cast<const __nv_bfloat16*>(p.a)[(int64_t(bh)*p.n+row)*D+x%D]:__float2bfloat16(0);
    }
    __syncthreads();
    kt::rt_bf<16,D> qreg;
    if(warp<8) kt::warp::load(qreg,shared.q[warp]);
    __syncthreads(); // q staging is now dead; ring may overwrite the union.
    if(warp>=8) {
        // Keep reallocation inside the long-lived role branches: an earlier
        // join makes the compiler conservatively use the smaller budget.
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if(warp==8) {
        for(int t=0;t*64<p.padded_n;++t) {
            int s=t%STAGES;
            if(t>=STAGES) wait(&shared.free[s],((t/STAGES)-1)&1);
            auto& slot=shared.slot[s];
            if(t*64+64<=p.n) {
                if(lane==0) {
                    expect(&shared.ready[s],sizeof(slot.k)+(OUTPUT?64*DV*2:0));
                    constexpr int S=D==32?32:64;
                    #pragma unroll
                    for(int k=0;k<D/S;++k) tma5(&km,slot.k.data+k*64*S,&shared.ready[s],bh*p.n+t*64,k);
                    if constexpr(OUTPUT) {
                        constexpr int V=DV==32?32:64;
                        #pragma unroll
                        for(int k=0;k<DV/V;++k) tma5(&vm,slot.v.data+k*64*V,&shared.ready[s],bh*p.n+t*64,k);
                    }
                } else arrive(&shared.ready[s]);
            } else {
                // The last logical tile cannot use the full-tile 5D map safely.
                for(int x=lane;x<64*D;x+=32) {
                    int j=t*64+logical_row(x/D);
                    slot.k[int2{x/D,x%D}]=j<p.n?static_cast<const __nv_bfloat16*>(p.b)[(int64_t(bh)*p.n+j)*D+x%D]:__float2bfloat16(0);
                }
                if constexpr(OUTPUT) for(int x=lane;x<64*DV;x+=32) {
                    int j=t*64+logical_row(x/DV);
                    slot.v[int2{x/DV,x%DV}]=j<p.n?static_cast<const __nv_bfloat16*>(p.v)[(int64_t(bh)*p.n+j)*DV+x%DV]:__float2bfloat16(0);
                }
                __syncwarp();
                // Every writer publishes its own stores to the ready barrier.
                arrive(&shared.ready[s]);
            }
        }
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
        // One lane per query row generates a decision, before the key loop.
        int decision=0;
        if(lane<16 && qbase+lane<p.n)
            decision=row_hard(p.seed,p.offset,uint64_t(bh)*p.n+qbase+lane,p.hard_prob);
        bool hard[2]={bool(__shfl_sync(0xffffffff,decision,lane/4)),
                      bool(__shfl_sync(0xffffffff,decision,8+lane/4))};
        Buffer::VState left;
        kt::rt_fl<16,DV> out{0.f};
        float maximum[2]{0,0}, denominator[2]{1,1};
        for(int t=0;t*64<p.padded_n;++t) {
            int s=t%STAGES, phase=(t/STAGES)&1;
            wait(&shared.ready[s],phase);
            Scalar scalar;
            {
                kt::rt_bf<64,D> kreg;
                kt::rt_fl<16,64> accum{0.f};
                kt::warp::load(kreg,shared.slot[s].k);
                kt::warp::wmma::mma_ABt(accum,qreg,kreg,accum);
                #pragma unroll
                for(int r=0;r<2;++r) {
                    #pragma unroll
                    for(int c=0;c<8;++c) {
                        auto x=accum.tiles[0][c/2].data[r+2*(c&1)];
                        auto a=Buffer::layout(r,c,0), b=Buffer::layout(r,c,1);
                        scalar.data[r][c].value={score(p,x.x,bh,qbase+a.first,t*64+a.second,hard[r]),score(p,x.y,bh,qbase+b.first,t*64+b.second,hard[r])};
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
                    if(i>=p.n || j>=p.n) { data.data[r][c].first.u0=0; data.data[r][c].second.u0=-INFINITY; }
                    if(i>=p.n || j+32>=p.n) { data.data[r][c].first.u1=0; data.data[r][c].second.u1=-INFINITY; }
                }
            }
            Buffer::HState top;
            if(warp>=4) {
                wait(&shared.mail_ready[warp-4][s],phase);
                top=Buffer::HState::load_shared(shared.mail[warp-4][s]);
                arrive(&shared.mail_free[warp-4][s]);
            } else if constexpr(OUTPUT) {
                if(checkpoint>0 && checkpoint<=p.checkpoints) {
                    int j=t*64+8*(lane&3)+6-lane/4;
                    int64_t off=(int64_t(bh)*p.checkpoints+checkpoint-1)*p.padded_n;
                    top.init[0].first={0,0};
                    top.init[0].second={j>=0?p.boundary[off+j]:-INFINITY,p.boundary[off+j+32]};
                }
            }
            Buffer::StatePair result;
            if constexpr(OUTPUT) result=data.inclusive_scan(left,top);
            else result=data.reduce_forward(left,top);
            left=result.first;
            if(warp<4) {
                if(t>=STAGES) wait(&shared.mail_free[warp][s],phase^1);
                result.second.store_shared(shared.mail[warp][s]);
                arrive(&shared.mail_ready[warp][s]);
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
                        if(qbase+pos.first>=p.n || t*64+pos.second>=p.n) x.u0=-INFINITY;
                        if(qbase+pos.first>=p.n || t*64+pos.second+32>=p.n) x.u1=-INFINITY;
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
                        dest[0]=__float2bfloat16(x.x*inverse[k%2]); dest[1]=__float2bfloat16(x.y*inverse[k%2]);
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
    __syncthreads(); // All consumers (including invalid rows) drain before CTA exit.
}

__global__ void passing(Args p) {
    int d=int(blockIdx.x*blockDim.x+threadIdx.x)-(p.checkpoints-1)*32;
    if(d>=p.padded_n) return;
    int bh=blockIdx.y; float x=-INFINITY;
    for(int s=0;s<p.checkpoints;++s) {
        int j=d+s*32;
        if(j>=0 && j<p.padded_n) {
            int64_t off=(int64_t(bh)*p.checkpoints+s)*p.padded_n+j;
            auto a=reinterpret_cast<const float2*>(p.summary)[off];
            x=logadd2(x+a.x,a.y); p.boundary[off]=x;
        }
    }
}

template<int D,int DV,bool OUTPUT> void launch(const Args& p,cudaStream_t stream) {
    auto km=permuted_map<D>(p.b,p.batch_heads*p.n);
    CUtensorMap vm{};
    if constexpr(OUTPUT) vm=permuted_map<DV>(p.v,p.batch_heads*p.n);
    constexpr int sm=sizeof(Shared<D,DV,OUTPUT>);
    auto err=cudaFuncSetAttribute(core<D,DV,OUTPUT>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm);
    if(err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    core<D,DV,OUTPUT><<<dim3((p.n+127)/128,p.batch_heads),384,sm,stream>>>(p,km,vm);
}
void launch_summary(const Args& p,int d,cudaStream_t stream) {
    switch(d) { case 32: launch<32,32,false>(p,stream);break; case 64: launch<64,32,false>(p,stream);break; case 128: launch<128,32,false>(p,stream);break; }
}
void launch_passing(const Args& p,cudaStream_t stream) {
    passing<<<dim3((p.padded_n+(p.checkpoints-1)*32+127)/128,p.batch_heads),128,0,stream>>>(p);
}
template<int D> void output_dim(const Args& p,int dv,cudaStream_t stream) {
    switch(dv) { case 32: launch<D,32,true>(p,stream);break; case 64: launch<D,64,true>(p,stream);break; case 128: launch<D,128,true>(p,stream);break; }
}
void launch_output(const Args& p,int d,int dv,cudaStream_t stream) {
    switch(d) { case 32: output_dim<32>(p,dv,stream);break; case 64: output_dim<64>(p,dv,stream);break; case 128: output_dim<128>(p,dv,stream);break; }
}
} // namespace dism_v2
