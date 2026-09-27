// Included inside dism_v2::embedding_bwd after the single-warp math helpers.
__device__ __forceinline__ void load_tma(const CUtensorMap* map,void* dst,uint64_t* bar,
                                        int feature,int row,int group) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5}], [%2];" :: "r"(smaddr(dst)),"l"(map),"r"(smaddr(bar)),
        "r"(feature),"r"(row),"r"(group):"memory");
}
template<int D,int T> CUtensorMap map(const void* src,int rows,int groups) {
    constexpr int S=D==32?32:64;
    cuuint64_t dims[]{D,cuuint64_t(rows),cuuint64_t(groups)},strides[]{D*2,cuuint64_t(rows)*D*2};
    cuuint32_t box[]{S,T,1},step[]{1,1,1};
    CUtensorMap result{};
    auto e=cuTensorMapEncodeTiled(&result,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,const_cast<void*>(src),
        dims,strides,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,
        D==32?CU_TENSOR_MAP_SWIZZLE_64B:CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(e!=CUDA_SUCCESS) throw std::runtime_error("embedding backward TMA descriptor failed");
    return result;
}
template<int D,int T> __device__ __forceinline__ void issue(
        const CUtensorMap* map,kt::st_bf<T,D>& dst,uint64_t* bar,int row,int group) {
    constexpr int S=D==32?32:64;
    #pragma unroll
    for(int f=0;f<D/S;++f) load_tma(map,dst.data+f*T*S,bar,f*S,row,group);
}
template<int D,int T> struct TokenWS {
    struct Init {kt::st_bf<16,D> x[8],u[4];};
    struct Slot {kt::st_bf<T,D> key,value;};
    union {Init init;Slot slot[2];};
    alignas(8) uint64_t ready[2],free[2];
};
template<int D,int T> __global__ __launch_bounds__(384,1) void token_ws(
        __grid_constant__ const Args a,__grid_constant__ const CUtensorMap km,
        __grid_constant__ const CUtensorMap vm) {
    extern __shared__ __align__(128) unsigned char mem[];
    auto& s=*reinterpret_cast<TokenWS<D,T>*>(mem);
    int warp=threadIdx.x/32,lane=threadIdx.x%32,bh=blockIdx.y,rb=blockIdx.x*64+warp%4*16;
    if(threadIdx.x==0) {
        for(int b=0;b<2;++b) {init_bar(&s.ready[b],32);init_bar(&s.free[b],256);}
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    if(warp<8) stage(s.init.x[warp],warp<4?a.x:a.y,int64_t(bh)*a.n+rb,a.n-rb,lane);
    if(warp<4) stage(s.init.u[warp],a.packed_u,int64_t(bh)*a.n+rb,a.n-rb,lane);
    __syncthreads();
    kt::rt_bf<16,D> x,u;
    if(warp<8) kt::warp::load(x,s.init.x[warp]);
    if(warp<4) kt::warp::load(u,s.init.u[warp]);
    __syncthreads();
    if(warp>=8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if(warp==8) for(int t=0;t*T<a.voc;++t) {
            int b=t%2,vb=t*T;
            if(t>=2) wait(&s.free[b],(t/2-1)&1);
            auto& slot=s.slot[b];
            if(vb+T<=a.voc) {
                if(lane==0) {
                    expect(&s.ready[b],2*T*D*2);
                    issue(&km,slot.key,&s.ready[b],vb,bh%a.heads);
                    issue(&vm,slot.value,&s.ready[b],vb,bh%a.heads);
                } else arrive(&s.ready[b]);
            } else {
                stage(slot.key,a.key,int64_t(bh%a.heads)*a.voc+vb,a.voc-vb,lane);
                stage(slot.value,a.value,int64_t(bh%a.heads)*a.voc+vb,a.voc-vb,lane);
                __syncwarp();arrive(&s.ready[b]);
            }
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
        float lse[2],corr[2];
        #pragma unroll
        for(int z=0;z<2;++z) {
            int row=rb+8*z+lane/4;int64_t i=int64_t(bh)*a.n+row;
            lse[z]=row<a.n?(warp<4?a.lx2[i]:a.ly2[i]):0.f;
            corr[z]=row<a.n?(warp<4?-a.delta[i]:a.lambda[i]):0.f;
        }
        kt::rt_fl<16,D> acc{0.f};
        for(int t=0;t*T<a.voc;++t) {
            int b=t%2;
            wait(&s.ready[b],(t/2)&1);
            if(warp<4) token_step<D,T,true>(x,u,acc,s.slot[b].key,s.slot[b].value,lse,corr,rb,t*T,a,lane);
            else token_step<D,T,false>(x,u,acc,s.slot[b].value,s.slot[b].value,lse,corr,rb,t*T,a,lane);
            __syncwarp();arrive(&s.free[b]);
        }
        store(acc,warp<4?a.dx:a.dy,int64_t(bh)*a.n+rb,a.n-rb,lane);
    }
    __syncthreads();
}
template<int D,int T> struct VocabWS {
    struct Init {kt::st_bf<16,D> key[4],value[4];};
    struct Slot {kt::st_bf<T,D> x,y,u;};
    union {Init init;Slot slot[2];};
    kt::st_bf<16,T> p[4][2];
    alignas(8) uint64_t ready[2],free[2],p_ready[4][2],p_free[4][2];
};
template<int D,int T,bool FULL> __device__ __forceinline__ void vocab_consume(
        const Args& a,VocabWS<D,T>& s,const kt::rt_bf<16,D>& key,const kt::rt_bf<16,D>& value,
        int pair,int lane,int h,int vb,int nt) {
        kt::rt_fl<16,D> acc{0.f};
        for(int t=0;t<a.batch*nt;++t) {
            int b=t%2,rb=t%nt*T,bh=t/nt*a.heads+h;
            wait(&s.ready[b],(t/2)&1);
            auto& slot=s.slot[b];
            if constexpr(FULL) {
                kt::rt_fl<16,T> p;
                dot(p,key,slot.x);probability(p,a.lx2,a,bh,rb,vb,lane);
                if(t>=2) wait(&s.p_free[pair][b],(t/2-1)&1);
                // Publish BF16 P before overwriting the FP32 tile with G.
                // Scalar stores use ordinary MMA logical coordinates.
                #pragma unroll
                for(int c=0;c<T/16;++c) {
                    #pragma unroll
                    for(int r=0;r<4;++r) {
                        int row=r%2*8+lane/4,col=c*16+r/2*8+2*(lane%4);
                        auto v=p.tiles[0][c].data[r];
                        s.p[pair][b][int2{row,col}]=__float2bfloat16_rn(v.x);
                        s.p[pair][b][int2{row,col+1}]=__float2bfloat16_rn(v.y);
                    }
                }
                __syncwarp();arrive(&s.p_ready[pair][b]);
                kt::rt_fl<16,T> dp;
                dot(dp,value,slot.u);
                #pragma unroll
                for(int c=0;c<T/16;++c) {
                    #pragma unroll
                    for(int r=0;r<4;++r) {
                        int col=rb+c*16+r/2*8+2*(lane%4);
                        auto& v=p.tiles[0][c].data[r];auto d=dp.tiles[0][c].data[r];
                        v.x*=d.x-(col<a.n?a.delta[int64_t(bh)*a.n+col]:0.f);
                        v.y*=d.y-(col+1<a.n?a.delta[int64_t(bh)*a.n+col+1]:0.f);
                    }
                }
                kt::rt_bf<16,T> g;
                kt::warp::copy(g,p);product(acc,g,slot.x,a.scale);
            } else {
                {
                    kt::rt_fl<16,T> p;
                    dot(p,value,slot.y);probability(p,a.ly2,a,bh,rb,vb,lane);
                    #pragma unroll
                    for(int c=0;c<T/16;++c) {
                        #pragma unroll
                        for(int r=0;r<4;++r) {
                            int col=rb+c*16+r/2*8+2*(lane%4);
                            auto& v=p.tiles[0][c].data[r];
                            v.x*=col<a.n?a.lambda[int64_t(bh)*a.n+col]:0.f;
                            v.y*=col+1<a.n?a.lambda[int64_t(bh)*a.n+col+1]:0.f;
                        }
                    }
                    kt::rt_bf<16,T> g;
                    kt::warp::copy(g,p);product(acc,g,slot.y,a.scale);
                }
                wait(&s.p_ready[pair][b],(t/2)&1);
                kt::rt_bf<16,T> p;
                kt::warp::load(p,s.p[pair][b]);
                __syncwarp();arrive(&s.p_free[pair][b]);
                product(acc,p,slot.u,1.f);
            }
            __syncwarp();arrive(&s.free[b]);
        }
        store(acc,FULL?a.dkey:a.dvalue,int64_t(h)*a.voc+vb,a.voc-vb,lane);
}
template<int D,int T> __global__ __launch_bounds__(384,1) void vocabulary_ws(
        __grid_constant__ const Args a,__grid_constant__ const CUtensorMap xm,
        __grid_constant__ const CUtensorMap ym,__grid_constant__ const CUtensorMap um) {
    extern __shared__ __align__(128) unsigned char mem[];
    auto& s=*reinterpret_cast<VocabWS<D,T>*>(mem);
    int warp=threadIdx.x/32,lane=threadIdx.x%32,h=blockIdx.y,pair=warp%4;
    int vb=blockIdx.x*64+pair*16,nt=(a.n+T-1)/T;
    if(threadIdx.x==0) {
        for(int b=0;b<2;++b) {
            init_bar(&s.ready[b],32);init_bar(&s.free[b],256);
            for(int w=0;w<4;++w) {init_bar(&s.p_ready[w][b],32);init_bar(&s.p_free[w][b],32);}
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    if(warp<4) {
        stage(s.init.key[pair],a.key,int64_t(h)*a.voc+vb,a.voc-vb,lane);
        stage(s.init.value[pair],a.value,int64_t(h)*a.voc+vb,a.voc-vb,lane);
    }
    __syncthreads();
    kt::rt_bf<16,D> key,value;
    if(warp<8) {kt::warp::load(key,s.init.key[pair]);kt::warp::load(value,s.init.value[pair]);}
    __syncthreads();
    if(warp>=8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if(warp==8) for(int t=0;t<a.batch*nt;++t) {
            int b=t%2,rb=t%nt*T,bh=t/nt*a.heads+h;
            if(t>=2) wait(&s.free[b],(t/2-1)&1);
            auto& slot=s.slot[b];
            if(rb+T<=a.n) {
                if(lane==0) {
                    expect(&s.ready[b],3*T*D*2);
                    issue(&xm,slot.x,&s.ready[b],rb,bh);
                    issue(&ym,slot.y,&s.ready[b],rb,bh);
                    issue(&um,slot.u,&s.ready[b],rb,bh);
                } else arrive(&s.ready[b]);
            } else {
                stage(slot.x,a.x,int64_t(bh)*a.n+rb,a.n-rb,lane);
                stage(slot.y,a.y,int64_t(bh)*a.n+rb,a.n-rb,lane);
                stage(slot.u,a.packed_u,int64_t(bh)*a.n+rb,a.n-rb,lane);
                __syncwarp();arrive(&s.ready[b]);
            }
        }
    } else {
        // Keep each budget inside its long-lived role branch: no low-budget join.
        if(warp<4) {
            asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
            vocab_consume<D,T,true>(a,s,key,value,pair,lane,h,vb,nt);
        } else {
            asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
            vocab_consume<D,T,false>(a,s,key,value,pair,lane,h,vb,nt);
        }
    }
    __syncthreads();
}
#include "embedding_bwd_symmetric.cuh"
template<int D,bool SYMMETRIC,int VT,bool SHARED> void dispatch_ws(Args a,cudaStream_t stream) {
    constexpr int T=D==128?32:64;
    // Sweep only the vocabulary scan; token-gradient tiling stays fixed.
    auto km=map<D,T>(a.key,a.voc,a.heads),vm=map<D,T>(a.value,a.voc,a.heads);
    auto xm=map<D,VT>(a.x,a.n,a.batch*a.heads),ym=map<D,VT>(a.y,a.n,a.batch*a.heads);
    auto um=map<D,VT>(a.packed_u,a.n,a.batch*a.heads);
    auto e=cudaFuncSetAttribute(token_ws<D,T>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(TokenWS<D,T>));
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    if constexpr(SYMMETRIC)
        e=cudaFuncSetAttribute(vocabulary_symmetric<D,VT,SHARED>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(VocabSymmetric<D,VT,SHARED>));
    else e=cudaFuncSetAttribute(vocabulary_ws<D,VT>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(VocabWS<D,VT>));
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    preprocess<D><<<(a.batch*a.heads*a.n+3)/4,128,0,stream>>>(a);
    token_ws<D,T><<<dim3((a.n+63)/64,a.batch*a.heads),384,sizeof(TokenWS<D,T>),stream>>>(a,km,vm);
    if constexpr(SYMMETRIC)
        vocabulary_symmetric<D,VT,SHARED><<<dim3((a.voc+127)/128,a.heads),384,sizeof(VocabSymmetric<D,VT,SHARED>),stream>>>(a,xm,ym,um);
    else vocabulary_ws<D,VT><<<dim3((a.voc+63)/64,a.heads),384,sizeof(VocabWS<D,VT>),stream>>>(a,xm,ym,um);
}
template<int D,bool SYMMETRIC,bool SHARED> void select_vocab_step(Args a,int vt,cudaStream_t stream) {
    if(vt==16) dispatch_ws<D,SYMMETRIC,16,SHARED>(a,stream);
    else if(vt==32) dispatch_ws<D,SYMMETRIC,32,SHARED>(a,stream);
    else {
        if constexpr(D==128 && !SYMMETRIC)
            throw std::runtime_error("paired D128 step64 exceeds shared capacity");
        else dispatch_ws<D,SYMMETRIC,64,SHARED>(a,stream);
    }
}
template<int D> void select_ws(Args a,bool symmetric,int vt,bool shared,cudaStream_t stream) {
    if(symmetric) {
        if constexpr(D==64) {
            if(shared) {select_vocab_step<D,true,true>(a,vt,stream);return;}
        }
        select_vocab_step<D,true,false>(a,vt,stream);
    } else select_vocab_step<D,false,false>(a,vt,stream);
}
