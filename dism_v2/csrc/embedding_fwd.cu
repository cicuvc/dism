#include <cuda.h>
#include <kittens.cuh>
#include <stdexcept>
#include <climits>
#include "pipeline.cuh"
namespace kt=kittens;
namespace dism_v2::embedding {
__device__ __forceinline__ float reciprocal(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


// The same physical shared tile supports row-layout score operands and
// column-layout PV operands. No GLX permutation or diagonal roll is involved.
template<int D> struct Scratch {
    kt::st_bf<16,D> token;
    kt::st_bf<64,D> key, value;
};

template<int D,int BV> __device__ __forceinline__ void step(
        const kt::rt_bf<16,D>& x, kt::rt_fl<16,D>& acc,
        float (&m)[2], float (&l)[2], int (&arg)[2],
        kt::st_bf<BV,D>& key_shared, kt::st_bf<BV,D>& value_shared,
        int vb, int vocab, float scale, int lane, int valid_rows) {
        kt::rt_fl<16,BV> score{0.f};
        #pragma unroll
        for(int f=0;f<D/32;++f) {
            kt::rt_bf<16,32> xp;
            #pragma unroll
            for(int c=0;c<2;++c) xp.tiles[0][c]=x.tiles[0][2*f+c];
            kt::rt_bf<BV,32> kp;
            auto view=key_shared.template subtile<BV,32>({0,f});
            kt::warp::load(kp,view);
            kt::warp::wmma::mma_ABt(score,xp,kp,score);
        }
        float mx[2]={-INFINITY,-INFINITY};
        int ix[2]={INT_MAX,INT_MAX};
        #pragma unroll
        for(int c=0;c<BV/16;++c) {
            #pragma unroll
            for(int r=0;r<4;++r) {
                auto& a=score.tiles[0][c].data[r];
                int j=vb+c*16+(r/2)*8+2*(lane%4), z=r%2;
                a.x=j<vocab?a.x*(scale*1.4426950408889634f):-INFINITY;
                a.y=j+1<vocab?a.y*(scale*1.4426950408889634f):-INFINITY;
                if(j<vocab && (a.x>mx[z] || (a.x==mx[z] && j<ix[z]))) { mx[z]=a.x; ix[z]=j; }
                if(j+1<vocab && (a.y>mx[z] || (a.y==mx[z] && j+1<ix[z]))) { mx[z]=a.y; ix[z]=j+1; }
            }
        }
        float alpha[2];
        #pragma unroll
        for(int z=0;z<2;++z) {
            #pragma unroll
            for(int mask=1;mask<=2;mask*=2) {
                float other=__shfl_xor_sync(0xffffffff,mx[z],mask);
                int oi=__shfl_xor_sync(0xffffffff,ix[z],mask);
                if(other>mx[z] || (other==mx[z] && oi<ix[z])) {mx[z]=other;ix[z]=oi;}
            }
            float next=fmaxf(m[z],mx[z]);
            alpha[z]=exp2f(m[z]-next);
            if(mx[z]>m[z] || (mx[z]==m[z] && ix[z]<arg[z])) arg[z]=ix[z];
            m[z]=next;
        }
        float sum[2]={0.f,0.f};
        #pragma unroll
        for(int c=0;c<BV/16;++c) {
            #pragma unroll
            for(int r=0;r<4;++r) {
                auto& a=score.tiles[0][c].data[r];
                bool valid=(r%2)*8+lane/4<valid_rows;
                a.x=valid?exp2f(a.x-m[r%2]):0.f;
                a.y=valid?exp2f(a.y-m[r%2]):0.f;
                sum[r%2]+=a.x+a.y;
            }
        }
        #pragma unroll
        for(int z=0;z<2;++z) {
            sum[z]+=__shfl_xor_sync(0xffffffff,sum[z],1);
            sum[z]+=__shfl_xor_sync(0xffffffff,sum[z],2);
            l[z]=alpha[z]*l[z]+sum[z];
        }
        kt::rt_bf<16,BV> p;
        kt::warp::copy(p,score);
        #pragma unroll
        for(int f=0;f<D/32;++f) {
            kt::rt_fl<16,32> ap;
            #pragma unroll
            for(int c=0;c<2;++c) {
                ap.tiles[0][c]=acc.tiles[0][f*2+c];
                #pragma unroll
                for(int r=0;r<4;++r) {
                    ap.tiles[0][c].data[r].x*=alpha[r%2];
                    ap.tiles[0][c].data[r].y*=alpha[r%2];
                }
            }
            kt::rt_bf<BV,32,kt::ducks::rt_layout::col> vp;
            auto view=value_shared.template subtile<BV,32>({0,f});
            kt::warp::load(vp,view);
            kt::warp::wmma::mma_AB(ap,p,vp,ap);
            #pragma unroll
            for(int c=0;c<2;++c) acc.tiles[0][f*2+c]=ap.tiles[0][c];
        }

}
template<int D> __device__ __forceinline__ void store_result(
        kt::rt_fl<16,D>& acc, const float (&m)[2], const float (&l)[2], const int (&arg)[2],
        __nv_bfloat16* output, float* lse, float* top, int* index,
        int row0, int bh, int n, int lane) {
    #pragma unroll
    for(int c=0;c<D/16;++c) {
        #pragma unroll
        for(int r=0;r<4;++r) {
            int row=row0+(r%2)*8+lane/4, col=c*16+(r/2)*8+2*(lane%4);
            auto a=acc.tiles[0][c].data[r];
            if(row<n) {
                output[(int64_t(bh)*n+row)*D+col]=__float2bfloat16_rn(a.x*reciprocal(l[r%2]));
                output[(int64_t(bh)*n+row)*D+col+1]=__float2bfloat16_rn(a.y*reciprocal(l[r%2]));
            }
        }
    }
    if(lane%4==0) {
        #pragma unroll
        for(int z=0;z<2;++z) {
            int row=row0+z*8+lane/4;
            if(row<n) {
                int64_t dst=int64_t(bh)*n+row;
                lse[dst]=(m[z]+log2f(l[z]))*0.6931471805599453f;
                top[dst]=reciprocal(l[z]); index[dst]=arg[z];
            }
        }
    }
}

template<int D> __global__ void single(const __nv_bfloat16* token,
        const __nv_bfloat16* key, const __nv_bfloat16* value, __nv_bfloat16* output,
        float* lse, float* top, int* index, int heads, int n, int vocab, float scale) {
    extern __shared__ __align__(128) unsigned char mem[];
    auto& s=*reinterpret_cast<Scratch<D>*>(mem);
    int lane=threadIdx.x, bh=blockIdx.y, row0=blockIdx.x*16, head=bh%heads;
    for(int i=lane;i<16*D;i+=32)
        s.token[int2{i/D,i%D}]=row0+i/D<n?token[(int64_t(bh)*n+row0)*D+i]:__float2bfloat16_rn(0.f);
    __syncwarp();
    kt::rt_bf<16,D> x;
    kt::warp::load(x,s.token);
    kt::rt_fl<16,D> acc{0.f};
    float m[2]={-INFINITY,-INFINITY}, l[2]={0.f,0.f};
    int arg[2]={INT_MAX,INT_MAX};
    for(int vb=0;vb<vocab;vb+=64) {
        for(int i=lane;i<64*D;i+=32) {
            int64_t pos=(int64_t(head)*vocab+vb)*D+i;
            s.key[int2{i/D,i%D}]=vb+i/D<vocab?key[pos]:__float2bfloat16_rn(0.f);
            s.value[int2{i/D,i%D}]=vb+i/D<vocab?value[pos]:__float2bfloat16_rn(0.f);
        }
        __syncwarp();
        step(x,acc,m,l,arg,s.key,s.value,vb,vocab,scale,lane,n-row0);
        __syncwarp();
    }
    store_result(acc,m,l,arg,output,lse,top,index,row0,bh,n,lane);
}

struct FusedArgs {
    const __nv_bfloat16 *q,*k,*eq,*ek;
    __nv_bfloat16 *oq,*ok;
    float *lk,*lq,*pk,*pq;
    int *ik,*iq;
    int h,n,voc;
    float scale;
};
template<int D,int BV> struct FusedScratch {
    struct Slot { kt::st_bf<BV,D> eq,ek; };
    union { kt::st_bf<16,D> token[8]; Slot slot[2]; };
    alignas(8) uint64_t ready[2],free[2];
};
__device__ __forceinline__ void tma3(const CUtensorMap* map,void* dst,uint64_t* bar,
                                    int feature,int row,int head) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%3, %4, %5}], [%2];" ::
                 "r"(smaddr(dst)),"l"(map),"r"(smaddr(bar)),"r"(feature),"r"(row),"r"(head):"memory");
}
template<int D,int BV> CUtensorMap ordinary_map(const void* p,int voc,int heads) {
    constexpr int S=D==32?32:64;
    cuuint64_t dims[]{D,cuuint64_t(voc),cuuint64_t(heads)};
    cuuint64_t strides[]{D*2,cuuint64_t(voc)*D*2};
    cuuint32_t box[]{S,BV,1},elem[]{1,1,1};
    CUtensorMap map{};
    auto e=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,const_cast<void*>(p),
        dims,strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,
        D==32?CU_TENSOR_MAP_SWIZZLE_64B:CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(e!=CUDA_SUCCESS) throw std::runtime_error("embedding TMA descriptor failed");
    return map;
}
template<int D,int BV> __global__ __launch_bounds__(384,1) void fused(
        __grid_constant__ const FusedArgs a,
        __grid_constant__ const CUtensorMap qm,__grid_constant__ const CUtensorMap km) {
    extern __shared__ __align__(128) unsigned char mem[];
    auto& s=*reinterpret_cast<FusedScratch<D,BV>*>(mem);
    int warp=threadIdx.x/32,lane=threadIdx.x%32,bh=blockIdx.y;
    int row0=blockIdx.x*64+(warp%4)*16,head=bh%a.h;
    if(threadIdx.x==0) {
        for(int b=0;b<2;++b) { init_bar(&s.ready[b],32); init_bar(&s.free[b],256); }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    if(warp<8) {
        auto token=warp<4?a.k:a.q;
        for(int i=lane;i<16*D;i+=32)
            s.token[warp][int2{i/D,i%D}]=row0+i/D<a.n?
                token[(int64_t(bh)*a.n+row0)*D+i]:__float2bfloat16_rn(0.f);
    }
    __syncthreads();
    kt::rt_bf<16,D> x;
    if(warp<8) kt::warp::load(x,s.token[warp]);
    __syncthreads(); // Staging is dead; producer may now overwrite the union.
    if(warp>=8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if(warp==8) for(int t=0;t*BV<a.voc;++t) {
            int b=t%2,vb=t*BV;
            if(t>=2) wait(&s.free[b],(t/2-1)&1);
            auto& slot=s.slot[b];
            if(vb+BV<=a.voc) {
                if(lane==0) {
                    expect(&s.ready[b],2*BV*D*2);
                    constexpr int S=D==32?32:64;
                    #pragma unroll
                    for(int f=0;f<D/S;++f) {
                        tma3(&qm,slot.eq.data+f*BV*S,&s.ready[b],f*S,vb,head);
                        tma3(&km,slot.ek.data+f*BV*S,&s.ready[b],f*S,vb,head);
                    }
                } else arrive(&s.ready[b]);
            } else {
                // Guarded unpadded tail, never read across a vocabulary head.
                for(int i=lane;i<BV*D;i+=32) {
                    int64_t pos=(int64_t(head)*a.voc+vb)*D+i;
                    slot.eq[int2{i/D,i%D}]=vb+i/D<a.voc?a.eq[pos]:__float2bfloat16_rn(0.f);
                    slot.ek[int2{i/D,i%D}]=vb+i/D<a.voc?a.ek[pos]:__float2bfloat16_rn(0.f);
                }
                __syncwarp();
                arrive(&s.ready[b]);
            }
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
        kt::rt_fl<16,D> acc{0.f};
        float m[2]={-INFINITY,-INFINITY},l[2]={0.f,0.f};
        int arg[2]={INT_MAX,INT_MAX};
        for(int t=0;t*BV<a.voc;++t) {
            int b=t%2;
            wait(&s.ready[b],(t/2)&1);
            auto& key=warp<4?s.slot[b].ek:s.slot[b].eq;
            auto& value=warp<4?s.slot[b].eq:s.slot[b].ek;
            step(x,acc,m,l,arg,key,value,t*BV,a.voc,a.scale,lane,a.n-row0);
            __syncwarp(); // Last PV read must finish before ring reuse.
            arrive(&s.free[b]);
        }
        store_result(acc,m,l,arg,warp<4?a.oq:a.ok,warp<4?a.lk:a.lq,
                     warp<4?a.pk:a.pq,warp<4?a.ik:a.iq,row0,bh,a.n,lane);
    }
    __syncthreads(); // Keep producer group alive until consumers are done.
}
template<int D,int BV> void launch_fused(FusedArgs a,int bh,cudaStream_t stream) {
    auto qm=ordinary_map<D,BV>(a.eq,a.voc,a.h),km=ordinary_map<D,BV>(a.ek,a.voc,a.h);
    auto e=cudaFuncSetAttribute(fused<D,BV>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(FusedScratch<D,BV>));
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    fused<D,BV><<<dim3((a.n+63)/64,bh),384,sizeof(FusedScratch<D,BV>),stream>>>(a,qm,km);
}
template<int D> void launch(const void* x,const void* k,const void* v,void* out,
        float* lse,float* top,int* idx,int bh,int h,int n,int voc,float scale,cudaStream_t stream) {
    auto e=cudaFuncSetAttribute(single<D>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Scratch<D>));
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    single<D><<<dim3((n+15)/16,bh),32,sizeof(Scratch<D>),stream>>>(
        (const __nv_bfloat16*)x,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,
        (__nv_bfloat16*)out,lse,top,idx,h,n,voc,scale);
}
}
void launch_embedding(const void* x,const void* k,const void* v,void* out,float* lse,
        float* top,int* idx,int bh,int h,int n,int voc,int d,float scale,cudaStream_t stream) {
    using namespace dism_v2::embedding;
    if(d==32) launch<32>(x,k,v,out,lse,top,idx,bh,h,n,voc,scale,stream);
    else if(d==64) launch<64>(x,k,v,out,lse,top,idx,bh,h,n,voc,scale,stream);
    else launch<128>(x,k,v,out,lse,top,idx,bh,h,n,voc,scale,stream);
}
void launch_embedding_fused(const void* q,const void* k,const void* eq,const void* ek,
        void* oq,void* ok,float* lk,float* lq,float* pk,float* pq,int* ik,int* iq,
        int bh,int h,int n,int voc,int d,int block_v,float scale,cudaStream_t stream) {
    using namespace dism_v2::embedding;
    FusedArgs a{(const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)eq,
                (const __nv_bfloat16*)ek,(__nv_bfloat16*)oq,(__nv_bfloat16*)ok,
                lk,lq,pk,pq,ik,iq,h,n,voc,scale};
    if(d==32 && block_v==128) launch_fused<32,128>(a,bh,stream);
    else if(d==64 && block_v==128) launch_fused<64,128>(a,bh,stream);
    else if(d==32) launch_fused<32,64>(a,bh,stream);
    else if(d==64) launch_fused<64,64>(a,bh,stream);
    else launch_fused<128,64>(a,bh,stream);
}
