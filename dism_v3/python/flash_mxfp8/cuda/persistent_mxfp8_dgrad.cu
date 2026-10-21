// dX[M,K] = dY[M,N] @ W[N,K], computed as (W^T @ dY^T)^T.
// Prequantized E4M3 dY uses per-row/N32 scales; W uses existing 32x32 scales.
// Native transposing LDSM feeds QMMA A directly; row LDSM feeds B directly.
// FP32 accumulation, BF16 output only; no global transpose or online quantization.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm::mxd {
template<class Tile> using Global=kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;
__device__ __forceinline__ int2 raster(int task,int columns,int rows,int group) {
    int span=group*columns,first=task/span*group,count=min(group,rows-first),local=task%span;
    return make_int2(first+local%count,local/count);
}
template<int R,int C,int BN,int S> struct Config {
    using A=kt::st_fp8e4m3<4*R,BN>;
    using B=kt::st_fp8e4m3<BN,C>;
    using O=kt::st_bf<4*R,C>;
    struct Packet {A a[2];B b;};
    struct Shared {Packet input[S];O output[2];};
    struct Args {
        Global<A> a;Global<B> b;Global<O> out;
        // dY: packed [ceil(N/128),roundup(M,16)]; W: bytes [ceil(N/32),K/32].
        const uint32_t *as;
        const uint8_t *bs;
        int m,k,n,nrows,ncols,total,steps,group,as_stride,bs_stride;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=101376);
    static_assert((BN/32)*(C/32)<=32);
    struct ScalePrefetch {
        uint32_t a[R/16],b;
        __device__ __forceinline__ void load(const Args &args,int nr,int kc,int step,int lane) {
            int block=step*(BN/32)/4;
#pragma unroll
            for(int i=0;i<R/16;i++) {
                int row=nr+i*16+lane/4+8*(lane&1);
                a[i]=(row<args.m && step<args.steps)
                    ?args.as[size_t(block)*args.as_stride+row]:0;
            }
            int row=step*(BN/32)+(lane/(C/32))%(BN/32),col=kc/32+lane%(C/32);
            b=(row<(args.n+31)/32 && col<args.bs_stride)
                ?args.bs[size_t(row)*args.bs_stride+col]:0;
        }
    };
};

template<int R,int C,int BN,int S>
__global__ __launch_bounds__(384,1) void persistent_mxfp8_dgrad(
    const __grid_constant__ typename Config<R,C,BN,S>::Args args) {
    using CF=Config<R,C,BN,S>;using Shared=typename CF::Shared;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> allocator(reinterpret_cast<int*>(storage));
    int group=kt::warpgroup::groupid(),warp=kt::warpgroup::warpid(),lane=kt::laneid();
    MbarrierRingPipe pipe(allocator,BufferSet{&Shared::input},group,
                         ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    auto &shared=allocator.allocate<Shared>();
    if(group==2) {
        kt::warpgroup::decrease_registers<40>();
        if(warp==0) {
            bool leader=kt::warp::elect_leader();int last=0;
            for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
                int2 coord=raster(task,args.ncols,args.nrows,args.group);
                int nr=coord.x*8*R,kc=coord.y*C;
                for(int step=0;step<args.steps;step++) {
                    auto p=pipe.waitBuffer(0,shared.input);auto &tile=p.template get<0>();
                    if(leader) {
                        auto &bar=p.getBarrier();
                        kt::tma::expect_bytes(bar,(8*R+C)*BN);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,nr,step*BN},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,nr+4*R,step*BN},bar);
                        kt::tma::load_async(tile.b,args.b,{0,0,step*BN,kc},bar);
                    }
                    last=p.SlotIdx;p.submitToNextAndTrigger();pipe.moveNext();
                }
            }
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        bool writer=warp==0 && kt::warp::elect_leader();
        for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
            int2 coord=raster(task,args.ncols,args.nrows,args.group);
            int nr=coord.x*8*R+group*4*R+warp*R,kc=coord.y*C;
            kt::rt_fl<C,R> accum;kt::warp::rt_maps::zero(accum);
            typename CF::ScalePrefetch next;
            next.load(args,nr,kc,0,lane);
            for(int step=0;step<args.steps;step++) {
                auto p=pipe.waitBuffer(2,shared.input);auto &tile=p.template get<0>();
                kt::rt_fp8e4m3<R,BN> ar;
                kt::rt<kt::fp8e4m3,BN,C,kt::ducks::rt_layout::col> br;
                kt::mxfp8_scale<R,BN> as;kt::mxfp8_scale<C,BN> bs;
                load_fp8_factored<4,true>(ar,tile.a[group]);
                // Native transposing LDSM is already in QMMA A register order.
                kt::warp::load(br,tile.b);
#pragma unroll
                for(int i=0;i<R/16;i++)
#pragma unroll
                    for(int t=0;t<BN/32;t++)
                        as.data[i][t]=next.a[i]>>(8*((step*(BN/32)+t)%4));
#pragma unroll
                for(int j=0;j<C/16;j++)
#pragma unroll
                    for(int t=0;t<BN/32;t++)
                        bs.data[j][t]=__shfl_sync(0xffffffff,next.b,t*(C/32)+j/2);
                next.load(args,nr,kc,step+1,lane);
                p.submitToNextAndTrigger();pipe.moveNext();
                kt::rt_fp8e4m3<C,BN> weight_a;
#pragma unroll
                for(int j=0;j<C/16;j++)
#pragma unroll
                    for(int t=0;t<BN/32;t++)
#pragma unroll
                        for(int v=0;v<4;v++)weight_a.tiles[j][t].data[v]=br.tiles[t][j].data[v];
                kt::warp::wmma::mma_ABt_block_scaled(accum,weight_a,ar,accum,bs,as);
            }
            if(writer)kt::tma::store_async_read_wait<0>();
            kt::warpgroup::sync(group+1);
            // Logical transpose plus BF16 conversion/STSM.trans restores [M,K].
            kt::rt<float,R,C,kt::ducks::rt_layout::col> transposed;
#pragma unroll
            for(int i=0;i<R/16;i++)
#pragma unroll
                for(int j=0;j<C/16;j++)
#pragma unroll
                    for(int v=0;v<4;v++)transposed.tiles[i][j].data[v]=accum.tiles[j][i].data[v];
            kt::warpgroup::store(shared.output[group],transposed);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            kt::warpgroup::sync(group+1);
            if(writer) {
                kt::tma::store_async(args.out,shared.output[group],
                                    {0,0,coord.x*8*R+group*4*R,kc});
                kt::tma::store_commit_group();
            }
        }
        if(writer)kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct PlanBase {
    virtual ~PlanBase()=default;
    virtual void launch(cudaStream_t)=0;
    virtual void info(int*)=0;
    virtual void rebind(void*a,void*b,void*as,void*bs,void*out)=0;
};
template<int R,int C,int BN,int S> struct Plan final:PlanBase {
    using CF=Config<R,C,BN,S>;typename CF::Args args;int grid,regs;
    Plan(void *a,void *b,void *as,void *bs,void *out,int m,int k,int n,int sms,int group):
        args{Global<typename CF::A>((kt::fp8e4m3*)a,1,1,m,n),
             Global<typename CF::B>((kt::fp8e4m3*)b,1,1,n,k),
             Global<typename CF::O>((kt::bf16*)out,1,1,m,k),
             (uint32_t*)as,(uint8_t*)bs,m,k,n,(m+8*R-1)/(8*R),(k+C-1)/C,
             ((m+8*R-1)/(8*R))*((k+C-1)/C),(n+BN-1)/BN,
             std::min(group,(m+8*R-1)/(8*R)),(m+15)/16*16,k/32} {
        if(n%CF::A::swizzle_bytes || k%CF::B::swizzle_bytes ||
           k%(CF::O::swizzle_bytes/2))throw std::runtime_error("N/K must align to TMA swizzle panels");
        int dev;check(cudaGetDevice(&dev));cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,dev));
        grid=std::min(args.total,sms>0?std::min(sms,prop.multiProcessorCount):prop.multiProcessorCount);
        check(cudaFuncSetAttribute(persistent_mxfp8_dgrad<R,C,BN,S>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr;check(cudaFuncGetAttributes(&attr,persistent_mxfp8_dgrad<R,C,BN,S>));
        regs=attr.numRegs;
        if(attr.localSizeBytes || regs<168)throw std::runtime_error("Unexpected register pool or local memory");
    }
    void launch(cudaStream_t s) override {
        persistent_mxfp8_dgrad<R,C,BN,S><<<grid,384,CF::smem,s>>>(args);check(cudaGetLastError());
    }
    void info(int *p) override {p[0]=R;p[1]=C;p[2]=BN;p[3]=S;p[4]=CF::smem;p[5]=regs;p[6]=grid;}
    void rebind(void*a,void*b,void*as,void*bs,void*out) override {
        args.a=Global<typename CF::A>((kt::fp8e4m3*)a,1,1,args.m,args.n);
        args.b=Global<typename CF::B>((kt::fp8e4m3*)b,1,1,args.n,args.k);
        args.out=Global<typename CF::O>((kt::bf16*)out,1,1,args.m,args.k);
        args.as=(uint32_t*)as;
        args.bs=(uint8_t*)bs;
    }
};
static thread_local std::string error;
}
extern "C" const char *tk_mxd_error(){return fp8gemm::mxd::error.c_str();}
extern "C" void *tk_mxd_create(void *a,void *b,void *as,void *bs,void *out,
    int m,int k,int n,int config,int sms,int group) {
    using namespace fp8gemm::mxd;
    try {
        if(!a||!b||!as||!bs||!out||m<=0||k<=0||n<=0||group<=0||sms<0)
            throw std::runtime_error("Invalid arguments");
#define CFG(I,R,C,BN,S) case I:return new Plan<R,C,BN,S>(a,b,as,bs,out,m,k,n,sms,group)
        switch(config) {
#include "mxfp8_dgrad_configs.inc"
            default:throw std::runtime_error("Invalid configuration");
        }
#undef CFG
    }catch(const std::exception &e){error=e.what();return nullptr;}
}
extern "C" int tk_mxd_launch(fp8gemm::mxd::PlanBase *p,void *s) {
    try {p->launch((cudaStream_t)s);return 0;}
    catch(const std::exception &e){fp8gemm::mxd::error=e.what();return -1;}
}
extern "C" int tk_mxd_rebind(fp8gemm::mxd::PlanBase *p,
    void*a,void*b,void*as,void*bs,void*out) {
    try {p->rebind(a,b,as,bs,out);return 0;}
    catch(const std::exception &e){fp8gemm::mxd::error=e.what();return -1;}
}
extern "C" void tk_mxd_info(fp8gemm::mxd::PlanBase *p,int *d){p->info(d);}
extern "C" void tk_mxd_destroy(fp8gemm::mxd::PlanBase *p){delete p;}
