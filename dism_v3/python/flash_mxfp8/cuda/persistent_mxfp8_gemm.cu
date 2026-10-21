// Persistent MXFP8 GEMM with prequantized A/B operands.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include "mxfp8_epilogue.cuh"
#include <stdexcept>
#include <string>
#ifndef FP8GEMM_FACTORED_STORES
#define FP8GEMM_FACTORED_STORES 0
#endif
namespace fp8gemm {
template<class ST,class RT>
__device__ __forceinline__ void mx_store_output(ST &dst,const RT &src) {
#if FP8GEMM_FACTORED_STORES
    store_factored<4>(dst,src);
#else
    kt::warpgroup::store(dst,src);
#endif
}
template<class T> using MXGlobal=kt::gl<typename T::dtype,-1,-1,-1,-1,T>;
__device__ __forceinline__ int2 mx_raster(int task,int ntiles,int mtiles,int g) {
    if(g==1)return make_int2(task/ntiles,task%ntiles);
    int span=g*ntiles,first=task/span*g,rows=min(g,mtiles-first),local=task%span;
    return make_int2(first+local%rows,local/rows);
}
template<int R,int C,int BK,int S,bool SaveQ=false,bool OnlyQ=false> struct MXConfig {
    static_assert(!OnlyQ || SaveQ);
    using ATile=kt::st_fp8e4m3<4*R,BK>;
    using BTile=kt::st_fp8e4m3<C,BK>;
    using OTile=kt::st_bf<4*R,C>;
    using QTile=kt::st_fp8e4m3<4*R,(C+31)/32*32>;
    union OutputStorage {OTile bf;QTile q;};
    struct QuantizedArgs {MXGlobal<QTile> out;uint32_t *scales;};
    struct NoQuantizedArgs {};
    static constexpr int SC=BK/32,SR=(C+31)/32;
    struct Packet {
        ATile a[2];BTile b;
    };
    struct Shared {
        Packet input[S];
        std::conditional_t<OnlyQ,QTile,std::conditional_t<SaveQ,OutputStorage,OTile>> output[2];
    };
    struct Args {
        MXGlobal<ATile> a;MXGlobal<BTile> b;
        [[no_unique_address]] std::conditional_t<OnlyQ,NoQuantizedArgs,MXGlobal<OTile>> out;
        const uint32_t *as; // packed [ceil(K/128),roundup(M,16)] uint32 words
        const uint8_t *bs; // bs[ceil(N/32),K/32]
        int m,n,sc,ntiles,mtiles,total,ktiles,raster_group;
        uint32_t as_stride;
        [[no_unique_address]] std::conditional_t<SaveQ,QuantizedArgs,NoQuantizedArgs> q;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=101376);
    struct ScalePrefetch {
        uint32_t a[R/16],b;
        __device__ __forceinline__ void load(const Args &args,int mr,int nc,int k,int lane) {
#pragma unroll
            for(int i=0;i<R/16;i++) {
                int row=mr+i*16+lane/4+8*(lane&1);
                a[i]=(row<args.m && k<args.ktiles)
                    ?args.as[size_t(uint32_t(k)*SC/4)*args.as_stride+uint32_t(row)]:0;
            }
            // Each lane fetches one byte; consumers broadcast the small set
            // of distinct B scales with shuffles after the prefetch latency.
            int row=nc/32+(lane/SC)%SR,col=k*SC+lane%SC;
            b=(row<(args.n+31)/32 && col<args.sc)
                ?args.bs[size_t(uint32_t(row))*uint32_t(args.sc)+uint32_t(col)]:0;
        }
    };
};
template<int R,int C,int BK,int S,bool SaveQ=false,bool OnlyQ=false>
__global__ __launch_bounds__(384,1) void persistent_mxfp8(
    const __grid_constant__ typename MXConfig<R,C,BK,S,SaveQ,OnlyQ>::Args args) {
    using CF=MXConfig<R,C,BK,S,SaveQ,OnlyQ>;using Shared=typename CF::Shared;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> allocator(reinterpret_cast<int*>(storage));
    int group=kt::warpgroup::groupid(),warp=kt::warpgroup::warpid(),lane=threadIdx.x%32;
    MbarrierRingPipe pipe(allocator,BufferSet{&Shared::input},group,
                         ProducerWarp<true,2>,ConsumerWarpGroup<false,0,1>);
    auto &shared=allocator.allocate<Shared>();
    if(group==2) {
        kt::warpgroup::decrease_registers<40>();
        if(warp==0) {
            bool leader=kt::warp::elect_leader();int last=0;
            for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
                int2 coord=mx_raster(task,args.ntiles,args.mtiles,args.raster_group);
                int mr=coord.x*(8*R),nc=coord.y*C;
                for(int k=0;k<args.ktiles;k++) {
                    auto packet=pipe.waitBuffer(0,shared.input);auto &tile=packet.template get<0>();
                    if(leader) {
                        auto &bar=packet.getBarrier();
                        kt::tma::expect_bytes(bar,(8*R+C)*BK);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,mr,k*BK},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,mr+4*R,k*BK},bar);
                        kt::tma::load_async(tile.b,args.b,{0,0,nc,k*BK},bar);
                    }
                    last=packet.SlotIdx;packet.submitToNextAndTrigger();pipe.moveNext();
                }
            }
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        bool writer=warp==0 && kt::warp::elect_leader();
        for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
            kt::rt_fl<R,C> accum;kt::warp::rt_maps::zero(accum);
            int2 coord=mx_raster(task,args.ntiles,args.mtiles,args.raster_group);
            int mr=coord.x*8*R+group*4*R+warp*R,nc=coord.y*C;
            typename CF::ScalePrefetch next;
            next.load(args,mr,nc,0,lane);
            for(int k=0;k<args.ktiles;k++) {
                auto packet=pipe.waitBuffer(2,shared.input);auto &tile=packet.template get<0>();
                kt::rt_fp8e4m3<R,BK> ar;kt::rt_fp8e4m3<C,BK> br;
                kt::mxfp8_scale<R,BK> as;kt::mxfp8_scale<C,BK> bs;
                load_fp8_factored<4,false>(ar,tile.a[group]);
                load_fp8_factored<1,true>(br,tile.b);
#pragma unroll
                for(int i=0;i<R/16;i++) {
                    uint32_t word=next.a[i];
#pragma unroll
                    for(int kk=0;kk<CF::SC;kk++)as.data[i][kk]=word >> (8*((k*CF::SC+kk)%4));
                }
#pragma unroll
                for(int i=0;i<C/16;i++) {
#pragma unroll
                    for(int kk=0;kk<CF::SC;kk++)
                        bs.data[i][kk]=__shfl_sync(0xffffffff,next.b,(i/2)*CF::SC+kk);
                }
                // Issue next-stage scale loads before this stage's QMMA.
                next.load(args,mr,nc,k+1,lane);
                packet.submitToNextAndTrigger();pipe.moveNext();
                kt::warp::wmma::mma_ABt_block_scaled(accum,ar,br,accum,as,bs);
            }
            if constexpr(SaveQ) {
                kt::rt_bf<R,C> rounded;
                kt::warp::copy(rounded,accum);
                if constexpr(!OnlyQ) {
                    if(writer)kt::tma::store_async_read_wait<0>();
                    kt::warpgroup::sync(group+1);
                    mx_store_output(shared.output[group].bf,rounded);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    kt::warpgroup::sync(group+1);
                    if(writer) {
                        kt::tma::store_async(args.out,shared.output[group].bf,
                                            {0,0,coord.x*8*R+group*4*R,nc});
                        kt::tma::store_commit_group();
                    }
                }
                // Without the BF16-store barrier, establish warp convergence
                // before quantization's shuffle reductions.
                if constexpr(OnlyQ)__syncwarp();
                // Quantization can overlap TMA reading the BF16 output.
#ifdef FP8GEMM_EPILOGUE_MOVM
                MXOutputTransposed<R,C> quantized;
#else
                kt::rt_fp8e4m3<R,C> quantized;
#endif
                kt::mxfp8_scale<R,C> scales;
#ifdef FP8GEMM_EPILOGUE_MOVM
                quantize_output_movmatrix(quantized,scales,rounded);
#else
                kt::online_quantize_mxfp8(quantized,scales,rounded);
#endif
                store_output_scales<R,C>(args.q.scales,scales,mr,nc,args.m,args.n);
                // Wait for the preceding output TMA read before reusing shared
                // storage: BF16 in dual-output mode, previous task in FP8-only mode.
                if(writer)kt::tma::store_async_read_wait<0>();
                kt::warpgroup::sync(group+1);
                auto &qshared=[&]() -> typename CF::QTile& {
                    if constexpr(OnlyQ)return shared.output[group];
                    else return shared.output[group].q;
                }();
#ifdef FP8GEMM_EPILOGUE_MOVM
                store_output_movmatrix<4,(FP8GEMM_FACTORED_STORES!=0)>(qshared,quantized);
#else
                mx_store_output(qshared,quantized);
#endif
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                kt::warpgroup::sync(group+1);
                if(writer) {
                    kt::tma::store_async(args.q.out,qshared,
                                        {0,0,coord.x*8*R+group*4*R,nc});
                    kt::tma::store_commit_group();
                }
            } else {
                if(writer)kt::tma::store_async_read_wait<0>();
                kt::warpgroup::sync(group+1);
                mx_store_output(shared.output[group],accum);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                kt::warpgroup::sync(group+1);
                if(writer) {
                    kt::tma::store_async(args.out,shared.output[group],
                                        {0,0,coord.x*8*R+group*4*R,nc});
                    kt::tma::store_commit_group();
                }
            }
        }
        if(writer)kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}
static void mx_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct MXPlanBase {
    virtual ~MXPlanBase()=default;
    virtual void launch(cudaStream_t)=0;
    virtual void info(int*)=0;
    virtual void rebind(void*a,void*b,void*as,void*bs,void*o,void*q,void*qs)=0;
};
template<int R,int C,int BK,int S,bool SaveQ=false,bool OnlyQ=false> struct MXPlan:MXPlanBase {
    using CF=MXConfig<R,C,BK,S,SaveQ,OnlyQ>;typename CF::Args args;int grid,regs;
    static auto bf16_args(void *o,int m,int n) {
        if constexpr(OnlyQ)return typename CF::NoQuantizedArgs{};
        else return MXGlobal<typename CF::OTile>((kt::bf16*)o,1,1,m,n);
    }
    static auto quantized_args(void *q,void *qs,int m,int n) {
        if constexpr(SaveQ) {
            static_assert(C%32==0);
            if(!q||!qs)throw std::runtime_error("Missing quantized output");
            // TK splits global columns into fixed-width swizzle panels. A
            // partial final panel is not masked by its five-dimensional map.
            if(n%CF::QTile::swizzle_bytes)
                throw std::runtime_error("N must align to FP8 output TMA swizzle panels");
            return typename CF::QuantizedArgs{
                MXGlobal<typename CF::QTile>((kt::fp8e4m3*)q,1,1,m,n),(uint32_t*)qs};
        } else return typename CF::NoQuantizedArgs{};
    }
    MXPlan(void *a,void *b,void *as,void *bs,void *o,int m,int n,int k,int sms,int g,
           void *q=nullptr,void *qs=nullptr):
        args{MXGlobal<typename CF::ATile>((kt::fp8e4m3*)a,1,1,m,k),
             MXGlobal<typename CF::BTile>((kt::fp8e4m3*)b,1,1,n,k),
             bf16_args(o,m,n),
             (uint32_t*)as,
             (uint8_t*)bs,m,n,(k+31)/32,(n+C-1)/C,(m+8*R-1)/(8*R),
             ((m+8*R-1)/(8*R))*((n+C-1)/C),(k+BK-1)/BK,std::min(g,(m+8*R-1)/(8*R)),
             uint32_t((m+15)/16*16),quantized_args(q,qs,m,n)} {
        if(k%CF::ATile::swizzle_bytes || k%CF::BTile::swizzle_bytes || (!OnlyQ && n%(CF::OTile::swizzle_bytes/2)))
            throw std::runtime_error("K/N must align to TMA swizzle panels");
        int dev;mx_check(cudaGetDevice(&dev));cudaDeviceProp prop;mx_check(cudaGetDeviceProperties(&prop,dev));
        grid=std::min(args.total,sms>0?std::min(sms,prop.multiProcessorCount):prop.multiProcessorCount);
        mx_check(cudaFuncSetAttribute(persistent_mxfp8<R,C,BK,S,SaveQ,OnlyQ>,cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr;mx_check(cudaFuncGetAttributes(&attr,persistent_mxfp8<R,C,BK,S,SaveQ,OnlyQ>));regs=attr.numRegs;
        if(regs<168 || attr.localSizeBytes)throw std::runtime_error("Unexpected register pool or local memory");
    }
    void launch(cudaStream_t stream) override {
        persistent_mxfp8<R,C,BK,S,SaveQ,OnlyQ><<<grid,384,CF::smem,stream>>>(args);mx_check(cudaGetLastError());
    }
    void info(int *d) override {d[0]=R;d[1]=C;d[2]=BK;d[3]=S;d[4]=CF::smem;d[5]=regs;d[6]=grid;}
    void rebind(void*a,void*b,void*as,void*bs,void*o,void*q,void*qs) override {
        args.a=MXGlobal<typename CF::ATile>((kt::fp8e4m3*)a,1,1,args.m,args.sc*32);
        args.b=MXGlobal<typename CF::BTile>((kt::fp8e4m3*)b,1,1,args.n,args.sc*32);
        args.out=bf16_args(o,args.m,args.n);
        args.as=(uint32_t*)as;
        args.bs=(uint8_t*)bs;
        args.q=quantized_args(q,qs,args.m,args.n);
    }
};
static thread_local std::string mx_error;
}
extern "C" const char *tk_mx_error(){return fp8gemm::mx_error.c_str();}
extern "C" void *tk_mx_create(void *a,void *b,void *as,void *bs,void *o,int m,int n,int k,int config,int sms,int g) {
    using namespace fp8gemm;
    try {
        if(!a||!b||!as||!bs||!o||m<=0||n<=0||k<=0||g<1)throw std::runtime_error("Invalid arguments");
#define CFG(I,R,C,K,S) case I:return new MXPlan<R,C,K,S>(a,b,as,bs,o,m,n,k,sms,g)
        switch(config) {
#include "mxfp8_configs.inc"
            default:throw std::runtime_error("Invalid configuration");
        }
#undef CFG
    }catch(const std::exception &e){mx_error=e.what();return nullptr;}
}
extern "C" int tk_mx_launch(fp8gemm::MXPlanBase *p,void *s){
    try{p->launch((cudaStream_t)s);return 0;}catch(const std::exception &e){fp8gemm::mx_error=e.what();return -1;}
}
extern "C" int tk_mx_rebind(fp8gemm::MXPlanBase *p,
    void*a,void*b,void*as,void*bs,void*o,void*q,void*qs) {
    try{p->rebind(a,b,as,bs,o,q,qs);return 0;}
    catch(const std::exception &e){fp8gemm::mx_error=e.what();return -1;}
}
extern "C" void *tk_mx_create_quantized(void *a,void *b,void *as,void *bs,void *o,
    void *q,void *qs,int m,int n,int k,int config,int sms,int g) {
    using namespace fp8gemm;
    try {
        if(!a||!b||!as||!bs||!q||!qs||m<=0||n<=0||k<=0||g<1)
            throw std::runtime_error("Invalid arguments");
#define CFG(I,R,C,K,S) case I: \
        if constexpr(C%32==0) { \
            if(o)return new MXPlan<R,C,K,S,true>(a,b,as,bs,o,m,n,k,sms,g,q,qs); \
            return new MXPlan<R,C,K,S,true,true>(a,b,as,bs,o,m,n,k,sms,g,q,qs); } \
        else throw std::runtime_error("Quantized epilogue requires C divisible by 32")
        switch(config) {
#include "mxfp8_configs.inc"
            default:throw std::runtime_error("Invalid configuration");
        }
#undef CFG
    }catch(const std::exception &e){mx_error=e.what();return nullptr;}
}
extern "C" void tk_mx_info(fp8gemm::MXPlanBase *p,int *d){p->info(d);}
extern "C" void tk_mx_destroy(fp8gemm::MXPlanBase *p){delete p;}
