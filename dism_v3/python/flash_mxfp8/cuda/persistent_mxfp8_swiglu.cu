// SM120a persistent fused up(x) * SiLU(gate(x)) with prequantized MXFP8
// operands. Both projections share the X tile and keep independent FP32
// accumulators; the activation and BF16 conversion stay in registers.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm::mxswiglu {
namespace kt=kittens;

template<class Tile> using Global=kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;

__device__ __forceinline__ int2 raster(int task,int ntiles,int mtiles,int group) {
    if(group==1)return make_int2(task/ntiles,task%ntiles);
    int span=group*ntiles,first=task/span*group,rows=min(group,mtiles-first),local=task%span;
    return make_int2(first+local%rows,local/rows);
}

// The gate accumulator has already been halved through its E8M0 block scale:
// up * gate * sigmoid(gate) = up * gate_half*(1+tanh(gate_half)).
__device__ __forceinline__ float swiglu(float gate_half,float up) {
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(gate_half));
    return fmaf(gate_half,t,gate_half)*up;
}

template<int R,int C,int BK,int S,bool Aux> struct Config {
    using ATile=kt::st_fp8e4m3<4*R,BK>;
    using BTile=kt::st_fp8e4m3<C,BK>;
    using OTile=kt::st_bf<4*R,C>;
    struct Packet {
        ATile a[2];
        BTile gate,up;
    };
    static constexpr int max_smem=101376;
    // Two tiles per consumer let gate and up TMA stores overlap. Fall back
    // to one tile, with serialized async stores plus read waits, when adding
    // the second tile would exceed the SM120 shared-memory limit.
    static constexpr int output_buffers=
        Aux && 1024+S*sizeof(Packet)+2*sizeof(OTile)+2*sizeof(OTile)<=max_smem
        ?2:1;
    struct Shared {
        Packet input[S];
        OTile output[2][output_buffers];
    };
    struct Args {
        Global<ATile> a;
        Global<BTile> gate,up;
        Global<OTile> out,gate_half,up_linear;
        const uint32_t *as; // [ceil(K/128),roundup(M,16)] packed E8M0 words
        const uint8_t *gs,*us; // [ceil(N/32),ceil(K/32)] E8M0 bytes
        int m,n,sc,ntiles,mtiles,total,ktiles,raster_group;
        uint32_t as_stride;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=max_smem);
};

template<int R,int C,int BK,int S,bool Aux>
__global__ __launch_bounds__(384,1) void persistent_swiglu(
    const __grid_constant__ typename Config<R,C,BK,S,Aux>::Args args) {
    using CF=Config<R,C,BK,S,Aux>;
    using Shared=typename CF::Shared;
    constexpr int SR=(C+31)/32,SC=BK/32;
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
                int2 coord=raster(task,args.ntiles,args.mtiles,args.raster_group);
                int mr=coord.x*(8*R),nc=coord.y*C;
                for(int k=0;k<args.ktiles;k++) {
                    auto packet=pipe.waitBuffer(0,shared.input);
                    auto &tile=packet.template get<0>();
                    if(leader) {
                        auto &bar=packet.getBarrier();
                        // Four FP8 TMA loads: two half-M X panels and one N
                        // panel for each of the gate/up projections.
                        kt::tma::expect_bytes(bar,(8*R+2*C)*BK);
                        kt::tma::load_async(tile.a[0],args.a,{0,0,mr,k*BK},bar);
                        kt::tma::load_async(tile.a[1],args.a,{0,0,mr+4*R,k*BK},bar);
                        kt::tma::load_async(tile.gate,args.gate,{0,0,nc,k*BK},bar);
                        kt::tma::load_async(tile.up,args.up,{0,0,nc,k*BK},bar);
                    }
                    last=packet.SlotIdx;packet.submitToNextAndTrigger();pipe.moveNext();
                }
            }
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<232>();
        bool writer=warp==0 && kt::warp::elect_leader();
        struct ScalePrefetch {
            uint32_t a[R/16],gate,up;
            __device__ __forceinline__ void load(
                const typename CF::Args &args,int mr,int nc,int k,int lane) {
#pragma unroll
                for(int i=0;i<R/16;i++) {
                    int row=mr+i*16+lane/4+8*(lane&1);
                    a[i]=(row<args.m && k<args.ktiles)
                        ?args.as[size_t(uint32_t(k)*SC/4)*args.as_stride+uint32_t(row)]:0;
                }
                int row=nc/32+(lane/SC)%SR,col=k*SC+lane%SC;
                bool valid=row<(args.n+31)/32 && col<args.sc;
                gate=valid?args.gs[size_t(uint32_t(row))*uint32_t(args.sc)+uint32_t(col)]:0;
                up=valid?args.us[size_t(uint32_t(row))*uint32_t(args.sc)+uint32_t(col)]:0;
            }
        };

        for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
            kt::rt_fl<R,C> gate_acc,up_acc;
            kt::warp::rt_maps::zero(gate_acc);
            kt::warp::rt_maps::zero(up_acc);
            int2 coord=raster(task,args.ntiles,args.mtiles,args.raster_group);
            int mr=coord.x*8*R+group*4*R+warp*R,nc=coord.y*C;
            ScalePrefetch next;
            next.load(args,mr,nc,0,lane);
            for(int k=0;k<args.ktiles;k++) {
                auto packet=pipe.waitBuffer(2,shared.input);
                auto &tile=packet.template get<0>();
                kt::rt_fp8e4m3<R,BK> ar;
                kt::rt_fp8e4m3<C,BK> gr,ur;
                kt::mxfp8_scale<R,BK> as,half_as;
                kt::mxfp8_scale<C,BK> gs,us;
                load_fp8_factored<4,false>(ar,tile.a[group]);
                load_fp8_factored<1,true>(gr,tile.gate);
                load_fp8_factored<1,true>(ur,tile.up);
#pragma unroll
                for(int i=0;i<R/16;i++) {
                    uint32_t word=next.a[i];
#pragma unroll
                    for(int kk=0;kk<SC;kk++) {
                        as.data[i][kk]=word >> (8*((k*SC+kk)%4));
                        // Decrementing E8M0 by one halves the entire gate
                        // projection. Keep the unmodified scale for up.
                        half_as.data[i][kk]=max(int(as.data[i][kk]&255)-1,0);
                    }
                }
#pragma unroll
                for(int i=0;i<C/16;i++) {
#pragma unroll
                    for(int kk=0;kk<SC;kk++) {
                        int source=(i/2)*SC+kk;
                        gs.data[i][kk]=__shfl_sync(0xffffffff,next.gate,source);
                        us.data[i][kk]=__shfl_sync(0xffffffff,next.up,source);
                    }
                }
                // Hide global scale latency behind both QMMA instructions.
                next.load(args,mr,nc,k+1,lane);
                packet.submitToNextAndTrigger();pipe.moveNext();
                kt::warp::wmma::mma_ABt_block_scaled(gate_acc,ar,gr,gate_acc,half_as,gs);
                kt::warp::wmma::mma_ABt_block_scaled(up_acc,ar,ur,up_acc,as,us);
            }
            kt::rt_bf<R,C> rounded;
            if constexpr(Aux) {
                // Save the half-gate used by the activation directly. This
                // avoids multiplying every gate value by two only to halve it
                // again in backward.
                kt::rt_bf<R,C> gate_half;
                kt::warp::copy(gate_half,gate_acc);
                if(writer)kt::tma::store_async_read_wait<0>();
                kt::warpgroup::sync(group+1);
                kt::warpgroup::store(shared.output[group][0],gate_half);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                kt::warpgroup::sync(group+1);
                if(writer) {
                    kt::tma::store_async(args.gate_half,shared.output[group][0],
                                        {0,0,coord.x*8*R+group*4*R,nc});
                    kt::tma::store_commit_group();
                }

                kt::rt_bf<R,C> up_linear;
                kt::warp::copy(up_linear,up_acc);
                if constexpr(CF::output_buffers==1) {
                    if(writer)kt::tma::store_async_read_wait<0>();
                    kt::warpgroup::sync(group+1);
                }
                kt::warpgroup::store(shared.output[group][CF::output_buffers-1],up_linear);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                kt::warpgroup::sync(group+1);
                if(writer) {
                    kt::tma::store_async(args.up_linear,
                                        shared.output[group][CF::output_buffers-1],
                                        {0,0,coord.x*8*R+group*4*R,nc});
                    kt::tma::store_commit_group();
                }
            }

#pragma unroll
            for(int r=0;r<gate_acc.height;r++) {
#pragma unroll
                for(int c=0;c<gate_acc.width;c++) {
#pragma unroll
                    for(int i=0;i<gate_acc.packed_per_tile;i++) {
                        float2 g=gate_acc.tiles[r][c].data[i];
                        float2 u=up_acc.tiles[r][c].data[i];
                        gate_acc.tiles[r][c].data[i]={swiglu(g.x,u.x),swiglu(g.y,u.y)};
                    }
                }
            }
            kt::warp::copy(rounded,gate_acc);
            if(writer)kt::tma::store_async_read_wait<0>();
            kt::warpgroup::sync(group+1);
            kt::warpgroup::store(shared.output[group][0],rounded);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            kt::warpgroup::sync(group+1);
            if(writer) {
                kt::tma::store_async(args.out,shared.output[group][0],
                                    {0,0,coord.x*8*R+group*4*R,nc});
                kt::tma::store_commit_group();
            }
        }
        if(writer)kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}

static void check(cudaError_t e) {
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}

struct PlanBase {
    virtual ~PlanBase()=default;
    virtual void launch(cudaStream_t)=0;
    virtual void info(int*)=0;
    virtual void rebind(
        void*x,void*xs,void*g,void*gs,void*u,void*us,void*o,void*go,void*uo)=0;
};

template<int R,int C,int BK,int S,bool Aux> struct Plan final:PlanBase {
    using CF=Config<R,C,BK,S,Aux>;
    typename CF::Args args;
    int grid,regs;
    Plan(void*x,void*xs,void*g,void*gs,void*u,void*us,void*o,void*go,void*uo,
         int m,int n,int k,int sms,int raster_group):args{
        Global<typename CF::ATile>(static_cast<kt::fp8e4m3*>(x),1,1,m,k),
        Global<typename CF::BTile>(static_cast<kt::fp8e4m3*>(g),1,1,n,k),
        Global<typename CF::BTile>(static_cast<kt::fp8e4m3*>(u),1,1,n,k),
        Global<typename CF::OTile>(static_cast<kt::bf16*>(o),1,1,m,n),
        Global<typename CF::OTile>(static_cast<kt::bf16*>(Aux?go:o),1,1,m,n),
        Global<typename CF::OTile>(static_cast<kt::bf16*>(Aux?uo:o),1,1,m,n),
        static_cast<uint32_t*>(xs),
        static_cast<uint8_t*>(gs),
        static_cast<uint8_t*>(us),
        m,n,(k+31)/32,(n+C-1)/C,(m+8*R-1)/(8*R),
        ((m+8*R-1)/(8*R))*((n+C-1)/C),(k+BK-1)/BK,
        std::min(raster_group,(m+8*R-1)/(8*R)),
        uint32_t((m+15)/16*16)} {
        if(k%CF::ATile::swizzle_bytes || k%CF::BTile::swizzle_bytes ||
           n%(CF::OTile::swizzle_bytes/2))
            throw std::runtime_error("K/N must align to TMA swizzle panels");
        int dev;check(cudaGetDevice(&dev));
        cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,dev));
        grid=std::min(args.total,sms>0?std::min(sms,prop.multiProcessorCount):prop.multiProcessorCount);
        check(cudaFuncSetAttribute(persistent_swiglu<R,C,BK,S,Aux>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr;
        check(cudaFuncGetAttributes(&attr,persistent_swiglu<R,C,BK,S,Aux>));
        regs=attr.numRegs;
        if(regs<168 || attr.localSizeBytes)
            throw std::runtime_error("Unexpected register pool or local memory");
    }
    void launch(cudaStream_t stream) override {
        persistent_swiglu<R,C,BK,S,Aux><<<grid,384,CF::smem,stream>>>(args);
        check(cudaGetLastError());
    }
    void info(int *d) override {
        d[0]=R;d[1]=C;d[2]=BK;d[3]=S;d[4]=CF::smem;
        d[5]=regs;d[6]=grid;d[7]=args.raster_group;d[8]=Aux;d[9]=CF::output_buffers;
    }
    void rebind(
        void*x,void*xs,void*g,void*gs,void*u,void*us,void*o,void*go,void*uo) override {
        args.a=Global<typename CF::ATile>((kt::fp8e4m3*)x,1,1,args.m,args.sc*32);
        args.gate=Global<typename CF::BTile>((kt::fp8e4m3*)g,1,1,args.n,args.sc*32);
        args.up=Global<typename CF::BTile>((kt::fp8e4m3*)u,1,1,args.n,args.sc*32);
        args.out=Global<typename CF::OTile>((kt::bf16*)o,1,1,args.m,args.n);
        args.gate_half=Global<typename CF::OTile>(
            (kt::bf16*)(Aux?go:o),1,1,args.m,args.n);
        args.up_linear=Global<typename CF::OTile>(
            (kt::bf16*)(Aux?uo:o),1,1,args.m,args.n);
        args.as=(uint32_t*)xs;
        args.gs=(uint8_t*)gs;
        args.us=(uint8_t*)us;
    }
};

static thread_local std::string error;
template<bool Aux> PlanBase*create(
    void*x,void*xs,void*g,void*gs,void*u,void*us,void*o,void*go,void*uo,
    int m,int n,int k,int config,int sms,int raster_group) {
#define CFG(I,R,C,K,S) case I:return new Plan<R,C,K,S,Aux>(x,xs,g,gs,u,us,o,go,uo,m,n,k,sms,raster_group)
    switch(config) {
#include "mxfp8_swiglu_configs.inc"
        default:throw std::runtime_error("Invalid configuration");
    }
#undef CFG
}
}

extern "C" const char *tk_mx_swiglu_error(){return fp8gemm::mxswiglu::error.c_str();}
extern "C" void *tk_mx_swiglu_create(
    void*x,void*xs,void*g,void*gs,void*u,void*us,void*o,
    int m,int n,int k,int config,int sms,int raster_group) {
    using namespace fp8gemm::mxswiglu;
    try {
        if(!x||!xs||!g||!gs||!u||!us||!o||m<=0||n<=0||k<=0||raster_group<1)
            throw std::runtime_error("Invalid arguments");
        return create<false>(x,xs,g,gs,u,us,o,o,o,m,n,k,config,sms,raster_group);
    }catch(const std::exception &e) {
        error=e.what();return nullptr;
    }
}
extern "C" void *tk_mx_swiglu_create_aux(
    void*x,void*xs,void*g,void*gs,void*u,void*us,void*o,void*go,void*uo,
    int m,int n,int k,int config,int sms,int raster_group) {
    using namespace fp8gemm::mxswiglu;
    try {
        if(!x||!xs||!g||!gs||!u||!us||!o||!go||!uo||
           m<=0||n<=0||k<=0||raster_group<1)
            throw std::runtime_error("Invalid arguments");
        return create<true>(x,xs,g,gs,u,us,o,go,uo,m,n,k,config,sms,raster_group);
    }catch(const std::exception &e) {
        error=e.what();return nullptr;
    }
}
extern "C" int tk_mx_swiglu_launch(fp8gemm::mxswiglu::PlanBase *p,void*stream) {
    try {p->launch(static_cast<cudaStream_t>(stream));return 0;}
    catch(const std::exception &e) {fp8gemm::mxswiglu::error=e.what();return -1;}
}
extern "C" int tk_mx_swiglu_rebind(
    fp8gemm::mxswiglu::PlanBase *p,
    void*x,void*xs,void*g,void*gs,void*u,void*us,void*o,void*go,void*uo) {
    try {p->rebind(x,xs,g,gs,u,us,o,go,uo);return 0;}
    catch(const std::exception &e) {fp8gemm::mxswiglu::error=e.what();return -1;}
}
extern "C" void tk_mx_swiglu_info(fp8gemm::mxswiglu::PlanBase *p,int*out) {
    p->info(out);
}
extern "C" void tk_mx_swiglu_destroy(fp8gemm::mxswiglu::PlanBase *p) {
    delete p;
}
