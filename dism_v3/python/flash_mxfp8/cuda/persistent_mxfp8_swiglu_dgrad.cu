// Fused dX[M,K] = dG[M,N] @ Wg[N,K] + dU[M,N] @ Wu[N,K].
// Both gradients use packed per-row/N32 E8M0 scales; both weights use the
// forward 32x32 scale grid. The swapAB path shares one FP32 accumulator and
// writes dX once as BF16 through TMA.
#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "fp8_shared_load.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm::mxswd {
template<class Tile> using Global=kt::gl<typename Tile::dtype,-1,-1,-1,-1,Tile>;

__device__ __forceinline__ int2 raster(int task,int columns,int rows,int group) {
    int span=group*columns,first=task/span*group,count=min(group,rows-first),local=task%span;
    return make_int2(first+local%count,local/count);
}

template<int R,int C,int BN,int S> struct Config {
    using A=kt::st_fp8e4m3<4*R,BN>;
    using B=kt::st_fp8e4m3<BN,C>;
    using O=kt::st_bf<4*R,C>;
    struct Packet {
        A gate[2],up[2];
        B weight_gate,weight_up;
    };
    struct Shared {
        Packet input[S];
        O output[2];
    };
    struct Args {
        Global<A> gate,up;
        Global<B> weight_gate,weight_up;
        Global<O> out;
        const uint32_t *gate_scale,*up_scale;
        const uint8_t *weight_gate_scale,*weight_up_scale;
        int m,k,n,nrows,ncols,total,steps,group,as_stride,bs_stride;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=101376);
    static_assert((BN/32)*(C/32)<=32);

    struct ScalePrefetch {
        uint32_t gate[R/16],up[R/16],weight_gate,weight_up;
        __device__ __forceinline__ void load(
            const Args &args,int nr,int kc,int step,int lane) {
            int block=step*(BN/32)/4;
#pragma unroll
            for(int i=0;i<R/16;i++) {
                int row=nr+i*16+lane/4+8*(lane&1);
                bool valid=row<args.m && step<args.steps;
                gate[i]=valid
                    ?args.gate_scale[size_t(block)*args.as_stride+row]:0;
                up[i]=valid
                    ?args.up_scale[size_t(block)*args.as_stride+row]:0;
            }
            int row=step*(BN/32)+(lane/(C/32))%(BN/32),col=kc/32+lane%(C/32);
            bool valid=row<(args.n+31)/32 && col<args.bs_stride;
            weight_gate=valid
                ?args.weight_gate_scale[size_t(row)*args.bs_stride+col]:0;
            weight_up=valid
                ?args.weight_up_scale[size_t(row)*args.bs_stride+col]:0;
        }
    };
};

template<int R,int C,int BN>
__device__ __forceinline__ void transpose_weight(
    kt::rt_fp8e4m3<C,BN> &weight_a,
    const kt::rt<kt::fp8e4m3,BN,C,kt::ducks::rt_layout::col> &weight_b) {
#pragma unroll
    for(int j=0;j<C/16;j++)
#pragma unroll
        for(int t=0;t<BN/32;t++)
#pragma unroll
            for(int v=0;v<4;v++)
                weight_a.tiles[j][t].data[v]=weight_b.tiles[t][j].data[v];
}

template<int R,int C,int BN,int S>
__global__ __launch_bounds__(384,1) void persistent_swiglu_dgrad(
    const __grid_constant__ typename Config<R,C,BN,S>::Args args) {
    using CF=Config<R,C,BN,S>;
    using Shared=typename CF::Shared;
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
                    auto p=pipe.waitBuffer(0,shared.input);
                    auto &tile=p.template get<0>();
                    if(leader) {
                        auto &bar=p.getBarrier();
                        kt::tma::expect_bytes(bar,(16*R+2*C)*BN);
                        kt::tma::load_async(tile.gate[0],args.gate,{0,0,nr,step*BN},bar);
                        kt::tma::load_async(tile.gate[1],args.gate,{0,0,nr+4*R,step*BN},bar);
                        kt::tma::load_async(tile.up[0],args.up,{0,0,nr,step*BN},bar);
                        kt::tma::load_async(tile.up[1],args.up,{0,0,nr+4*R,step*BN},bar);
                        kt::tma::load_async(tile.weight_gate,args.weight_gate,
                                            {0,0,step*BN,kc},bar);
                        kt::tma::load_async(tile.weight_up,args.weight_up,
                                            {0,0,step*BN,kc},bar);
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
            kt::rt_fl<C,R> accum;
            kt::warp::rt_maps::zero(accum);
            typename CF::ScalePrefetch next;
            next.load(args,nr,kc,0,lane);
            for(int step=0;step<args.steps;step++) {
                auto p=pipe.waitBuffer(2,shared.input);
                auto &tile=p.template get<0>();
                typename CF::ScalePrefetch current=next;
                next.load(args,nr,kc,step+1,lane);
                {
                    kt::rt_fp8e4m3<R,BN> a;
                    kt::rt<kt::fp8e4m3,BN,C,kt::ducks::rt_layout::col> b;
                    kt::mxfp8_scale<R,BN> as;
                    kt::mxfp8_scale<C,BN> bs;
                    load_fp8_factored<4,true>(a,tile.gate[group]);
                    kt::warp::load(b,tile.weight_gate);
#pragma unroll
                    for(int i=0;i<R/16;i++)
#pragma unroll
                        for(int t=0;t<BN/32;t++)
                            as.data[i][t]=
                                current.gate[i]>>(8*((step*(BN/32)+t)%4));
#pragma unroll
                    for(int j=0;j<C/16;j++)
#pragma unroll
                        for(int t=0;t<BN/32;t++)
                            bs.data[j][t]=__shfl_sync(
                                0xffffffff,current.weight_gate,t*(C/32)+j/2);
                    kt::rt_fp8e4m3<C,BN> weight_a;
                    transpose_weight<R,C,BN>(weight_a,b);
                    kt::warp::wmma::mma_ABt_block_scaled(
                        accum,weight_a,a,accum,bs,as);
                }
                {
                    kt::rt_fp8e4m3<R,BN> a;
                    kt::rt<kt::fp8e4m3,BN,C,kt::ducks::rt_layout::col> b;
                    kt::mxfp8_scale<R,BN> as;
                    kt::mxfp8_scale<C,BN> bs;
                    load_fp8_factored<4,true>(a,tile.up[group]);
                    kt::warp::load(b,tile.weight_up);
#pragma unroll
                    for(int i=0;i<R/16;i++)
#pragma unroll
                        for(int t=0;t<BN/32;t++)
                            as.data[i][t]=
                                current.up[i]>>(8*((step*(BN/32)+t)%4));
#pragma unroll
                    for(int j=0;j<C/16;j++)
#pragma unroll
                        for(int t=0;t<BN/32;t++)
                            bs.data[j][t]=__shfl_sync(
                                0xffffffff,current.weight_up,t*(C/32)+j/2);
                    kt::rt_fp8e4m3<C,BN> weight_a;
                    transpose_weight<R,C,BN>(weight_a,b);
                    p.submitToNextAndTrigger();pipe.moveNext();
                    kt::warp::wmma::mma_ABt_block_scaled(
                        accum,weight_a,a,accum,bs,as);
                }
            }
            if(writer)kt::tma::store_async_read_wait<0>();
            kt::warpgroup::sync(group+1);
            kt::rt<float,R,C,kt::ducks::rt_layout::col> transposed;
#pragma unroll
            for(int i=0;i<R/16;i++)
#pragma unroll
                for(int j=0;j<C/16;j++)
#pragma unroll
                    for(int v=0;v<4;v++)
                        transposed.tiles[i][j].data[v]=accum.tiles[j][i].data[v];
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

static void check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}

struct PlanBase {
    virtual ~PlanBase()=default;
    virtual void launch(cudaStream_t)=0;
    virtual void info(int*)=0;
    virtual void rebind(
        void*gate,void*up,void*weight_gate,void*weight_up,
        void*gate_scale,void*up_scale,
        void*weight_gate_scale,void*weight_up_scale,void*out)=0;
};

template<int R,int C,int BN,int S> struct Plan final:PlanBase {
    using CF=Config<R,C,BN,S>;
    typename CF::Args args;
    int grid,regs;
    Plan(
        void*gate,void*up,void*weight_gate,void*weight_up,
        void*gate_scale,void*up_scale,
        void*weight_gate_scale,void*weight_up_scale,
        void*out,int m,int k,int n,int sms,int group):
        args{
            Global<typename CF::A>((kt::fp8e4m3*)gate,1,1,m,n),
            Global<typename CF::A>((kt::fp8e4m3*)up,1,1,m,n),
            Global<typename CF::B>((kt::fp8e4m3*)weight_gate,1,1,n,k),
            Global<typename CF::B>((kt::fp8e4m3*)weight_up,1,1,n,k),
            Global<typename CF::O>((kt::bf16*)out,1,1,m,k),
            (uint32_t*)gate_scale,(uint32_t*)up_scale,
            (uint8_t*)weight_gate_scale,(uint8_t*)weight_up_scale,
            m,k,n,(m+8*R-1)/(8*R),(k+C-1)/C,
            ((m+8*R-1)/(8*R))*((k+C-1)/C),(n+BN-1)/BN,
            std::min(group,(m+8*R-1)/(8*R)),(m+15)/16*16,k/32} {
        if(n%CF::A::swizzle_bytes || k%CF::B::swizzle_bytes ||
           k%(CF::O::swizzle_bytes/2))
            throw std::runtime_error("N/K must align to TMA swizzle panels");
        int device;check(cudaGetDevice(&device));
        cudaDeviceProp properties;
        check(cudaGetDeviceProperties(&properties,device));
        grid=std::min(args.total,sms>0
            ?std::min(sms,properties.multiProcessorCount)
            :properties.multiProcessorCount);
        check(cudaFuncSetAttribute(
            persistent_swiglu_dgrad<R,C,BN,S>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attributes;
        check(cudaFuncGetAttributes(
            &attributes,persistent_swiglu_dgrad<R,C,BN,S>));
        regs=attributes.numRegs;
        if(attributes.localSizeBytes || regs<168)
            throw std::runtime_error("Unexpected register pool or local memory");
    }
    void launch(cudaStream_t stream) override {
        persistent_swiglu_dgrad<R,C,BN,S><<<grid,384,CF::smem,stream>>>(args);
        check(cudaGetLastError());
    }
    void info(int*out) override {
        out[0]=R;out[1]=C;out[2]=BN;out[3]=S;
        out[4]=CF::smem;out[5]=regs;out[6]=grid;out[7]=args.group;
    }
    void rebind(
        void*gate,void*up,void*weight_gate,void*weight_up,
        void*gate_scale,void*up_scale,
        void*weight_gate_scale,void*weight_up_scale,void*out) override {
        args.gate=Global<typename CF::A>((kt::fp8e4m3*)gate,1,1,args.m,args.n);
        args.up=Global<typename CF::A>((kt::fp8e4m3*)up,1,1,args.m,args.n);
        args.weight_gate=Global<typename CF::B>(
            (kt::fp8e4m3*)weight_gate,1,1,args.n,args.k);
        args.weight_up=Global<typename CF::B>(
            (kt::fp8e4m3*)weight_up,1,1,args.n,args.k);
        args.out=Global<typename CF::O>((kt::bf16*)out,1,1,args.m,args.k);
        args.gate_scale=(uint32_t*)gate_scale;
        args.up_scale=(uint32_t*)up_scale;
        args.weight_gate_scale=(uint8_t*)weight_gate_scale;
        args.weight_up_scale=(uint8_t*)weight_up_scale;
    }
};

static thread_local std::string error;
}

extern "C" const char *tk_swd_error() {
    return fp8gemm::mxswd::error.c_str();
}
extern "C" void *tk_swd_create(
    void*gate,void*up,void*weight_gate,void*weight_up,
    void*gate_scale,void*up_scale,
    void*weight_gate_scale,void*weight_up_scale,
    void*out,int m,int k,int n,int config,int sms,int group) {
    using namespace fp8gemm::mxswd;
    try {
        if(!gate||!up||!weight_gate||!weight_up||!gate_scale||!up_scale||
           !weight_gate_scale||!weight_up_scale||!out||
           m<=0||k<=0||n<=0||group<=0||sms<0)
            throw std::runtime_error("Invalid arguments");
#define CFG(I,R,C,K,S) case I:return new Plan<R,C,K,S>(gate,up,weight_gate,weight_up,gate_scale,up_scale,weight_gate_scale,weight_up_scale,out,m,k,n,sms,group)
        switch(config) {
#include "mxfp8_swiglu_dgrad_configs.inc"
            default:throw std::runtime_error("Invalid configuration");
        }
#undef CFG
    }catch(const std::exception &exception) {
        error=exception.what();return nullptr;
    }
}
extern "C" int tk_swd_launch(fp8gemm::mxswd::PlanBase *plan,void*stream) {
    try {
        plan->launch(static_cast<cudaStream_t>(stream));
        return 0;
    }catch(const std::exception &exception) {
        fp8gemm::mxswd::error=exception.what();return -1;
    }
}
extern "C" int tk_swd_rebind(
    fp8gemm::mxswd::PlanBase *plan,
    void*gate,void*up,void*weight_gate,void*weight_up,
    void*gate_scale,void*up_scale,
    void*weight_gate_scale,void*weight_up_scale,void*out) {
    try {
        plan->rebind(
            gate,up,weight_gate,weight_up,
            gate_scale,up_scale,weight_gate_scale,weight_up_scale,out);
        return 0;
    }catch(const std::exception &exception) {
        fp8gemm::mxswd::error=exception.what();return -1;
    }
}
extern "C" void tk_swd_info(fp8gemm::mxswd::PlanBase *plan,int*out) {
    plan->info(out);
}
extern "C" void tk_swd_destroy(fp8gemm::mxswd::PlanBase *plan) {
    delete plan;
}
