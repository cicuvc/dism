#include "kittens.cuh"
#include "linear_tma_buffer.cuh"
#include "mxfp8_quantize.cuh"
#include "tma_ring_pipe.cuh"
#include <stdexcept>
#include <string>

namespace fp8gemm::weight_quant {
namespace kt=kittens;
template<class Tile> using Global=kt::gl<typename Tile::dtype,-1,-1,-1,-1,kt::tma::descriptor<Tile,false>>;

template<int CW,int Stages,int Cols> struct Config {
    static_assert(CW==4 || CW==8);
    static_assert(Cols==32 || Cols==128);
    using Input=LinearTile<float,CW*32,Cols>;
    using Output=LinearTile<kt::fp8e4m3,128,Cols>;
    struct Shared {Input input[Stages]; Output output[CW/4][2];};
    struct Args {
        Global<Input> input;
        Global<Output> output;
        uint8_t *scales;
        int block_rows, block_cols, tile_cols, total;
    };
    static constexpr int smem=1024+sizeof(Shared);
    static_assert(smem<=101376);
};

__device__ __forceinline__ float4 load128(uint32_t address) {
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w) : "r"(address) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t pack_e4m3(float4 v,float reciprocal) {
    uint16_t lo,hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(v.y*reciprocal),"f"(v.x*reciprocal));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(v.w*reciprocal),"f"(v.z*reciprocal));
    return uint32_t(lo)|(uint32_t(hi)<<16);
}

template<int CW,int Stages,bool Redux,bool RoundBF16,int Cols=32>
__global__ __launch_bounds__((CW+4)*32,1) void persistent_quantize(
    const __grid_constant__ typename Config<CW,Stages,Cols>::Args args) {
    using CF=Config<CW,Stages,Cols>; using Shared=typename CF::Shared;
    constexpr int Vectors=Cols/4, ConsumerRegs=Cols==128 ? 232 : 128;
    extern __shared__ __align__(1024) unsigned char storage[];
    kt::shared_allocator<1024> alloc(reinterpret_cast<int*>(storage));
    int wid=threadIdx.x/32,group=kt::warpgroup::groupid(),warp=kt::warpgroup::warpid(),lane=threadIdx.x%32;
    bool producer=wid>=CW;
    MbarrierRingPipe pipe(alloc,BufferSet{&Shared::input},int(producer),
                         ProducerWarp<true,1>,CommEndInfo<CW,false,0>{});
    auto &shared=alloc.allocate<Shared>();
    if(producer) {
        kt::warpgroup::decrease_registers<32>();
        if(warp==0) {
            bool leader=kt::warp::elect_leader(); int last=0;
            for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
                auto packet=pipe.waitBuffer(0,shared.input);
                if(leader) {
                    kt::tma::expect_bytes(packet.getBarrier(),CW*32*Cols*4);
                    kt::tma::load_async(packet.template get<0>(),args.input,
                        {0,0,(task/args.tile_cols)*CW*32,(task%args.tile_cols)*Cols},packet.getBarrier());
                }
                last=packet.SlotIdx;
                packet.submitToNextAndTrigger(); pipe.moveNext();
            }
            pipe.waitSlot(last);
        }
    } else {
        kt::warpgroup::increase_registers<ConsumerRegs>();
        bool writer=warp==0 && kt::warp::elect_leader();
        int output_slot=0;
        for(int task=blockIdx.x;task<args.total;task+=gridDim.x) {
            auto packet=pipe.waitBuffer(1,shared.input);
            uint32_t input_base=uint32_t(__cvta_generic_to_shared(packet.template get<0>().data));
            float4 values[Vectors];
            float maximum=0.f;
#pragma unroll
            for(int i=0;i<Vectors;++i) {
                // Lane l loads elements 4*l..4*l+3 of a contiguous 128-float
                // segment. Cols=32 spans four rows/instruction; Cols=128 spans
                // one row/instruction and assigns each 32-column block 8 lanes.
                float4 v=load128(input_base+4*(wid*32*Cols+i*128+lane*4));
                if constexpr(RoundBF16) {
                    v.x=__bfloat162float(__float2bfloat16_rn(v.x));
                    v.y=__bfloat162float(__float2bfloat16_rn(v.y));
                    v.z=__bfloat162float(__float2bfloat16_rn(v.z));
                    v.w=__bfloat162float(__float2bfloat16_rn(v.w));
                }
                values[i]=v;
                float a=fmaxf(fabsf(v.x),fabsf(v.y));
                float b=fmaxf(fabsf(v.z),fabsf(v.w));
                maximum=fmaxf(maximum,fmaxf(a,b));
            }
            packet.submitToNextAndTrigger(); pipe.moveNext();
            // Finite, nonnegative FP32 amax has an unsigned-monotone exponent.
            uint32_t e=__float_as_uint(maximum)>>23;
            if constexpr(Redux) {
                uint32_t mask=Cols==128 ? 0xffu << (lane&24) : 0xffffffffu;
                asm("redux.sync.max.u32 %0, %1, %2;" : "=r"(e) : "r"(e),"r"(mask));
            } else {
#pragma unroll
                for(int d=(Cols==128 ? 4 : 16);d;d>>=1) e=max(e,__shfl_xor_sync(0xffffffff,e,d));
            }
            // E4M3 has 8 maximum exponent bits. Use the upper safe E4M3
            // exponent range instead of normalizing amax near one.
            uint8_t scale_e=uint8_t(quant::shift_e8m0_byte(e));
            float reciprocal=__uint_as_float(
                scale_e==254 ? 0x00400000u : (254u-scale_e)<<23);
            uint32_t q[Vectors];
#pragma unroll
            for(int i=0;i<Vectors;++i) q[i]=pack_e4m3(values[i],reciprocal);
            int block_row=(task/args.tile_cols)*CW+wid;
            int block_col=(task%args.tile_cols)*(Cols/32)+(Cols==128 ? lane/8 : 0);
            if((Cols==128 ? lane%8==0 : lane==0) && block_row<args.block_rows && block_col<args.block_cols)
                args.scales[size_t(block_row)*args.block_cols+block_col]=scale_e;

            // Double-buffer output. The previous tile may still be in flight;
            // only the older group must finish reading before slot reuse.
            if(writer) kt::tma::store_async_read_wait<1>();
            kt::warpgroup::sync(group+1);
            auto &out=shared.output[group][output_slot];
            uint32_t output_base=uint32_t(__cvta_generic_to_shared(out.data));
#pragma unroll
            for(int i=0;i<Vectors;++i) {
                uint32_t address=output_base+warp*32*Cols+i*128+lane*4;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(address),"r"(q[i]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            kt::warpgroup::sync(group+1);
            if(writer) {
                kt::tma::store_async(args.output,out,
                    {0,0,(task/args.tile_cols)*CW*32+group*128,(task%args.tile_cols)*Cols});
                kt::tma::store_commit_group();
            }
            output_slot^=1;
        }
        if(writer) kt::tma::store_async_wait<0>();
    }
    kt::warpgroup::sync(group+1);
}

static void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
struct PlanBase {virtual ~PlanBase()=default; virtual void launch(cudaStream_t)=0; virtual void info(int*)=0;};
template<int CW,int S,bool Redux,bool RoundBF16,int Cols=32> struct Plan final:PlanBase {
    using CF=Config<CW,S,Cols>;
    typename CF::Args args;
    int grid,regs;
    Plan(void *w,void *q,void *s,int n,int k,int ws,int qs,int sms):args{
        Global<typename CF::Input>(static_cast<float*>(w),1,1,n,k,kt::gl_strides{size_t(ws)*n,size_t(ws)*n,size_t(ws)}),
        Global<typename CF::Output>(static_cast<kt::fp8e4m3*>(q),1,1,n,k,kt::gl_strides{size_t(qs)*n,size_t(qs)*n,size_t(qs)}),
        static_cast<uint8_t*>(s),(n+31)/32,(k+31)/32,(k+Cols-1)/Cols,((n+CW*32-1)/(CW*32))*((k+Cols-1)/Cols)} {
        int device; check(cudaGetDevice(&device)); cudaDeviceProp prop; check(cudaGetDeviceProperties(&prop,device));
        check(cudaFuncSetAttribute(persistent_quantize<CW,S,Redux,RoundBF16,Cols>,cudaFuncAttributeMaxDynamicSharedMemorySize,CF::smem));
        cudaFuncAttributes attr; check(cudaFuncGetAttributes(&attr,persistent_quantize<CW,S,Redux,RoundBF16,Cols>));
        regs=attr.numRegs;
        if(attr.localSizeBytes || regs*(CW+4)<(Cols==128 ? 232 : 128)*CW+32*4)
            throw std::runtime_error("Unexpected local memory or insufficient WASP register pool");
        int resident;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
            persistent_quantize<CW,S,Redux,RoundBF16,Cols>,(CW+4)*32,CF::smem));
        grid=std::min(args.total,sms>0 ? sms : prop.multiProcessorCount*resident);
    }
    void launch(cudaStream_t stream) override {
        persistent_quantize<CW,S,Redux,RoundBF16,Cols><<<grid,(CW+4)*32,CF::smem,stream>>>(args);
        check(cudaGetLastError());
    }
    void info(int *d) override {d[0]=CW;d[1]=S;d[2]=Redux;d[3]=RoundBF16;d[4]=CF::smem;d[5]=regs;d[6]=grid;d[7]=Cols;}
};
static thread_local std::string error;
template<bool RoundBF16> PlanBase *create(void *w,void *q,void *s,int n,int k,int ws,int qs,int config,int sms) {
#define CFG(ID,CW,S,REDUX) case ID:return new Plan<CW,S,REDUX,RoundBF16>(w,q,s,n,k,ws,qs,sms)
    switch(config) {
        CFG(0,4,2,true); CFG(1,4,3,true); CFG(2,8,2,true);
        CFG(3,4,2,false); CFG(4,4,3,false); CFG(5,8,2,false);
        case 6:return new Plan<4,1,true,RoundBF16,128>(w,q,s,n,k,ws,qs,sms);
        case 7:return new Plan<4,1,false,RoundBF16,128>(w,q,s,n,k,ws,qs,sms);
        default:throw std::runtime_error("Unknown quantizer configuration");
    }
#undef CFG
}
}
extern "C" const char *tk_quant_error() {return fp8gemm::weight_quant::error.c_str();}
extern "C" void *tk_quant_create(void *w,void *q,void *s,int n,int k,int ws,int qs,int config,int bf16,int sms) {
    using namespace fp8gemm::weight_quant;
    try {
        if(!w || !q || !s || n<=0 || k<=0 || ws<k || qs<k || ws%4 || qs%16)
            throw std::runtime_error("Invalid shape/stride; input/output row strides must be 16-byte aligned");
        return bf16 ? create<true>(w,q,s,n,k,ws,qs,config,sms) : create<false>(w,q,s,n,k,ws,qs,config,sms);
    } catch(const std::exception& e) {error=e.what(); return nullptr;}
}
extern "C" int tk_quant_launch(void *p,void *stream) {
    try {static_cast<fp8gemm::weight_quant::PlanBase*>(p)->launch(static_cast<cudaStream_t>(stream));return 0;}
    catch(const std::exception& e) {fp8gemm::weight_quant::error=e.what();return -1;}
}
extern "C" void tk_quant_info(void *p,int *d) {static_cast<fp8gemm::weight_quant::PlanBase*>(p)->info(d);}
extern "C" void tk_quant_destroy(void *p) {delete static_cast<fp8gemm::weight_quant::PlanBase*>(p);}
