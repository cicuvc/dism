// Fused BF16 SwiGLU backward and two-direction MXFP8 quantization.
// Each warp walks 32x32 tiles. Inputs use a double-buffered cp.async pipeline:
// a warp copies each tile into a transposed, padded shared layout using
// half-warp-coalesced 4-byte transfers, then lane r reads row r without
// shared-memory bank conflicts. No BF16 gradients are written to global
// memory; only E4M3 data and E8M0 scales are stored.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "mxfp8_quantize.cuh"
#include "silu_bwd_common.cuh"

namespace fp8gemm::silu_bwd_quantize {
using fp8gemm::silu_bwd::PairGrad;
using fp8gemm::silu_bwd::backward_pair;
using fp8gemm::silu_bwd::bf16x2_scale;
using fp8gemm::silu_bwd::bf16x2_scale_pair;
using fp8gemm::silu_bwd::cvt_e4m3x2_bf16x2;
using fp8gemm::quant::shift_e8m0;
using fp8gemm::quant::shift_e8m0_pair;

static thread_local std::string error;

constexpr int kChunks=4;                  // 16-byte chunks in one row
constexpr int kRowWords=16;               // 4-byte words in one row
constexpr int kTileWords=32*kRowWords;
constexpr int kStages=2;
constexpr int kWarps=8;
template<bool Direct>
constexpr int kInputs=Direct?1:3;
template<bool Direct>
constexpr int kSharedBytes=
    kWarps*kStages*kInputs<Direct>*kTileWords*sizeof(uint32_t);

struct Args {
    const uint16_t *dy,*half_gate,*up;
    uint8_t *gate_row,*up_row,*gate_col,*up_col;
    uint8_t *gate_row_scale,*up_row_scale,*gate_col_scale,*up_col_scale;
    int m,n,mp,mp16,nblocks,tile_columns,total_tiles;
};

__device__ __forceinline__ uint32_t max_exponent(
    uint32_t values,uint32_t current=0) {
    current=max(current,values&0x7f80u);
    return max(current,(values>>16)&0x7f80u);
}

__device__ __forceinline__ uint32_t swizzled_addr(
    const uint32_t *base,int row,int column) {
    uint32_t address=uint32_t(__cvta_generic_to_shared(base))
        +uint32_t((row*32+column)*sizeof(uint16_t));
    return address^((address>>2)&0x70u);
}

__device__ __forceinline__ uint4 load_shared_v4(uint32_t address) {
    uint4 out;
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
                 : "r"(address));
    return out;
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int Pending>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(Pending) : "memory");
}

// Shared layout is logically row-major. Every 16-byte chunk is placed at
// address ^ ((address >> 2) & 0x70), preserving 16-byte alignment.
template<bool Direct,bool Transposed=false>
__device__ __forceinline__ void load_tile(
    uint32_t *shared,const Args &args,int tile,int lane) {
    int tile_col=tile%args.tile_columns,tile_row=tile/args.tile_columns;
    int block_row=tile_row*32,block_col=tile_col*32;
    uint32_t *dy_src=shared;
    uint32_t *half_gate_src=Direct?nullptr:shared+kTileWords;
    uint32_t *up_src=Direct?nullptr:shared+2*kTileWords;
#pragma unroll
    for(int iteration=0;iteration<4;iteration++) {
        int linear=lane+iteration*32;
        int row=linear/kChunks;
        int chunk=linear%kChunks;
        int global_row=block_row+row;
        size_t global_col=size_t(block_col+chunk*8);
        uint32_t dy_addr=swizzled_addr(dy_src,row,chunk*8);
        if(global_row<args.m) {
            uint32_t shared_addr;
            shared_addr=dy_addr;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                         :: "r"(shared_addr),
                            "l"(args.dy+size_t(global_row)*args.n+global_col)
                         : "memory");
            if constexpr(!Direct) {
                shared_addr=swizzled_addr(up_src,row,chunk*8);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                             :: "r"(shared_addr),
                                "l"(args.up+size_t(global_row)*args.n+global_col)
                             : "memory");
                shared_addr=swizzled_addr(half_gate_src,row,chunk*8);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                             :: "r"(shared_addr),
                                "l"(args.half_gate+size_t(global_row)*args.n+global_col)
                             : "memory");
            }
        } else {
            uint4 zero={0,0,0,0};
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :: "r"(dy_addr), "r"(zero.x), "r"(zero.y),
                            "r"(zero.z), "r"(zero.w) : "memory");
            if constexpr(!Direct) {
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                             :: "r"(swizzled_addr(up_src,row,chunk*8)),
                                "r"(zero.x), "r"(zero.y),
                                "r"(zero.z), "r"(zero.w) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                             :: "r"(swizzled_addr(half_gate_src,row,chunk*8)),
                                "r"(zero.x), "r"(zero.y),
                                "r"(zero.z), "r"(zero.w) : "memory");
            }
        }
    }
    cp_async_commit();
}

template<bool Direct,bool Transposed=false>
__device__ __forceinline__ void process_tile(
    const uint32_t *shared,const Args &args,int tile,int lane) {
    int tile_col=tile%args.tile_columns,tile_row=tile/args.tile_columns;
    int block_row=tile_row*32,block_col=tile_col*32;
    int row=block_row+lane;
    const uint32_t *dy_src=shared;
    const uint32_t *half_gate_src=Direct?nullptr:shared+kTileWords;
    const uint32_t *up_src=Direct?nullptr:shared+2*kTileWords;

    uint32_t gate_values[16],up_values[16]={0};
    uint32_t gate_exponent=0,up_exponent=0;
#pragma unroll
    for(int vector=0;vector<4;vector++) {
        int column=vector*8;
        uint4 dy=load_shared_v4(swizzled_addr(dy_src,lane,column));
        PairGrad p0,p1,p2,p3;
        if constexpr(Direct) {
            p0={dy.x,0};p1={dy.y,0};p2={dy.z,0};p3={dy.w,0};
        } else {
            uint4 half_gate=load_shared_v4(
                swizzled_addr(half_gate_src,lane,column));
            uint4 up=load_shared_v4(swizzled_addr(up_src,lane,column));
            p0=backward_pair(dy.x,half_gate.x,up.x);
            p1=backward_pair(dy.y,half_gate.y,up.y);
            p2=backward_pair(dy.z,half_gate.z,up.z);
            p3=backward_pair(dy.w,half_gate.w,up.w);
        }
        gate_values[vector*4]=p0.gate;
        gate_values[vector*4+1]=p1.gate;
        gate_values[vector*4+2]=p2.gate;
        gate_values[vector*4+3]=p3.gate;
        up_values[vector*4]=p0.up;
        up_values[vector*4+1]=p1.up;
        up_values[vector*4+2]=p2.up;
        up_values[vector*4+3]=p3.up;
        if(row<args.m) {
            gate_exponent=max_exponent(p0.gate,gate_exponent);
            gate_exponent=max_exponent(p1.gate,gate_exponent);
            gate_exponent=max_exponent(p2.gate,gate_exponent);
            gate_exponent=max_exponent(p3.gate,gate_exponent);
            if constexpr(!Direct) {
                up_exponent=max_exponent(p0.up,up_exponent);
                up_exponent=max_exponent(p1.up,up_exponent);
                up_exponent=max_exponent(p2.up,up_exponent);
                up_exponent=max_exponent(p3.up,up_exponent);
            }
        }
    }

    uint8_t gate_row_exp=uint8_t(shift_e8m0(gate_exponent)>>7);
    uint8_t up_row_exp=uint8_t(shift_e8m0(up_exponent)>>7);
    if(row<args.m) {
        uint16_t gate_out[16],up_out[16];
#pragma unroll
        for(int pair=0;pair<16;pair++) {
            gate_out[pair]=cvt_e4m3x2_bf16x2(
                bf16x2_scale(gate_values[pair],gate_row_exp));
            if constexpr(!Direct)
                up_out[pair]=cvt_e4m3x2_bf16x2(
                    bf16x2_scale(up_values[pair],up_row_exp));
        }
        auto *gate_dst=reinterpret_cast<uint4*>(
            args.gate_row+size_t(row)*args.n+block_col);
        auto *up_dst=reinterpret_cast<uint4*>(
            args.up_row+size_t(row)*args.n+block_col);
#pragma unroll
        for(int vector=0;vector<2;vector++) {
            int pair=vector*8;
            gate_dst[vector]={uint32_t(gate_out[pair])|(uint32_t(gate_out[pair+1])<<16),
                              uint32_t(gate_out[pair+2])|(uint32_t(gate_out[pair+3])<<16),
                              uint32_t(gate_out[pair+4])|(uint32_t(gate_out[pair+5])<<16),
                              uint32_t(gate_out[pair+6])|(uint32_t(gate_out[pair+7])<<16)};
            if constexpr(!Direct)
                up_dst[vector]={uint32_t(up_out[pair])|(uint32_t(up_out[pair+1])<<16),
                                uint32_t(up_out[pair+2])|(uint32_t(up_out[pair+3])<<16),
                                uint32_t(up_out[pair+4])|(uint32_t(up_out[pair+5])<<16),
                                uint32_t(up_out[pair+6])|(uint32_t(up_out[pair+7])<<16)};
        }
    }

    if(row<args.mp16) {
        size_t offset=size_t(tile_col/4)*args.mp16*4+size_t(row)*4+(tile_col&3);
        args.gate_row_scale[offset]=gate_row_exp;
        if constexpr(!Direct)args.up_row_scale[offset]=up_row_exp;
    }

    if constexpr(Direct) {
        uint16_t column_out[16];
#pragma unroll
        for(int pair=0;pair<16;pair++) {
            uint32_t low=gate_values[pair]&0xffffu;
            uint32_t high=(gate_values[pair]>>16)&0xffffu;
            uint32_t low_exp=__reduce_max_sync(0xffffffffu,low&0x7f80u);
            uint32_t high_exp=__reduce_max_sync(0xffffffffu,high&0x7f80u);
            uint32_t shifted_pair=shift_e8m0_pair(low_exp|(high_exp<<16));
            uint8_t low_scale=uint8_t(shifted_pair>>7);
            uint8_t high_scale=uint8_t(shifted_pair>>23);
            int feature_low=block_col+pair*2;
            if(lane==0) {
                size_t offset_low=
                    size_t(tile_row/4)*args.n*4+size_t(feature_low)*4+(tile_row&3);
                args.gate_col_scale[offset_low]=low_scale;
                args.gate_col_scale[offset_low+4]=high_scale;
            }
            if constexpr(Transposed) {
                uint32_t low_next=__shfl_down_sync(0xffffffffu,low,1);
                uint32_t high_next=__shfl_down_sync(0xffffffffu,high,1);
                if((lane&1)==0) {
                    uint16_t low_q=cvt_e4m3x2_bf16x2(
                        bf16x2_scale(low|(low_next<<16),low_scale));
                    uint16_t high_q=cvt_e4m3x2_bf16x2(
                        bf16x2_scale(high|(high_next<<16),high_scale));
                    size_t index=size_t(feature_low)*args.mp+block_row+lane;
                    *reinterpret_cast<uint16_t*>(args.gate_col+index)=low_q;
                    *reinterpret_cast<uint16_t*>(
                        args.gate_col+index+args.mp)=high_q;
                }
            } else {
                column_out[pair]=cvt_e4m3x2_bf16x2(
                    bf16x2_scale_pair(gate_values[pair],low_scale,high_scale));
            }
        }
        if constexpr(!Transposed) {
            auto *column_dst=reinterpret_cast<uint4*>(
                args.gate_col+size_t(block_row+lane)*args.n+block_col);
#pragma unroll
            for(int vector=0;vector<2;vector++) {
                int pair=vector*8;
                column_dst[vector]={
                    uint32_t(column_out[pair])|
                        (uint32_t(column_out[pair+1])<<16),
                    uint32_t(column_out[pair+2])|
                        (uint32_t(column_out[pair+3])<<16),
                    uint32_t(column_out[pair+4])|
                        (uint32_t(column_out[pair+5])<<16),
                    uint32_t(column_out[pair+6])|
                        (uint32_t(column_out[pair+7])<<16)};
            }
        }
    } else {
        // Fused backward gradients use feature-major [N,MP] for wgrad A.
#pragma unroll
        for(int pair=0;pair<16;pair++) {
#pragma unroll
            for(int half=0;half<2;half++) {
                uint32_t gate_value=(gate_values[pair]>>(16*half))&0xffffu;
                uint32_t up_value=(up_values[pair]>>(16*half))&0xffffu;
                uint32_t gate_exp=__reduce_max_sync(
                    0xffffffffu,gate_value&0x7f80u);
                uint32_t up_exp=__reduce_max_sync(
                    0xffffffffu,up_value&0x7f80u);
                uint32_t shifted_pair=shift_e8m0_pair(gate_exp|(up_exp<<16));
                uint8_t gate_scale=uint8_t(shifted_pair>>7);
                uint8_t up_scale=uint8_t(shifted_pair>>23);
                uint32_t gate_next=__shfl_down_sync(0xffffffffu,gate_value,1);
                uint32_t up_next=__shfl_down_sync(0xffffffffu,up_value,1);
                int feature=block_col+pair*2+half;
                if(lane==0) {
                    size_t offset=
                        size_t(tile_row/4)*args.n*4+size_t(feature)*4+(tile_row&3);
                    args.gate_col_scale[offset]=gate_scale;
                    args.up_col_scale[offset]=up_scale;
                }
                if((lane&1)==0) {
                    uint32_t gate_pair=gate_value|(gate_next<<16);
                    uint32_t up_pair=up_value|(up_next<<16);
                    uint16_t gate_q=cvt_e4m3x2_bf16x2(
                        bf16x2_scale(gate_pair,gate_scale));
                    uint16_t up_q=cvt_e4m3x2_bf16x2(
                        bf16x2_scale(up_pair,up_scale));
                    size_t gate_index=size_t(feature)*args.mp+block_row+lane;
                    size_t up_index=size_t(feature)*args.mp+block_row+lane;
                    *reinterpret_cast<uint16_t*>(args.gate_col+gate_index)=gate_q;
                    *reinterpret_cast<uint16_t*>(args.up_col+up_index)=up_q;
                }
            }
        }
    }
}

template<bool Direct,bool Transposed=false>
__global__ __launch_bounds__(256) void silu_bwd_quantize_kernel(Args args) {
    extern __shared__ __align__(16) uint8_t storage[];
    int warp=threadIdx.x/32,lane=threadIdx.x&31;
    int stride=gridDim.x*kWarps;
    int tile=blockIdx.x*kWarps+warp;
    if(tile>=args.total_tiles)return;
    auto *warp_shared=reinterpret_cast<uint32_t*>(
        storage+warp*kStages*kInputs<Direct>*kTileWords*sizeof(uint32_t));

    int stage=0,next=tile+stride;
    load_tile<Direct,Transposed>(warp_shared,args,tile,lane);
    while(tile<args.total_tiles) {
        bool has_next=next<args.total_tiles;
        if(has_next)
            load_tile<Direct,Transposed>(
                warp_shared+(stage^1)*kInputs<Direct>*kTileWords,
                args,next,lane);
        if(has_next)cp_async_wait<1>();
        else cp_async_wait<0>();
        __syncwarp();
        process_tile<Direct,Transposed>(
            warp_shared+stage*kInputs<Direct>*kTileWords,args,tile,lane);
        __syncwarp();
        tile=next;next+=stride;stage^=1;
    }
}

static void check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}

struct PlanBase {
    virtual ~PlanBase()=default;
    virtual void launch(cudaStream_t)=0;
    virtual void info(int*)=0;
    virtual void rebind(
        void*dy,void*half_gate,void*up,
        void*gate_row,void*gate_row_scale,void*up_row,void*up_row_scale,
        void*gate_col,void*gate_col_scale,void*up_col,void*up_col_scale)=0;
};

template<bool Direct,bool Transposed=false>
struct Plan final:PlanBase {
    Args args;
    int blocks;
    Plan(
        void*dy,void*half_gate,void*up,
        void*gate_row,void*gate_row_scale,void*up_row,void*up_row_scale,
        void*gate_col,void*gate_col_scale,void*up_col,void*up_col_scale,
        int m,int n):
        args{
            static_cast<const uint16_t*>(dy),
            static_cast<const uint16_t*>(half_gate),
            static_cast<const uint16_t*>(up),
            static_cast<uint8_t*>(gate_row),
            static_cast<uint8_t*>(up_row),
            static_cast<uint8_t*>(gate_col),
            static_cast<uint8_t*>(up_col),
            static_cast<uint8_t*>(gate_row_scale),
            static_cast<uint8_t*>(up_row_scale),
            static_cast<uint8_t*>(gate_col_scale),
            static_cast<uint8_t*>(up_col_scale),
            m,n,(m+127)/128*128,(m+15)/16*16,n/32,n/32,
            (n/32)*((m+127)/128*128/32)} {
        int device;check(cudaGetDevice(&device));
        cudaDeviceProp properties;check(cudaGetDeviceProperties(&properties,device));
        int tile_blocks=(args.total_tiles+kWarps-1)/kWarps;
        blocks=std::min(
            tile_blocks,
            Direct?properties.multiProcessorCount*3
                  :properties.multiProcessorCount);
        check(cudaFuncSetAttribute(
            silu_bwd_quantize_kernel<Direct,Transposed>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes<Direct>));
    }
    void launch(cudaStream_t stream) override {
        silu_bwd_quantize_kernel<Direct,Transposed>
            <<<blocks,256,kSharedBytes<Direct>,stream>>>(args);
    }
    void info(int*out) override {
        out[0]=blocks;out[1]=256;out[2]=kSharedBytes<Direct>;
    }
    void rebind(
        void*dy,void*half_gate,void*up,
        void*gate_row,void*gate_row_scale,void*up_row,void*up_row_scale,
        void*gate_col,void*gate_col_scale,void*up_col,void*up_col_scale) override {
        args.dy=static_cast<const uint16_t*>(dy);
        args.half_gate=static_cast<const uint16_t*>(half_gate);
        args.up=static_cast<const uint16_t*>(up);
        args.gate_row=static_cast<uint8_t*>(gate_row);
        args.up_row=static_cast<uint8_t*>(up_row);
        args.gate_col=static_cast<uint8_t*>(gate_col);
        args.up_col=static_cast<uint8_t*>(up_col);
        args.gate_row_scale=static_cast<uint8_t*>(gate_row_scale);
        args.up_row_scale=static_cast<uint8_t*>(up_row_scale);
        args.gate_col_scale=static_cast<uint8_t*>(gate_col_scale);
        args.up_col_scale=static_cast<uint8_t*>(up_col_scale);
    }
};

}

extern "C" const char *tk_silu_bwd_q_error() {
    return fp8gemm::silu_bwd_quantize::error.c_str();
}

extern "C" void *tk_silu_bwd_q_create(
    void*dy,void*half_gate,void*up,
    void*gate_row,void*gate_row_scale,void*up_row,void*up_row_scale,
    void*gate_col,void*gate_col_scale,void*up_col,void*up_col_scale,
    int m,int n) {
    using namespace fp8gemm::silu_bwd_quantize;
    try {
        if(!dy||!half_gate||!up||!gate_row||!gate_row_scale||
           !up_row||!up_row_scale||!gate_col||!gate_col_scale||
           !up_col||!up_col_scale||m<=0||n<=0||n%32)
            throw std::runtime_error("Invalid arguments or N not multiple of 32");
        return new Plan<false>(
            dy,half_gate,up,
            gate_row,gate_row_scale,up_row,up_row_scale,
            gate_col,gate_col_scale,up_col,up_col_scale,m,n);
    }catch(const std::exception &exception) {
        error=exception.what();return nullptr;
    }
}

extern "C" void *tk_silu_bwd_q_create_direct(
    void*x,void*row_fp8,void*row_scale,void*column_fp8,void*column_scale,
    int m,int n) {
    using namespace fp8gemm::silu_bwd_quantize;
    try {
        if(!x||!row_fp8||!row_scale||!column_fp8||!column_scale||
           m<=0||n<=0||n%32)
            throw std::runtime_error("Invalid arguments or N not multiple of 32");
        return new Plan<true>(
            x,x,x,row_fp8,row_scale,row_fp8,row_scale,
            column_fp8,column_scale,column_fp8,column_scale,m,n);
    }catch(const std::exception &exception) {
        error=exception.what();return nullptr;
    }
}

extern "C" void *tk_silu_bwd_q_create_direct_transposed(
    void*x,void*row_fp8,void*row_scale,void*column_fp8,void*column_scale,
    int m,int n) {
    using namespace fp8gemm::silu_bwd_quantize;
    try {
        if(!x||!row_fp8||!row_scale||!column_fp8||!column_scale||
           m<=0||n<=0||n%32)
            throw std::runtime_error("Invalid arguments or N not multiple of 32");
        return new Plan<true,true>(
            x,x,x,row_fp8,row_scale,row_fp8,row_scale,
            column_fp8,column_scale,column_fp8,column_scale,m,n);
    }catch(const std::exception &exception) {
        error=exception.what();return nullptr;
    }
}

extern "C" int tk_silu_bwd_q_launch(void*opaque,void*stream) {
    using namespace fp8gemm::silu_bwd_quantize;
    try {
        static_cast<PlanBase*>(opaque)->launch(static_cast<cudaStream_t>(stream));
        check(cudaGetLastError());
        return 0;
    }catch(const std::exception &exception) {
        error=exception.what();return -1;
    }
}
extern "C" int tk_silu_bwd_q_rebind(
    void*opaque,void*dy,void*half_gate,void*up,
    void*gate_row,void*gate_row_scale,void*up_row,void*up_row_scale,
    void*gate_col,void*gate_col_scale,void*up_col,void*up_col_scale) {
    using namespace fp8gemm::silu_bwd_quantize;
    try {
        static_cast<PlanBase*>(opaque)->rebind(
            dy,half_gate,up,
            gate_row,gate_row_scale,up_row,up_row_scale,
            gate_col,gate_col_scale,up_col,up_col_scale);
        return 0;
    }catch(const std::exception &exception) {
        error=exception.what();return -1;
    }
}

extern "C" void tk_silu_bwd_q_info(void*opaque,int*out) {
    static_cast<fp8gemm::silu_bwd_quantize::PlanBase*>(opaque)->info(out);
}

extern "C" void tk_silu_bwd_q_destroy(void*opaque) {
    delete static_cast<fp8gemm::silu_bwd_quantize::PlanBase*>(opaque);
}
