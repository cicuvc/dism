// BF16 elementwise backward for y = silu(gate) * up.
// Inputs are dY, half_gate=0.5*gate, and up. Every intermediate remains BF16;
// packed tanh/HFMA2/HMUL2 instructions process two values per thread.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "silu_bwd_common.cuh"

namespace fp8gemm::silu_bwd {
static thread_local std::string error;

static void check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}

__global__ __launch_bounds__(256) void silu_bwd_bf16_kernel(
    const uint4 *__restrict__ dy4,
    const uint4 *__restrict__ half_gate4,
    const uint4 *__restrict__ up4,
    uint4 *__restrict__ dgate4,
    uint4 *__restrict__ dup4,
    int64_t vectors,
    int64_t elements) {
    int64_t vector=blockIdx.x*blockDim.x+threadIdx.x;
    int64_t stride=int64_t(gridDim.x)*blockDim.x;
    for(;vector<vectors;vector+=stride) {
        uint4 dy=dy4[vector],half_gate=half_gate4[vector],up=up4[vector];
        uint4 dgate,dup;
        PairGrad p0=backward_pair(dy.x,half_gate.x,up.x);
        PairGrad p1=backward_pair(dy.y,half_gate.y,up.y);
        PairGrad p2=backward_pair(dy.z,half_gate.z,up.z);
        PairGrad p3=backward_pair(dy.w,half_gate.w,up.w);
        dgate={p0.gate,p1.gate,p2.gate,p3.gate};
        dup={p0.up,p1.up,p2.up,p3.up};
        dgate4[vector]=dgate;
        dup4[vector]=dup;
    }

    int64_t tail=elements-vectors*8;
    if(blockIdx.x==0 && threadIdx.x<tail) {
        int64_t element=vectors*8+threadIdx.x;
        auto *dyh=reinterpret_cast<const uint16_t*>(dy4);
        auto *gh=reinterpret_cast<const uint16_t*>(half_gate4);
        auto *uh=reinterpret_cast<const uint16_t*>(up4);
        auto *dgh=reinterpret_cast<uint16_t*>(dgate4);
        auto *duh=reinterpret_cast<uint16_t*>(dup4);
        PairGrad grad=backward_pair(dyh[element],gh[element],uh[element]);
        dgh[element]=uint16_t(grad.gate);
        duh[element]=uint16_t(grad.up);
    }
}

struct Plan {
    const uint4 *dy,*half_gate,*up;
    uint4 *dgate,*dup;
    int64_t elements,vectors;
    int grid;
    Plan(void*dy_,void*half_gate_,void*up_,void*dgate_,void*dup_,int64_t elements_):
        dy(static_cast<const uint4*>(dy_)),
        half_gate(static_cast<const uint4*>(half_gate_)),
        up(static_cast<const uint4*>(up_)),
        dgate(static_cast<uint4*>(dgate_)),
        dup(static_cast<uint4*>(dup_)),
        elements(elements_),
        vectors(elements_/8) {
        int device;check(cudaGetDevice(&device));
        cudaDeviceProp properties;check(cudaGetDeviceProperties(&properties,device));
        int64_t work=(vectors+255)/256;
        grid=int(std::min<int64_t>(work,properties.multiProcessorCount*8));
        if(grid<1)grid=1;
    }
    void launch(cudaStream_t stream) {
        silu_bwd_bf16_kernel<<<grid,256,0,stream>>>(
            dy,half_gate,up,dgate,dup,vectors,elements);
    }
};

}

extern "C" void *tk_silu_bwd_create(
    void*dy,void*half_gate,void*up,void*dgate,void*dup,int64_t elements) {
    using namespace fp8gemm::silu_bwd;
    try {
        if(!dy||!half_gate||!up||!dgate||!dup||elements<=0)
            throw std::runtime_error("Invalid arguments");
        auto *plan=new Plan(dy,half_gate,up,dgate,dup,elements);
        auto error=cudaGetLastError();
        if(error!=cudaSuccess) {
            delete plan;
            throw std::runtime_error(cudaGetErrorString(error));
        }
        return plan;
    }catch(const std::exception &exception) {
        fp8gemm::silu_bwd::error=exception.what();
        return nullptr;
    }
}

extern "C" const char *tk_silu_bwd_error() {
    return fp8gemm::silu_bwd::error.c_str();
}

extern "C" int tk_silu_bwd_launch(void*opaque,void*stream) {
    using namespace fp8gemm::silu_bwd;
    try {
        static_cast<Plan*>(opaque)->launch(static_cast<cudaStream_t>(stream));
        auto error=cudaGetLastError();
        if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
        return 0;
    }catch(const std::exception &exception) {
        fp8gemm::silu_bwd::error=exception.what();
        return -1;
    }
}

extern "C" void tk_silu_bwd_info(void*opaque,int*out) {
    auto *plan=static_cast<fp8gemm::silu_bwd::Plan*>(opaque);
    out[0]=plan->grid;
    out[1]=256;
}

extern "C" void tk_silu_bwd_destroy(void*opaque) {
    delete static_cast<fp8gemm::silu_bwd::Plan*>(opaque);
}
