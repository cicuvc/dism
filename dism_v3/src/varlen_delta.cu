#include "summary/primitives.cuh"
#include "varlen/layout.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace dism_varlen {
template<class Output>
__global__ void varlen_delta_kernel(const Output* out,const kt::bf16* dout,float* delta,
        const int64_t* table,int sequences,int heads,int tokens) {
    int lane=threadIdx.x%32;
    int local=blockIdx.x*8+threadIdx.x/32;
    for (int s=blockIdx.y;s<sequences;s+=gridDim.y) {
        const int64_t* row=table+s*Fields;
        int n=row[Length];
        if (local>=int64_t(n)*heads) continue;
        float sum=0.f;
#pragma unroll
        for (int c=lane;c<ActiveConfig::DV;c+=32) {
            int64_t offset=(row[Begin]*heads+local)*ActiveConfig::DV+c;
            sum=fmaf(float(out[offset]),float(dout[offset]),sum);
        }
#pragma unroll
        for (int shift=16;shift>0;shift/=2) sum+=__shfl_down_sync(0xffffffff,sum,shift);
        int head=local%heads,query=local/heads;
        if (lane==0) delta[int64_t(head)*tokens+row[Begin]+query]=sum;
    }
}
} // namespace dism_varlen

const void* varlen_delta_address_0() { return reinterpret_cast<const void*>(dism_varlen::varlen_delta_kernel<float>); }
const void* varlen_delta_address_1() { return reinterpret_cast<const void*>(dism_varlen::varlen_delta_kernel<kt::bf16>); }
} // namespace DISM_VARIANT
