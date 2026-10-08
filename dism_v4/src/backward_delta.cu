#include "summary/primitives.cuh"
#include "dimensions.cuh"

#include "variant.cuh"
namespace DISM_VARIANT {

namespace {
constexpr int Channels=DISM_KC_HEAD_DIM;
template<class Output>
__global__ void row_delta(const Output* out,const __nv_bfloat16* dout,float* delta,
                           int n,int heads,int rows) {
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;
    if (row>=rows) return;
    float sum=0.f;
#pragma unroll
    for (int c=lane;c<Channels;c+=32) {
        int64_t off=int64_t(row)*Channels+c;
        sum=fmaf(float(out[off]),float(dout[off]),sum);
    }
#pragma unroll
    for (int shift=16;shift>0;shift/=2) sum+=__shfl_down_sync(0xffffffff,sum,shift);
    int head=row%heads,query=(row/heads)%n,batch=row/(heads*n);
    if (lane==0) delta[(int64_t(batch)*heads+head)*n+query]=sum;
}
}

const void* backward_delta_address_0() { return reinterpret_cast<const void*>(row_delta<float>); }
const void* backward_delta_address_1() { return reinterpret_cast<const void*>(row_delta<kt::bf16>); }
} // namespace DISM_VARIANT
