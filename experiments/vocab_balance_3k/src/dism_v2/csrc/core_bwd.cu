#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace dism_v2 {
template<int DV>
__global__ void backward_delta(const __nv_bfloat16* dout,const __nv_bfloat16* out,float* delta,int rows) {
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;
    float value=0;
    if(row<rows) {
        #pragma unroll
        for(int d=lane;d<DV;d+=32) {
            int64_t offset=int64_t(row)*DV+d;
            value=fmaf(__bfloat162float(dout[offset]),__bfloat162float(out[offset]),value);
        }
    }
    #pragma unroll
    for(int shift=16;shift>0;shift/=2) value+=__shfl_down_sync(0xffffffff,value,shift);
    if(lane==0 && row<rows) delta[row]=value;
}
void launch_backward_delta(const void* dout,const void* out,float* delta,int rows,int dv,cudaStream_t stream) {
    auto a=static_cast<const __nv_bfloat16*>(dout),b=static_cast<const __nv_bfloat16*>(out);
    switch(dv) {
        case 32: backward_delta<32><<<(rows+7)/8,256,0,stream>>>(a,b,delta,rows); break;
        case 64: backward_delta<64><<<(rows+7)/8,256,0,stream>>>(a,b,delta,rows); break;
        case 128: backward_delta<128><<<(rows+7)/8,256,0,stream>>>(a,b,delta,rows); break;
    }
}
} // namespace dism_v2
