#include <cuda_runtime.h>
#include <cstdint>
namespace dism_v2 {
namespace scalar_bwd {
__global__ void reduce_tau(const float* partial,float* output,int batch,int heads,int count) {
    __shared__ float warps[8];
    int tid=threadIdx.x,lane=tid&31,warp=tid/32,head=blockIdx.x;
    float x=0;
    // Explicit batch loop avoids device 64-bit division helpers (SASS CALL).
    for(int b=0;b<batch;++b) {
        for(int64_t c=tid;c<count;c+=256)
            x+=partial[(int64_t(b)*heads+head)*count+c];
    }
    #pragma unroll
    for(int shift=16;shift>0;shift/=2) x+=__shfl_down_sync(0xffffffff,x,shift);
    if(lane==0) warps[warp]=x;
    __syncthreads();
    if(warp==0) {
        x=lane<8?warps[lane]:0.f;
        #pragma unroll
        for(int shift=16;shift>0;shift/=2) x+=__shfl_down_sync(0xffffffff,x,shift);
        if(lane==0) output[head]=x;
    }
}
}
void launch_tau_reduce(const float* partial,float* output,int batch,int heads,int count,cudaStream_t stream) {
    scalar_bwd::reduce_tau<<<heads,256,0,stream>>>(partial,output,batch,heads,count);
}
}
