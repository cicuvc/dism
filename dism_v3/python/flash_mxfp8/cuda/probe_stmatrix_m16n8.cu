#include <cuda_runtime.h>
#include <cstdint>
__global__ void stmatrix_m16n8_probe(uint8_t *out) {
    __shared__ __align__(16) uint8_t tile[128];
    int lane=threadIdx.x,r=lane/4,c=2*(lane%4);
    uint32_t word=(r*8+c)|((r*8+c+1)<<8)|(((r+8)*8+c)<<16)|(((r+8)*8+c+1)<<24);
    uint32_t addr=__cvta_generic_to_shared(tile+(lane%8)*16);
    asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared::cta.b8 [%0], {%1};"
                 ::"r"(addr),"r"(word):"memory");
    __syncthreads();
    for(int i=lane;i<128;i+=32)out[i]=tile[i];
}
extern "C" int stmatrix_probe(uint8_t *out,cudaStream_t stream) {
    stmatrix_m16n8_probe<<<1,32,0,stream>>>(out);
    return int(cudaGetLastError());
}
