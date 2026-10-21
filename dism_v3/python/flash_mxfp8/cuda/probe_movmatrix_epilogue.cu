#include "kittens.cuh"
#include "tma_ring_pipe.cuh"
#include "mxfp8_epilogue.cuh"

template<int C,bool Movmatrix>
__global__ void probe_output(const kittens::bf16 *x,kittens::fp8e4m3 *q,uint32_t *scales) {
    namespace kt=kittens;
    using ST=kt::st_fp8e4m3<16,C>;
    __shared__ ST output;
    kt::rt_bf<16,C> src;
    int lane=threadIdx.x,rr=lane/4,cc=2*(lane%4);
#pragma unroll
    for(int j=0;j<C/16;j++) {
#pragma unroll
        for(int p=0;p<4;p++)
            src.tiles[0][j].data[p]=*reinterpret_cast<const kt::bf16_2*>(
                x+(rr+8*(p%2))*C+j*16+cc+8*(p/2));
    }
    kt::mxfp8_scale<16,C> s;
    if constexpr(Movmatrix) {
        fp8gemm::MXOutputTransposed<16,C> d;
        fp8gemm::quantize_output_movmatrix(d,s,src);
        fp8gemm::store_output_movmatrix<1>(output,d);
    } else {
        kt::rt_fp8e4m3<16,C> d;
        kt::online_quantize_mxfp8(d,s,src);
        kt::warp::store(output,d);
    }
    fp8gemm::store_output_scales<16,C>(scales,s,0,0,16,C);
    __syncthreads();
    for(int i=lane;i<16*C;i+=32)
        q[i]=*ST::idx(output.data,{i/C,i%C});
}
extern "C" int probe_movmatrix(const void *x,void *q,void *s,int c,int mode,void *stream) {
#define CASE(C) case C: \
    if(mode)probe_output<C,true><<<1,32,0,(cudaStream_t)stream>>>((const kittens::bf16*)x,(kittens::fp8e4m3*)q,(uint32_t*)s); \
    else probe_output<C,false><<<1,32,0,(cudaStream_t)stream>>>((const kittens::bf16*)x,(kittens::fp8e4m3*)q,(uint32_t*)s);break
    switch(c) {CASE(32);CASE(64);CASE(128);CASE(256);default:return -1;}
#undef CASE
    return int(cudaGetLastError());
}
