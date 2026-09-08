#include "log_affine.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void probe(float* result) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=32001) return;
    float x=-i*0.001f;
    result[3*i]=dism_v2::tile_logadd2(x,0.f);
    result[3*i+1]=dism_v2::logadd2(x,0.f);
    float identity=dism_v2::tile_logadd2(-INFINITY,x);
    float zero=dism_v2::tile_logadd2(-INFINITY,-INFINITY);
    result[3*i+2]=(identity==x && zero==-INFINITY &&
        dism_v2::tile_logadd2(x,-INFINITY)==x)?0.f:1.f;
}
int main() {
    float* result;
    if(cudaMallocManaged(&result,32001*3*sizeof(float))!=cudaSuccess) return 2;
    probe<<<126,256>>>(result);
    if(cudaDeviceSynchronize()!=cudaSuccess) return 3;
    double max_approx=0,max_passing=0,max_formula=0;
    for(int i=0;i<32001;++i) {
        float x=-i*0.001f;
        double exact=std::log2(1+std::exp2(double(x)));
        double formula=1.81089463*(1-std::tanh(std::abs(double(x))*.34114549+.48232999));
        max_approx=std::fmax(max_approx,std::abs(result[3*i]-exact));
        max_passing=std::fmax(max_passing,std::abs(result[3*i+1]-exact));
        max_formula=std::fmax(max_formula,std::abs(result[3*i]-formula));
        if(result[3*i+2]!=0 || !std::isfinite(result[3*i])) return 4;
    }
    std::printf("approx_error=%.9g passing_error=%.9g formula_error=%.9g\n",max_approx,max_passing,max_formula);
    cudaFree(result);
    // MUFU.TANH is itself approximate: on sm120 its contribution relative to
    // the host tanh formula reaches 1.37e-5, not FP32 rounding alone.
    return max_approx<.00060 && max_passing<3e-7 && max_formula<2e-5?0:5;
}
