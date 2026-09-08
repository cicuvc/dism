#pragma once
#include <cuda_runtime.h>
namespace dism_v2::embedding_bwd {
// Canonical full branch: O=softmax(s X K^T) V; opposite branch: LSE(s Y V^T).
struct Args {
    const void *x,*y,*key,*value,*out;
    const float *u,*lx,*ly,*lambda;
    void* packed_u;
    float *delta,*dx,*dy,*dkey,*dvalue;
    float *lx2,*ly2;
    int batch,heads,n,voc;
    float scale,scale2;
};
void launch(Args,int,bool,cudaStream_t);
}
