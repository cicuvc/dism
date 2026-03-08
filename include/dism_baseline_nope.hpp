#pragma once

#include <cuda.h>
#include <cuda_bf16.h>

namespace {

template <int N_HEADDIM, int N_KEYDIM = N_HEADDIM>
struct BaselineNoPEAttnStateImpl {
    struct FwdPreprocessArgs{
        int Batch, Head, Seq;
        nv_bfloat16 *Q, *K, *V;
        half *QCache, *KCache;
        float *VBuffer, *HBuffer;
    };

    static void invokeFwdPreprocess(const FwdPreprocessArgs&);
};

} // namespace 