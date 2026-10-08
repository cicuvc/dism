#pragma once

#include <array>

#include <cuda.h>
#include <cuda_bf16.h>


template <int N_HEADDIM, int N_KEYDIM = N_HEADDIM>
struct BaselineNoPEAttnStateImpl {
    struct FwdPreprocessArgs{
        size_t Batch, Seqlen, Head;
        half *Q, *K;
        nv_bfloat16 *V, *O;
        float *VBuffer, *HBuffer, *RcpTau, *FwdMax;
    };

    struct BwdPreprocessArgs{
        size_t Batch, Seqlen, Head;
        half *Q, *K;
        nv_bfloat16 *V, *dO, *dV;
        float *VBuffer, *HBuffer, *RcpTau, *FwdMax;
    };

    static std::array<size_t, 4> getVHBufferShape(size_t batch, size_t head, size_t seqlen);
    static void invokeFwdPreprocess(const FwdPreprocessArgs&);
    static void invokeFwd(const FwdPreprocessArgs&);
    static void invokeBwdPreprocess(const BwdPreprocessArgs&);
};