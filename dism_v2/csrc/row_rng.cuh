#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace dism_v2 {
// Philox4x32-10: counter=(offset/4 low, high, logical row low, high).
// Offset counts 32-bit words in each independent row subsequence.
__device__ __forceinline__ uint32_t row_bits(uint64_t seed, uint64_t offset, uint64_t row) {
    uint4 c=make_uint4(uint32_t(offset/4),uint32_t((offset/4)>>32),uint32_t(row),uint32_t(row>>32));
    uint32_t k0=uint32_t(seed),k1=uint32_t(seed>>32);
    #pragma unroll
    for(int round=0;round<10;++round) {
        uint32_t lo0=0xD2511F53u*c.x,hi0=__umulhi(0xD2511F53u,c.x);
        uint32_t lo1=0xCD9E8D57u*c.z,hi1=__umulhi(0xCD9E8D57u,c.z);
        c=make_uint4(hi1^c.y^k0,lo1,hi0^c.w^k1,lo0);
        k0+=0x9E3779B9u; k1+=0xBB67AE85u;
    }
    return c.x;
}
__device__ __forceinline__ bool row_hard(uint64_t seed,uint64_t offset,uint64_t row,float probability) {
    if(probability==0.f) return false;
    if(probability==1.f) return true;
    return float(row_bits(seed,offset,row)>>8)*0x1p-24f<probability;
}
} // namespace dism_v2
